"""Training budget and imported-policy selection must preserve return optimization."""

import numpy as np
import pytest
import torch

from ai_trader.grpo.grpo import GRPOTrainer


class ShortEpisodeTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_sizes = []

    def collect_rollouts(self, num_episodes):
        self.rollout_sizes.append(num_episodes)
        self.total_timesteps += 2 * num_episodes
        return [
            {'rewards': np.zeros(2, dtype=np.float32),
             'metadata': {'episode_reward': 0.0, 'episode_steps': 2,
                          'initial_market_indicators': [0, 0, 0]}}
            for _ in range(num_episodes)
        ]

    def update_policy(self, episodes, advantages):
        with torch.no_grad():
            self.policy.weight.sub_(1)
        self.num_updates += 1
        return {'policy_loss': 0.0}


def make_trainer(**kwargs):
    policy = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.weight.fill_(5)
    return ShortEpisodeTrainer(policy, object(), episodes_per_group=2, num_groups=1, **kwargs)


@pytest.mark.parametrize('saved_timesteps', [0, 100])
def test_short_episodes_do_not_end_before_actual_timestep_budget(tmp_path, saved_timesteps):
    evaluations = []
    trainer = make_trainer(
        evaluation_interval=100,
        evaluation_callback=lambda current: evaluations.append(float(current.weight.item()))
        or {'mean_net_return': float(current.weight.item())},
    )
    trainer.total_timesteps = saved_timesteps
    summaries = []
    metrics = trainer.train(
        total_episodes=2,  # Initial estimate assumed five steps; actual episodes have two.
        max_timesteps=saved_timesteps + 10,
        checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'),
        on_iteration_end=lambda iteration, summary: summaries.append(summary),
    )
    assert trainer.rollout_sizes == [2, 2, 1]
    assert metrics == {'total_timesteps': saved_timesteps + 10, 'num_updates': 3}
    assert evaluations == [2.0]  # Evaluate when the real budget ends, not the old estimate.
    assert summaries[-1]['total_iterations'] == summaries[-1]['iteration'] == 3
    checkpoint = torch.load(tmp_path / 'checkpoint_iter3.pt', weights_only=True)
    assert checkpoint['total_timesteps'] == saved_timesteps + 10


def test_episode_count_still_limits_runs_without_timestep_budget():
    trainer = make_trainer()
    metrics = trainer.train(total_episodes=3)
    assert trainer.rollout_sizes == [2, 1]
    assert metrics == {'total_timesteps': 6, 'num_updates': 2}


def test_fine_tuning_can_keep_better_imported_weights_without_old_best(tmp_path):
    evaluations = []
    trainer = make_trainer(
        evaluation_callback=lambda current: evaluations.append(float(current.weight.item()))
        or {'mean_net_return': float(current.weight.item())},
    )
    unrelated_best = {'policy_state_dict': {'weight': torch.tensor([[99.0]])}}
    trainer.train(
        total_episodes=2,
        preserve_initial_policy=True,
        resume=False,
        resume_best_checkpoints=[unrelated_best],
        checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'),
    )
    selected = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    assert evaluations == [5.0, 4.0]
    assert selected['policy_state_dict']['weight'].item() == 5.0
    assert selected['extra_state']['best_validation_return'] == 5.0
    assert selected['total_timesteps'] == selected['num_updates'] == selected['iteration'] == 0
    assert trainer.policy.weight.item() == 4.0  # Best selection does not interrupt updates.


def test_no_progress_fails_instead_of_looping_forever():
    trainer = make_trainer()
    original_collect = trainer.collect_rollouts

    def no_progress(num_episodes):
        episodes = original_collect(num_episodes)
        trainer.total_timesteps = 0
        return episodes

    trainer.collect_rollouts = no_progress
    with pytest.raises(RuntimeError, match='no timestep progress'):
        trainer.train(total_episodes=2, max_timesteps=10)


def test_undiscounted_returns_keep_profitable_delayed_cashflow_positive():
    rewards = np.zeros(300, dtype=np.float32)
    rewards[0], rewards[-1] = -0.2, 1.0
    discounted = make_trainer(gamma=0.99)._compute_returns(rewards)
    undiscounted = make_trainer(gamma=1.0)._compute_returns(rewards)
    assert discounted[0] < 0  # +0.8% net can have negative discounted value.
    assert undiscounted[0] == pytest.approx(0.8)
    np.testing.assert_allclose(undiscounted, np.cumsum(rewards[::-1])[::-1])


def liquidation_metrics(net_return, *, complete=True):
    return {'mean_net_return': net_return,
            'incomplete_liquidation_episodes': 0 if complete else 1,
            'max_open_quantity': 0 if complete else 9}


def test_residual_inventory_cannot_outrank_realized_profit(tmp_path):
    evaluations = iter([liquidation_metrics(2.0), liquidation_metrics(20.0, complete=False)])
    trainer = make_trainer(selection_require_liquidation=True,
                           evaluation_callback=lambda current: next(evaluations))
    trainer.train(total_episodes=4, checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    selected = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    assert selected['policy_state_dict']['weight'].item() == 4.0
    assert selected['extra_state']['best_validation_return'] == 2.0
    assert selected['extra_state']['selection_require_liquidation'] is True
    assert trainer.policy.weight.item() == 3.0


def test_all_ineligible_preserves_diagnostics_and_never_uses_stale_best(tmp_path):
    trainer = make_trainer(selection_require_liquidation=True,
                           evaluation_callback=lambda current: liquidation_metrics(20.0, complete=False))
    stale_path = tmp_path / 'checkpoint_best.pt'
    torch.save({'stale': True}, stale_path)
    stale_bytes = stale_path.read_bytes()
    with pytest.raises(RuntimeError, match='No eligible validation checkpoint'):
        trainer.train(total_episodes=2, revert_to_best_patience=1,
                      checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    assert stale_path.read_bytes() == stale_bytes
    diagnostic = torch.load(tmp_path / 'checkpoint_iter1.pt', weights_only=True)
    assert diagnostic['policy_state_dict']['weight'].item() == 4.0
    assert diagnostic['total_timesteps'] == 4


@pytest.mark.parametrize('eligible_initial', [True, False])
def test_resume_revalidates_liquidation_for_both_source_and_old_best(tmp_path, eligible_initial):
    evaluated = []

    def evaluate(current):
        weight = float(current.weight.item())
        evaluated.append(weight)
        eligible = (weight == 5 and eligible_initial) or (weight == 99 and not eligible_initial)
        return liquidation_metrics(2.0 if eligible else 20.0, complete=eligible)

    trainer = make_trainer(selection_require_liquidation=True, evaluation_callback=evaluate)
    candidate = {'policy_state_dict': {'weight': torch.tensor([[99.0]])},
                 'observation_schema': None, 'iteration': 8,
                 'extra_state': {'best_validation_return': 999.0}}
    trainer.train(total_episodes=2, resume=True, resume_best_checkpoints=[candidate],
                  checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    selected = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    assert evaluated == [5.0, 99.0, 4.0]
    assert selected['policy_state_dict']['weight'].item() == (5.0 if eligible_initial else 99.0)
    assert selected['extra_state']['best_validation_return'] == 2.0
    assert trainer.policy.weight.item() == 4.0


def test_ineligible_imported_fine_tune_baseline_can_be_replaced_by_realized_profit(tmp_path):
    evaluations = iter([liquidation_metrics(20.0, complete=False), liquidation_metrics(2.0)])
    trainer = make_trainer(selection_require_liquidation=True,
                           evaluation_callback=lambda current: next(evaluations))
    trainer.train(total_episodes=2, preserve_initial_policy=True,
                  checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    selected = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    assert selected['policy_state_dict']['weight'].item() == 4.0
    assert selected['extra_state']['best_validation_return'] == 2.0


@pytest.mark.parametrize('diagnostics', [
    {}, {'incomplete_liquidation_episodes': 0}, {'max_open_quantity': 0},
    {'incomplete_liquidation_episodes': 0, 'max_open_quantity': float('nan')},
])
def test_strict_selection_requires_both_finite_liquidation_fields(diagnostics):
    trainer = make_trainer(selection_require_liquidation=True,
                           evaluation_callback=lambda current: {'mean_net_return': 20.0, **diagnostics})
    with pytest.raises(RuntimeError, match='No eligible validation checkpoint'):
        trainer.train(total_episodes=2, preserve_initial_policy=True)
