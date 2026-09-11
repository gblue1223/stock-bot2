"""Reward-mixture controls and frozen-rollout diagnostics need no extra inference."""
import copy

import numpy as np
import pytest
import torch

from ai_trader.grpo.grpo import GRPOTrainer
from test_grpo_memory_batches import RecordingPolicy, make_episode


STAGE_SCHEMA = {'stage_fields': ['is_active', 'profit_rate', 'holding_fraction']}


@pytest.mark.parametrize('coefficient', [-1, np.nan, np.inf, -np.inf, True, '1'])
def test_invalid_group_coefficient_is_rejected(coefficient):
    with pytest.raises(ValueError, match='group_advantage_coef'):
        GRPOTrainer(RecordingPolicy(), object(), group_advantage_coef=coefficient)


def test_disabling_both_reward_learning_sources_is_rejected():
    with pytest.raises(ValueError, match='requires use_gae=True'):
        GRPOTrainer(RecordingPolicy(), object(), use_gae=False, group_advantage_coef=0.)


def test_zero_coefficient_ignores_group_rank_without_changing_critic_or_extra_forwards():
    torch.set_num_threads(1)
    left = RecordingPolicy()
    right = copy.deepcopy(left)
    episode, values = make_episode(left, 6)
    episode['values'] = values
    trainers = [GRPOTrainer(policy, object(), batch_size=6, num_epochs=1, group_advantage_coef=0.)
                for policy in (left, right)]
    outputs = []
    for trainer, advantages in zip(trainers, (np.zeros(6), np.arange(6) * 100.)):
        trainer.policy.forward_batches.clear()
        np.random.seed(53)
        outputs.append(trainer.update_policy([episode], [advantages]))
        assert trainer.policy.forward_batches == [(6, True)]
    for name, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[name])
    assert outputs[0]['value_loss'] == outputs[1]['value_loss']
    assert outputs[1]['learning_signal/all/group_component_std'] == 0.
    assert outputs[1]['learning_signal/position_available'] == 0
    assert outputs[1]['learning_signal/position/unknown/sample_count'] == 6


def test_group_coefficient_roundtrips_and_legacy_resume_restores_one(tmp_path):
    trainer = GRPOTrainer(RecordingPolicy(), object(), group_advantage_coef=.25)
    checkpoint_path = tmp_path / 'checkpoint.pt'
    trainer.save_checkpoint(str(checkpoint_path), 4)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint['config']['group_advantage_coef'] == .25
    target = GRPOTrainer(RecordingPolicy(), object(), group_advantage_coef=0.)
    target.load_checkpoint(str(checkpoint_path))
    assert target.group_advantage_coef == .25
    del checkpoint['config']['group_advantage_coef']
    target.restore_training_progress(checkpoint)
    assert target.group_advantage_coef == 1.
    checkpoint['config']['group_advantage_coef'] = np.nan
    with pytest.raises(ValueError, match='group_advantage_coef'):
        target.restore_training_progress(checkpoint)


def test_action_position_breakdown_preserves_raw_and_normalized_signal():
    trainer = GRPOTrainer(RecordingPolicy(), object(), observation_schema=STAGE_SCHEMA)
    states = np.zeros((6, 1, 16), np.float32)
    states[3:, -1, -15] = 1.
    actions = np.array([0, 1, 0, 2, 0, 2])
    gae = np.array([-2., 1., 3., -1., 0., 5.])
    group = np.array([.1, -.1, .2, -.2, .3, -.3])
    before = gae + group
    after = (before - before.mean()) / (before.std() + 1e-8)
    targets = np.arange(1., 7.)
    metrics = trainer._learning_signal_metrics(states, actions, gae, group, before, after,
                                               targets, np.zeros(6))
    buy = metrics['action']['buy']
    assert buy['sample_count'] == 1
    assert buy['raw_gae_mean'] == 1.
    assert buy['group_component_mean'] == -.1
    assert buy['pre_normalized_advantage_mean'] == .9
    assert buy['normalized_advantage_mean'] == pytest.approx(after[1])
    assert buy['critic_target_error_mae'] == 2.
    assert buy['critic_explained_variance_available'] == 0
    assert metrics['position']['holding']['sample_count'] == 3
    assert metrics['action_position']['sell']['holding']['sample_count'] == 2
    assert metrics['action_position']['sell']['flat']['available'] == 0
    assert metrics['all']['critic_explained_variance'] == pytest.approx(0.)
    assert metrics['all']['group_to_gae_std_ratio'] == pytest.approx(group.std() / gae.std())
    assert all(np.isfinite(value) for _, value in trainer._numeric_metric_leaves(metrics))


def test_constant_or_unavailable_gae_has_explicit_ratio_availability():
    trainer = GRPOTrainer(RecordingPolicy(), object())
    for gae in (None, np.ones(3)):
        metrics = trainer._learning_signal_metrics(np.zeros((3, 1, 2)), np.array([0, 1, 2]),
                                                   gae, np.ones(3), np.ones(3), np.zeros(3),
                                                   np.zeros(3), None)
        assert metrics['all']['group_to_gae_std_ratio_available'] == 0
        assert metrics['all']['group_to_gae_std_ratio'] == 0.
        assert metrics['all']['critic_sample_count'] == 0


def test_nested_exit_diagnostics_and_update_metrics_reach_tensorboard():
    policy = RecordingPolicy()
    episodes = [make_episode(policy, 4)[0], make_episode(policy, 4)[0]]
    episodes[0]['metadata']['exit_reasons'] = {'policy': {'round_trips': 2, 'net_pnl': -3.}}
    episodes[1]['metadata']['exit_reasons'] = {'policy': {'round_trips': 0, 'net_pnl': 0.}}
    episodes[0]['metadata']['reason_label'] = 'policy'

    class Writer:
        def __init__(self):
            self.values = {}

        def add_scalar(self, name, value, iteration):
            self.values[name] = value

    trainer = GRPOTrainer(policy, object())
    trainer.writer = Writer()
    trainer._log_metrics(1, episodes, {0: episodes}, {'nested': {'count': 3}})
    values = trainer.writer.values
    assert values['train/episode_diagnostics/exit_reasons/policy/round_trips/mean'] == 1.
    assert values['train/episode_diagnostics/exit_reasons/policy/net_pnl/mean'] == -1.5
    assert values['train/episode_diagnostics/exit_reasons/policy/net_pnl/available_episodes'] == 2
    assert values['train/nested/count'] == 3
    assert not any('reason_label' in name for name in values)


def test_cost_aware_delayed_returns_move_entry_probability_in_opposite_directions():
    """One deterministic update credits entry for the later net liquidation result.

    This is a small learning-signal regression test, not evidence that historical
    market observations identify profitable trading opportunities.
    """
    class ContextPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(4, 3))
            self.values = torch.nn.Parameter(torch.zeros(4))
            self.forward_count = 0

        def evaluate_actions(self, states, actions):
            self.forward_count += 1
            indices = (states[:, -1, 0] > 0).long() * 2 + states[:, -1, -15].long()
            distribution = torch.distributions.Categorical(logits=self.logits[indices])
            return distribution.log_prob(actions), distribution.entropy(), self.values[indices]

    policy = ContextPolicy()
    episodes = []
    # Costs are debited at entry; proceeds arrive one decision later. Holding
    # cash earns zero in both contexts. Net entry returns are +0.3 and -0.3.
    for context, exit_return in ((1., .4), (-1., -.2)):
        for enter in (False, True):
            states = np.zeros((2, 1, 16), np.float32)
            states[:, :, 0] = context
            states[1, -1, -15] = int(enter)
            actions = np.array([1, 2] if enter else [0, 0])
            with torch.no_grad():
                logs, _, values = policy.evaluate_actions(torch.from_numpy(states), torch.from_numpy(actions))
            episodes.append({'states': states, 'actions': actions, 'log_probs': logs.numpy(),
                             'values': values.numpy(), 'dones': np.array([False, True]),
                             'rewards': np.array([-.1, exit_return] if enter else [0., 0.], np.float32)})
    trainer = GRPOTrainer(policy, object(), gamma=1., lambda_gae=1., group_advantage_coef=0.,
                          learning_rate=.05, entropy_coef=0., value_coef=0., batch_size=8,
                          num_epochs=1, observation_schema=STAGE_SCHEMA)
    policy.forward_count = 0
    before = policy.logits.softmax(-1).detach().clone()
    metrics = trainer.update_policy(episodes, [np.zeros(2)] * len(episodes))
    after = policy.logits.softmax(-1).detach()
    assert metrics['optimizer_steps'] == 1 and policy.forward_count == 1
    assert after[2, 1] > before[2, 1]  # Buy after a net-positive delayed result.
    assert after[0, 1] < before[0, 1]  # Avoid the cost-inclusive losing entry.
    assert metrics['learning_signal/action/buy/raw_gae_positive_fraction'] == .5
    assert metrics['learning_signal/action_position/buy/flat/sample_count'] == 2
