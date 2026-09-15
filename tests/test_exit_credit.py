"""Position-only exit preferences, quote accounting, and PPO sign regressions."""
import copy
import json

import numpy as np
import pytest
import torch

from ai_trader.grpo.entry_pattern import DEFAULT_ENTRY_PATTERN, pattern_actor_signals
from ai_trader.grpo.exit_credit import exit_preference
from ai_trader.grpo.environments.execution import ExecutionSimulator
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.gpu_tuning import sample_rollouts
from ai_trader.grpo.update_diagnostic import prepare_rollouts, BUNDLE_FORMAT, BUNDLE_VERSION
from ai_trader.grpo.diagnose_entry_credit import analyze_bundle
from ai_trader.grpo.evaluation import _additional_metrics, evaluation_signature, compatible_resume_best
from test_scalping_accounting import ReplayFixture
from test_grpo_memory_batches import RecordingPolicy
from test_short_horizon_net import actor_episode

CFG = dict(DEFAULT_ENTRY_PATTERN)


@pytest.mark.parametrize('age,net,desired,reason', [
    (.999, 10, 0, 'before_target_window'),
    (1., 10, 2, 'net_profit_available'),
    (4.999, -10, 0, 'await_net_profit'),
    (3., 0, 0, 'await_net_profit'),
    (3., None, None, 'quote_unavailable'),
    (5., -10, 2, 'horizon_elapsed'),
    (300., -10, 2, 'horizon_elapsed'),
    (5., None, 2, 'horizon_elapsed'),
])
def test_explicit_exit_preferences(age, net, desired, reason):
    assert exit_preference(age, net, CFG) == (desired, reason)


def test_quote_uses_remaining_depth_fees_slippage_and_ticks_without_mutation():
    sim = ExecutionSimulator({'slippage_bps': 10, 'tick_size': .01})
    snap = sim.make_snapshot(2, 105, bids=[(103, 5), (90, 5)], asks=[(106, 20)], quote_timestamp=2)
    sim.process(snap)
    before = copy.deepcopy(sim.__dict__)
    proceeds = sim.quoted_sell_proceeds(10, .002)
    expected = (5 * sim.execution_price(103, 'sell') + 5 * sim.execution_price(90, 'sell')) * .998
    assert proceeds == pytest.approx(expected)
    assert proceeds - 1000 < 0  # Last price and top bid alone would wrongly imply profit.
    assert sim.__dict__ == before
    sim._remaining['sell'][103] = 4
    assert sim.quoted_sell_proceeds(10, .002) is None
    sim.process(sim.make_snapshot(4, 105, bids=[(103, 10)], asks=[(106, 20)], quote_timestamp=2))
    assert sim.quoted_sell_proceeds(1, .002) is None


class ExitFixture(ReplayFixture):
    def _sample_episode_start(self, max_attempts=100):
        result = super()._sample_episode_start(max_attempts)
        self._episode_entry_signal = np.ones(len(self.input_prices), bool)
        self._entry_target_times = self.input_seconds + 9 * 3600
        self._entry_target_prices = self.input_prices
        return result


def trajectory(mode, actions, latency=0):
    env = ExitFixture([100] * 12, entry_pattern_config={**CFG, 'target_mode': mode},
                      execution_action_mask=True, max_stages=1, initial_cash=1000,
                      transaction_cost_rate=.001, sell_tax_rate=.0018,
                      execution_config={'order_latency_ms': latency, 'slippage_bps': 2, 'spread_bps': 10})
    traces = []
    observation, info = env.reset()
    for action in actions:
        before = observation.copy()
        observation, reward, done, truncated, info = env.step(action)
        traces.append((before, reward, info['fills'], info['action_mask'].copy()))
        if done or truncated:
            break
    assert done or truncated
    return env, traces, info['episode']


def test_fixed_actions_preserve_observations_nav_masks_and_execution_and_do_not_force_5s_exit():
    actions = [1] + [0] * 10
    old, a, ep_old = trajectory('short_horizon_net', actions)
    new, b, ep = trajectory('short_horizon_trade', actions)
    assert len(a) == len(b)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:3] == right[1:3]
        np.testing.assert_array_equal(left[3], right[3])
    assert ep['net_return'] == ep_old['net_return']
    assert ep['realized_net_pnl'] == ep_old['realized_net_pnl']
    assert sum(row[1] for row in b) == pytest.approx(ep['net_return'])
    assert ep['exit_episode_end_round_trip_count'] == 1
    assert ep['quantity_weighted_holding_time'] > 5
    report = ep['entry_pattern']
    assert not report['exit_policy_mask'][0]  # flat BUY
    assert not report['exit_policy_mask'][-1]  # controller owns the terminal exit
    assert report['exit_target_summary']['overdue_hold_count'] > 0
    for row in report['exit_decisions']:
        if row['age_seconds'] >= 5:
            assert row['target'] == -1 and row['desired_action'] == 2
    json.dumps(report, allow_nan=False)
    summary = _additional_metrics([ep])['entry_pattern']['exit_target_summary']
    assert summary['overdue_hold_count'] == report['exit_target_summary']['overdue_hold_count']
    old.close()
    new.close()


def test_deadline_sell_is_positive_even_with_realized_loss_and_delayed_fill():
    env, _, ep = trajectory('short_horizon_trade', [1] + [0] * 6 + [2] + [0] * 3, latency=1500)
    rows = ep['entry_pattern']['exit_decisions']
    sell = next(row for row in rows if row['action'] == 2)
    assert sell['age_seconds'] == 5
    assert sell['quoted_net_pnl'] < 0
    assert sell['target'] == 1 and sell['reason'] == 'horizon_elapsed'
    assert ep['realized_net_pnl'] < 0
    assert ep['quantity_weighted_holding_time'] > 5  # real latency remains in force
    env.close()


def make_actor_episode(policy, action, target, coef=1.):
    ep = actor_episode(policy, target)
    ep['actions'][0] = action
    with torch.no_grad():
        logs, _, _ = policy.evaluate_actions(torch.from_numpy(ep['states']), torch.from_numpy(ep['actions']))
    ep['log_probs'] = logs.numpy()
    ep['metadata']['entry_pattern'] = {
        'config': {**CFG, 'policy_coef': coef}, 'policy_credit': [0.] * 8,
        'exit_policy_credit': [target * coef] + [0.] * 7,
        'exit_policy_mask': [True] + [False] * 7,
        'exit_decisions': [{'decision_index': 0, 'action': action, 'desired_action': action if target > 0 else 2-action,
                            'entry_order_id': 1, 'age_seconds': 5., 'quoted_net_pnl': -10., 'target': target,
                            'reason': 'horizon_elapsed'}]}
    return ep


@pytest.mark.parametrize('action,target', [(2, 1.), (0, -1.), (0, 1.), (2, -1.)])
def test_exit_actor_update_overrides_opposing_gae_without_changing_critic_target(action, target):
    torch.set_num_threads(1)
    policy = RecordingPolicy()
    policy.actor.bias.requires_grad_(False)
    ep = make_actor_episode(policy, action, target)
    trainer = GRPOTrainer(policy, object(), batch_size=8, num_epochs=1, use_gae=True,
                          gamma=1., lambda_gae=1., group_advantage_coef=0.,
                          value_coef=0., entropy_coef=0., learning_rate=.001)
    before = policy.actor.weight.detach().clone()
    trainer.update_policy([ep], [np.zeros(8, np.float32)])
    row = trainer.last_entry_credit_report['holding_decisions'][0]
    assert row['normalized_advantage'] * target < -1
    assert row['actor_base_advantage'] == 0
    assert row['initial_actor_signal'] == target
    assert row['return_target'] == ep['rewards'][0]
    assert (policy.actor.weight[action, 0] - before[action, 0]).item() * target > 0
    with torch.no_grad():
        logs, _, _ = policy.evaluate_actions(torch.from_numpy(ep['states']), torch.from_numpy(ep['actions']))
    assert (logs[0].item() - ep['log_probs'][0]) * target > 0


def test_only_marked_position_decisions_replace_gae_and_zero_coef_restores_nav():
    p = RecordingPolicy()
    ep = make_actor_episode(p, 0, -1.)
    original = np.arange(8, dtype=np.float32)
    base, credit, local = pattern_actor_signals([ep], original)
    assert local.tolist() == [True] + [False] * 7
    np.testing.assert_array_equal(base[1:], original[1:])
    assert credit[0] == -1
    disabled = make_actor_episode(p, 0, -1., coef=0)
    base, credit, local = pattern_actor_signals([disabled], original)
    np.testing.assert_array_equal(base, original)
    assert not credit.any() and not local.any()
    del ep['metadata']['entry_pattern']['exit_policy_mask']
    with pytest.raises(ValueError, match='collect new rollouts'):
        pattern_actor_signals([ep], original)


def test_capture_offline_and_gpu_trials_preserve_position_mask_and_signals():
    trainer = GRPOTrainer(RecordingPolicy(), object(), gamma=1., lambda_gae=1., group_advantage_coef=0.)
    ep = make_actor_episode(trainer.policy, 2, 1.)
    saved, adv = prepare_rollouts([ep], [np.zeros(8, np.float32)], action_dim=3, masked=False)
    report = analyze_bundle({'format': BUNDLE_FORMAT, 'format_version': BUNDLE_VERSION,
        'trainer_hyperparameters': {k: getattr(trainer, k) for k in ('gamma', 'lambda_gae', 'use_gae', 'group_advantage_coef')},
        'episodes': saved, 'advantages': adv})['report']
    assert report['holding_decisions'][0]['initial_actor_signal'] == 1.
    sampled, _ = sample_rollouts([ep], [np.zeros(8, np.float32)], 2)
    assert sampled[0]['metadata']['entry_pattern']['exit_policy_mask'] == [True, False]
    assert pattern_actor_signals(sampled, np.array([-4., 1.]))[1][0] == 1


def test_old_short_horizon_checkpoint_remains_old_and_cannot_supply_new_best():
    from test_profit_experiment_config import legacy_best_checkpoint
    checkpoint, signature = legacy_best_checkpoint()
    config = checkpoint['extra_state']['training_config']
    config['entry_pattern_config'] = {**CFG, 'target_mode': 'short_horizon_net'}
    saved = evaluation_signature(config, signature['date_splits'], signature['observation_schema'])
    checkpoint['extra_state']['evaluation_signature'] = saved
    assert compatible_resume_best(checkpoint, checkpoint, saved)
    new = evaluation_signature({**config, 'entry_pattern_config': CFG}, signature['date_splits'], signature['observation_schema'])
    assert not compatible_resume_best(checkpoint, checkpoint, new)
