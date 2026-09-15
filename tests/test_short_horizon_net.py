"""Joint horizon/net labels and trade-local BUY actor regression coverage."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_trader.grpo.entry_pattern import DEFAULT_ENTRY_PATTERN, pattern_actor_signals
from ai_trader.grpo.environments.scalping_env_e2e import GRPOScalpingEnv
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.diagnose_entry_credit import analyze_bundle
from ai_trader.grpo.update_diagnostic import prepare_rollouts, BUNDLE_FORMAT, BUNDLE_VERSION
from ai_trader.grpo.gpu_tuning import sample_rollouts
from test_grpo_memory_batches import RecordingPolicy


CFG = {**DEFAULT_ENTRY_PATTERN, 'target_mode': 'short_horizon_net'}


def report_for(prices, *, times=None, pnl=5., fills=None, sold=None, active=False):
    env = object.__new__(GRPOScalpingEnv)
    env.entry_pattern_config = CFG
    env.episode_rewards = [0., 0.]
    env._pattern_fills = fills or [(7, 0., 100., 10)]
    env._entry_order_decisions = {7: 0}
    env._entry_target_times = np.asarray(times if times is not None else np.arange(len(prices)), dtype=float)
    env._entry_target_prices = np.asarray(prices, dtype=float)
    env.simulator = SimpleNamespace(orders=[SimpleNamespace(order_id=7, active=active)])
    quantity = sum(fill[3] for fill in env._pattern_fills)
    env.episode_trades = [{'entry_order_id': 7, 'quantity': quantity if sold is None else sold, 'net_pnl': pnl}]
    return env._entry_pattern_report()


@pytest.mark.parametrize('prices,pnl,expected', [
    ([99, 100, 99, 99, 99, 99, 200], 5., 1.),   # exact lower bound
    ([99, 99, 99, 99, 99, 100, 200], 5., 1.),   # exact upper bound
    ([99, 99, 99, 99, 99, 99, 200], 5., -1.),   # only a late recovery
    ([200, 99, 99, 99, 99, 99, 200], 5., -1.),  # only before and after window
    ([99, 100, 99, 99, 99, 99, 200], -5., -1.), # price hit, cost-inclusive loss
    ([99, 100, 99, 99, 99, 99, 200], 0., 0.),
    ([99, 99, 99, 99, 99, 99, 200], 0., -1.),
])
def test_joint_target_requires_both_observed_horizon_hit_and_order_net_profit(prices, pnl, expected):
    report = report_for(prices, pnl=pnl)
    assert report['policy_credit'] == [expected, 0.]
    assert report['orders'][0]['short_horizon_net_target'] == expected
    assert report['censored_quantity'] == 0
    assert report['labeled_buy_decisions'] == 1


@pytest.mark.parametrize('kwargs', [
    {'prices': [100, 101]},                                  # early hit, incomplete horizon
    {'prices': [100, 101], 'times': [0, 6]},                 # no observed row in window
    {'prices': [100] * 7, 'sold': 5},                       # partial exit
    {'prices': [100] * 7, 'active': True},                  # pending additional fill
    {'prices': [100] * 6, 'fills': [(7, 0., 100., 5), (7, .1, 100., 5)]},
])
def test_unknown_outcomes_never_become_positive_or_negative_labels(kwargs):
    report = report_for(**kwargs)
    assert report['policy_credit'] == [0., 0.]
    assert report['orders'][0]['short_horizon_net_target'] is None
    assert report['censored_quantity'] == 10
    assert report['labeled_buy_decisions'] == 0


def test_partial_fills_use_their_own_execution_prices_and_times():
    fills = [(7, 0., 100., 9), (7, 2., 102., 1)]
    report = report_for([99, 101, 200, 101, 101, 101, 101, 101], fills=fills)
    # The high price is too early for fill two, even though 90% of quantity hit.
    order = report['orders'][0]
    assert order['legacy_credit'] == pytest.approx(.8)
    assert order['short_horizon_success'] is False
    assert report['policy_credit'][0] == -1.


@pytest.mark.parametrize('mode', ['short_horizon_net', 'realized_net', 'legacy_price'])
@pytest.mark.parametrize('coef', [0., .5, 1.])
def test_buy_routing_keeps_legacy_and_nav_ablation_and_does_not_modify_inputs(mode, coef):
    ep = {'actions': np.array([0, 1, 2, 1, 1]), 'metadata': {'entry_pattern': {
        'config': {**CFG, 'target_mode': mode, 'policy_coef': coef},
        'policy_credit': [0, coef, 0, -coef, 0]}}}
    normalized = np.array([.4, -4.101, .7, 3., 2.], dtype=np.float32)
    original = normalized.copy()
    base, credit, local = pattern_actor_signals([ep], normalized)
    np.testing.assert_array_equal(normalized, original)
    if mode == 'short_horizon_net' and coef > 0:
        np.testing.assert_array_equal(local, [False, True, False, True, True])
        np.testing.assert_array_equal(base[[0, 2]], normalized[[0, 2]])
        np.testing.assert_array_equal((base + credit)[[1, 3, 4]], [coef, -coef, 0])
    else:
        assert not local.any()
        np.testing.assert_array_equal(base, normalized)


def actor_episode(policy, target):
    states = np.zeros((8, 2, 2), dtype=np.float32)
    states[0, :, 0] = 1.
    states[1:, :, 1] = 1.
    actions = np.array([1, 0, 2, 0, 2, 0, 2, 0])
    with torch.no_grad():
        logs, _, _ = policy.evaluate_actions(torch.from_numpy(states), torch.from_numpy(actions))
    rewards = np.zeros(8, np.float32)
    rewards[0] = -10 if target > 0 else 10
    return {'states': states, 'actions': actions, 'log_probs': logs.numpy(),
            'values': np.zeros(8, np.float32), 'rewards': rewards,
            'dones': np.arange(8) == 7, 'metadata': {'entry_pattern': {
                'config': CFG.copy(), 'policy_credit': [target] + [0.] * 7}}}


@pytest.mark.parametrize('target', [1., -1., 0.])
def test_actual_optimizer_follows_own_trade_even_when_nav_signal_opposes_it(target):
    torch.set_num_threads(1)
    policy = RecordingPolicy()
    # Isolate BUY state's actor gradient from other states' shared bias/critic.
    policy.actor.bias.requires_grad_(False)
    ep = actor_episode(policy, target)
    before = policy.actor.weight.detach().clone()
    trainer = GRPOTrainer(policy, object(), batch_size=8, num_epochs=1, use_gae=True,
                          gamma=1., lambda_gae=1., group_advantage_coef=0.,
                          value_coef=0., entropy_coef=0., learning_rate=.001)
    metrics = trainer.update_policy([ep], [np.zeros(8, np.float32)])
    summary = trainer.last_entry_credit_report['action_summary']['buy']
    assert summary['actor_base_advantage_mean'] == 0.
    assert summary['initial_actor_signal_mean'] == target
    assert summary['return_target_mean'] == ep['rewards'][0]
    assert metrics['entry_pattern/trade_local_buy_decisions'] == 1
    if target:
        assert summary['normalized_advantage_mean'] * target < -1
        assert (policy.actor.weight[1, 0] - before[1, 0]).item() * target > 0
        with torch.no_grad():
            logs, _, _ = policy.evaluate_actions(torch.from_numpy(ep['states']), torch.from_numpy(ep['actions']))
        assert (logs[0].item() - ep['log_probs'][0]) * target > 0
    else:
        torch.testing.assert_close(policy.actor.weight[:, 0], before[:, 0], rtol=0, atol=0)


def test_capture_offline_diagnostics_and_gpu_sample_preserve_new_actor_objective():
    trainer = GRPOTrainer(RecordingPolicy(), object(), gamma=1., lambda_gae=1., group_advantage_coef=0.)
    source = actor_episode(trainer.policy, 1.)
    advantages = [np.zeros(8, np.float32)]
    saved, groups = prepare_rollouts([source], advantages, action_dim=3, masked=False)
    assert saved[0]['metadata']['entry_pattern'] == source['metadata']['entry_pattern']
    bundle = {'format': BUNDLE_FORMAT, 'format_version': BUNDLE_VERSION,
              'trainer_hyperparameters': {name: getattr(trainer, name) for name in
                                         ('gamma', 'lambda_gae', 'use_gae', 'group_advantage_coef')},
              'episodes': saved, 'advantages': groups}
    offline = analyze_bundle(bundle)['report']
    assert offline['action_summary']['buy']['initial_actor_signal_mean'] == 1.
    assert offline['action_summary']['buy']['normalized_advantage_mean'] < -1.
    sampled, _ = sample_rollouts([source], advantages, 2)
    assert sampled[0]['metadata']['entry_pattern']['config'] == CFG
    sampled[0]['metadata']['entry_pattern']['config']['target_mode'] = 'legacy_price'
    assert source['metadata']['entry_pattern']['config'] == CFG
    saved[0]['metadata']['entry_pattern']['config']['target_mode'] = 'realized_net'
    assert source['metadata']['entry_pattern']['config'] == CFG


def test_previous_realized_net_best_is_not_reused_for_new_objective():
    from test_profit_experiment_config import legacy_best_checkpoint
    from ai_trader.grpo.evaluation import evaluation_signature, compatible_resume_best
    checkpoint, signature = legacy_best_checkpoint()
    config = checkpoint['extra_state']['training_config']
    config['entry_pattern_config'] = {**CFG, 'target_mode': 'realized_net'}
    saved = evaluation_signature(config, signature['date_splits'], signature['observation_schema'])
    checkpoint['extra_state']['evaluation_signature'] = saved
    assert compatible_resume_best(checkpoint, checkpoint, saved)
    changed = evaluation_signature({**config, 'entry_pattern_config': CFG},
                                   signature['date_splits'], signature['observation_schema'])
    assert not compatible_resume_best(checkpoint, checkpoint, changed)
