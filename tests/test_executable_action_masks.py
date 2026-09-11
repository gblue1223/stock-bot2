"""Execution eligibility must match live/replay and the cached PPO likelihood."""
import numpy as np
import pytest
import torch

from ai_trader.grpo.environments import DummyVecEnv
from ai_trader.grpo.evaluation import evaluate_policy
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.inference.enhanced_grpo_infer_xlstm import (
    GRPOInferenceE2EXLSTM, EnhancedGRPOInferenceXLSTM, Position)
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from lib.action_masks import executable_action_mask
from lib.observations import ObservationBuilder
from test_scalping_accounting import ReplayFixture


def eligibility(**changes):
    state = dict(filled_stages=0, occupied_stages=0, pending_buy=False, pending_sell=False,
                 exit_active=False, within_trade_limit=True, buy_budget=1000., buy_unit_cost=100.)
    state.update(changes)
    return state


def small_policy(obs_dim=17, **kwargs):
    return GRPOPolicyE2EXLSTM(obs_dim=obs_dim, cnn_channels=4, rnn_hidden_dim=4,
                              fc_hidden_dim=8, max_stages=2, execution_action_mask=True, **kwargs)


@pytest.mark.parametrize('changes,expected', [
    ({}, [True, True, False]),
    ({'buy_budget': 99.99}, [True, False, False]),
    ({'pending_buy': True, 'occupied_stages': 1}, [True, False, False]),
    ({'pending_buy': True, 'occupied_stages': 1, 'filled_stages': 1}, [True, False, True]),
    ({'occupied_stages': 2, 'filled_stages': 2}, [True, False, True]),
    ({'occupied_stages': 1, 'filled_stages': 1, 'pending_sell': True}, [True, False, False]),
    ({'occupied_stages': 1, 'filled_stages': 1, 'exit_active': True}, [True, False, False]),
    ({'occupied_stages': 1, 'filled_stages': 1, 'within_trade_limit': False}, [True, False, True]),
])
def test_shared_execution_eligibility(changes, expected):
    np.testing.assert_array_equal(executable_action_mask(eligibility(**changes), 2), expected)


@pytest.mark.parametrize('state', [None, {}, eligibility(buy_budget=np.nan),
                                  eligibility(buy_budget='100'), eligibility(buy_unit_cost=None),
                                  eligibility(pending_buy=1), eligibility(occupied_stages=0, filled_stages=1)])
def test_live_mask_contract_rejects_missing_or_invalid_execution_state(state):
    with pytest.raises(ValueError):
        executable_action_mask(state, 2)


def test_environment_pending_partial_fill_reserves_one_stage_and_sell_can_cancel_buy():
    env = ReplayFixture([100] * 8, max_stages=2, initial_cash=1000., execution_action_mask=True,
                        max_episode_steps=5, liquidation_max_steps=2,
                        execution_config={'order_latency_ms': 0, 'fallback_depth': 2,
                                          'slippage_bps': 0, 'spread_bps': 0})
    _, info = env.reset()
    assert info['action_mask'].tolist() == [True, True, False]
    _, _, _, _, info = env.step(1)
    state = env.action_mask_state()
    assert state['filled_stages'] == state['occupied_stages'] == 1
    assert state['pending_buy']
    assert info['action_mask'].tolist() == [True, False, True]
    env.step(2)
    assert env.quantity == 0 and not env._pending('buy')
    assert env.action_masks().tolist() == [True, True, False]
    assert env._signal_exit_order_id is None


def test_environment_cash_and_trade_limit_only_block_entry_not_exit():
    env = ReplayFixture([100] * 6, max_stages=2, initial_cash=1000., execution_action_mask=True,
                        max_episode_steps=3, liquidation_max_steps=2)
    env.reset()
    env.cash = 99.99
    assert not env.action_masks()[1]
    env.cash = 1000
    env.step(1)
    env.max_trades_per_episode = 1
    env.episode_trades.append({})
    assert env.action_masks().tolist() == [True, False, True]


def test_full_budget_fee_roundoff_does_not_invalidate_mask_but_overdraft_does():
    price = 16.495495495495497
    env = ReplayFixture([price] * 5, max_stages=1, initial_cash=10 * (price * 1.00015),
                        transaction_cost_rate=.00015, account_observations=True,
                        execution_action_mask=True)
    env.reset()
    _, _, _, _, info = env.step(1)
    assert -1e-12 < env.cash / env.initial_cash < 0
    assert info['action_mask'].tolist() == [True, False, True]
    assert env.action_mask_state()['buy_budget'] == 0.
    env.cash = -.1
    with pytest.raises(ValueError, match='buy_budget'):
        env.action_masks()


def test_mask_read_is_pure_and_does_not_block_automatic_risk_liquidation():
    env = ReplayFixture([100, 100, 80, 80, 80, 80], max_stages=1, initial_cash=1000.,
                        execution_action_mask=True, stop_loss_pct=5, max_episode_steps=4,
                        liquidation_max_steps=1)
    env.reset()
    env.step(1)
    env.step(0)
    count = len(env.simulator.orders)
    assert env.action_masks().tolist() == [True, False, False]
    assert len(env.simulator.orders) == count and not env._exit_requested
    env.step(0)
    assert env.quantity == 0
    assert env.episode_trades[0]['exit_reason'] == 'stop_loss'


def test_policy_requires_explicit_boolean_masks_and_rejects_forbidden_cached_action():
    policy = small_policy()
    states = torch.zeros(2, 8, 17)
    with pytest.raises(ValueError, match='requires explicit'):
        policy(states)
    with pytest.raises(ValueError, match='boolean'):
        policy(states, action_masks=torch.ones(2, 3))
    masks = torch.tensor([[True, False, False], [True, True, False]])
    actions, old_logs, _ = policy.get_action_with_value(states, action_masks=masks)
    logs, _, _ = policy.evaluate_actions(states, actions, action_masks=masks)
    torch.testing.assert_close(logs, old_logs)
    assert actions[0] == 0
    with pytest.raises(ValueError, match='forbidden'):
        policy.evaluate_actions(states, torch.tensor([1, 1]), action_masks=masks)


def test_vector_autoreset_masks_cached_updates_and_batched_evaluation():
    def factory():
        return ReplayFixture([100, 100, 101, 100, 101, 100, 101, 100], initial_cash=1000.,
                             max_stages=2, max_episode_steps=4, liquidation_max_steps=2,
                             execution_action_mask=True)
    vector = DummyVecEnv([factory, factory])
    policy = small_policy()
    trainer = GRPOTrainer(policy, vector, episodes_per_group=2, num_groups=1, batch_size=3,
                          num_epochs=1, group_advantage_coef=0., device='cpu')
    episodes = trainer.collect_rollouts(4)
    assert all(ep['action_masks'].shape == (4, 3) for ep in episodes)
    for episode in episodes:
        assert episode['action_masks'][0].tolist() == [True, True, False]
        with torch.no_grad():
            logs, _, _ = policy.evaluate_actions(torch.from_numpy(episode['states']),
                torch.from_numpy(episode['actions']), action_masks=episode['action_masks'])
        np.testing.assert_allclose(logs.numpy(), episode['log_probs'], atol=1e-6)
    # Exercise the uncached-critic fallback using exactly the same masks.
    episodes[0].pop('values')
    metrics = trainer.update_policy(episodes, [np.zeros(4, np.float32)] * 4)
    assert metrics['optimizer_steps'] > 0
    from test_profit_training_diagnostics import Writer
    trainer.writer = Writer()
    trainer._log_metrics(0, episodes, {0: episodes}, metrics)
    assert trainer.writer.metrics['group_0/diagnostic_samples'] > 0
    validators = [factory(), factory()]
    serial = evaluate_policy(policy, validators[0], num_episodes=3)
    batched = evaluate_policy(policy, validators, num_episodes=3)
    assert serial['episode_net_returns'] == pytest.approx(batched['episode_net_returns'])
    assert serial['diagnostics']['action_counts'] == batched['diagnostics']['action_counts']
    episodes[0].pop('action_masks')
    with pytest.raises(ValueError, match='cached boolean'):
        trainer.update_policy(episodes, [np.zeros(4, np.float32)] * 4)
    vector.close()


def test_live_and_replay_use_identical_masks_actions_and_confidences(tmp_path):
    env = ReplayFixture([100] * 8, max_stages=2, initial_cash=1000., execution_action_mask=True,
                        max_episode_steps=5, liquidation_max_steps=2,
                        execution_config={'order_latency_ms': 0, 'fallback_depth': 2,
                                          'slippage_bps': 0, 'spread_bps': 0})
    observation, _ = env.reset()
    policy = small_policy().eval()
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([0., 10., 1.]))
    path = tmp_path / 'replay.pt'
    torch.save({'policy_state_dict': policy.state_dict(), 'observation_schema': env.observation_schema,
                'config': {'execution_action_mask': True}}, path)
    live = GRPOInferenceE2EXLSTM(path, device='cpu')
    for index in range(2):
        mask_state = env.action_mask_state()
        np.testing.assert_array_equal(live.build_action_mask(env.stages, mask_state), env.action_masks())
        raw = env.episode_data[env.current_step:env.current_step + 1]
        action, confidence = live.predict(raw, stages=env.stages, current_price=env.current_price,
                                         current_time_seconds=env.current_time_seconds,
                                         action_mask_state=mask_state)
        with torch.no_grad():
            expected, _, probabilities = policy.get_action_with_probabilities(
                torch.from_numpy(observation).unsqueeze(0), deterministic=True,
                action_masks=env.action_masks()[None, :])
        assert action == expected.item()
        assert confidence == pytest.approx(probabilities[0, action].item())
        if index == 0:
            observation, _, _, _, _ = env.step(action)


def test_live_masked_checkpoint_requires_affordability_and_keeps_risk_exits(tmp_path):
    builder = ObservationBuilder(['현재가'], seq_len=8, rolling_min_samples=1, max_stages=2)
    policy = small_policy(obs_dim=16)
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([0., 10., 1.]))
    path = tmp_path / 'masked.pt'
    torch.save({'policy_state_dict': policy.state_dict(), 'observation_schema': builder.schema,
                'config': {'execution_action_mask': True}}, path)
    live = GRPOInferenceE2EXLSTM(path, device='cpu')
    raw = np.full((8, 1), 100.)
    with pytest.raises(ValueError, match='explicit action_mask_state'):
        live.predict(raw)
    assert live.predict(raw, action_mask_state=eligibility())[0] == 1
    assert live.predict(raw, action_mask_state=eligibility(buy_budget=0.))[0] == 0
    enhanced = EnhancedGRPOInferenceXLSTM(live)
    action, _, info = enhanced.predict(raw, Position(100., 0., 100., 0.), 95., current_time_seconds=1.,
        action_mask_state=eligibility(filled_stages=1, occupied_stages=1, pending_sell=True, exit_active=True))
    assert action == 2 and info['risk_exit_all']
