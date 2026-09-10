"""Real replay suffixes, truthful account observations and completed trade metrics."""
import json

import numpy as np
import pytest
import torch

from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from ai_trader.grpo.evaluation import evaluate_policy
from lib.observations import ACCOUNT_FIELDS
from test_scalping_accounting import ReplayFixture, finish


def book(bids, sizes, asks=None, ask_sizes=None):
    bids = np.asarray(bids, dtype=float)
    n = len(bids)
    return {
        'bid_prices': bids.reshape(n, -1),
        'bid_sizes': np.asarray(sizes).reshape(n, -1),
        'ask_prices': np.asarray(asks if asks is not None else bids + 1).reshape(n, -1),
        'ask_sizes': np.asarray(ask_sizes if ask_sizes is not None else np.full(n, 100)).reshape(n, -1),
    }


def account(obs, env):
    values = obs[-1, env.expected_features:env.expected_features + len(ACCOUNT_FIELDS)]
    return dict(zip(ACCOUNT_FIELDS, values))


def test_closeout_waits_for_latency_and_includes_tail_loss_and_fees_in_reward():
    env = ReplayFixture([100, 100, 100, 100, 90], seconds=[0, .1, .2, .25, .31],
                        max_stages=1, initial_cash=1000, max_episode_steps=2,
                        liquidation_max_steps=3, transaction_cost_rate=.001,
                        execution_config={'order_latency_ms': 100, 'spread_bps': 0, 'slippage_bps': 0})
    env.reset()
    _, first, done, truncated, _ = env.step(1)
    assert not done and not truncated
    _, final, done, truncated, info = env.step(0)
    ep = info['episode']
    assert done and not truncated
    assert ep['liquidation_complete'] and ep['liquidation_stop_reason'] == 'flat'
    assert ep['liquidation_steps'] == 2 and ep['steps_taken'] == 2
    assert ep['market_steps_taken'] == 4
    assert ep['trades'][0]['exit_price'] == 90
    assert first + final == pytest.approx(ep['net_return'])
    assert ep['net_return'] == pytest.approx(ep['realized_net_pnl'] / 10)
    assert ep['total_fees'] == pytest.approx(9 * 100 * .001 + 9 * 90 * .001)
    assert ep['gross_realized_pnl'] == pytest.approx(-90)
    assert ep['max_drawdown'] == pytest.approx(-ep['net_return'])
    # Two policy rewards cover four actual events; closeout latency contributes
    # its zero-return event instead of collapsing the tail into one observation.
    event_changes = np.array([-.18, 0, 0, -8.991])
    assert ep['sharpe_ratio'] == pytest.approx(event_changes.mean() / event_changes.std())
    assert ep['sharpe_ratio_kind'] == 'unannualized_event_nav_changes'


@pytest.mark.parametrize('tail_steps,prices,reason,used', [
    (1, [100] * 6, 'step_limit', 1),
    (5, [100] * 4, 'end_of_data', 1),
    (5, [100] * 3, 'end_of_data', 0),
])
def test_unfillable_closeout_stops_at_bound_or_data_end(tail_steps, prices, reason, used):
    n = len(prices)
    env = ReplayFixture(prices, max_stages=1, initial_cash=1000,
                        max_episode_steps=2, liquidation_max_steps=tail_steps,
                        execution=book([99] * n, [100, 100] + [0] * (n - 2), asks=[100] * n))
    reward, ep = finish(env, [1, 0])
    assert ep['liquidation_steps'] == used and ep['liquidation_stop_reason'] == reason
    assert not ep['liquidation_complete'] and ep['open_quantity'] == 10
    assert ep['round_trip_count'] == ep['fill_count'] == 0
    assert reward == pytest.approx(ep['net_return'])
    assert all(order.status != 'filled' for order in env.simulator.orders if order.side == 'sell')


def test_closeout_retries_partial_expired_sell_and_prices_actual_remaining_depth():
    env = ReplayFixture([100, 100, 100, 120, 90, 90],
                        max_stages=1, initial_cash=1000, max_episode_steps=2,
                        liquidation_max_steps=3,
                        execution=book([99, 99, 99, 120, 90, 90], [100, 100, 100, 1, 100, 100],
                                       asks=[100, 100, 100, 121, 91, 91]))
    reward, ep = finish(env, [1, 0])
    assert ep['liquidation_complete']
    assert [(t['quantity'], t['exit_price']) for t in ep['trades']] == [(1, 120), (9, 90)]
    assert ep['round_trip_count'] == 1 and ep['fill_count'] == 2
    assert reward == pytest.approx(-7)
    assert reward == pytest.approx(ep['realized_net_pnl'] / 10)


def test_closeout_resubmits_sell_after_order_expires_across_market_gap():
    env = ReplayFixture([100] * 5, seconds=[0, 1, 2, 10, 11],
                        max_stages=1, initial_cash=1000, max_episode_steps=2,
                        liquidation_max_steps=2)
    reward, ep = finish(env, [1, 0])
    assert ep['expired_orders'] == 1 and ep['submitted_orders'] == 3
    assert ep['liquidation_complete'] and ep['liquidation_steps'] == 2
    assert ep['round_trip_count'] == 1 and reward == 0


def test_last_policy_action_executes_before_closeout_and_pending_buy_cancel_race_is_drained():
    env = ReplayFixture([100] * 5, seconds=[0, .1, .11, .2, .3],
                        max_stages=1, initial_cash=1000, max_episode_steps=1,
                        liquidation_max_steps=4,
                        execution=book([99] * 5, [100] * 5, asks=[100] * 5,
                                       ask_sizes=[1, 1, 100, 100, 100]))
    reward, ep = finish(env, [1])
    assert ep['buy_action_outcomes']['submitted'] == 1
    assert ep['liquidation_complete'] and not env._pending('buy')
    assert ep['round_trip_count'] == 1
    assert sum(t['quantity'] for t in ep['trades']) == 10
    assert reward == pytest.approx(-1)


def test_one_completed_entry_with_fragmented_exits_reports_one_round_trip():
    env = ReplayFixture([100] * 7, max_stages=1, initial_cash=1000,
                        execution=book([99] * 7, [1, 1, 1, 1, 2, 20, 20], asks=[100] * 7))
    _, ep = finish(env, [1, 0, 2, 0, 0, 0])
    assert [t['quantity'] for t in ep['trades']] == [1, 1, 8]
    assert ep['round_trip_count'] == 1 and ep['num_trades'] == ep['fill_count'] == 3
    assert ep['round_trip_win_rate'] == 0
    assert ep['fill_avg_holding_time'] == pytest.approx(10 / 3)
    assert ep['quantity_weighted_holding_time'] == pytest.approx(4.5)
    assert ep['round_trips'][0]['net_pnl'] == pytest.approx(ep['realized_net_pnl'])


def test_partial_entry_times_are_quantity_weighted_in_new_holding_metric():
    env = ReplayFixture([100] * 5, max_stages=1, initial_cash=1000,
                        execution=book([99] * 5, [100] * 5, asks=[100] * 5,
                                       ask_sizes=[1, 1, 10, 10, 10]))
    _, ep = finish(env, [1, 0, 2, 0])
    assert ep['round_trip_count'] == 1
    # One share enters at t=1, nine at t=2; all exit at t=3.
    assert ep['quantity_weighted_holding_time'] == pytest.approx(1.1)
    assert ep['fill_avg_holding_time'] == pytest.approx(2)


def test_v3_observes_cash_exposure_pending_orders_and_decision_horizon_only():
    env = ReplayFixture([100] * 7, max_stages=1, initial_cash=1000,
                        max_episode_steps=3, liquidation_max_steps=3, account_observations=True,
                        execution_config={'order_latency_ms': 1500, 'spread_bps': 0, 'slippage_bps': 0})
    obs, _ = env.reset()
    assert obs.shape == (1, 28) and env.observation_schema['version'] == 3
    assert account(obs, env)['remaining_steps_ratio'] == 1
    obs, _, _, _, _ = env.step(1)
    values = account(obs, env)
    assert values['pending_buy_value_ratio'] == pytest.approx(1)
    assert values['position_value_ratio'] == 0
    assert values['remaining_steps_ratio'] == pytest.approx(2 / 3)
    obs, _, _, _, _ = env.step(0)
    values = account(obs, env)
    assert values['pending_buy_value_ratio'] == 0 and values['cash_ratio'] == 0
    assert values['position_value_ratio'] == values['stage_1_value_ratio'] == 1
    assert values['remaining_steps_ratio'] == pytest.approx(1 / 3)
    env._signal_exit_order_id = env.stages[0]['order_id']
    env._request_stage_exit(env.current_time_seconds)
    values = account(env._get_current_observation(), env)
    assert values['exit_pending'] == 1 and values['pending_sell_value_ratio'] == 1


def test_v3_zero_cash_tolerates_only_fee_multiplication_roundoff():
    price = 16.495495495495497
    initial = 10 * (price * (1 + .00015))
    env = ReplayFixture([price] * 4, max_stages=1, initial_cash=initial,
                        transaction_cost_rate=.00015, account_observations=True)
    env.reset()
    obs, _, _, _, _ = env.step(1)
    assert env.quantity == 10 and abs(env.cash) < 1e-10
    assert account(obs, env)['cash_ratio'] == 0
    env.cash = -1
    with pytest.raises(ValueError, match='cannot be negative'):
        env._get_current_observation()


def test_xlstm_sampling_reserves_aligned_tail_without_exposing_future_rows(tmp_path):
    columns = ['현재가', '등락률', '누적거래대금']
    features = np.column_stack((np.arange(100, 110), np.zeros(10), np.arange(10))).astype(np.float32)
    metadata = np.asarray([['TEST', '20260908', str(90000000 + i * 1000)] for i in range(10)])
    np.savez(tmp_path / 'episode.npz', features=features, metadata=metadata, execution_last_price=features[:, 0])
    manifest = {'metadata': {'feature_columns': columns, 'price_unit': 'krw'},
                'episodes': [{'file_path': 'episode.npz', 'stock_code': 'TEST', 'date': '20260908', 'length': 10}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    env = GRPOScalpingEnvXLSTM(extracted_dir=str(tmp_path), expected_features=3, seq_len=2,
                             max_episode_steps=3, liquidation_max_steps=4)
    env.reset(options={'episode_index': 0, 'start_index': 1})
    assert env.episode_length == 9 and env.decision_steps == 3
    assert env.current_price == 102
    assert len(env.episode_execution['last_price']) == len(env.episode_metadata) == 9
    _, ep = finish(env, [0, 0, 0])
    assert ep['steps_taken'] == 3 and ep['liquidation_steps'] == 0
    env.close()


def test_batched_evaluation_matches_serial_real_v3_episodes_and_closeout(tmp_path):
    columns = ['현재가', '등락률', '누적거래대금']
    features = np.column_stack((np.arange(100, 111), np.zeros(11), np.arange(11))).astype(np.float32)
    metadata = np.asarray([['TEST', '20260908', str(90000000 + i * 1000)] for i in range(11)])
    np.savez(tmp_path / 'episode.npz', features=features, metadata=metadata, execution_last_price=features[:, 0])
    manifest = {'metadata': {'feature_columns': columns, 'price_unit': 'krw'},
                'episodes': [{'file_path': 'episode.npz', 'stock_code': 'TEST', 'date': '20260908', 'length': 11}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    class EntryThenHold(torch.nn.Module):
        def get_action(self, obs, deterministic=False):
            assert obs.shape[-1] == 29  # Three market + eleven account + fifteen stage fields.
            actions = (obs[:, -1, -15] == 0).long()
            return actions, torch.zeros(len(obs))

    def make_env():
        return GRPOScalpingEnvXLSTM(
            extracted_dir=str(tmp_path), expected_features=3, seq_len=1,
            max_episode_steps=3, liquidation_max_steps=4, account_observations=True,
            initial_cash=1000, transaction_cost_rate=0, sell_tax_rate=0,
            execution_config={'order_latency_ms': 100, 'spread_bps': 0, 'slippage_bps': 0})

    serial_env = make_env()
    batch_envs = [make_env(), make_env(), make_env()]
    try:
        serial = evaluate_policy(EntryThenHold(), serial_env, num_episodes=5, seed=24)
        batched = evaluate_policy(EntryThenHold(), batch_envs, num_episodes=5, seed=24)
        assert serial == batched
        assert serial['diagnostics']['steps'] == 15
        assert serial['incomplete_liquidation_episodes'] == 0
        assert serial['round_trip_count'] == 5
        assert serial['liquidation_stop_reasons'] == {'flat': 5}
        assert all(ep['liquidation_steps'] > 0 for ep in serial['diagnostics']['episodes'])
    finally:
        for env in [serial_env, *batch_envs]:
            env.close()


@pytest.mark.parametrize('invalid', [-1, True, 1.5])
def test_invalid_liquidation_budget_is_rejected(invalid):
    with pytest.raises(ValueError, match='liquidation_max_steps'):
        ReplayFixture([100] * 3, liquidation_max_steps=invalid)
