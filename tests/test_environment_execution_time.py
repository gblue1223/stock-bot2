"""Execution attribution, first exit reasons and policy-time replay contracts."""
import json

import numpy as np
import pytest

from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from lib.observations import EXECUTION_FIELDS
from test_scalping_accounting import ReplayFixture, finish
from test_environment_closeout import book


def test_execution_attribution_reconciles_partial_depth_fills_fees_and_tick_rounding():
    n = 7
    execution = {
        'bid_prices': np.array([[99, 98], [99, 98], [103, 102], [103, 102], [104, 103], [105, 104], [105, 104]]),
        'ask_prices': np.array([[101, 102], [101, 102], [105, 106], [105, 106], [106, 107], [107, 108], [107, 108]]),
        'bid_sizes': np.tile([2, 100], (n, 1)),
        'ask_sizes': np.tile([2, 100], (n, 1)),
    }
    env = ReplayFixture([100, 100, 104, 104, 105, 106, 106], execution=execution,
                        initial_cash=1000, max_stages=1, transaction_cost_rate=.001,
                        execution_config={'order_latency_ms': 0, 'slippage_bps': 10, 'tick_size': .01})
    reward, ep = finish(env, [1, 0, 2, 0, 0, 0])
    assert ep['round_trip_count'] == 1 and ep['fill_count'] > 1
    assert ep['spread_cost'] > 0 and ep['depth_cost'] > 0 and ep['slippage_cost'] > 0
    assert ep['gross_realized_pnl'] == pytest.approx(
        ep['mid_price_pnl'] - ep['spread_cost'] - ep['depth_cost'] - ep['slippage_cost'])
    assert ep['matched_fees'] == pytest.approx(ep['total_fees'])
    assert ep['pnl_attribution_residual'] == pytest.approx(0, abs=1e-10)
    assert reward == pytest.approx(ep['realized_net_pnl'] / 10)
    assert ep['exit_signal_round_trip_count'] == 1
    assert ep['exit_signal_net_pnl'] == pytest.approx(ep['realized_net_pnl'])


def test_expired_stop_loss_retries_keep_first_reason_through_terminal_tail():
    env = ReplayFixture([100, 100, 90, 90, 90, 90], seconds=[0, 1, 2, 10, 11, 12],
                        max_stages=1, initial_cash=1000, stop_loss_pct=5,
                        max_episode_steps=3, liquidation_max_steps=3)
    _, ep = finish(env, [1, 0, 0])
    assert ep['expired_orders'] == 1
    assert ep['exit_stop_loss_round_trip_count'] == 1
    assert ep['exit_episode_end_round_trip_count'] == 0
    assert all(trade['exit_reason'] == 'stop_loss' for trade in ep['trades'])
    assert all(order.reason == 'stop_loss' for order in env.simulator.orders if order.side == 'sell')


def test_subsecond_quantity_uses_entry_lots_instead_of_average_holding_threshold():
    env = ReplayFixture([100] * 6, seconds=[0, .1, 1, 1.2, 1.3, 1.4],
                        initial_cash=1000, max_stages=1,
                        execution=book([99] * 6, [100] * 6, asks=[100] * 6,
                                       ask_sizes=[5, 5, 10, 10, 10, 10]))
    _, ep = finish(env, [1, 0, 2, 0, 0])
    # Half entered at .1s, half at 1s, all sold at 1.2s.
    assert ep['quantity_weighted_holding_time'] == pytest.approx(.65)
    assert ep['subsecond_exit_quantity_ratio'] == pytest.approx(.5)
    assert ep['subsecond_round_trip_ratio'] == 1


def test_timed_decision_processes_stop_loss_between_policy_observations():
    env = ReplayFixture([100, 100, 80, 80, 80, 80, 80], seconds=[0, .1, .2, .3, .7, 1, 2],
                        initial_cash=1000, max_stages=1, stop_loss_pct=5,
                        decision_interval_seconds=1, max_episode_steps=2)
    env.reset()
    obs, reward, done, truncated, info = env.step(1)
    assert not done and not truncated
    assert env.current_step == 5 and env.simulator.event_index == 5
    assert env.quantity == 0 and len(env.episode_rewards) == 1
    assert [f['timestamp'] - env.episode_start_time_seconds for f in info['fills']] == pytest.approx([.1, .3])
    assert env.episode_trades[0]['exit_reason'] == 'stop_loss'
    assert reward == pytest.approx(-20)
    assert len(env.equity_history) == 6


def test_timed_decision_preserves_order_latency_expiry_and_gap_retry():
    env = ReplayFixture([100] * 7, seconds=[0, .1, .15, .2, .7, .8, 1],
                        initial_cash=1000, max_stages=1, max_holding_seconds=.1,
                        decision_interval_seconds=1, max_episode_steps=1,
                        liquidation_max_steps=2,
                        execution_config={'order_latency_ms': 100, 'order_ttl_seconds': .2,
                                          'spread_bps': 0, 'slippage_bps': 0})
    _, ep = finish(env, [1])
    assert ep['expired_orders'] >= 1
    assert ep['liquidation_complete']
    assert ep['exit_max_holding_round_trip_count'] == 1
    assert ep['steps_taken'] == 1 and ep['market_steps_taken'] == 6


def test_duration_timer_ends_policy_then_replays_tail_and_rewards_reconcile():
    env = ReplayFixture([100] * 8, seconds=[0, .1, .2, .3, .4, .5, .6, .7],
                        max_stages=1, initial_cash=1000, decision_interval_seconds=.2,
                        episode_duration_seconds=.35, max_episode_steps=10,
                        liquidation_max_steps=2,
                        execution_config={'order_latency_ms': 100, 'spread_bps': 0, 'slippage_bps': 0})
    reward, ep = finish(env, [1, 0])
    assert ep['steps_taken'] == 2 and ep['market_steps_taken'] == 5
    assert ep['policy_duration_seconds'] == pytest.approx(.4)
    assert ep['episode_duration_seconds'] == pytest.approx(.5)
    assert ep['liquidation_steps'] == 1 and ep['liquidation_complete']
    assert ep['exit_episode_end_round_trip_count'] == 1
    assert reward == pytest.approx(ep['net_return'])


def test_decision_cap_still_limits_policy_actions_with_elapsed_time_enabled():
    env = ReplayFixture([100] * 11, seconds=np.arange(11) / 10,
                        decision_interval_seconds=.2, episode_duration_seconds=10,
                        max_episode_steps=2, liquidation_max_steps=2)
    _, ep = finish(env, [0, 0])
    assert ep['steps_taken'] == 2 and ep['market_steps_taken'] == 4
    assert ep['policy_duration_seconds'] == pytest.approx(.4)


def test_duplicate_timestamps_are_all_processed_and_gaps_do_not_invent_timer_fills():
    env = ReplayFixture([100] * 8, seconds=[0, 0, .1, .1, .3, 3, 3.1, 3.2],
                        initial_cash=1000, max_stages=1, decision_interval_seconds=.2,
                        episode_duration_seconds=.5, max_episode_steps=10,
                        liquidation_max_steps=2,
                        execution_config={'order_latency_ms': 100, 'order_ttl_seconds': .5,
                                          'spread_bps': 0, 'slippage_bps': 0})
    reward, ep = finish(env, [1, 0])
    assert ep['steps_taken'] == 2
    assert ep['expired_orders'] == 1  # .5s timer sell has expired at the 3s quote.
    assert ep['trades'][0]['holding_time'] == pytest.approx(3)
    assert ep['liquidation_steps'] == 1
    assert ep['market_steps_taken'] == 6 and env.simulator.event_index == 6
    assert reward == 0


def test_unknown_quote_timestamp_and_empty_book_remain_finite_without_fabricated_depth():
    env = ReplayFixture([100] * 3, account_observations=True, execution_observations=True,
                        execution=book([99] * 3, [0] * 3, asks=[101] * 3, ask_sizes=[0] * 3))
    obs, _ = env.reset()
    state = env._execution_state()
    assert np.isfinite(obs).all()
    assert state['quote_timestamp_known'] == state['quote_age_fraction'] == 0
    assert state['buy_depth_ratio'] == state['sell_depth_ratio'] == 0


@pytest.mark.parametrize('name,value', [('decision_interval_seconds', -1), ('episode_duration_seconds', np.inf),
                                        ('episode_duration_seconds', True)])
def test_invalid_time_configuration_fails(name, value):
    with pytest.raises(ValueError, match=name):
        ReplayFixture([100, 100], **{name: value})


def test_v4_cost_observations_match_remaining_cost_basis_and_quote_freshness():
    env = ReplayFixture([100] * 5, seconds=[0, .1, .2, .3, .4], initial_cash=1000, max_stages=1,
                        account_observations=True, execution_observations=True,
                        transaction_cost_rate=.001, sell_tax_rate=.002,
                        episode_duration_seconds=1,
                        execution={**book([99] * 5, [100] * 5, asks=[101] * 5),
                                   'quote_timestamp': np.array([32400, 32400.1, 32400.1, 32400.1, 32400.4])},
                        execution_config={'order_latency_ms': 0, 'slippage_bps': 2,
                                          'max_quote_age_seconds': .05})
    obs, _ = env.reset()
    state = env._execution_state()
    assert obs.shape == (1, 38) and env.observation_schema['version'] == 4
    assert state['spread_bps'] == pytest.approx(200)
    assert state['liquidation_return'] == state['breakeven_return'] == 0
    obs, _, _, _, _ = env.step(1)
    state = env._execution_state()
    basis = sum(s['entry_price'] * s['quantity'] + s['entry_fees'] for s in env.stages)
    proceeds = env.quantity * 99 * .9998 * .997
    assert state['liquidation_return'] == pytest.approx(proceeds / basis - 1)
    assert state['breakeven_return'] == pytest.approx(basis / proceeds - 1)
    assert state['remaining_time_fraction'] == pytest.approx(.9)
    assert state['quote_timestamp_known'] == 1
    env.step(0)
    assert env._execution_state()['quote_age_fraction'] == pytest.approx(2)
    assert env._execution_state()['sell_depth_ratio'] == 0


def test_time_sampling_reserves_actual_elapsed_events_and_aligned_tail(tmp_path):
    columns = ['현재가', '등락률', '누적거래대금']
    n = 20
    features = np.column_stack((np.full(n, 100), np.zeros(n), np.arange(n))).astype(np.float32)
    metadata = np.asarray([['TEST', '20260908', str(90000000 + i * 100)] for i in range(n)])
    np.savez(tmp_path / 'episode.npz', features=features, metadata=metadata, execution_last_price=features[:, 0])
    manifest = {'metadata': {'feature_columns': columns, 'price_unit': 'krw'},
                'episodes': [{'file_path': 'episode.npz', 'stock_code': 'TEST', 'date': '20260908', 'length': n}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    env = GRPOScalpingEnvXLSTM(extracted_dir=str(tmp_path), seq_len=2, expected_features=3,
                             max_episode_steps=3, decision_interval_seconds=.3,
                             episode_duration_seconds=1, liquidation_max_steps=4)
    try:
        env.reset(options={'episode_index': 0, 'start_index': 1})
        assert env.episode_length >= 15  # history + 9 decision events + 4 tail events.
        assert len(env.episode_execution['last_price']) == env.episode_length
        assert env.current_time_seconds - 32400 == pytest.approx(.2)
        for _ in range(3):
            _, _, done, truncated, info = env.step(0)
        assert done and not truncated and info['episode']['steps_taken'] == 3
        assert info['episode']['policy_duration_seconds'] >= .9 - 1e-9
    finally:
        env.close()


def test_sampling_accounts_for_repeated_rounding_to_irregular_event_times():
    env = ReplayFixture([100] * 20, seconds=np.arange(20) * .6, seq_len=2,
                        max_episode_steps=3, decision_interval_seconds=1,
                        liquidation_max_steps=4)
    env.reset()
    metadata = env.episode_metadata.copy()
    env._reset_options = {'start_index': 8}
    start, end = env._episode_slice_bounds(metadata)
    assert start == 8 and end == 20
    # Nominal 3 seconds wrongly admits start=9; actual three intervals take 3.6s.
    env._reset_options = {'start_index': 9}
    with pytest.raises(ValueError, match='insufficient policy time'):
        env._episode_slice_bounds(metadata)
