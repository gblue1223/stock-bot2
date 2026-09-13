"""BUY decisions retain their actual order and realized FIFO accounting lineage."""
import json

import numpy as np
import pytest

from test_environment_closeout import book
from test_scalping_accounting import ReplayFixture, finish


def assert_entry_accounting(episode):
    trace = episode['entry_diagnostics']
    assert trace['version'] == 1
    assert trace['decision_count'] == episode['steps_taken']
    submitted = [decision for decision in trace['buy_decisions'] if decision['entry_order_id'] is not None]
    assert [(decision['entry_order_id'], decision['decision_index']) for decision in submitted] == [
        (entry['entry_order_id'], entry['decision_index']) for entry in trace['entries']]
    for outcome, count in episode['buy_action_outcomes'].items():
        assert sum(decision['outcome'] == outcome for decision in trace['buy_decisions']) == count
    for entry in trace['entries']:
        assert entry['submitted_quantity'] >= entry['filled_quantity']
        assert entry['filled_quantity'] == entry['sold_quantity'] + entry['open_quantity']
        assert entry['net_pnl'] == pytest.approx(
            entry['mid_price_pnl'] - entry['spread_cost'] - entry['depth_cost']
            - entry['slippage_cost'] - entry['matched_fees'] + entry['pnl_attribution_residual'])
        assert entry['pnl_attribution_residual'] == pytest.approx(0, abs=1e-9)
    for entry_key, episode_key in (
        ('net_pnl', 'realized_net_pnl'), ('mid_price_pnl', 'mid_price_pnl'),
        ('spread_cost', 'spread_cost'), ('depth_cost', 'depth_cost'), ('slippage_cost', 'slippage_cost'),
        ('matched_fees', 'matched_fees'), ('entry_fees', 'total_entry_fees'), ('exit_fees', 'total_exit_fees'),
        ('open_quantity', 'open_quantity'),
    ):
        assert sum(entry[entry_key] for entry in trace['entries']) == pytest.approx(episode[episode_key])
    assert sum(entry['complete'] for entry in trace['entries']) == episode['round_trip_count']
    # Terminal diagnostics are small JSON values, without simulator objects or ndarrays.
    assert json.loads(json.dumps(trace, allow_nan=False)) == trace


def test_submitted_and_blocked_buy_decisions_have_distinct_order_mapping():
    env = ReplayFixture([100] * 4, max_stages=1)
    _, episode = finish(env, [1, 1, 0])
    trace = episode['entry_diagnostics']
    assert trace['buy_decisions'] == [
        {'decision_index': 0, 'entry_order_id': 1, 'outcome': 'submitted'},
        {'decision_index': 1, 'entry_order_id': None, 'outcome': 'max_stages'},
    ]
    assert trace['entries'][0]['complete']
    assert_entry_accounting(episode)


def test_unaffordable_and_risk_blocked_buy_decisions_do_not_invent_orders():
    env = ReplayFixture([100] * 4, initial_cash=100, max_stages=5)
    _, episode = finish(env, [1, 2, 1])
    assert episode['entry_diagnostics']['buy_decisions'] == [
        {'decision_index': 0, 'entry_order_id': None, 'outcome': 'insufficient_budget_for_one_share'},
        {'decision_index': 2, 'entry_order_id': None, 'outcome': 'risk_exit_active'},
    ]
    assert episode['entry_diagnostics']['entries'] == []
    assert_entry_accounting(episode)


def test_unfilled_order_and_pending_buy_rejection_are_separate():
    env = ReplayFixture([100] * 5, execution_config={
        'order_latency_ms': 10000, 'order_ttl_seconds': 20, 'slippage_bps': 0, 'spread_bps': 0})
    _, episode = finish(env, [1, 1, 0, 0])
    trace = episode['entry_diagnostics']
    assert trace['buy_decisions'][1] == {
        'decision_index': 1, 'entry_order_id': None, 'outcome': 'pending_buy'}
    entry, = trace['entries']
    assert entry['submitted_quantity'] > 0
    assert entry['filled_quantity'] == entry['sold_quantity'] == entry['open_quantity'] == 0
    assert entry['order_status'] == 'cancelled' and not entry['complete']
    assert entry['net_pnl'] == entry['matched_fees'] == 0
    assert entry['exit_reason'] == 'unfilled'
    assert_entry_accounting(episode)


def test_partially_filled_cancelled_entry_is_complete_when_all_actual_fills_are_sold():
    env = ReplayFixture([100] * 5, seconds=[0, .1, .2, .3, .4],
                        initial_cash=1000, max_stages=1,
                        execution=book([99] * 5, [100] * 5, asks=[100] * 5,
                                       ask_sizes=[1, 1, 100, 100, 100]))
    _, episode = finish(env, [1, 2, 0, 0])
    entry, = episode['entry_diagnostics']['entries']
    assert entry['submitted_quantity'] == 10
    assert entry['filled_quantity'] == entry['sold_quantity'] == 1
    assert entry['open_quantity'] == 0
    assert entry['order_status'] == 'cancelled' and entry['complete']
    assert entry['exit_reason'] == 'signal'
    assert_entry_accounting(episode)


def test_late_entry_fill_and_split_exits_still_link_to_the_original_buy_decision():
    env = ReplayFixture([100] * 6, seconds=[0, .1, .11, .2, .3, .4],
                        initial_cash=1000, max_stages=1,
                        execution=book([99] * 6, [100] * 6, asks=[100] * 6,
                                       ask_sizes=[1, 1, 100, 100, 100, 100]))
    _, episode = finish(env, [1, 2, 1, 0, 0])
    assert len(episode['trades']) == 2
    assert [trade['quantity'] for trade in episode['trades']] == [1, 9]
    trace = episode['entry_diagnostics']
    assert trace['buy_decisions'][1] == {
        'decision_index': 2, 'entry_order_id': None, 'outcome': 'signal_exit_active'}
    entry, = trace['entries']
    assert entry['entry_order_id'] == 1 and entry['decision_index'] == 0
    assert entry['filled_quantity'] == entry['sold_quantity'] == 10
    assert entry['complete']
    assert entry['quantity_weighted_holding_time'] == pytest.approx(
        episode['round_trips'][0]['quantity_weighted_holding_time'])
    assert_entry_accounting(episode)


def test_open_entry_fees_are_not_counted_as_realized_matched_costs():
    env = ReplayFixture([100] * 5, initial_cash=1000, max_stages=1,
                        transaction_cost_rate=.001,
                        execution=book([99] * 5, [1] * 5, asks=[100] * 5))
    _, episode = finish(env, [1, 2, 0, 0])
    entry, = episode['entry_diagnostics']['entries']
    assert entry['filled_quantity'] == 9 and entry['sold_quantity'] == 1
    assert entry['open_quantity'] == 8 and not entry['complete']
    assert entry['entry_fees'] == pytest.approx(.9)
    assert entry['exit_fees'] == pytest.approx(.099)
    assert entry['matched_fees'] == pytest.approx(.199)
    assert entry['net_pnl'] == pytest.approx(-1.199)
    assert_entry_accounting(episode)


def test_entry_with_no_sales_remains_open_without_a_realized_loss_label():
    env = ReplayFixture([100] * 4, initial_cash=1000, max_stages=1,
                        transaction_cost_rate=.001,
                        execution=book([99] * 4, [100, 100, 0, 0], asks=[100] * 4))
    _, episode = finish(env, [1, 0, 0])
    entry, = episode['entry_diagnostics']['entries']
    assert entry['filled_quantity'] == entry['open_quantity'] == 9
    assert entry['sold_quantity'] == 0 and not entry['complete']
    assert entry['net_pnl'] == entry['matched_fees'] == 0
    assert entry['entry_fees'] > 0
    assert_entry_accounting(episode)


def test_multiple_entries_keep_order_ids_separate_from_policy_indices_and_reset():
    env = ReplayFixture([100] * 7, initial_cash=1000, max_stages=1)
    _, episode = finish(env, [1, 2, 0, 1, 2, 0])
    trace = episode['entry_diagnostics']
    assert [(entry['entry_order_id'], entry['decision_index']) for entry in trace['entries']] == [(1, 0), (3, 3)]
    assert_entry_accounting(episode)
    snapshot = json.dumps(trace, sort_keys=True)
    _, empty_episode = finish(env, [0] * 6)
    assert empty_episode['entry_diagnostics'] == {
        'version': 1, 'decision_count': 6, 'buy_decisions': [], 'entries': []}
    assert json.dumps(trace, sort_keys=True) == snapshot
    assert_entry_accounting(empty_episode)
    _, third_episode = finish(env, [0, 1, 2, 0, 0, 0])
    entry, = third_episode['entry_diagnostics']['entries']
    assert entry['entry_order_id'] == 1 and entry['decision_index'] == 1
    assert_entry_accounting(third_episode)


def test_timed_market_events_and_closeout_tail_do_not_increment_policy_indices():
    env = ReplayFixture([100] * 9, seconds=[0, .1, .2, .3, .7, 1, 1.1, 2, 3],
                        initial_cash=1000, max_stages=1, decision_interval_seconds=1,
                        max_episode_steps=2, liquidation_max_steps=2)
    _, episode = finish(env, [0, 1])
    trace = episode['entry_diagnostics']
    assert trace['decision_count'] == 2 and episode['market_steps_taken'] == 8
    assert trace['buy_decisions'] == [{'decision_index': 1, 'entry_order_id': 1, 'outcome': 'submitted'}]
    assert trace['entries'][0]['complete']
    assert_entry_accounting(episode)


def test_entry_cost_decomposition_reconciles_depth_slippage_fees_and_exit_fragments():
    n = 7
    execution = {
        'bid_prices': np.array([[99, 98], [99, 98], [103, 102], [103, 102], [104, 103], [105, 104], [105, 104]]),
        'ask_prices': np.array([[101, 102], [101, 102], [105, 106], [105, 106], [106, 107], [107, 108], [107, 108]]),
        'bid_sizes': np.tile([2, 100], (n, 1)), 'ask_sizes': np.tile([2, 100], (n, 1)),
    }
    env = ReplayFixture([100, 100, 104, 104, 105, 106, 106], execution=execution,
                        initial_cash=1000, max_stages=1, transaction_cost_rate=.001,
                        execution_config={'order_latency_ms': 0, 'slippage_bps': 10, 'tick_size': .01})
    _, episode = finish(env, [1, 0, 2, 0, 0, 0])
    entry, = episode['entry_diagnostics']['entries']
    assert len(episode['trades']) > 1 and entry['complete']
    assert entry['mid_price_pnl'] > 0
    assert all(entry[key] > 0 for key in ('spread_cost', 'depth_cost', 'slippage_cost', 'matched_fees'))
    assert_entry_accounting(episode)
