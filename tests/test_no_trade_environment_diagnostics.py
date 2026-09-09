import pytest

from ai_trader.grpo.environments.execution import ExecutionSimulator
from test_scalping_accounting import ReplayFixture, finish


def quote(sim, timestamp, *, asks=((101, 10),), bids=((99, 10),), **kwargs):
    return sim.make_snapshot(timestamp, 100, asks=asks, bids=bids, **kwargs)


def test_unaffordable_buy_and_terminal_buy_have_distinct_outcomes_and_reset():
    env = ReplayFixture([100] * 4, initial_cash=100, max_stages=5)
    reward, episode = finish(env, [1, 2, 1])

    outcomes = episode['buy_action_outcomes']
    assert outcomes['insufficient_budget_for_one_share'] == 1
    assert outcomes['risk_exit_active'] == 1
    assert sum(outcomes.values()) == 2
    assert episode['submitted_orders'] == episode['filled_quantity'] == 0
    assert reward == episode['net_return'] == 0
    assert env.cash == 100

    env.reset()
    assert all(count == 0 for count in env.buy_action_outcomes.values())
    # Report dictionaries must remain snapshots after another reset/step.
    env.step(1)
    assert outcomes['insufficient_budget_for_one_share'] == 1
    assert outcomes['risk_exit_active'] == 1


def test_buy_while_fully_invested_reports_stage_limit_without_new_order():
    env = ReplayFixture([100] * 4, max_stages=1)
    reward, episode = finish(env, [1, 1, 0])

    assert episode['buy_action_outcomes']['submitted'] == 1
    assert episode['buy_action_outcomes']['max_stages'] == 1
    assert sum(episode['buy_action_outcomes'].values()) == 2
    assert episode['submitted_orders'] == 2  # One entry and terminal liquidation.
    assert episode['num_trades'] == 1
    assert episode['open_quantity'] == 0
    assert reward == episode['net_return'] == 0


def test_pending_buy_prevents_another_order_and_reset_clears_execution_checks():
    env = ReplayFixture([100] * 5, execution_config={
        'order_latency_ms': 3000, 'order_ttl_seconds': 10,
        'slippage_bps': 0, 'spread_bps': 0})
    env.reset()
    env.step(1)
    env.step(1)

    episode = env._calculate_episode_metadata()
    assert episode['buy_action_outcomes']['submitted'] == 1
    assert episode['buy_action_outcomes']['pending_buy'] == 1
    assert sum(episode['buy_action_outcomes'].values()) == 2
    assert episode['submitted_orders'] == 1
    assert episode['filled_quantity'] == 0
    assert episode['execution_blocked_checks']['order_latency'] == 2

    env.reset()
    fresh = env._calculate_episode_metadata()
    assert fresh['submitted_orders'] == 0
    assert all(value == 0 for value in fresh['buy_action_outcomes'].values())
    assert all(value == 0 for value in fresh['execution_blocked_checks'].values())
    assert episode['execution_blocked_checks']['order_latency'] == 2


def test_buy_after_trade_limit_with_remaining_stage_reports_trade_limit():
    env = ReplayFixture([100] * 7, max_stages=2, max_trades_per_episode=1)
    env.reset()
    for action in [1, 1, 2, 1]:
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated

    assert env.quantity > 0
    assert len(env.episode_trades) == 1
    assert env.buy_action_outcomes['submitted'] == 2
    assert env.buy_action_outcomes['max_trades'] == 1
    assert sum(env.buy_action_outcomes.values()) == 3
    assert env.simulator.summary()['submitted_orders'] == 3


def test_repeated_stale_checks_are_not_distinct_orders_and_expiry_takes_precedence():
    sim = ExecutionSimulator({
        'order_latency_ms': 0, 'slippage_bps': 0,
        'max_quote_age_seconds': .1, 'order_ttl_seconds': .4})
    sim.process(quote(sim, 0))
    order = sim.submit('buy', 1, 0)
    for timestamp in [.2, .3, .5]:
        assert sim.process(quote(sim, timestamp, quote_timestamp=0)) == []

    summary = sim.summary()
    assert summary['submitted_orders'] == 1
    assert summary['execution_blocked_checks']['stale_quote'] == 2
    assert summary['expired_orders'] == 1
    assert summary['filled_quantity'] == 0
    assert order.status == 'expired'

    sim.reset()
    assert sim.summary()['submitted_orders'] == 0
    assert all(value == 0 for value in sim.summary()['execution_blocked_checks'].values())
    assert summary['execution_blocked_checks']['stale_quote'] == 2


@pytest.mark.parametrize('side,limit_price,asks,resources,reason', [
    ('buy', None, (), {}, 'no_displayed_depth'),
    ('buy', None, ((101, 10), (102, 10)), {'cash_available': 100}, 'insufficient_cash'),
    ('sell', None, ((101, 10),), {'sell_available': 0}, 'insufficient_inventory'),
    ('buy', 100, ((101, 10),), {}, 'limit_price_not_marketable'),
])
def test_execution_block_reasons_count_once_per_order_event(
        side, limit_price, asks, resources, reason):
    sim = ExecutionSimulator({'order_latency_ms': 0, 'slippage_bps': 0})
    sim.process(quote(sim, 0, asks=asks))
    order = sim.submit(side, 1, 0, limit_price=limit_price)

    assert sim.process(quote(sim, .1, asks=asks), **resources) == []
    assert order.active
    summary = sim.summary()
    assert summary['execution_blocked_checks'][reason] == 1
    assert sum(summary['execution_blocked_checks'].values()) == 1
    assert summary['submitted_orders'] == 1
    assert summary['filled_quantity'] == 0


def test_consumed_displayed_depth_is_reported_without_refilling_same_quote():
    sim = ExecutionSimulator({'order_latency_ms': 0, 'slippage_bps': 0})
    sim.process(quote(sim, 0, asks=((101, 1),)))
    sim.submit('buy', 1, 0)
    fills = sim.process(quote(sim, .1, asks=((101, 1),)))
    assert [(fill.price, fill.quantity) for fill in fills] == [(101, 1)]
    sim.submit('buy', 1, .1)

    assert sim.process(quote(sim, .2, asks=((101, 1),))) == []
    summary = sim.summary()
    assert summary['execution_blocked_checks']['displayed_depth_exhausted'] == 1
    assert summary['submitted_orders'] == 2
    assert summary['filled_quantity'] == 1
