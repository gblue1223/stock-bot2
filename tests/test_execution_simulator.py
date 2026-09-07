import pytest

from ai_trader.grpo.environments.execution import ExecutionSimulator


def engine(**kwargs):
    return ExecutionSimulator(dict(order_latency_ms=0, slippage_bps=0, **kwargs))


def quote(sim, time, bids=((99, 10),), asks=((101, 10),), **kwargs):
    return sim.make_snapshot(time, 100, bids=bids, asks=asks, **kwargs)


def test_latency_uses_later_quote_and_never_submission_snapshot():
    sim = ExecutionSimulator({'order_latency_ms': 100, 'slippage_bps': 0})
    sim.process(quote(sim, 0))
    order = sim.submit('buy', 2, 0)
    assert sim.fills == []
    assert sim.process(quote(sim, .05)) == []
    fills = sim.process(quote(sim, .11, asks=((103, 10),)))
    assert fills[0].price == 103
    assert order.status == 'filled'


def test_shared_depth_partial_fills_and_no_unchanged_snapshot_refill():
    sim = engine()
    sim.process(quote(sim, 0, asks=((101, 3), (102, 2))))
    first = sim.submit('buy', 4, 0)
    second = sim.submit('buy', 4, 0)
    fills = sim.process(quote(sim, .1, asks=((101, 3), (102, 2))))
    assert [(f.price, f.quantity) for f in fills] == [(101, 3), (102, 1), (102, 1)]
    assert first.status == 'filled' and second.status == 'partial'
    assert sim.process(quote(sim, .2, asks=((101, 3), (102, 2)))) == []
    fills = sim.process(quote(sim, .3, asks=((101, 5), (102, 2))))
    assert sum(f.quantity for f in fills) == 2  # Only the new displayed increment.


def test_limit_order_does_not_fill_on_last_trade_touch():
    sim = engine()
    sim.process(quote(sim, 0))
    order = sim.submit('buy', 2, 0, limit_price=100)
    assert sim.process(quote(sim, .1)) == []
    assert order.active
    fills = sim.process(quote(sim, .2, bids=((98, 10),), asks=((100, 10),)))
    assert fills[0].price == 100


def test_external_depth_decrease_does_not_restore_our_consumed_liquidity():
    sim = engine()
    sim.process(quote(sim, 0, asks=((101, 10),)))
    sim.submit('buy', 5, 0)
    assert sum(f.quantity for f in sim.process(quote(sim, .1, asks=((101, 10),)))) == 5
    sim.submit('buy', 10, .1)
    assert sim.process(quote(sim, .2, asks=((101, 5),))) == []
    assert sum(f.quantity for f in sim.process(quote(sim, .3, asks=((101, 10),)))) == 5


def test_cancellation_latency_allows_a_fill_before_cancel_arrives():
    sim = engine(cancel_latency_ms=50)
    sim.process(quote(sim, 0))
    order = sim.submit('buy', 2, 0)
    sim.cancel(order.order_id, .01)
    assert len(sim.process(quote(sim, .03))) == 1
    assert order.status == 'filled'


def test_cancel_arrives_before_fill_and_order_expiry():
    sim = engine(cancel_latency_ms=50, order_ttl_seconds=.2)
    sim.process(quote(sim, 0))
    cancelled = sim.submit('buy', 2, 0)
    expired = sim.submit('buy', 2, 0, limit_price=90)
    sim.cancel(cancelled.order_id, 0)
    assert sim.process(quote(sim, .1)) == []
    assert cancelled.status == 'cancelled'
    sim.process(quote(sim, .3))
    assert expired.status == 'expired'


def test_stale_book_does_not_fill_and_cash_cannot_go_negative():
    sim = engine(max_quote_age_seconds=.1)
    sim.process(quote(sim, 0))
    order = sim.submit('buy', 100, 0)
    assert sim.process(quote(sim, .2, quote_timestamp=0)) == []
    fills = sim.process(quote(sim, .3), cash_available=250, buy_fee_rate=.01)
    assert sum(f.quantity for f in fills) == 2
    assert order.remaining == 98


def test_short_selling_is_capped_and_ioc_remainder_cancelled():
    sim = engine()
    sim.process(quote(sim, 0))
    order = sim.submit('sell', 10, 0, time_in_force='IOC')
    fills = sim.process(quote(sim, .1), sell_available=3)
    assert sum(f.quantity for f in fills) == 3
    assert order.status == 'cancelled' and order.remaining == 7


def test_missing_real_book_is_explicit_and_can_be_rejected():
    sim = engine(spread_bps=20)
    snap = sim.make_snapshot(0, 100)
    assert snap.synthetic and snap.bids[0][0] == pytest.approx(99.9)
    strict = engine(require_order_book=True)
    with pytest.raises(ValueError, match='real order book'):
        strict.make_snapshot(0, 100)


def test_invalid_or_backwards_market_data_rejected():
    sim = engine()
    sim.process(quote(sim, 2))
    with pytest.raises(ValueError, match='backwards'):
        sim.process(quote(sim, 1))
    with pytest.raises(ValueError, match='locked/crossed'):
        quote(sim, 3, bids=((102, 10),))


def test_empty_book_is_not_replaced_with_synthetic_liquidity():
    sim = engine()
    sim.process(quote(sim, 0, bids=(), asks=()))
    sim.submit('buy', 5, 0)
    assert sim.process(quote(sim, .1, bids=(), asks=())) == []
    assert not sim.snapshot.synthetic


def test_invalid_tick_never_improves_a_sell_above_the_bid():
    sim = engine(tick_size=100)
    with pytest.raises(ValueError, match='tick_size'):
        sim.execution_price(99, 'sell')
