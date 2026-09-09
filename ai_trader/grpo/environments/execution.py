"""Deterministic order replay against displayed depth, with explicit fallbacks.

Passive queue fills are deliberately not inferred from last-trade prices. Limit
orders execute only when the opposite displayed book is marketable. Unchanged
displayed liquidity cannot be consumed repeatedly by this simulated account.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import math


@dataclass(frozen=True)
class ExecutionConfig:
    order_latency_ms: float = 100.0
    cancel_latency_ms: float = 50.0
    order_ttl_seconds: float = 2.0
    spread_bps: float = 10.0
    slippage_bps: float = 2.0
    fallback_depth: int = 100
    require_order_book: bool = False
    max_quote_age_seconds: float = 1.0
    tick_size: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == 'require_order_book':
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.order_ttl_seconds <= 0 or self.fallback_depth < 1:
            raise ValueError('order_ttl_seconds and fallback_depth must be positive')
        if self.spread_bps >= 10000 or self.slippage_bps >= 10000:
            raise ValueError('spread/slippage must be less than 10000 bps')


@dataclass
class Order:
    order_id: int
    side: str
    quantity: int
    submitted_at: float
    submitted_event: int
    eligible_at: float
    expires_at: float
    limit_price: Optional[float] = None
    time_in_force: str = 'GTC'
    reason: str = 'signal'
    filled_quantity: int = 0
    status: str = 'pending'
    cancel_at: Optional[float] = None

    @property
    def remaining(self):
        return self.quantity - self.filled_quantity

    @property
    def active(self):
        return self.status in ('pending', 'partial')


@dataclass(frozen=True)
class Fill:
    order_id: int
    side: str
    quantity: int
    price: float
    timestamp: float
    reason: str


@dataclass
class MarketSnapshot:
    timestamp: float
    last_price: float
    bids: list = field(default_factory=list)  # [(price in KRW, shares)]
    asks: list = field(default_factory=list)
    quote_timestamp: Optional[float] = None
    synthetic: bool = False


class ExecutionSimulator:
    def __init__(self, config=None):
        self.config = config if isinstance(config, ExecutionConfig) else ExecutionConfig(**(config or {}))
        self.reset()

    def reset(self):
        self.orders = []
        self.fills = []
        self.event_index = -1
        self.snapshot = None
        self._next_id = 1
        self._synthetic_events = 0
        self._real_events = 0
        self._displayed = {'buy': {}, 'sell': {}}
        self._remaining = {'buy': {}, 'sell': {}}
        self._execution_blocked_checks = dict.fromkeys((
            'stale_quote', 'not_next_event', 'order_latency', 'no_displayed_depth',
            'displayed_depth_exhausted', 'insufficient_cash', 'insufficient_inventory',
            'limit_price_not_marketable'), 0)

    def submit(self, side, quantity, timestamp, *, limit_price=None, time_in_force='GTC', reason='signal'):
        if side not in ('buy', 'sell') or time_in_force not in ('GTC', 'IOC'):
            raise ValueError('unsupported side or time_in_force')
        if not math.isfinite(quantity) or int(quantity) != quantity or quantity <= 0:
            raise ValueError('quantity must be positive whole shares')
        if not math.isfinite(timestamp):
            raise ValueError('invalid order timestamp')
        if self.snapshot and timestamp < self.snapshot.timestamp:
            raise ValueError('cannot submit an order in the past')
        if limit_price is not None and (not math.isfinite(limit_price) or limit_price <= 0):
            raise ValueError('invalid limit price')
        order = Order(self._next_id, side, int(quantity), float(timestamp), self.event_index,
                      timestamp + self.config.order_latency_ms / 1000,
                      timestamp + self.config.order_ttl_seconds, limit_price, time_in_force, reason)
        self._next_id += 1
        self.orders.append(order)
        return order

    def cancel(self, order_id, timestamp):
        order = next(o for o in self.orders if o.order_id == order_id)
        if order.active:
            cancel_at = timestamp + self.config.cancel_latency_ms / 1000
            order.cancel_at = min(order.cancel_at, cancel_at) if order.cancel_at is not None else cancel_at

    def cancel_all(self, timestamp, *, side=None, end_of_replay=False):
        for order in self.orders:
            if order.active and (side is None or order.side == side):
                if end_of_replay:
                    order.status = 'cancelled'
                else:
                    self.cancel(order.order_id, timestamp)

    @staticmethod
    def _levels(levels, reverse):
        aggregate = {}
        for price, size in levels:
            price, size = float(price), float(size)
            if not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size < 0:
                raise ValueError('invalid displayed price/depth')
            if size >= 1:
                aggregate[price] = aggregate.get(price, 0) + int(size)
        return sorted(aggregate.items(), reverse=reverse)

    def make_snapshot(self, timestamp, last_price, *, bids=None, asks=None, quote_timestamp=None):
        if not math.isfinite(timestamp) or not math.isfinite(last_price) or last_price <= 0:
            raise ValueError('invalid market timestamp/last price')
        has_book = bids is not None and asks is not None
        if not has_book:
            if self.config.require_order_book:
                raise ValueError('real order book required; regenerate cache with execution arrays')
            half = self.config.spread_bps / 20000
            bids = [(last_price * (1 - half), self.config.fallback_depth)]
            asks = [(last_price * (1 + half), self.config.fallback_depth)]
        bids = self._levels(bids, reverse=True)
        asks = self._levels(asks, reverse=False)
        if bids and asks and (bids[0][0] > asks[0][0] or (has_book and bids[0][0] == asks[0][0])):
            raise ValueError('locked/crossed book is not continuous-session executable depth')
        if quote_timestamp is not None and (not math.isfinite(quote_timestamp) or quote_timestamp > timestamp):
            raise ValueError('invalid quote timestamp')
        return MarketSnapshot(float(timestamp), float(last_price), bids, asks, quote_timestamp, not has_book)

    def _set_snapshot(self, snapshot):
        if self.snapshot and snapshot.timestamp < self.snapshot.timestamp:
            raise ValueError('market time moved backwards')
        self.snapshot = snapshot
        self.event_index += 1
        self._synthetic_events += int(snapshot.synthetic)
        self._real_events += int(not snapshot.synthetic)
        for side, levels in [('buy', snapshot.asks), ('sell', snapshot.bids)]:
            displayed = dict(levels)
            previous = self._displayed[side]
            remaining = self._remaining[side]
            if snapshot.synthetic:
                available = displayed.copy()  # Explicit per-event fallback assumption.
            else:
                available = {
                    p: max(0, min(q, remaining.get(p, 0) + q - previous.get(p, 0)))
                    for p, q in displayed.items()
                }
            self._displayed[side] = displayed
            self._remaining[side] = available

    def execution_price(self, price, side):
        factor = 1 + (1 if side == 'buy' else -1) * self.config.slippage_bps / 10000
        price *= factor
        tick = self.config.tick_size
        if tick > 0:
            price = (math.ceil(price / tick) if side == 'buy' else math.floor(price / tick)) * tick
        if price <= 0:
            raise ValueError('tick_size rounds this execution price to zero; configure valid tick units')
        return price

    def liquidation_mark(self):
        if self.snapshot is None:
            raise RuntimeError('market snapshot required')
        bid = self.snapshot.bids[0][0] if self.snapshot.bids else self.snapshot.last_price * (1 - self.config.spread_bps / 20000)
        return self.execution_price(bid, 'sell')

    def process(self, snapshot, *, cash_available=math.inf, sell_available=math.inf,
                buy_fee_rate=0.0, sell_fee_rate=0.0):
        self._set_snapshot(snapshot)
        new_fills = []
        stale = snapshot.quote_timestamp is not None and snapshot.timestamp - snapshot.quote_timestamp > self.config.max_quote_age_seconds
        for order in self.orders:
            if not order.active:
                continue
            if order.cancel_at is not None and snapshot.timestamp >= order.cancel_at:
                order.status = 'cancelled'
                continue
            if snapshot.timestamp > order.expires_at:
                order.status = 'expired'
                continue
            # Even zero-latency orders cannot consume the already observed event.
            if stale:
                self._execution_blocked_checks['stale_quote'] += 1
                continue
            if self.event_index <= order.submitted_event:
                self._execution_blocked_checks['not_next_event'] += 1
                continue
            if snapshot.timestamp < order.eligible_at:
                self._execution_blocked_checks['order_latency'] += 1
                continue
            levels = snapshot.asks if order.side == 'buy' else snapshot.bids
            blocked_reasons = set()
            if not levels:
                blocked_reasons.add('no_displayed_depth')
            for book_price, _ in levels:
                price = self.execution_price(book_price, order.side)
                if order.limit_price is not None:
                    if (order.side == 'buy' and price > order.limit_price) or (order.side == 'sell' and price < order.limit_price):
                        blocked_reasons.add('limit_price_not_marketable')
                        break
                available = self._remaining[order.side].get(book_price, 0)
                qty = min(order.remaining, available)
                if order.side == 'buy' and math.isfinite(cash_available):
                    qty = min(qty, max(0, int(cash_available / (price * (1 + buy_fee_rate)))))
                if order.side == 'sell' and math.isfinite(sell_available):
                    qty = min(qty, max(0, int(sell_available)))
                if qty <= 0:
                    if available <= 0:
                        blocked_reasons.add('displayed_depth_exhausted')
                    elif order.side == 'buy':
                        blocked_reasons.add('insufficient_cash')
                    else:
                        blocked_reasons.add('insufficient_inventory')
                    continue
                fill = Fill(order.order_id, order.side, qty, price, snapshot.timestamp, order.reason)
                order.filled_quantity += qty
                self._remaining[order.side][book_price] -= qty
                if order.side == 'buy':
                    cash_available -= qty * price * (1 + buy_fee_rate)
                    sell_available += qty
                else:
                    cash_available += qty * price * (1 - sell_fee_rate)
                    sell_available -= qty
                new_fills.append(fill)
                order.status = 'filled' if order.remaining == 0 else 'partial'
                if not order.active:
                    break
            for reason in blocked_reasons:
                self._execution_blocked_checks[reason] += 1
            if order.active and order.time_in_force == 'IOC':
                order.status = 'cancelled'
        self.fills.extend(new_fills)
        return new_fills

    def summary(self):
        """Include blocked checks counted once per reason per active-order/event.

        These are not distinct order totals: repeated events may count the same
        order again, and a partial-fill event may encounter multiple reasons.
        Expiry/cancellation and earlier eligibility guards take precedence.
        """
        submitted = sum(o.quantity for o in self.orders)
        filled = sum(o.filled_quantity for o in self.orders)
        return {
            'submitted_orders': len(self.orders), 'submitted_quantity': submitted,
            'filled_quantity': filled, 'fill_ratio': filled / submitted if submitted else 0.0,
            'partial_orders': sum(0 < o.filled_quantity < o.quantity for o in self.orders),
            'cancelled_orders': sum(o.status == 'cancelled' for o in self.orders),
            'expired_orders': sum(o.status == 'expired' for o in self.orders),
            'execution_blocked_checks': self._execution_blocked_checks.copy(),
            'execution_model': ('mixed' if self._synthetic_events and self._real_events else
                                'synthetic_spread' if self._synthetic_events else 'displayed_depth'),
            'synthetic_book_events': self._synthetic_events,
            'real_book_events': self._real_events,
        }
