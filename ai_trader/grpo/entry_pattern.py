"""Causal entry selection and post-rollout, fill-based short-horizon targets.

Selection uses only past prices and actual cumulative BUY execution notional.
Future prices are used exclusively for training labels, never observations/masks.
"""
from collections import deque

import numpy as np


DEFAULT_ENTRY_PATTERN = {
    'min_buy_notional_krw': 3_000_000_000.0,
    'min_price_return': 0.01,
    'min_window_seconds': 5.0,
    'max_window_seconds': 60.0,
    'target_min_seconds': 1.0,
    'target_max_seconds': 5.0,
    'policy_coef': 1.0,
}


def cumulative_buy_from_signed_trades(prices_krw, signed_volume, cumulative_volume):
    """Negative volume is SELL; every nonnegative value is BUY (zero adds zero).

    An entirely unsigned/nonnegative dataset is accepted as all BUY. Merged
    quote/trader rows repeat the last trade's volume: count only rows where
    cumulative volume increases, requiring a matching reported trade quantity.
    The first row establishes a baseline; earlier trades are unknown/excluded.
    """
    p, v, c = (np.asarray(x, dtype=np.float64) for x in
               (prices_krw, signed_volume, cumulative_volume))
    if (p.ndim != 1 or not len(p) or v.shape != p.shape or c.shape != p.shape
            or not all(np.isfinite(x).all() for x in (p, v, c))
            or np.any(p <= 0) or np.any(c < 0)):
        raise ValueError('Invalid signed-trade source arrays')
    delta = np.diff(c, prepend=c[0])
    changed = delta > 0
    if np.any(delta < 0) or not np.allclose(delta[changed], np.abs(v[changed]), rtol=0, atol=1e-6):
        raise ValueError('Cumulative volume increments must match signed trade quantity; trade direction is incomplete')
    return np.cumsum(np.where(changed & (v >= 0), p * delta, 0.0))


def validate_entry_pattern(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) - set(DEFAULT_ENTRY_PATTERN):
        raise ValueError('Invalid entry_pattern_config keys')
    result = {**DEFAULT_ENTRY_PATTERN, **config}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'entry_pattern_config.{key} must be finite and positive')
    if not 5 <= result['min_window_seconds'] <= result['max_window_seconds'] <= 60:
        raise ValueError('Entry selection windows must lie in [5, 60] seconds')
    if not 1 <= result['target_min_seconds'] <= result['target_max_seconds'] <= 5:
        raise ValueError('Entry target window must lie in [1, 5] seconds')
    return result


def entry_signal(timestamps, prices, cumulative_buy_notional_krw, config):
    """Find any common lookback window satisfying BOTH notional and price gain.

Consider every historical event boundary plus the as-of 60-second boundary.
The cumulative series makes the notional constraint monotonic; a deque queries
the minimum starting price among eligible boundaries in O(N) time.
No partial history shorter than min_window_seconds is accepted.
"""
    cfg = validate_entry_pattern(config)
    t, p, c = (np.asarray(x, dtype=np.float64) for x in
               (timestamps, prices, cumulative_buy_notional_krw))
    if (t.ndim != 1 or not len(t) or p.shape != t.shape or c.shape != t.shape
            or not all(np.isfinite(x).all() for x in (t, p, c))
            or np.any(np.diff(t) < 0) or np.any(np.diff(c) < 0)
            or np.any(p <= 0) or np.any(c < 0)):
        raise ValueError('Entry signal requires chronological prices and nondecreasing cumulative BUY notional')
    upper = np.minimum(np.searchsorted(t, t - cfg['min_window_seconds'], side='right') - 1,
                       np.searchsorted(c, c - cfg['min_buy_notional_krw'], side='right') - 1)
    lower = np.maximum(0, np.searchsorted(t, t - cfg['max_window_seconds'], side='right') - 1)
    signal = np.zeros(len(t), dtype=bool)
    candidates, added = deque(), -1
    for i, right in enumerate(upper):
        while added < right:
            added += 1
            while candidates and p[candidates[-1]] >= p[added]:
                candidates.pop()
            candidates.append(added)
        while candidates and candidates[0] < lower[i]:
            candidates.popleft()
        if candidates:
            signal[i] = p[i] >= p[candidates[0]] * (1 + cfg['min_price_return'])
    return signal


def signal_ranges(signal):
    """Compact inclusive ranges used to reject unsuitable files before sampling."""
    edges = np.diff(np.r_[False, np.asarray(signal, dtype=bool), False].astype(np.int8))
    return [[int(a), int(b - 1)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def fill_target(timestamps, prices, fill_time, fill_price, config):
    """Return +1/-1, or None for a censored/no-observation target.

Use actual observed last-trade prices in the inclusive [1, 5] second window
after this fill. Require data through the entire horizon even for early hits.
"""
    cfg = validate_entry_pattern(config)
    end = fill_time + cfg['target_max_seconds']
    if timestamps[-1] < end:
        return None
    left = np.searchsorted(timestamps, fill_time + cfg['target_min_seconds'], side='left')
    right = np.searchsorted(timestamps, end, side='right')
    if left >= right:
        return None
    return 1.0 if np.max(prices[left:right]) >= fill_price else -1.0


def pattern_policy_credit(episodes):
    """Separate PPO auxiliary signal; NAV rewards/critic targets stay unchanged."""
    result = []
    for ep in episodes:
        actions = np.asarray(ep['actions'])
        report = ep.get('metadata', {}).get('entry_pattern')
        credit = np.zeros(len(actions), dtype=np.float32)
        if report is not None:
            credit = np.asarray(report['policy_credit'], dtype=np.float32)
            if (credit.shape != actions.shape or not np.isfinite(credit).all()
                    or np.any(credit[actions != 1] != 0)):
                raise ValueError('Invalid entry pattern policy credit')
        result.append(credit)
    return np.concatenate(result)
