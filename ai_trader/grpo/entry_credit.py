"""Join realized entry outcomes to frozen actor/critic targets, using CPU arrays only.

This describes the observed rollout, not counterfactual profits or the gradient of
the shared neural network. It never reads observations, recomputes GAE, changes
targets, or runs a policy. Money-valued entry attribution and percent-NAV rewards
are deliberately kept in separate fields.
"""
from collections import Counter
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


_MONEY_FIELDS = (
    'net_pnl', 'mid_price_pnl', 'spread_cost', 'depth_cost', 'slippage_cost',
    'matched_fees', 'pnl_attribution_residual', 'entry_fees', 'exit_fees',
)
_QUANTITY_FIELDS = ('submitted_quantity', 'filled_quantity', 'sold_quantity', 'open_quantity')
_SIGNAL_FIELDS = (
    'raw_gae', 'group_component', 'pre_normalized_advantage', 'normalized_advantage',
    'return_target', 'cached_value', 'critic_target_error', 'immediate_reward',
    'bootstrap_value_delta', 'immediate_td_error', 'future_td_trace',
    'gae_trace_consistency_residual',
)


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'entry credit: {name} must be an integer >= {minimum}')
    return int(value)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f'entry credit: {name} must be a finite number')
    return float(value)


def _vector(value, length, name, *, optional=False):
    if value is None and optional:
        return None
    array = np.asarray(value)
    if (array.shape != (length,) or array.dtype.kind not in 'iuf'
            or not np.isfinite(array).all()):
        raise ValueError(f'entry credit: {name} must be a finite numeric vector of length {length}')
    return array


def _episode_key(metadata):
    """Copy optional primitive identity fields without retaining other metadata."""
    start = metadata.get('episode_start', {}) if isinstance(metadata, Mapping) else {}
    key = start.get('episode_key', {}) if isinstance(start, Mapping) else {}
    if not isinstance(key, Mapping):
        return {}
    result = {}
    for name in ('stock_code', 'date', 'start_index'):
        if name not in key:
            continue
        value = key[name]
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int)):
            result[name] = value
        elif isinstance(value, float) and np.isfinite(value):
            result[name] = value
    return result


def _summary(rows):
    result = {'sample_count': len(rows), 'available': int(bool(rows))}
    for field in _SIGNAL_FIELDS:
        selected = np.asarray([row[field] for row in rows if row[field] is not None], dtype=np.float64)
        result[f'{field}_available'] = int(bool(selected.size))
        result[f'{field}_sample_count'] = int(selected.size)
        result[f'{field}_mean'] = float(selected.mean()) if selected.size else 0.0
        result[f'{field}_std'] = float(selected.std()) if selected.size else 0.0
        result[f'{field}_negative_count'] = int((selected < 0).sum())
        result[f'{field}_negative_fraction'] = float((selected < 0).mean()) if selected.size else 0.0
        result[f'{field}_positive_fraction'] = float((selected > 0).mean()) if selected.size else 0.0
    return result


def _trace_entries(trace, actions):
    """Validate version-1 metadata without silently treating broken data as legacy."""
    if not isinstance(trace, Mapping) or _integer(trace.get('version'), 'version') != 1:
        raise ValueError('entry credit: entry_diagnostics must have version 1')
    if _integer(trace.get('decision_count'), 'decision_count') != len(actions):
        raise ValueError('entry credit: decision_count does not match the episode')
    decisions, entries = trace.get('buy_decisions'), trace.get('entries')
    if not isinstance(decisions, list) or not isinstance(entries, list):
        raise ValueError('entry credit: buy_decisions and entries must be lists')
    decision_map, order_map = {}, {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError('entry credit: each buy decision must be an object')
        index = _integer(decision.get('decision_index'), 'decision_index')
        if index >= len(actions) or actions[index] != 1 or index in decision_map:
            raise ValueError('entry credit: BUY decision indices must be unique and select BUY actions')
        order_id = decision.get('entry_order_id')
        if order_id is not None:
            order_id = _integer(order_id, 'entry_order_id')
            if order_id in order_map:
                raise ValueError('entry credit: a submitted order must identify exactly one BUY decision')
            order_map[order_id] = index
        outcome = decision.get('outcome')
        if not isinstance(outcome, str) or not outcome:
            raise ValueError('entry credit: BUY outcome must be a nonempty string')
        decision_map[index] = dict(decision_index=index, entry_order_id=order_id, outcome=outcome)
    if set(decision_map) != set(np.flatnonzero(actions == 1).tolist()):
        raise ValueError('entry credit: buy_decisions must cover every selected BUY action')

    checked, seen = [], set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError('entry credit: each entry must be an object')
        order_id = _integer(entry.get('entry_order_id'), 'entry_order_id')
        index = _integer(entry.get('decision_index'), 'entry decision_index')
        if order_id in seen or order_map.get(order_id) != index:
            raise ValueError('entry credit: entry order/decision linkage is inconsistent')
        seen.add(order_id)
        row = dict(entry_order_id=order_id, decision_index=index)
        row.update({field: _integer(entry.get(field), field) for field in _QUANTITY_FIELDS})
        submitted, filled, sold, open_quantity = (row[field] for field in _QUANTITY_FIELDS)
        if submitted <= 0 or filled > submitted or sold + open_quantity != filled:
            raise ValueError('entry credit: entry quantities are inconsistent')
        if not isinstance(entry.get('complete'), (bool, np.bool_)):
            raise ValueError('entry credit: complete must be boolean')
        row['complete'] = bool(entry['complete'])
        if row['complete'] and (filled == 0 or open_quantity != 0):
            raise ValueError('entry credit: a complete entry must have filled and fully sold quantity')
        for field in ('order_status', 'exit_reason'):
            if not isinstance(entry.get(field), str):
                raise ValueError(f'entry credit: {field} must be a string')
            row[field] = entry[field]
        row.update({field: _number(entry.get(field), field) for field in _MONEY_FIELDS})
        holding = _number(entry.get('quantity_weighted_holding_time'), 'quantity_weighted_holding_time')
        if holding < 0:
            raise ValueError('entry credit: holding time cannot be negative')
        row['quantity_weighted_holding_time'] = holding
        checked.append(row)
    if seen != set(order_map):
        raise ValueError('entry credit: every submitted BUY order must have an entry record')
    return decision_map, checked


def analyze_entry_credit(episodes, *, raw_gae, group_component, pre_normalized,
                         normalized, returns, cached_values, gamma, lambda_gae):
    """Return finite flat ``metrics`` plus JSON-compatible entry/action diagnostics.

    Array arguments are one-dimensional and use the trainer's episode-concatenation
    order. ``raw_gae`` and ``cached_values`` may be None (non-GAE training). The
    version-1 entry trace lives in ``episode['metadata']['entry_diagnostics']``;
    an absent trace is supported and explicitly unavailable, while malformed
    present traces raise ValueError. Complete-entry profitability uses realized
    net money PnL including matched fees; it is not the percent-NAV BUY reward.

    Future TD trace is the supplied raw GAE minus the immediate TD error. Its
    consistency residual checks the next supplied GAE, respecting terminals and
    episode boundaries, without recalculating targets or normalization.
    """
    gamma, lambda_gae = _number(gamma, 'gamma'), _number(lambda_gae, 'lambda_gae')
    if not 0 <= gamma <= 1 or not 0 <= lambda_gae <= 1:
        raise ValueError('entry credit: gamma and lambda_gae must be within [0, 1]')
    episodes = list(episodes)
    episode_arrays = []
    for episode in episodes:
        actions = np.asarray(episode['actions'])
        if actions.ndim != 1 or actions.dtype.kind not in 'iu' or not np.isin(actions, (0, 1, 2)).all():
            raise ValueError('entry credit: expected one-dimensional HOLD=0, BUY=1, SELL=2 actions')
        rewards = _vector(episode['rewards'], len(actions), 'episode rewards')
        dones = None
        if raw_gae is not None and cached_values is not None:
            dones = np.asarray(episode.get('dones'))
            if (dones.shape != actions.shape or dones.dtype.kind not in 'biuf'
                    or not np.isin(dones, (0, 1)).all()):
                raise ValueError('entry credit: GAE decomposition requires boolean or 0/1 episode dones')
            dones = dones.astype(np.bool_, copy=False)
        episode_arrays.append((actions, rewards, dones))
    length = sum(len(actions) for actions, _, _ in episode_arrays)
    arrays = {
        'raw_gae': _vector(raw_gae, length, 'raw_gae', optional=True),
        'group_component': _vector(group_component, length, 'group_component'),
        'pre_normalized_advantage': _vector(pre_normalized, length, 'pre_normalized'),
        'normalized_advantage': _vector(normalized, length, 'normalized'),
        'return_target': _vector(returns, length, 'returns'),
        'cached_value': _vector(cached_values, length, 'cached_values', optional=True),
    }
    action_rows = {name: [] for name in ('hold', 'buy', 'sell')}
    entries, buy_decisions = [], []
    available_episodes = 0
    offset = 0
    for episode_index, (episode, (actions, rewards, dones)) in enumerate(zip(episodes, episode_arrays)):
        signals = []
        for index, action in enumerate(actions):
            global_index = offset + index
            signal = {name: float(values[global_index]) if values is not None else None
                      for name, values in arrays.items()}
            signal.update(immediate_reward=float(rewards[index]), critic_target_error=None,
                          bootstrap_value_delta=None, immediate_td_error=None,
                          future_td_trace=None, gae_trace_consistency_residual=None)
            if cached_values is not None:
                signal['critic_target_error'] = signal['return_target'] - signal['cached_value']
            if raw_gae is not None and cached_values is not None:
                has_next = index + 1 < len(actions) and not dones[index]
                next_value = float(arrays['cached_value'][global_index + 1]) if has_next else 0.0
                bootstrap = gamma * next_value - signal['cached_value']
                td = signal['immediate_reward'] + bootstrap
                future = signal['raw_gae'] - td
                next_trace = gamma * lambda_gae * float(arrays['raw_gae'][global_index + 1]) if has_next else 0.0
                signal.update(bootstrap_value_delta=bootstrap, immediate_td_error=td,
                              future_td_trace=future, gae_trace_consistency_residual=future - next_trace)
            signals.append(signal)
            action_rows[('hold', 'buy', 'sell')[int(action)]].append(signal)
        metadata = episode.get('metadata')
        if isinstance(metadata, Mapping) and 'entry_diagnostics' in metadata:
            decisions, traced_entries = _trace_entries(metadata['entry_diagnostics'], actions)
            available_episodes += 1
            key = _episode_key(metadata)
            for index, decision in sorted(decisions.items()):
                buy_decisions.append({**decision, 'episode_index': episode_index,
                                      'episode_key': dict(key), 'global_decision_index': offset + index,
                                      **signals[index]})
            for entry in traced_entries:
                row = {**entry, 'episode_index': episode_index,
                       'episode_key': dict(key),
                       'global_decision_index': offset + entry['decision_index'],
                       **signals[entry['decision_index']]}
                row['buy_outcome'] = decisions[entry['decision_index']]['outcome']
                filled, sold = row['filled_quantity'], row['sold_quantity']
                row['realized_pnl_available'] = bool(sold)
                row['complete_pnl_available'] = row['complete']
                row['partial_entry_fill'] = 0 < filled < row['submitted_quantity']
                row['outcome'] = ('unfilled' if not filled else 'incomplete' if not row['complete']
                                  else 'profitable' if row['net_pnl'] > 0 else 'loss' if row['net_pnl'] < 0
                                  else 'break_even')
                row['computed_pnl_attribution_residual'] = row['net_pnl'] - (
                    row['mid_price_pnl'] - row['spread_cost'] - row['depth_cost']
                    - row['slippage_cost'] - row['matched_fees'])
                row['recorded_residual_difference'] = (row['computed_pnl_attribution_residual']
                                                       - row['pnl_attribution_residual'])
                entries.append(row)
        offset += len(actions)

    availability = {
        'entry_attribution_available': bool(available_episodes),
        'entry_attribution_complete': bool(episodes) and available_episodes == len(episodes),
        'available_episode_count': available_episodes,
        'missing_episode_count': len(episodes) - available_episodes,
        'reason': ('complete' if episodes and available_episodes == len(episodes) else
                   'partial_legacy_metadata' if available_episodes else 'entry_trace_unavailable'),
    }
    metrics = {key: int(value) for key, value in availability.items() if key != 'reason'}
    metrics.update(episode_count=len(episodes), sample_count=length, entry_count=len(entries),
                   traced_buy_decision_count=len(buy_decisions),
                   blocked_buy_decision_count=sum(row['entry_order_id'] is None for row in buy_decisions),
                   complete_entry_count=sum(row['complete'] for row in entries),
                   partial_fill_entry_count=sum(row['partial_entry_fill'] for row in entries))
    action_summary = {action: _summary(rows) for action, rows in action_rows.items()}
    for action, summary in action_summary.items():
        metrics.update({f'action/{action}/{key}': value for key, value in summary.items()})
    outcome_summary = {}
    for outcome in ('profitable', 'loss', 'break_even', 'incomplete', 'unfilled'):
        selected = [row for row in entries if row['outcome'] == outcome]
        summary = _summary(selected)
        completed = [row for row in selected if row['complete']]
        sold = [row for row in selected if row['realized_pnl_available']]
        summary['complete_net_pnl_available'] = int(bool(completed))
        summary['complete_net_pnl_mean'] = float(np.mean([row['net_pnl'] for row in completed])) if completed else 0.0
        summary['realized_pnl_entry_count'] = len(sold)
        summary['realized_components_available'] = int(bool(sold))
        summary['realized_components_entry_count'] = len(sold)
        for field in ('mid_price_pnl', 'spread_cost', 'depth_cost', 'slippage_cost',
                      'matched_fees', 'net_pnl', 'computed_pnl_attribution_residual'):
            summary[f'{field}_sum'] = float(sum(row[field] for row in sold))
        summary['holding_time_available'] = int(bool(sold))
        summary['holding_time_mean'] = float(np.mean([row['quantity_weighted_holding_time'] for row in sold])) if sold else 0.0
        summary['pnl_attribution_residual_max_abs'] = max((abs(row['computed_pnl_attribution_residual']) for row in sold), default=0.0)
        summary['recorded_residual_difference_max_abs'] = max((abs(row['recorded_residual_difference']) for row in sold), default=0.0)
        reasons = dict(Counter(row['exit_reason'] for row in sold))
        outcome_summary[outcome] = {**summary, 'exit_reason_counts': reasons}
        metrics.update({f'outcome/{outcome}/{key}': value for key, value in summary.items()})
        metrics.update({f'outcome/{outcome}/exit_reason/{reason}/entry_count': count
                        for reason, count in reasons.items()})
    return {
        'version': 1,
        'scope': ('Observed rollout entry outcomes and frozen actor/critic targets; not counterfactual '
                  'profit, updated-policy performance, or the shared-network gradient. Incomplete '
                  'entry PnL covers sold quantity only. Entry PnL is money; rewards and GAE use the '
                  'training reward units. Zero summary values require their availability/count flags.'),
        'gamma': gamma, 'lambda_gae': lambda_gae, 'metrics': metrics,
        'availability': availability, 'action_summary': action_summary,
        'outcome_summary': outcome_summary, 'entries': entries, 'buy_decisions': buy_decisions,
    }
