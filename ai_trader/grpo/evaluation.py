"""Chronological holdouts and deterministic, cost-inclusive policy evaluation."""

from datetime import datetime
from copy import deepcopy
from collections import Counter
from typing import Iterable, Optional

import numpy as np
import torch


_ORDER_METRICS = ('submitted_orders', 'submitted_quantity', 'filled_quantity',
                  'partial_orders', 'cancelled_orders', 'expired_orders')
_EXIT_REASONS = ('signal', 'stop_loss', 'max_holding', 'episode_end')
_PNL_COMPONENTS = ('mid_price_pnl', 'spread_cost', 'depth_cost', 'slippage_cost',
                   'matched_fees', 'pnl_attribution_residual')
_TIME_DIAGNOSTICS = ('episode_start_time_seconds', 'episode_end_time_seconds',
                     'episode_duration_seconds', 'policy_duration_seconds',
                     'decision_interval_seconds', 'configured_episode_duration_seconds')
_TRADE_DIAGNOSTICS = ('round_trip_count', 'round_trip_win_rate',
                      'quantity_weighted_holding_time', 'fill_count', 'fill_win_rate',
                      'liquidation_complete',
                      'fill_avg_holding_time', 'round_trips', 'liquidation_steps',
                      'liquidation_seconds', 'liquidation_stop_reason', 'market_steps_taken',
                      'total_entry_fees', 'total_exit_fees', 'total_fees', 'gross_realized_pnl',
                      *_PNL_COMPONENTS, *_TIME_DIAGNOSTICS,
                      'subsecond_exit_quantity_ratio', 'subsecond_round_trip_ratio',
                      'attribution_reference_kind', 'entry_pattern',
                      *(f'exit_{reason}_{metric}' for reason in _EXIT_REASONS
                        for metric in ('round_trip_count', 'quantity', 'net_pnl', 'holding_seconds')))


def _action_name(index):
    return ('hold', 'buy', 'sell')[index] if 0 <= index < 3 else f'action_{index}'


class _ActionDiagnostics:
    def __init__(self):
        self.counts = Counter(dict.fromkeys(('hold', 'buy', 'sell'), 0))
        self.probability_steps = 0
        self.probability_sum = self.probability_max = None

    def record(self, action, probabilities):
        self.counts[_action_name(action)] += 1
        if probabilities is not None:
            if self.probability_sum is None:
                self.probability_sum = np.zeros_like(probabilities, dtype=np.float64)
                self.probability_max = np.zeros_like(probabilities, dtype=np.float64)
            self.probability_sum += probabilities
            self.probability_max = np.maximum(self.probability_max, probabilities)
            self.probability_steps += 1

    def merge(self, other):
        self.counts.update(other.counts)
        if other.probability_steps:
            if self.probability_sum is None:
                self.probability_sum = other.probability_sum.copy()
                self.probability_max = other.probability_max.copy()
            else:
                self.probability_sum += other.probability_sum
                self.probability_max = np.maximum(self.probability_max, other.probability_max)
            self.probability_steps += other.probability_steps

    def summary(self):
        steps = sum(self.counts.values())
        return {
            'steps': steps,
            'action_counts': dict(self.counts),
            'action_rates': {key: value / steps if steps else 0.0
                             for key, value in self.counts.items()},
            'probability_steps': self.probability_steps,
            'mean_action_probabilities': (
                {_action_name(i): float(value / self.probability_steps)
                 for i, value in enumerate(self.probability_sum)}
                if self.probability_steps else None),
            'max_action_probabilities': (
                {_action_name(i): float(value) for i, value in enumerate(self.probability_max)}
                if self.probability_steps else None),
        }


def _execution_diagnostics(episode):
    # Missing counters are unknown, never evidence that no order was submitted.
    metrics = {key: int(episode[key]) if key in episode else None for key in _ORDER_METRICS}
    submitted, filled = metrics['submitted_quantity'], metrics['filled_quantity']
    metrics['fill_ratio'] = (filled / submitted if submitted else 0.0
                             ) if submitted is not None and filled is not None else None
    for name in ('buy_action_outcomes', 'execution_blocked_checks'):
        metrics[name] = dict(episode[name]) if name in episode else None
    return metrics


def _sum_execution_diagnostics(episodes):
    metrics = {}
    for key in _ORDER_METRICS:
        values = [episode[key] for episode in episodes]
        metrics[key] = sum(values) if all(value is not None for value in values) else None
    for name in ('buy_action_outcomes', 'execution_blocked_checks'):
        values = [episode[name] for episode in episodes]
        if any(value is None for value in values):
            metrics[name] = None
        else:
            counts = Counter()
            for value in values:
                counts.update(value)
            metrics[name] = dict(counts)
    return _execution_diagnostics({key: value for key, value in metrics.items() if value is not None})


def _additional_metrics(episodes):
    """Keep missing legacy trade/tail diagnostics unknown, not apparent zeros."""
    def values(key):
        result = [episode.get(key) for episode in episodes]
        return result if all(value is not None for value in result) else None

    result = {}
    reports = [episode.get('entry_pattern') for episode in episodes]
    if all(report is not None for report in reports):
        totals = {key: int(sum(report[key] for report in reports)) for key in
                  ('success_quantity', 'failure_quantity', 'censored_quantity', 'labeled_buy_decisions')}
        labeled = totals['success_quantity'] + totals['failure_quantity']
        result['entry_pattern'] = {**totals, 'success_rate': totals['success_quantity'] / labeled if labeled else None}
    for count_key, rate_key in (('round_trip_count', 'round_trip_win_rate'),
                                ('fill_count', 'fill_win_rate')):
        counts, rates = values(count_key), values(rate_key)
        total = sum(counts) if counts is not None else None
        result[count_key] = int(total) if total is not None else None
        result[f'mean_{count_key}'] = float(np.mean(counts)) if counts is not None else None
        result[rate_key] = (float(np.dot(counts, rates) / total) if total else 0.0
                            ) if counts is not None and rates is not None else None
    for key in ('quantity_weighted_holding_time', 'fill_avg_holding_time',
                'liquidation_steps', 'liquidation_seconds',
                'episode_duration_seconds', 'policy_duration_seconds',
                'subsecond_exit_quantity_ratio', 'subsecond_round_trip_ratio'):
        observations = values(key)
        result[f'mean_{key}'] = float(np.mean(observations)) if observations is not None else None
    for key in ('total_entry_fees', 'total_exit_fees', 'total_fees', 'gross_realized_pnl', *_PNL_COMPONENTS):
        observations = values(key)
        result[key] = float(sum(observations)) if observations is not None else None
        result[f'mean_{key}'] = float(np.mean(observations)) if observations is not None else None
    result['exit_reasons'] = {}
    for reason in _EXIT_REASONS:
        counts = values(f'exit_{reason}_round_trip_count')
        quantities = values(f'exit_{reason}_quantity')
        pnls = values(f'exit_{reason}_net_pnl')
        holding = values(f'exit_{reason}_holding_seconds')
        total = sum(counts) if counts is not None else None
        quantity = sum(quantities) if quantities is not None else None
        result['exit_reasons'][reason] = {
            'round_trip_count': int(total) if total is not None else None,
            'quantity': float(quantity) if quantity is not None else None,
            'net_pnl': float(sum(pnls)) if pnls is not None else None,
            'quantity_weighted_holding_seconds': (float(np.dot(quantities, holding) / quantity) if quantity else 0.0
                                                  ) if quantities is not None and holding is not None else None,
        }
    stops = values('liquidation_stop_reason')
    result['liquidation_stop_reasons'] = dict(Counter(stops)) if stops is not None else None
    references = values('attribution_reference_kind')
    result['attribution_reference_kinds'] = dict(Counter(references)) if references is not None else None
    # A bought but unliquidated position is still trading, even with zero exits.
    filled = values('filled_quantity')
    result['no_trade_episode_fraction'] = (
        float(np.mean(np.asarray(filled) == 0)) if filled is not None else None)
    return result


def normalize_date(value) -> str:
    text = str(value).replace("-", "").split(".")[0]
    return datetime.strptime(text, "%Y%m%d").strftime("%Y%m%d")


def chronological_date_split(
    dates: Iterable, train_end_date: Optional[str] = None,
    validation_end_date: Optional[str] = None,
    validation_fraction: float = 0.2, test_fraction: float = 0.2,
    embargo_dates: int = 0,
) -> dict:
    """Split entire market dates; optional embargo removes boundary sessions.

    Fractions select boundaries only. No observation window crosses a date or
    appears in another partition. Explicit boundaries are inclusive.
    """
    unique = sorted({normalize_date(day) for day in dates})
    if len(unique) < 3:
        raise ValueError("At least three distinct dates are required for train/validation/test.")
    if embargo_dates < 0:
        raise ValueError("embargo_dates must be nonnegative")
    if (train_end_date is None) != (validation_end_date is None):
        raise ValueError("Specify both train_end_date and validation_end_date")
    if train_end_date is not None:
        train_end, val_end = normalize_date(train_end_date), normalize_date(validation_end_date)
        if train_end >= val_end:
            raise ValueError("train_end_date must precede validation_end_date")
        train = [day for day in unique if day <= train_end]
        validation = [day for day in unique if train_end < day <= val_end]
        test = [day for day in unique if day > val_end]
    else:
        if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
                and validation_fraction + test_fraction < 1):
            raise ValueError("Invalid validation/test fractions")
        test_count = max(1, int(len(unique) * test_fraction))
        validation_count = max(1, int(len(unique) * validation_fraction))
        train_count = len(unique) - test_count - validation_count
        train = unique[:train_count]
        validation = unique[train_count:train_count + validation_count]
        test = unique[train_count + validation_count:]
    validation, test = validation[embargo_dates:], test[embargo_dates:]
    if not train or not validation or not test:
        raise ValueError("Every split must contain at least one date after embargo")
    return {"train": train, "validation": validation, "test": test}


def evaluate_policy(policy, env, num_episodes: int = 8, seed: int = 42,
                    device: str = "cpu", max_steps: int = 100000,
                    collect_diagnostics: bool = True) -> dict:
    """Evaluate exactly these weights on a repeatable set of held-out paths.

    ``env`` may be one environment or a list/tuple of independent environments.
    Episodes always use seed + episode_index, regardless of batch size or their
    completion order. The caller owns and closes the environments.

    Only environment-reported cost-inclusive NAV returns determine performance.
    The evaluator never updates parameters or uses shaped reward as a fallback.
    Diagnostics record decisions and execution outcomes on these same paths;
    they do not change actions, cost settings, or the selection metric.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    environments = list(env) if isinstance(env, (list, tuple)) else [env]
    if not environments or len({id(item) for item in environments}) != len(environments):
        raise ValueError("Evaluation requires independent, nonempty environments")
    environments = environments[:num_episodes]
    was_training = policy.training
    policy.eval()
    completed = [None] * num_episodes
    probability_action = getattr(policy, 'get_action_with_probabilities', None)

    def start_episode(environment, index):
        obs, _ = environment.reset(seed=seed + index)
        return {'environment': environment, 'index': index, 'obs': obs,
                'steps': 0, 'actions': _ActionDiagnostics()}

    try:
        with torch.no_grad():
            slots = [start_episode(environment, index)
                     for index, environment in enumerate(environments)]
            next_index = len(slots)
            while any(slot is not None for slot in slots):
                active = [(index, slot) for index, slot in enumerate(slots) if slot is not None]
                tensor = torch.as_tensor(np.stack([slot['obs'] for _, slot in active]),
                                         dtype=torch.float32, device=device)
                mask_kwargs = {}
                if getattr(policy, 'execution_action_mask', False):
                    if not all(getattr(slot['environment'], 'execution_action_mask', False)
                               for _, slot in active):
                        raise ValueError('Masked policy evaluation requires enabled environment action masks')
                    mask_kwargs['action_masks'] = torch.as_tensor(
                        np.stack([slot['environment'].action_masks() for _, slot in active]),
                        dtype=torch.bool, device=device)
                probabilities = None
                if collect_diagnostics and callable(probability_action):
                    actions, _, probabilities = probability_action(tensor, deterministic=True, **mask_kwargs)
                    probabilities = probabilities.detach().cpu().numpy()
                    if probabilities.ndim == 1 and len(active) == 1:
                        probabilities = probabilities[None, :]
                    if (probabilities.ndim != 2 or len(probabilities) != len(active)
                            or not np.isfinite(probabilities).all()):
                        raise ValueError("Evaluation requires finite batched action probabilities")
                else:
                    actions, _ = policy.get_action(tensor, deterministic=True, **mask_kwargs)
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().numpy()
                actions = np.asarray(actions).reshape(-1)
                if (len(actions) != len(active) or not np.isfinite(actions).all()
                        or (actions != np.floor(actions)).any()):
                    raise ValueError("Evaluation requires one finite integer action per environment")
                for row, (slot_index, slot) in enumerate(active):
                    action_index = int(actions[row])
                    if collect_diagnostics:
                        slot['actions'].record(action_index, None if probabilities is None else probabilities[row])
                    obs, _, terminated, truncated, info = slot['environment'].step(action_index)
                    slot['steps'] += 1
                    if not (terminated or truncated):
                        if slot['steps'] >= max_steps:
                            raise RuntimeError("Evaluation episode exceeded max_steps")
                        slot['obs'] = obs
                        continue
                    episode = deepcopy(info.get('episode', {}))
                    value = episode.get('net_return', episode.get('total_return'))
                    if value is None or not np.isfinite(value):
                        raise ValueError("Evaluation requires finite episode net_return (%)")
                    for key in ('num_trades', 'max_drawdown', 'avg_holding_time',
                                'open_quantity', 'realized_net_pnl', *_ORDER_METRICS,
                                *[name for name in _TRADE_DIAGNOSTICS
                                  if name not in ('round_trips', 'liquidation_stop_reason', 'attribution_reference_kind', 'entry_pattern')]):
                        if episode.get(key) is not None and not np.isfinite(episode[key]):
                            raise ValueError(f"Evaluation requires finite episode {key}")
                    episode['net_return'] = float(value)
                    completed[slot['index']] = (episode, slot['actions'])
                    slots[slot_index] = (start_episode(slot['environment'], next_index)
                                         if next_index < num_episodes else None)
                    if slots[slot_index] is not None:
                        next_index += 1
    finally:
        policy.train(was_training)
    # Aggregate in episode order so unequal lengths or batch sizes cannot change
    # the path order, mean-of-episodes weighting, or diagnostic summation order.
    episodes = [episode for episode, _ in completed]
    returns = [episode['net_return'] for episode in episodes]
    trades = [int(episode.get('num_trades', 0)) for episode in episodes]
    drawdowns = [float(episode.get('max_drawdown', 0.0)) for episode in episodes]
    holding = [float(episode.get('avg_holding_time', 0.0)) for episode in episodes]
    residual_quantities = [float(episode.get('open_quantity', 0.0)) for episode in episodes]
    realized_pnls = [float(episode.get('realized_net_pnl', 0.0)) for episode in episodes]
    incomplete_liquidations = sum(not episode.get('liquidation_complete', True) for episode in episodes)
    execution_models = dict(Counter(str(episode.get('execution_model', 'unknown')) for episode in episodes))
    result = {
        "mean_net_return": float(np.mean(returns)),
        "std_net_return": float(np.std(returns)),
        "min_net_return": float(np.min(returns)),
        "mean_num_trades": float(np.mean(trades)),
        "max_drawdown": float(np.max(drawdowns)),
        "avg_holding_time": float(np.mean(holding)),
        "num_episodes": num_episodes,
        "seed": seed,
        "episode_net_returns": returns,
        "incomplete_liquidation_episodes": incomplete_liquidations,
        "max_open_quantity": float(np.max(residual_quantities)),
        "mean_realized_net_pnl": float(np.mean(realized_pnls)),
        "execution_models": execution_models,
        "synthetic_execution_episodes": sum(
            count for name, count in execution_models.items() if 'synthetic' in name),
        **_additional_metrics(episodes),
    }
    no_trade = result['no_trade_episode_fraction']
    result['no_trade_reference_net_return'] = 0.0
    result['profitable_with_trades'] = (bool(result['mean_net_return'] > 0
                                           and no_trade < 1 and incomplete_liquidations == 0)
                                      if no_trade is not None else None)
    result['evaluation_outcome'] = ('incomplete_liquidation' if incomplete_liquidations else
                                    'unknown_execution' if no_trade is None else
                                    'no_trade' if no_trade == 1 else
                                    'profitable' if result['mean_net_return'] > 0 else 'nonprofitable')
    if collect_diagnostics:
        total_actions = _ActionDiagnostics()
        diagnostic_episodes = []
        for index, (episode, actions) in enumerate(completed):
            total_actions.merge(actions)
            diagnostic_episodes.append({
                'episode_index': index, 'seed': seed + index,
                'episode_key': deepcopy(episode.get('episode_key', {})),
                'net_return': returns[index], 'num_trades': trades[index],
                'open_quantity': residual_quantities[index],
                **actions.summary(), **_execution_diagnostics(episode),
                **{key: deepcopy(episode.get(key)) for key in _TRADE_DIAGNOSTICS},
            })
        result['diagnostics'] = {
            **total_actions.summary(), **_sum_execution_diagnostics(diagnostic_episodes),
            'execution_blocked_checks_unit': 'order/event checks, not distinct orders',
            'episodes': diagnostic_episodes,
        }
    return result


def validate_checkpoint_dates(checkpoint: dict, date_splits: dict) -> None:
    """Pretraining/resume must not import weights trained on the holdouts."""
    history = checkpoint.get('extra_state', {}).get('date_splits')
    if not isinstance(history, dict) or not history.get('train'):
        raise ValueError("Checkpoint has no training date lineage; retrain with chronological splits")
    trained = {normalize_date(value) for value in history['train']}
    held_out = {normalize_date(value) for key in ('validation', 'test') for value in date_splits[key]}
    train_end = max(normalize_date(value) for value in date_splits['train'])
    if trained & held_out or max(trained) > train_end:
        raise ValueError("Checkpoint was trained on dates reserved for validation/test or later")
    previously_validated = {normalize_date(value) for value in history.get('validation', [])}
    test_start = min(normalize_date(value) for value in date_splits['test'])
    if previously_validated and max(previously_validated) >= test_start:
        raise ValueError("Checkpoint model selection already used the new test dates")


def evaluation_signature(config: dict, date_splits: dict, observation_schema: dict) -> dict:
    """Settings that must match before inheriting another checkpoint's best."""
    defaults = {
        'episode_steps': 600, 'evaluation_episodes': 8, 'evaluation_seed': 42,
        'selection_require_liquidation': False,  # Historical checkpoints used marked NAV without this filter.
        'transaction_cost_rate': .00015, 'buy_tax_rate': 0.0, 'sell_tax_rate': .0018,
        'initial_cash': 1_000_000.0, 'stop_loss_pct': 2.0,
        'liquidation_max_steps': 0,  # Historical checkpoints had no liquidation tail.
        'decision_interval_seconds': 0.0, 'episode_duration_seconds': 0.0,
        'execution_action_mask': False,
        'entry_pattern_config': None,
        'max_trades_per_episode': None, 'base_price': 100000.0, 'price_scale': 1.0,
        'execution_config': {'order_latency_ms': 100, 'cancel_latency_ms': 50,
                             'order_ttl_seconds': 2, 'spread_bps': 10, 'slippage_bps': 2,
                             'fallback_depth': 100, 'require_order_book': False},
    }
    return deepcopy({
        'version': 1,
        'observation_schema': observation_schema,
        'date_splits': date_splits,
        'settings': {key: config.get(key, default) for key, default in defaults.items()},
        'deterministic': True,
        'metric': 'mean_net_return',
    })


def compatible_resume_best(candidate: dict, source: dict, current_signature: dict) -> bool:
    """Reject unrelated experiments; legacy metadata is checked, never trusted as a score.

    Callers restrict candidates to the source/output checkpoint directories and
    re-evaluate accepted weights on the current fixed validation paths.
    """
    source_extra, candidate_extra = source.get('extra_state', {}), candidate.get('extra_state', {})
    source_config = source_extra.get('training_config')
    candidate_config = candidate_extra.get('training_config')
    if not isinstance(source_config, dict) or not isinstance(candidate_config, dict):
        return False
    source_weights, candidate_weights = source.get('policy_state_dict'), candidate.get('policy_state_dict')
    if not isinstance(source_weights, dict) or not isinstance(candidate_weights, dict):
        return False
    if source_weights.keys() != candidate_weights.keys() or any(
            getattr(source_weights[key], 'shape', None) != getattr(candidate_weights[key], 'shape', None)
            for key in source_weights):
        return False
    # These identify the source experiment even when the new output directory
    # or runtime's copy of the dataset has moved.
    for key in ('output_dir', 'extracted_dir', 'db_path', 'table_name'):
        if candidate_config.get(key) != source_config.get(key):
            return False
    for checkpoint, extra, config in ((source, source_extra, source_config),
                                      (candidate, candidate_extra, candidate_config)):
        schema = checkpoint.get('observation_schema')
        splits = extra.get('date_splits')
        if schema != current_signature['observation_schema'] or not isinstance(splits, dict):
            return False
        if splits != source_extra.get('date_splits'):
            return False
        signature = deepcopy(extra.get('evaluation_signature')) or evaluation_signature(config, splits, schema)
        if not isinstance(signature, dict):
            return False
        # Before liquidation tails existed, version-1 signatures omitted the
        # setting. They remain compatible only with the historical disabled tail.
        if signature.get('version') == 1 and isinstance(signature.get('settings'), dict):
            if 'entry_pattern_config' not in signature['settings'] and config.get('entry_pattern_config') is None:
                signature['settings']['entry_pattern_config'] = None
            for name in ('liquidation_max_steps', 'decision_interval_seconds', 'episode_duration_seconds',
                         'execution_action_mask'):
                if name not in signature['settings'] and config.get(name, 0) == 0:
                    signature['settings'][name] = 0
        if signature != current_signature:
            return False
        expected_settings = evaluation_signature(config, signature['date_splits'], schema)['settings']
        if signature['settings'] != expected_settings:
            return False
    return True
