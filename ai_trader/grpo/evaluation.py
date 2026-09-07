"""Chronological holdouts and deterministic, cost-inclusive policy evaluation."""

from datetime import datetime
from copy import deepcopy
from typing import Iterable, Optional

import numpy as np
import torch


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
                    device: str = "cpu", max_steps: int = 100000) -> dict:
    """Evaluate exactly these weights on a repeatable set of held-out paths.

    Only environment-reported cost-inclusive NAV returns determine performance.
    The evaluator never updates parameters or uses shaped reward as a fallback.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    was_training = policy.training
    policy.eval()
    returns, trades, drawdowns, holding = [], [], [], []
    incomplete_liquidations = 0
    residual_quantities, realized_pnls = [], []
    execution_models = {}
    try:
        with torch.no_grad():
            for index in range(num_episodes):
                obs, _ = env.reset(seed=seed + index)
                for _ in range(max_steps):
                    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action, _ = policy.get_action(tensor, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(int(action.item()))
                    if terminated or truncated:
                        episode = info.get("episode", {})
                        value = episode.get("net_return", episode.get("total_return"))
                        if value is None or not np.isfinite(value):
                            raise ValueError("Evaluation requires finite episode net_return (%)")
                        returns.append(float(value))
                        trades.append(int(episode.get("num_trades", 0)))
                        drawdowns.append(float(episode.get("max_drawdown", 0.0)))
                        holding.append(float(episode.get("avg_holding_time", 0.0)))
                        incomplete_liquidations += int(not episode.get("liquidation_complete", True))
                        residual_quantities.append(float(episode.get("open_quantity", 0.0)))
                        realized_pnls.append(float(episode.get("realized_net_pnl", 0.0)))
                        execution_model = str(episode.get("execution_model", "unknown"))
                        execution_models[execution_model] = execution_models.get(execution_model, 0) + 1
                        break
                else:
                    raise RuntimeError("Evaluation episode exceeded max_steps")
    finally:
        policy.train(was_training)
    return {
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
    }


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
        'transaction_cost_rate': .00015, 'buy_tax_rate': 0.0, 'sell_tax_rate': .0018,
        'initial_cash': 1_000_000.0, 'stop_loss_pct': 2.0,
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
        signature = extra.get('evaluation_signature') or evaluation_signature(config, splits, schema)
        if signature != current_signature:
            return False
        expected_settings = evaluation_signature(config, signature['date_splits'], schema)['settings']
        if signature['settings'] != expected_settings:
            return False
    return True
