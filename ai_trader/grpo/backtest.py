"""Replay a validated checkpoint on its recorded held-out dates.

Example: python -m ai_trader.grpo.backtest --policy models/run/checkpoints/checkpoint_best.pt
         --output models/run/replay.json
"""
import argparse
import json
from pathlib import Path

import torch

from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from ai_trader.grpo.evaluation import evaluate_policy
from ai_trader.grpo.inference.enhanced_grpo_infer_xlstm import GRPOInferenceE2EXLSTM


def run_backtest(policy_path, *, extracted_dir=None, partition='test', episodes=None,
                 seed=None, device='cpu', execution_overrides=None):
    if partition not in ('validation', 'test'):
        raise ValueError('Replay partition must be validation or test')
    checkpoint = torch.load(policy_path, map_location='cpu', weights_only=False)
    extra = checkpoint.get('extra_state', {})
    config, splits = extra.get('training_config'), extra.get('date_splits')
    if not config or not splits or not splits.get(partition):
        raise ValueError('A current training checkpoint with simulation settings and date_splits is required')
    inference = GRPOInferenceE2EXLSTM(policy_path, device=device)
    schema = inference.observation_schema
    execution = dict(config['execution_config'])
    execution.update(execution_overrides or {})
    environment = GRPOScalpingEnvXLSTM(
        db_path=config.get('db_path'), extracted_dir=extracted_dir or config.get('extracted_dir'),
        table_name=config.get('table_name', 'datasets'),
        allowed_dates=splits[partition], expected_features=len(schema['feature_columns']),
        seq_len=schema['seq_len'], max_episode_steps=config['episode_steps'],
        rolling_window_size=schema['rolling_window_size'], rolling_min_samples=schema['rolling_min_samples'],
        max_stages=schema['max_stages'],
        account_observations=schema.get('version') in (3, 4),
        execution_observations=schema.get('version') == 4,
        execution_action_mask=config.get('execution_action_mask', False),
        entry_pattern_config=config.get('entry_pattern_config'),
        liquidation_max_steps=config.get('liquidation_max_steps', 0),
        decision_interval_seconds=config.get('decision_interval_seconds', 0.0),
        episode_duration_seconds=config.get('episode_duration_seconds', 0.0),
        max_holding_seconds=schema['max_holding_seconds'], initial_cash=config['initial_cash'],
        transaction_cost_rate=config['transaction_cost_rate'], buy_tax_rate=config['buy_tax_rate'],
        sell_tax_rate=config['sell_tax_rate'], stop_loss_pct=config['stop_loss_pct'],
        base_price=config.get('base_price', 100000), price_scale=config.get('price_scale', 1.0),
        max_trades_per_episode=config.get('max_trades_per_episode'), execution_config=execution)
    try:
        environment.observation_builder.validate_schema(schema)
        metrics = evaluate_policy(inference.policy, environment,
                                  config.get('evaluation_episodes', 8) if episodes is None else episodes,
                                  config.get('evaluation_seed', 42) if seed is None else seed,
                                  str(inference.device))
        return {
            'checkpoint': str(Path(policy_path).resolve()), 'partition': partition,
            'dates': splits[partition], 'observation_schema': schema,
            'execution_config': execution,
            'decision_timing': {key: config.get(key, 0.0) for key in
                                ('decision_interval_seconds', 'episode_duration_seconds')},
            'costs': {key: config[key] for key in ('transaction_cost_rate', 'buy_tax_rate', 'sell_tax_rate')},
            'no_trade_reference_net_return': 0.0, 'metrics': metrics,
            'valuation': 'Liquidation-marked NAV (% of initial cash); residual inventory is reported separately',
            'passive_limit_fills': 'Only marketable opposite depth is executed; queue fills are not inferred',
        }
    finally:
        environment.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--extracted-dir')
    parser.add_argument('--partition', choices=('validation', 'test'), default='test')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--order-latency-ms', type=float)
    parser.add_argument('--spread-bps', type=float)
    parser.add_argument('--slippage-bps', type=float)
    parser.add_argument('--require-order-book', action='store_true')
    args = parser.parse_args(argv)
    overrides = {name: getattr(args, name) for name in ('order_latency_ms', 'spread_bps', 'slippage_bps')
                 if getattr(args, name) is not None}
    if args.require_order_book:
        overrides['require_order_book'] = True
    report = run_backtest(args.policy, extracted_dir=args.extracted_dir, partition=args.partition,
                          episodes=args.episodes, seed=args.seed, device=args.device,
                          execution_overrides=overrides)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
