"""Compare an original policy and exported LR candidates on paired validation paths."""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re

import torch

from .diagnose_xlstm import _device_name, _saved_splits, run_diagnostics
from .evaluation import evaluation_signature, normalize_date
from .local_check_inputs import _state_equal
from .runtime_precision import precision_metadata, restore_precision, set_tf32, snapshot_precision
from .update_diagnostic import restore_rng, snapshot_rng


logger = logging.getLogger(__name__)
# Include every environment behavior read by create_environment. Local paths and
# cache limits can differ, but all policies receive the same explicit dataset.
ENVIRONMENT_FIELDS = (
    'table_name', 'seq_len', 'features', 'episode_steps', 'use_raw_data',
    'account_observations', 'execution_observations', 'execution_action_mask',
    'liquidation_max_steps', 'decision_interval_seconds', 'episode_duration_seconds',
    'base_price', 'price_scale', 'no_trade_penalty', 'max_trades_per_episode',
    'step_reward_scale', 'win_bonus', 'loss_penalty', 'buy_signal_bonus',
    'rolling_window_size', 'rolling_min_samples', 'initial_cash', 'max_stages',
    'max_holding_seconds', 'stop_loss_pct', 'execution_config',
    'transaction_cost_rate', 'buy_tax_rate', 'sell_tax_rate',
    'cnn_channels', 'rnn_hidden_dim', 'hidden_dim', 'action_dim', 'checkpoint_segments',
)
METRICS = ('mean_net_return', 'total_fees', 'mean_realized_net_pnl', 'round_trip_count',
           'mean_num_trades', 'max_drawdown', 'no_trade_episode_fraction',
           'incomplete_liquidation_episodes', 'max_open_quantity')
ORDER_METRICS = ('submitted_orders', 'submitted_quantity', 'filled_quantity',
                 'partial_orders', 'cancelled_orders', 'expired_orders')


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path, expected):
    if not isinstance(expected, str) or re.fullmatch(r'[0-9a-fA-F]{64}', expected) is None:
        raise ValueError(f'Missing or invalid SHA256 for {path}')
    actual = _hash(path)
    if actual != expected.lower():
        raise ValueError(f'Checkpoint SHA256 mismatch: {path}')
    return actual


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite and numeric')
    return float(value)


def _conditions(checkpoint, episodes, seed):
    splits = _saved_splits(checkpoint)
    config = checkpoint.get('extra_state', {}).get('training_config')
    schema = checkpoint.get('observation_schema')
    if not isinstance(config, dict) or not config or not isinstance(schema, dict):
        raise ValueError('Checkpoint requires training_config and observation_schema')
    signature = evaluation_signature({**config, 'evaluation_episodes': episodes,
                                      'evaluation_seed': seed}, splits, schema)
    signature['environment'] = {key: config.get(key) for key in ENVIRONMENT_FIELDS}
    return signature


def _prepare_inputs(checkpoint, comparison_report, episodes, seed):
    source = Path(checkpoint).expanduser().resolve()
    comparison_path = Path(comparison_report).expanduser().resolve()
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    if (comparison.get('format') != 'grpo_learning_rate_comparison'
            or comparison.get('format_version') != 1 or not comparison.get('model_exported')):
        raise ValueError('An LR comparison report with exported candidates is required')
    source_hash = _verify_hash(source, comparison.get('source_checkpoint_sha256'))
    original = torch.load(source, map_location='cpu', weights_only=True)
    conditions = _conditions(original, episodes, seed)
    variants = comparison.get('variants')
    if not isinstance(variants, list) or not variants:
        raise ValueError('No exported LR candidates')
    entries = [{'label': 'baseline', 'checkpoint': source, 'sha256': source_hash,
                'learning_rate': None}]
    paths = {source}
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict) or variant.get('status') != 'ok':
            raise ValueError('Every LR variant must have completed successfully before comparison')
        rate = _finite(variant.get('learning_rate'), 'learning_rate')
        if rate <= 0:
            raise ValueError('learning_rate must be positive')
        path = variant.get('candidate_checkpoint')
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError('Successful variant requires an absolute candidate_checkpoint')
        path = Path(path).resolve()
        if path in paths:
            raise ValueError('Candidate checkpoint paths must be distinct from source and each other')
        paths.add(path)
        digest = _verify_hash(path, variant.get('candidate_sha256'))
        candidate = torch.load(path, map_location='cpu', weights_only=True)
        provenance = candidate.get('extra_state', {}).get('export_provenance', {})
        if (provenance.get('source_checkpoint_sha256') != source_hash
                or provenance.get('learning_rate') != rate):
            raise ValueError('Candidate export provenance differs from the source or learning rate')
        if _conditions(candidate, episodes, seed) != conditions:
            raise ValueError('Candidate validation conditions differ from the original checkpoint')
        entries.append({'label': f'candidate_{index:02d}', 'checkpoint': path,
                        'sha256': digest, 'learning_rate': rate})
    return entries, conditions, comparison_path


def _validate_evaluation(report, conditions, episodes, seed, device):
    if (report.get('split') != 'validation' or report.get('deterministic') is not True
            or report.get('device') != device or report.get('date_splits') != conditions['date_splits']
            or report.get('dates') != conditions['date_splits']['validation']
            or report.get('observation_schema') != conditions['observation_schema']):
        raise ValueError('Evaluation changed the validation split, schema or execution mode')
    reconstructed = {'extra_state': {'training_config': report.get('saved_training_config'),
                                    'date_splits': report.get('date_splits')},
                     'observation_schema': report.get('observation_schema')}
    if _conditions(reconstructed, episodes, seed) != conditions:
        raise ValueError('Evaluation effective environment settings differ')
    expected = conditions['settings']
    if (report.get('execution_config') != expected['execution_config']
            or report.get('costs') != {key: expected[key] for key in
                                     ('transaction_cost_rate', 'buy_tax_rate', 'sell_tax_rate')}):
        raise ValueError('Evaluation costs or execution configuration differ')
    metrics = report.get('metrics', {})
    if metrics.get('num_episodes') != episodes or metrics.get('seed') != seed:
        raise ValueError('Evaluation episode count or seed differs')
    diagnostic = metrics.get('diagnostics', {})
    details = diagnostic.get('episodes')
    returns = metrics.get('episode_net_returns')
    if (not isinstance(details, list) or len(details) != episodes
            or not isinstance(returns, list) or len(returns) != episodes):
        raise ValueError('Paired comparison requires every episode identity and return')
    identities, checked_returns = [], []
    for index, (episode, value) in enumerate(zip(details, returns)):
        key = episode.get('episode_key', {})
        stock, date, start = key.get('stock_code'), key.get('date'), key.get('start_index')
        if (not isinstance(stock, str) or not stock or isinstance(start, bool)
                or not isinstance(start, int) or start < 0 or episode.get('seed') != seed + index
                or episode.get('episode_index') != index):
            raise ValueError('Incomplete or misordered validation episode identity')
        date = normalize_date(date)
        if date not in conditions['date_splits']['validation']:
            raise ValueError('Evaluation episode is outside validation dates')
        value = _finite(value, 'episode_net_return')
        if not math.isclose(value, _finite(episode.get('net_return'), 'diagnostic net_return'), abs_tol=1e-10):
            raise ValueError('Per-episode diagnostic return disagrees with evaluation')
        checked_returns.append(value)
        identities.append({'stock_code': stock, 'date': date, 'start_index': start, 'seed': seed + index})
    summary = {key: _finite(metrics.get(key), key) for key in METRICS}
    if not math.isclose(summary['mean_net_return'], math.fsum(checked_returns) / episodes, abs_tol=1e-10):
        raise ValueError('Mean return disagrees with paired episode returns')
    summary.update({key: metrics.get(key) for key in ('profitable_with_trades', 'evaluation_outcome')})
    summary['num_episodes'] = episodes
    summary.update({key: diagnostic.get(key) for key in
                    ('action_counts', 'action_rates', 'mean_action_probabilities', 'max_action_probabilities',
                     'buy_action_outcomes', 'execution_blocked_checks', *ORDER_METRICS)})
    return identities, checked_returns, summary


def _write_json(path, report):
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def run_comparison(checkpoint_path, comparison_report, *, output, extracted_dir,
                   episodes=8, seed=42, device='auto'):
    """Persist paired results; fail closed on input/condition/path mismatches."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError('episodes must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    destination = Path(output).expanduser()
    if os.path.lexists(destination):
        raise FileExistsError(f'Comparison output already exists: {destination}')
    destination = destination.resolve()
    if destination.suffix.lower() != '.json' or destination.name.lower() == 'evaluation_report.json':
        raise ValueError('Use a separate new comparison .json filename')
    details_dir = destination.with_name(destination.stem + '_evaluations')
    if os.path.lexists(details_dir):
        raise FileExistsError(f'Evaluation directory already exists: {details_dir}')
    selected_device = _device_name(device)
    data = Path(extracted_dir).expanduser().resolve()
    if not data.is_dir():
        raise FileNotFoundError(f'Extracted dataset not found: {data}')
    entries, conditions, comparison_path = _prepare_inputs(checkpoint_path, comparison_report, episodes, seed)
    details_dir.mkdir(parents=True, exist_ok=False)
    original_precision, original_rng = snapshot_precision(), snapshot_rng()
    report = {'format': 'grpo_paired_validation_comparison', 'format_version': 1, 'status': 'running',
              'comparison_report': str(comparison_path), 'device': selected_device, 'episodes': episodes,
              'seed': seed, 'split': 'validation', 'holdout_test_evaluated': False,
              'conditions': conditions, 'baseline': None, 'candidates': [],
              'units': {'mean_net_return': 'percent of initial cash',
                        'mean_net_return_delta_pp': 'percentage points', 'max_drawdown': 'percent',
                        'total_fees': 'account currency, summed over episodes',
                        'mean_realized_net_pnl': 'account currency, averaged over episodes'},
              'evidence_scope': 'One fixed update and observed validation paths; not proven profitability.',
              'small_sample': episodes < 64}
    baseline_keys = baseline_returns = None
    try:
        set_tf32(False)
        report['precision'] = precision_metadata()
        for entry in entries:
            _verify_hash(entry['checkpoint'], entry['sha256'])
            restore_rng(original_rng)
            logger.info('Evaluating %s on %d fixed validation episodes (LR=%s)',
                        entry['label'], episodes, entry['learning_rate'])
            evaluation = run_diagnostics(entry['checkpoint'], split='validation', episodes=episodes,
                                         seed=seed, device=selected_device, extracted_dir=data)
            _verify_hash(entry['checkpoint'], entry['sha256'])
            detail_path = details_dir / (entry['label'] + '.json')
            _write_json(detail_path, evaluation)
            keys, returns, metrics = _validate_evaluation(evaluation, conditions, episodes, seed, selected_device)
            item = {'checkpoint': str(entry['checkpoint']), 'checkpoint_sha256': entry['sha256'],
                    'learning_rate': entry['learning_rate'], 'evaluation_report': str(detail_path), 'metrics': metrics}
            if entry['label'] == 'baseline':
                baseline_keys, baseline_returns = keys, returns
                report['baseline'] = item
                report['episode_identities'] = keys
            else:
                if keys != baseline_keys:
                    raise ValueError('Candidate episode keys differ from baseline; paired comparison refused')
                deltas = [value - base for value, base in zip(returns, baseline_returns)]
                delta = math.fsum(deltas) / episodes
                item['paired'] = {'mean_net_return_delta_pp': delta, 'episode_deltas_pp': deltas,
                                  'improved_episodes': sum(value > 0 for value in deltas),
                                  'worse_episodes': sum(value < 0 for value in deltas),
                                  'unchanged_episodes': sum(value == 0 for value in deltas),
                                  'both_no_trade': metrics['no_trade_episode_fraction'] == 1
                                      and report['baseline']['metrics']['no_trade_episode_fraction'] == 1,
                                  'observed_mean_return_improved': delta > 0}
                report['candidates'].append(item)
            logger.info('%s validation: return=%.6f%%, round_trips=%s, no_trade_fraction=%s',
                        entry['label'], metrics['mean_net_return'], metrics['round_trip_count'],
                        metrics['no_trade_episode_fraction'])
        all_items = [report['baseline'], *report['candidates']]
        report['summary'] = {
            'candidate_count': len(report['candidates']),
            'baseline_mean_net_return': report['baseline']['metrics']['mean_net_return'],
            'all_no_trade': all(item['metrics']['no_trade_episode_fraction'] == 1 for item in all_items),
            'improved_mean_return_candidate_count': sum(item['paired']['observed_mean_return_improved']
                                                        for item in report['candidates']),
            'profitable_with_trades_candidate_count': sum(item['metrics']['profitable_with_trades'] is True
                                                         for item in report['candidates']),
            'evidence_scope': report['evidence_scope'], 'small_sample': report['small_sample'],
        }
        report['status'] = 'completed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        restore_precision(original_precision)
        restore_rng(original_rng)
        report['precision_restored'] = precision_metadata() == original_precision
        report['rng_restored'] = _state_equal(snapshot_rng(), original_rng)
        _write_json(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--comparison-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--extracted-dir', required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_comparison(args.checkpoint, args.comparison_report, output=args.output,
                   extracted_dir=args.extracted_dir, episodes=args.episodes, seed=args.seed, device=args.device)
    print(f'Paired validation comparison: {Path(args.output).resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
