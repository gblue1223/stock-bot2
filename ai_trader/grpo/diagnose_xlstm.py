"""Diagnose a saved xLSTM policy on its recorded evaluation dates, without training.

Example: python -m ai_trader.grpo.diagnose_xlstm --checkpoint models/run/checkpoints/checkpoint_best.pt
         --output models/run/validation_diagnostics.json --device cpu
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from ai_trader.grpo.evaluation import evaluate_policy, normalize_date, validate_checkpoint_dates
from lib.observations import ObservationBuilder


def _training_factories():
    # Import the shared constructors only when replay starts. The training entry
    # point and trainer/optimizer restoration are deliberately never invoked.
    from ai_trader.grpo.train_xlstm import TrainingConfig, create_environment, create_policy
    return TrainingConfig, create_environment, create_policy


def _saved_splits(checkpoint):
    splits = checkpoint.get('extra_state', {}).get('date_splits')
    if not isinstance(splits, dict) or any(
            not isinstance(splits.get(name), list) or not splits[name]
            for name in ('train', 'validation', 'test')):
        raise ValueError('Checkpoint requires nonempty train/validation/test date_splits lineage')
    normalized = {name: [normalize_date(day) for day in splits[name]]
                  for name in ('train', 'validation', 'test')}
    if (max(normalized['train']) >= min(normalized['validation']) or
            max(normalized['validation']) >= min(normalized['test'])):
        raise ValueError('Saved date_splits must be chronological and disjoint')
    validate_checkpoint_dates(checkpoint, splits)
    return deepcopy(splits)


def _device_name(device):
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu')
    return device


def run_diagnostics(checkpoint_path, *, split='validation', episodes=None, seed=None,
                    device='auto', extracted_dir=None, db_path=None):
    """Replay exactly the requested checkpoint; do not search for newer weights."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation or test')
    if extracted_dir is not None and db_path is not None:
        raise ValueError('Specify only one of extracted_dir and db_path')
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    # This command accepts a trusted local training checkpoint, including its
    # saved configuration. It does not deserialize downloaded/untrusted models.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('A full training checkpoint is required')
    extra = checkpoint.get('extra_state', {})
    saved_config = extra.get('training_config') if isinstance(extra, dict) else None
    if not isinstance(saved_config, dict) or not saved_config:
        raise ValueError('Checkpoint requires extra_state.training_config')
    schema = checkpoint.get('observation_schema')
    schema_builder = ObservationBuilder.from_schema(schema)
    splits = _saved_splits(checkpoint)
    weights = checkpoint.get('policy_state_dict')
    if not isinstance(weights, dict) or not weights:
        raise ValueError('Checkpoint requires policy_state_dict')
    selected_episodes = saved_config.get('evaluation_episodes', 8) if episodes is None else episodes
    selected_seed = saved_config.get('evaluation_seed', 42) if seed is None else seed
    if (isinstance(selected_episodes, bool) or not isinstance(selected_episodes, int)
            or selected_episodes <= 0):
        raise ValueError('episodes must be a positive integer')
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int) or selected_seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    selected_device = _device_name(device)
    config_type, create_environment, create_policy = _training_factories()
    config = config_type()
    for name, value in deepcopy(saved_config).items():
        if not name.startswith('_'):
            setattr(config, name, value)
    # New training defaults must not change the replay of a legacy checkpoint.
    config.account_observations = schema.get('version') in (3, 4)
    config.execution_observations = schema.get('version') == 4
    config.liquidation_max_steps = saved_config.get('liquidation_max_steps', 0)
    config.decision_interval_seconds = saved_config.get('decision_interval_seconds', 0.0)
    config.episode_duration_seconds = saved_config.get('episode_duration_seconds', 0.0)
    config.group_advantage_coef = saved_config.get('group_advantage_coef', 1.0)
    overrides = {}
    if extracted_dir is not None:
        config.extracted_dir = str(Path(extracted_dir).resolve())
        config.db_path = None
        overrides.update(extracted_dir=config.extracted_dir, db_path=None)
    elif db_path is not None:
        config.db_path = str(Path(db_path).resolve())
        config.extracted_dir = None
        overrides.update(db_path=config.db_path, extracted_dir=None)
    if config.extracted_dir:
        if not Path(config.extracted_dir).is_dir():
            raise FileNotFoundError('Saved extracted_dir is unavailable; supply --extracted-dir or --db-path')
    elif not config.db_path or not Path(config.db_path).is_file():
        raise FileNotFoundError('Saved database is unavailable; supply --extracted-dir or --db-path')
    if episodes is not None:
        overrides['evaluation_episodes'] = episodes
    if seed is not None:
        overrides['evaluation_seed'] = seed
    if selected_device != saved_config.get('device'):
        overrides['device'] = selected_device
    config.device = selected_device
    environment = create_environment(config, selected_device, allowed_dates=splits[split])
    try:
        schema_builder.validate_schema(environment.observation_schema)
        # A moved dataset must still contain every selected date. Do not silently
        # reduce the replay population to whatever happens to be available.
        date_index = 2 if config.extracted_dir else 1
        available = {normalize_date(key[date_index]) for key in environment.valid_keys}
        expected = {normalize_date(day) for day in splits[split]}
        if available != expected:
            raise ValueError(f'Environment dates differ from saved {split} split: '
                             f'missing={sorted(expected - available)}, extra={sorted(available - expected)}')
        policy = create_policy(config, environment, selected_device)
        policy.load_state_dict(weights, strict=True)
        metrics = evaluate_policy(policy, environment, num_episodes=selected_episodes,
                                  seed=selected_seed, device=selected_device,
                                  collect_diagnostics=True)
        return {
            'checkpoint': str(path),
            'checkpoint_iteration': checkpoint.get('iteration'),
            'checkpoint_total_timesteps': checkpoint.get('total_timesteps'),
            'split': split, 'dates': splits[split], 'date_splits': splits,
            'deterministic': True, 'device': selected_device,
            'observation_schema': schema,
            'saved_training_config': deepcopy(saved_config),
            'overrides': overrides,
            'execution_config': deepcopy(config.execution_config),
            'costs': {key: getattr(config, key) for key in
                      ('transaction_cost_rate', 'buy_tax_rate', 'sell_tax_rate')},
            'metrics': metrics,
        }
    finally:
        environment.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted local training checkpoint to replay')
    parser.add_argument('--output', required=True, help='New diagnostic JSON file; existing files are refused')
    parser.add_argument('--split', choices=('validation', 'train', 'test'), default='validation')
    parser.add_argument('--episodes', type=int, help='Defaults to the saved evaluation_episodes')
    parser.add_argument('--seed', type=int, help='Defaults to the saved evaluation_seed')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    data = parser.add_mutually_exclusive_group()
    data.add_argument('--extracted-dir', help='Replacement location of the same extracted dataset')
    data.add_argument('--db-path', help='Replacement location of the same DuckDB dataset')
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.name.lower() == 'evaluation_report.json':
        raise ValueError('Use a separate diagnostic filename, not evaluation_report.json')
    if output.suffix.lower() != '.json':
        raise ValueError('Diagnostic output must have a .json extension')
    if output == Path(args.checkpoint).resolve() or output.exists():
        raise FileExistsError(f'Diagnostic output already exists or is the checkpoint: {output}')
    report = run_diagnostics(args.checkpoint, split=args.split, episodes=args.episodes,
                             seed=args.seed, device=args.device,
                             extracted_dir=args.extracted_dir, db_path=args.db_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against another process writing during replay.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(f'Diagnostic report saved: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
