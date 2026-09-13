"""Read-only preparation for bounded local checks of a recorded training run."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from .diagnose_xlstm import _device_name, _saved_splits, _training_factories
from .evaluation import normalize_date
from .likelihood_failure import portable_metadata
from lib.observations import ObservationBuilder


MAX_LOCAL_CACHE_BYTES = 64 * 1024 * 1024
LEGACY_DEFAULTS = {
    'liquidation_max_steps': 0, 'lambda_gae': .95, 'decision_interval_seconds': 0.0,
    'episode_duration_seconds': 0.0, 'execution_action_mask': False, 'no_trade_patience': 0,
    'no_trade_max_validations': 0, 'policy_update_checks': False,
    'rollout_logprob_tolerance': 1e-3, 'kl_probe_samples': 32,
    'profitable_min_round_trips': 20, 'profitable_min_traded_dates': 3,
    'group_advantage_coef': 1.0, 'training_seed': 42,
}
REQUIRED_CONFIG = (
    'seq_len', 'features', 'episode_steps', 'hidden_dim', 'cnn_channels', 'rnn_hidden_dim',
    'action_dim', 'episodes_per_group', 'num_groups', 'batch_size', 'num_epochs', 'checkpoint_segments',
    'gamma', 'clip', 'kl_target', 'entropy_coef', 'value_coef', 'max_grad_norm', 'use_gae',
    'use_raw_data', 'rolling_window_size', 'rolling_min_samples', 'max_stages', 'max_holding_seconds',
    'initial_cash', 'stop_loss_pct', 'execution_config', 'base_price', 'price_scale',
    'transaction_cost_rate', 'buy_tax_rate', 'sell_tax_rate', 'no_trade_penalty',
    'max_trades_per_episode', 'step_reward_scale', 'win_bonus', 'loss_penalty', 'buy_signal_bonus',
    'train_end_date', 'validation_end_date', 'validation_fraction', 'test_fraction', 'embargo_dates',
)
LOCAL_EXECUTION_SETTINGS = frozenset({
    'device', 'num_workers', 'cache_max_bytes', 'extracted_dir', 'db_path', 'output_dir',
    'load_policy', 'resume', 'resume_lr', 'capture_update_bundle',
})
# restore_training_progress restores these from the trainer checkpoint, even
# when an older extra_state.training_config still contains different metadata.
RESTORED_TRAINER_SETTINGS = (
    'group_advantage_coef', 'no_trade_patience', 'no_trade_max_validations',
    'profitable_min_round_trips', 'profitable_min_traded_dates',
    'policy_update_checks', 'rollout_logprob_tolerance', 'kl_probe_samples',
)


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _sha256(value, source):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
        raise ValueError(f'{source} must contain a SHA256 hexadecimal digest')
    return value.lower()


def _manifest_expectations(checkpoint_path, expected_manifest):
    expectations = []
    if expected_manifest is not None:
        expectations.append({'source': 'explicit expected_manifest',
                             'sha256': _sha256(expected_manifest, 'expected_manifest')})
    candidates = [checkpoint_path.parent / 'colab_run.json']
    if checkpoint_path.parent.name == 'checkpoints':
        candidates.append(checkpoint_path.parent.parent / 'colab_run.json')
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(encoding='utf-8') as stream:
                metadata = json.load(stream)
            if not isinstance(metadata, dict):
                raise ValueError(f'Invalid source run metadata: {candidate}')
            if 'manifest_sha256' in metadata:
                expectations.append({'source': str(candidate),
                                     'sha256': _sha256(metadata['manifest_sha256'], str(candidate))})
    return expectations


def _validate_training_state(checkpoint, config, builder):
    weights = checkpoint.get('policy_state_dict')
    if not isinstance(weights, dict) or not weights or any(
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('Checkpoint requires a finite policy_state_dict')
    expected_shapes = {
        'conv1.weight': (config.cnn_channels, builder.obs_dim, 5),
        'xlstm.cells.0.w_q.weight': (config.rnn_hidden_dim, config.cnn_channels),
        'fc1.weight': (config.hidden_dim, config.rnn_hidden_dim),
        'policy_head.weight': (config.action_dim, config.hidden_dim),
        'value_head.weight': (1, config.hidden_dim),
    }
    for key, shape in expected_shapes.items():
        if key not in weights or tuple(weights[key].shape) != shape:
            raise ValueError(f'Saved policy shape disagrees with training configuration/schema: {key}')
    optimizer = checkpoint.get('optimizer_state_dict')
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict)
            or not isinstance(optimizer.get('param_groups'), list) or not optimizer['param_groups']):
        raise ValueError('Checkpoint requires an Adam optimizer_state_dict with parameter groups')
    parameter_ids = []
    rates = []
    for group in optimizer['param_groups']:
        if not isinstance(group, dict) or not isinstance(group.get('params'), list) or not group['params']:
            raise ValueError('Saved Adam parameter groups must be nonempty')
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in group['params']):
            raise ValueError('Saved Adam parameter IDs must be nonnegative integers')
        parameter_ids.extend(group['params'])
        rate = group.get('lr')
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Saved Adam learning rate must be finite and positive')
        rates.append(float(rate))
    if len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError('Saved Adam parameter IDs must be unique')
    if any(index not in parameter_ids for index in optimizer['state']):
        raise ValueError('Saved Adam state has an unknown parameter ID')
    for state in optimizer['state'].values():
        if not isinstance(state, dict) or any(name not in state for name in ('step', 'exp_avg', 'exp_avg_sq')):
            raise ValueError('Saved optimizer state lacks Adam moments or step counters')
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError(f'Saved Adam {name} must be finite')
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid saved Adam {name}')
    for name in ('iteration', 'num_updates', 'total_timesteps'):
        value = checkpoint.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'Full training checkpoint requires nonnegative integer {name}')
    return rates


def _verify_episode_files(episodes, data_path):
    bytes_checked, total_bytes = 0, 0
    for episode in episodes:
        relative = episode.get('file_path')
        if not isinstance(relative, str) or not relative:
            raise ValueError('Manifest episode requires file_path')
        path = (data_path / relative).resolve()
        if not path.is_relative_to(data_path):
            raise ValueError(f'Manifest episode escapes the extracted dataset: {relative}')
        if not path.is_file():
            raise FileNotFoundError(f'Manifest episode file is missing: {path}')
        actual_bytes = path.stat().st_size
        if 'bytes' in episode:
            declared = episode['bytes']
            if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
                raise ValueError(f'Invalid manifest episode byte count: {relative}')
            if declared != actual_bytes:
                raise ValueError(f'Manifest episode size differs: {relative}, expected={declared}, actual={actual_bytes}')
            bytes_checked += 1
        total_bytes += actual_bytes
    return {'files_checked': len(episodes), 'file_sizes_checked': bytes_checked, 'total_file_bytes': total_bytes}


def prepare_local_config(checkpoint_path, extracted_dir, output_dir, *, device='cuda', num_workers=2,
                         expected_manifest=None):
    """Preserve recorded learning behavior; adjust only local execution locations/resources.

    expected_manifest is an optional SHA256 string. A colab_run.json beside the
    source checkpoint/run is checked independently when it records a manifest
    hash. Episode existence/sizes are checked; their contents and model execution are not.
    """
    workers = _positive_integer('num_workers', num_workers)
    selected_device = _device_name(device)
    checkpoint_path = Path(checkpoint_path).resolve()
    data_path, destination = Path(extracted_dir).resolve(), Path(output_dir).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    if not data_path.is_dir():
        raise FileNotFoundError(f'Extracted dataset not found: {data_path}')
    manifest_path = data_path / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Dataset manifest not found: {manifest_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('A full training checkpoint is required')
    extra = checkpoint.get('extra_state')
    saved = extra.get('training_config') if isinstance(extra, dict) else None
    if not isinstance(saved, dict) or not saved:
        raise ValueError('Checkpoint requires extra_state.training_config')
    saved = portable_metadata(saved)
    missing = [name for name in REQUIRED_CONFIG if name not in saved]
    if missing:
        raise ValueError(f'Saved training_config is incomplete; refusing new training defaults: {missing}')
    schema = checkpoint.get('observation_schema')
    builder = ObservationBuilder.from_schema(schema)
    saved_splits = _saved_splits(checkpoint)
    config_type, _, _ = _training_factories()
    config = config_type()
    known_settings = set(config.__dict__)
    unknown = sorted(set(saved) - known_settings)
    if unknown:
        raise ValueError(f'Saved training_config has unsupported settings: {unknown}')
    for name, value in deepcopy(saved).items():
        setattr(config, name, value)
    checkpoint_config = checkpoint.get('config', {})
    if not isinstance(checkpoint_config, dict):
        raise ValueError('Invalid saved checkpoint config')
    legacy_defaults = {}
    for name, default in LEGACY_DEFAULTS.items():
        if name not in saved:
            value = checkpoint_config.get(name, default)
            setattr(config, name, value)
            legacy_defaults[name] = value
    account, execution = schema.get('version') in (3, 4), schema.get('version') == 4
    for name, expected in (('account_observations', account), ('execution_observations', execution)):
        if name in saved and saved[name] != expected:
            raise ValueError(f'Saved {name} contradicts the observation schema')
        setattr(config, name, expected)
    for name in ('seq_len', 'rolling_window_size', 'rolling_min_samples', 'max_holding_seconds', 'max_stages'):
        if getattr(config, name) != schema[name]:
            raise ValueError(f'Saved {name} contradicts the observation schema')
    if config.features != len(builder.feature_columns):
        raise ValueError('Saved features count contradicts the observation schema')
    rates = _validate_training_state(checkpoint, config, builder)
    cache = saved.get('cache_max_bytes', MAX_LOCAL_CACHE_BYTES)
    if isinstance(cache, bool) or not isinstance(cache, int) or cache < 0:
        raise ValueError('Saved cache_max_bytes must be a nonnegative integer')
    changes = {
        'device': selected_device, 'num_workers': workers, 'cache_max_bytes': min(cache, MAX_LOCAL_CACHE_BYTES),
        'extracted_dir': str(data_path), 'db_path': None, 'output_dir': str(destination),
        'load_policy': str(checkpoint_path), 'resume': True, 'resume_lr': None,
        'capture_update_bundle': None, 'lr': rates[0],
    }
    overrides = {}
    for name, value in changes.items():
        before = getattr(config, name)
        if before != value:
            overrides[name] = {'before': before, 'after': value}
        setattr(config, name, value)
    config.validate()
    raw_manifest = manifest_path.read_bytes()
    actual_hash = hashlib.sha256(raw_manifest).hexdigest()
    expectations = _manifest_expectations(checkpoint_path, expected_manifest)
    for expectation in expectations:
        if actual_hash != expectation['sha256']:
            raise ValueError(f"Local manifest SHA256 differs from {expectation['source']}: "
                             f"expected={expectation['sha256']}, actual={actual_hash}")
    manifest = json.loads(raw_manifest)
    episodes = manifest.get('episodes') if isinstance(manifest, dict) else None
    if not isinstance(episodes, list) or not episodes or any(not isinstance(episode, dict) for episode in episodes):
        raise ValueError('Manifest requires a nonempty episodes list')
    metadata = manifest.get('metadata', {})
    if not isinstance(metadata, dict) or metadata.get('feature_columns') != builder.feature_columns:
        raise ValueError('Manifest feature order differs from the checkpoint observation schema')
    file_checks = _verify_episode_files(episodes, data_path)
    from .train_xlstm import prepare_date_splits
    regenerated = prepare_date_splits(config)
    for split in ('train', 'validation', 'test'):
        expected = [normalize_date(day) for day in saved_splits[split]]
        actual = [normalize_date(day) for day in regenerated[split]]
        if expected != actual:
            raise ValueError(f'Regenerated {split} date split differs from saved lineage; '
                             'refusing to change checkpoint population')
    date_count = len({normalize_date(episode['date']) for episode in episodes})
    samples = config.episodes_per_group * config.num_groups * config.episode_steps
    weight_bytes = sum(value.numel() * value.element_size() for value in checkpoint['policy_state_dict'].values())
    optimizer_bytes = sum(value.numel() * value.element_size()
                          for state in checkpoint['optimizer_state_dict']['state'].values()
                          for value in state.values() if isinstance(value, torch.Tensor))
    return {
        'config': portable_metadata(config.__dict__), 'checkpoint': str(checkpoint_path),
        'manifest': {'path': str(manifest_path), 'sha256': actual_hash,
                     'expected_sha256': expectations[0]['sha256'] if expectations else None,
                     'expectations': expectations, 'match_verified': bool(expectations),
                     'episode_count': len(episodes), 'date_count': date_count,
                     'episode_contents_verified': False, **file_checks},
        'date_splits': deepcopy(saved_splits), 'observation_schema': deepcopy(schema),
        'overrides': overrides, 'legacy_defaults': portable_metadata(legacy_defaults),
        'optimizer_learning_rates': rates,
        'memory_estimate': {
            'planned_rollout_samples': samples,
            'rollout_observation_bytes': samples * config.seq_len * builder.obs_dim * 4,
            'policy_state_bytes': weight_bytes, 'optimizer_state_bytes': optimizer_bytes,
            'cache_max_bytes': config.cache_max_bytes, 'worker_cache_bytes': workers * config.cache_max_bytes,
            'notice': 'Partial estimate from checkpoint tensors and configured rollout count. '
                      'Excludes copies, liquidation tail, execution arrays and GPU activations; not a peak-memory guarantee.',
        },
    }


def _state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_state_equal(value, right[key]) for key, value in left.items()))
    if isinstance(left, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_state_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _config_equal(left, right):
    """Compare numeric configuration values without treating True as numeric one."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and left == right
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_config_equal(value, right[key]) for key, value in left.items()))
    if isinstance(left, list):
        return (isinstance(right, list) and len(left) == len(right)
                and all(_config_equal(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _validate_bundle_settings(bundle, checkpoint):
    from .update_diagnostic import TRAINER_FIELDS

    saved = checkpoint.get('extra_state', {}).get('training_config')
    captured = bundle.get('training_config')
    settings = checkpoint.get('config', {})
    if not isinstance(saved, dict) or not saved or not isinstance(captured, dict) or not isinstance(settings, dict):
        raise ValueError('Bundle and source require saved training configuration')
    source, captured, settings = portable_metadata(saved), portable_metadata(captured), portable_metadata(settings)
    missing = [name for name in REQUIRED_CONFIG if name not in source]
    if missing:
        raise ValueError(f'Source training configuration is incomplete: {missing}')
    for name, default in LEGACY_DEFAULTS.items():
        if name not in source:
            source[name] = settings.get(name, default)
    for name in RESTORED_TRAINER_SETTINGS:
        source[name] = settings.get(name, LEGACY_DEFAULTS[name])
    schema = checkpoint['observation_schema']
    source['account_observations'] = schema.get('version') in (3, 4)
    source['execution_observations'] = schema.get('version') == 4
    groups = checkpoint['optimizer_state_dict'].get('param_groups')
    if not isinstance(groups, list) or not groups or not isinstance(groups[0], dict):
        raise ValueError('Source checkpoint requires Adam parameter groups')
    rate = groups[0].get('lr')
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Source Adam learning rate must be finite and positive')
    source['lr'] = float(rate)
    permitted = {}
    # The local capture command explicitly enables the diagnostic safeguards.
    # Permit only this one strengthening; never permit disabling saved checks.
    if source['policy_update_checks'] is False and captured.get('policy_update_checks') is True:
        permitted['policy_update_checks'] = {'before': False, 'after': True,
                                            'reason': 'Local capture explicitly requires update safeguards'}
        source['policy_update_checks'] = True
    compared = sorted((set(source) | set(captured)) - LOCAL_EXECUTION_SETTINGS)
    for name in compared:
        if name not in source or name not in captured or not _config_equal(source[name], captured[name]):
            raise ValueError(f'Fixed update bundle has different source training/environment setting: {name}')
    hyperparameters = bundle.get('trainer_hyperparameters')
    if not isinstance(hyperparameters, dict) or set(hyperparameters) != set(TRAINER_FIELDS):
        raise ValueError('Fixed update bundle requires complete trainer_hyperparameters')
    hyperparameters = portable_metadata(hyperparameters)
    for name in TRAINER_FIELDS:
        config_name = {'learning_rate': 'lr', 'clip_epsilon': 'clip'}.get(name, name)
        if not _config_equal(hyperparameters[name], source[config_name]):
            raise ValueError(f'Fixed update bundle has different effective trainer hyperparameter: {name}')
    reconstruction = bundle.get('policy_config', {}).get('kwargs')
    expected_policy = {
        'obs_dim': ObservationBuilder.from_schema(schema).obs_dim,
        'cnn_channels': source['cnn_channels'], 'rnn_hidden_dim': source['rnn_hidden_dim'],
        'fc_hidden_dim': source['hidden_dim'], 'action_dim': source['action_dim'],
        'checkpoint_segments': source['checkpoint_segments'], 'max_stages': source['max_stages'],
        'execution_action_mask': source['execution_action_mask'],
    }
    if not _config_equal(expected_policy, reconstruction):
        raise ValueError('Fixed update bundle policy reconstruction differs from source configuration')
    local_differences = {name: {'before': saved.get(name), 'after': captured.get(name)}
                         for name in sorted(LOCAL_EXECUTION_SETTINGS)
                         if not _config_equal(saved.get(name), captured.get(name))}
    return {'semantic_config_match': True, 'trainer_hyperparameters_match': True,
            'compared_settings': compared, 'permitted_capture_overrides': permitted,
            'local_execution_differences': portable_metadata(local_differences)}


def validate_bundle_source(bundle_path, checkpoint_path):
    """Reject reuse of a rollout captured from different weights, Adam or lineage.

    Memory mapping keeps the potentially large rollout observations untouched;
    only small model/optimizer metadata pages are needed for source comparison.
    """
    from .update_diagnostic import BUNDLE_FORMAT, BUNDLE_VERSION

    bundle_path, checkpoint_path = Path(bundle_path).resolve(), Path(checkpoint_path).resolve()
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=True, mmap=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (not isinstance(bundle, dict) or bundle.get('format') != BUNDLE_FORMAT
            or bundle.get('format_version') != BUNDLE_VERSION):
        raise ValueError('Unsupported fixed update bundle format/version')
    if not isinstance(checkpoint, dict):
        raise ValueError('A full source checkpoint is required')
    for key in ('policy_state_dict', 'optimizer_state_dict', 'observation_schema'):
        if not isinstance(bundle.get(key), dict) or not bundle[key] or not isinstance(checkpoint.get(key), dict):
            raise ValueError(f'Bundle and source checkpoint require {key}')
        if not _state_equal(bundle[key], checkpoint[key]):
            raise ValueError(f'Fixed update bundle does not match source checkpoint {key}')
    progress = bundle.get('progress')
    if not isinstance(progress, dict):
        raise ValueError('Fixed update bundle requires progress')
    for key in ('num_updates', 'total_timesteps'):
        for source in (progress, checkpoint):
            value = source.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'Invalid bundle/source {key}')
    if progress['num_updates'] != checkpoint['num_updates']:
        raise ValueError('Fixed update bundle num_updates differs from source checkpoint')
    if progress['total_timesteps'] < checkpoint['total_timesteps']:
        raise ValueError('Fixed update bundle timesteps precede source checkpoint')
    saved_splits = _saved_splits(checkpoint)
    configuration_match = _validate_bundle_settings(bundle, checkpoint)
    return {'bundle': str(bundle_path), 'checkpoint': str(checkpoint_path), 'matched': True,
            'checkpoint_num_updates': checkpoint['num_updates'], 'bundle_num_updates': progress['num_updates'],
            'checkpoint_total_timesteps': checkpoint['total_timesteps'],
            'bundle_total_timesteps': progress['total_timesteps'], 'date_splits': saved_splits,
            **configuration_match}
