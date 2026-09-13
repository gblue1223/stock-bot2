"""Local preparation preserves training semantics and rejects a different dataset lineage."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from ai_trader.grpo.evaluation import chronological_date_split
from ai_trader.grpo.local_check_inputs import MAX_LOCAL_CACHE_BYTES, prepare_local_config, validate_bundle_source
from ai_trader.grpo.train_xlstm import TrainingConfig
from lib.observations import ObservationBuilder


@pytest.fixture
def saved_run(tmp_path):
    torch.set_num_threads(1)
    dataset = tmp_path / 'episodes'
    dataset.mkdir()
    dates = [f'202401{day:02d}' for day in range(1, 11)]
    builder = ObservationBuilder(['first', 'second'], seq_len=8, rolling_window_size=4,
                                 rolling_min_samples=2, account_observations=True, execution_observations=True)
    manifest = {'metadata': {'feature_columns': builder.feature_columns},
                'episodes': [{'date': day, 'length': 100, 'file_path': f'episode_{day}.npz', 'bytes': 5} for day in dates]}
    for episode in manifest['episodes']:
        (dataset / episode['file_path']).write_bytes(b'12345')  # Only stat is authorized; never parse these files.
    manifest_path = dataset / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config = TrainingConfig()
    config.seq_len, config.features = 8, 2
    config.rolling_window_size, config.rolling_min_samples = 4, 2
    config.cnn_channels, config.rnn_hidden_dim, config.hidden_dim = 4, 4, 8
    config.episode_steps, config.batch_size = 10, 4
    config.lr = 3e-5
    config.resume_lr, config.capture_update_bundle = .4, '/old/capture.pt'
    config.cache_max_bytes = 128 * 1024 * 1024
    config.extracted_dir, config.output_dir = '/colab/data', '/colab/run'
    weights = {
        'conv1.weight': torch.zeros(4, builder.obs_dim, 5),
        'xlstm.cells.0.w_q.weight': torch.zeros(4, 4),
        'fc1.weight': torch.zeros(8, 4), 'policy_head.weight': torch.zeros(3, 8),
        'value_head.weight': torch.zeros(1, 8),
    }
    checkpoint = {
        'policy_state_dict': weights,
        'optimizer_state_dict': {'state': {0: {'step': torch.tensor(5.), 'exp_avg': torch.ones(4),
                                              'exp_avg_sq': torch.ones(4)}},
                                 'param_groups': [{'lr': 1e-5, 'params': [0]}]},
        'observation_schema': builder.schema,
        'iteration': 19, 'num_updates': 19, 'total_timesteps': 400,
        'config': {},
        'extra_state': {'training_config': deepcopy(config.__dict__),
                        'date_splits': chronological_date_split(dates)},
    }
    run_path = tmp_path / 'run'
    checkpoint_path = run_path / 'checkpoints' / 'checkpoint_iter19.pt'
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(checkpoint, checkpoint_path)
    (run_path / 'colab_run.json').write_text(json.dumps({'manifest_sha256': digest}), encoding='utf-8')
    return checkpoint_path, dataset, tmp_path / 'local_output', checkpoint, digest


def test_prepare_is_read_only_preserves_learning_config_and_uses_safe_load(saved_run, monkeypatch):
    checkpoint_path, dataset, output, checkpoint, digest = saved_run
    before_bytes = checkpoint_path.read_bytes()
    original_load = torch.load
    loads = []

    def safe_load(*args, **kwargs):
        loads.append(kwargs.get('weights_only'))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', safe_load)
    result = prepare_local_config(checkpoint_path, dataset, output, device='cpu', expected_manifest=digest)
    assert loads == [True]
    assert checkpoint_path.read_bytes() == before_bytes
    assert not output.exists()
    assert result['date_splits'] == checkpoint['extra_state']['date_splits']
    assert result['manifest']['sha256'] == digest and result['manifest']['match_verified']
    assert result['manifest']['episode_count'] == result['manifest']['date_count'] == 10
    assert result['manifest']['episode_contents_verified'] is False
    assert result['manifest']['files_checked'] == result['manifest']['file_sizes_checked'] == 10
    config = result['config']
    assert config['lr'] == 1e-5
    assert config['resume'] is True and config['load_policy'] == str(checkpoint_path.resolve())
    assert config['resume_lr'] is None and config['capture_update_bundle'] is None
    assert config['num_workers'] == 2 and config['cache_max_bytes'] == MAX_LOCAL_CACHE_BYTES
    assert config['db_path'] is None and config['extracted_dir'] == str(dataset.resolve())
    allowed = {'device', 'num_workers', 'cache_max_bytes', 'extracted_dir', 'db_path', 'output_dir',
               'load_policy', 'resume', 'resume_lr', 'capture_update_bundle', 'lr'}
    for name, original in checkpoint['extra_state']['training_config'].items():
        if name not in allowed:
            assert config[name] == original, name
    assert result['memory_estimate']['worker_cache_bytes'] == 2 * MAX_LOCAL_CACHE_BYTES
    assert result['memory_estimate']['rollout_observation_bytes'] == 4 * 4 * 10 * 8 * 38 * 4


def test_source_manifest_mismatch_rejects_even_when_explicit_hash_matches_local(saved_run):
    checkpoint_path, dataset, output, _, digest = saved_run
    (checkpoint_path.parent.parent / 'colab_run.json').write_text(json.dumps({'manifest_sha256': 'a' * 64}), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest SHA256 differs'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu', expected_manifest=digest)
    assert not output.exists()


def test_recomputed_splits_must_exactly_equal_saved_population(saved_run):
    checkpoint_path, dataset, output, checkpoint, _ = saved_run
    checkpoint['extra_state']['date_splits']['train'].pop(0)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='Regenerated train date split differs'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu')


def test_missing_optimizer_rejected_without_touching_data(saved_run):
    checkpoint_path, dataset, output, checkpoint, _ = saved_run
    del checkpoint['optimizer_state_dict']
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='optimizer_state_dict'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu')


def test_invalid_actual_optimizer_lr_rejected(saved_run):
    checkpoint_path, dataset, output, checkpoint, _ = saved_run
    checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = float('nan')
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='learning rate must be finite'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu')


def test_policy_schema_shape_mismatch_rejected(saved_run):
    checkpoint_path, dataset, output, checkpoint, _ = saved_run
    checkpoint['policy_state_dict']['conv1.weight'] = torch.zeros(4, 37, 5)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='policy shape disagrees'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu')


def test_legacy_guards_remain_disabled_and_cache_is_not_increased(saved_run):
    checkpoint_path, dataset, output, checkpoint, _ = saved_run
    for name in ('policy_update_checks', 'rollout_logprob_tolerance', 'kl_probe_samples', 'no_trade_max_validations'):
        checkpoint['extra_state']['training_config'].pop(name)
    checkpoint['extra_state']['training_config']['cache_max_bytes'] = 1024
    torch.save(checkpoint, checkpoint_path)
    result = prepare_local_config(checkpoint_path, dataset, output, device='cpu')
    assert result['config']['policy_update_checks'] is False
    assert result['legacy_defaults']['policy_update_checks'] is False
    assert result['config']['no_trade_max_validations'] == 0
    assert result['config']['cache_max_bytes'] == 1024


def test_manifest_without_provenance_is_reported_unverified_and_bad_inputs_rejected(saved_run):
    checkpoint_path, dataset, output, _, _ = saved_run
    (checkpoint_path.parent.parent / 'colab_run.json').unlink()
    result = prepare_local_config(checkpoint_path, dataset, output, device='cpu')
    assert result['manifest']['expected_sha256'] is None
    assert result['manifest']['match_verified'] is False
    with pytest.raises(ValueError, match='positive integer'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu', num_workers=True)
    with pytest.raises(ValueError, match='SHA256'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu', expected_manifest='invalid')


def test_episode_file_missing_or_size_changed_rejected(saved_run):
    checkpoint_path, dataset, output, _, _ = saved_run
    path = dataset / 'episode_20240101.npz'
    path.write_bytes(b'truncated')
    with pytest.raises(ValueError, match='episode size differs'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu')
    path.unlink()
    with pytest.raises(FileNotFoundError, match='episode file is missing'):
        prepare_local_config(checkpoint_path, dataset, output, device='cpu')


def bundle_for_source(path, checkpoint):
    from ai_trader.grpo.update_diagnostic import BUNDLE_FORMAT, BUNDLE_VERSION, TRAINER_FIELDS

    bundle = {key: deepcopy(checkpoint[key]) for key in ('policy_state_dict', 'optimizer_state_dict', 'observation_schema')}
    bundle.update(format=BUNDLE_FORMAT, format_version=BUNDLE_VERSION,
                  progress={'num_updates': checkpoint['num_updates'], 'total_timesteps': checkpoint['total_timesteps'] + 160},
                  training_config=deepcopy(checkpoint['extra_state']['training_config']))
    config = bundle['training_config']
    config['lr'] = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    bundle['trainer_hyperparameters'] = {
        name: config[{'learning_rate': 'lr', 'clip_epsilon': 'clip'}.get(name, name)] for name in TRAINER_FIELDS}
    bundle['policy_config'] = {'kwargs': {
        'obs_dim': ObservationBuilder.from_schema(checkpoint['observation_schema']).obs_dim,
        'cnn_channels': config['cnn_channels'], 'rnn_hidden_dim': config['rnn_hidden_dim'],
        'fc_hidden_dim': config['hidden_dim'], 'action_dim': config['action_dim'],
        'checkpoint_segments': config['checkpoint_segments'], 'max_stages': config['max_stages'],
        'execution_action_mask': config['execution_action_mask'],
    }}
    torch.save(bundle, path)
    return bundle


def test_reused_bundle_matches_exact_source_weights_adam_and_progress(saved_run, monkeypatch):
    checkpoint_path, _, output, checkpoint, _ = saved_run
    bundle_path = output.parent / 'fixed.pt'
    bundle_for_source(bundle_path, checkpoint)
    calls = []
    original_load = torch.load

    def recorded_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', recorded_load)
    result = validate_bundle_source(bundle_path, checkpoint_path)
    assert result['matched']
    assert result['bundle_num_updates'] == result['checkpoint_num_updates'] == 19
    assert result['bundle_total_timesteps'] > result['checkpoint_total_timesteps']
    assert result['semantic_config_match'] and result['trainer_hyperparameters_match']
    assert calls[0]['weights_only'] is True and calls[0]['mmap'] is True
    assert calls[1]['weights_only'] is True


def test_reused_bundle_rejects_different_weights_adam_schema_or_update_count(saved_run):
    checkpoint_path, _, output, checkpoint, _ = saved_run
    bundle_path = output.parent / 'fixed.pt'
    original = bundle_for_source(bundle_path, checkpoint)
    for mutate, message in (
        (lambda bundle: bundle['policy_state_dict']['policy_head.weight'].add_(1), 'policy_state_dict'),
        (lambda bundle: bundle['optimizer_state_dict']['state'][0]['exp_avg'].add_(1), 'optimizer_state_dict'),
        (lambda bundle: bundle['optimizer_state_dict']['param_groups'][0].update(lr=3e-6), 'optimizer_state_dict'),
        (lambda bundle: bundle['observation_schema'].update(seq_len=16), 'observation_schema'),
        (lambda bundle: bundle['progress'].update(num_updates=20), 'num_updates'),
        (lambda bundle: bundle['progress'].update(total_timesteps=1), 'timesteps'),
    ):
        bundle = deepcopy(original)
        mutate(bundle)
        torch.save(bundle, bundle_path)
        with pytest.raises(ValueError, match=message):
            validate_bundle_source(bundle_path, checkpoint_path)


@pytest.mark.parametrize('name,value', [
    ('transaction_cost_rate', .002), ('sell_tax_rate', .001), ('step_reward_scale', 2.0),
    ('episode_steps', 20), ('episode_duration_seconds', 60.0), ('liquidation_max_steps', 0),
    ('gamma', .5), ('lambda_gae', .9), ('batch_size', 8), ('num_epochs', 3),
    ('episodes_per_group', 8), ('group_advantage_coef', 0.0), ('kl_target', .1),
    ('rollout_logprob_tolerance', .1), ('kl_probe_samples', 8), ('execution_action_mask', False),
    ('validation_fraction', .1), ('training_seed', 4),
])
def test_reused_bundle_rejects_semantic_training_and_environment_changes(saved_run, name, value):
    checkpoint_path, _, output, checkpoint, _ = saved_run
    path = output.parent / 'fixed.pt'
    bundle = bundle_for_source(path, checkpoint)
    bundle['training_config'][name] = value
    if name in bundle['trainer_hyperparameters']:
        bundle['trainer_hyperparameters'][name] = value
    torch.save(bundle, path)
    with pytest.raises(ValueError, match=f'training/environment setting: {name}'):
        validate_bundle_source(path, checkpoint_path)


def test_nested_execution_settings_and_hidden_effective_hyperparameters_are_checked(saved_run):
    checkpoint_path, _, output, checkpoint, _ = saved_run
    path = output.parent / 'fixed.pt'
    original = bundle_for_source(path, checkpoint)
    mutations = [
        (lambda bundle: bundle['training_config']['execution_config'].update(slippage_bps=100), 'execution_config'),
        (lambda bundle: bundle['trainer_hyperparameters'].update(gamma=.5), 'effective trainer hyperparameter: gamma'),
        (lambda bundle: bundle['trainer_hyperparameters'].update(learning_rate=3e-5), 'effective trainer hyperparameter: learning_rate'),
        (lambda bundle: bundle['trainer_hyperparameters'].update(use_gae=1), 'effective trainer hyperparameter: use_gae'),
        (lambda bundle: bundle['policy_config']['kwargs'].update(execution_action_mask=False), 'policy reconstruction'),
    ]
    for mutate, message in mutations:
        bundle = deepcopy(original)
        mutate(bundle)
        torch.save(bundle, path)
        with pytest.raises(ValueError, match=message):
            validate_bundle_source(path, checkpoint_path)


def test_local_execution_changes_and_actual_adam_lr_are_allowed(saved_run):
    checkpoint_path, _, output, checkpoint, _ = saved_run
    path = output.parent / 'fixed.pt'
    bundle = bundle_for_source(path, checkpoint)
    # Saved metadata lr is stale (3e-5); capture uses actual Adam lr (1e-5).
    assert checkpoint['extra_state']['training_config']['lr'] != bundle['training_config']['lr']
    local = dict(device='cpu', num_workers=1, cache_max_bytes=512, extracted_dir='/local/data',
                 db_path=None, output_dir='/local/out', load_policy='/local/source.pt', resume=True,
                 resume_lr=None, capture_update_bundle='/local/new_bundle.pt')
    bundle['training_config'].update(local)
    torch.save(bundle, path)
    result = validate_bundle_source(path, checkpoint_path)
    assert result['matched'] and result['semantic_config_match']
    assert result['local_execution_differences']['device']['after'] == 'cpu'
    assert result['local_execution_differences']['capture_update_bundle']['after'] == '/local/new_bundle.pt'


def test_legacy_guard_enable_is_recorded_but_disabling_saved_guards_is_rejected(saved_run):
    checkpoint_path, _, output, checkpoint, _ = saved_run
    path = output.parent / 'fixed.pt'
    original = bundle_for_source(path, checkpoint)
    for name in ('policy_update_checks', 'no_trade_max_validations', 'rollout_logprob_tolerance', 'kl_probe_samples'):
        checkpoint['extra_state']['training_config'].pop(name)
    torch.save(checkpoint, checkpoint_path)
    result = validate_bundle_source(path, checkpoint_path)
    assert result['permitted_capture_overrides']['policy_update_checks']['before'] is False
    assert result['permitted_capture_overrides']['policy_update_checks']['after'] is True
    checkpoint['config']['policy_update_checks'] = True
    torch.save(checkpoint, checkpoint_path)
    original['training_config']['policy_update_checks'] = False
    original['trainer_hyperparameters']['policy_update_checks'] = False
    torch.save(original, path)
    with pytest.raises(ValueError, match='training/environment setting: policy_update_checks'):
        validate_bundle_source(path, checkpoint_path)


def test_restored_trainer_checkpoint_settings_override_stale_config_metadata(saved_run):
    checkpoint_path, _, output, checkpoint, _ = saved_run
    path = output.parent / 'fixed.pt'
    bundle = bundle_for_source(path, checkpoint)
    checkpoint['config'].update(group_advantage_coef=0.0, kl_probe_samples=8)
    torch.save(checkpoint, checkpoint_path)
    bundle['training_config'].update(group_advantage_coef=0.0, kl_probe_samples=8)
    bundle['trainer_hyperparameters'].update(group_advantage_coef=0.0, kl_probe_samples=8)
    torch.save(bundle, path)
    assert validate_bundle_source(path, checkpoint_path)['trainer_hyperparameters_match']
