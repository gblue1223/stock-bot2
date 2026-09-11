"""Experiment settings must survive CLI, checkpoint and environment boundaries."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ai_trader.grpo.train_xlstm as training
from ai_trader.grpo.evaluation import compatible_resume_best, evaluation_signature
from lib.observations import ObservationBuilder


def valid_config(tmp_path, **overrides):
    config = training.TrainingConfig()
    config.extracted_dir = str(tmp_path)
    config.__dict__.update(overrides)
    return config


def test_training_defaults_enable_cost_observations_without_combining_other_experiments(tmp_path):
    config = valid_config(tmp_path)
    config.validate()
    assert config.account_observations and config.execution_observations
    assert config.decision_interval_seconds == config.episode_duration_seconds == 0.
    assert config.group_advantage_coef == 1. and config.lambda_gae == .95
    assert config.training_seed == 42
    builder = ObservationBuilder([f'feature_{i}' for i in range(config.features)],
                                 account_observations=config.account_observations,
                                 execution_observations=config.execution_observations)
    assert builder.obs_dim == 63


@pytest.mark.parametrize('field', ['decision_interval_seconds', 'episode_duration_seconds', 'group_advantage_coef'])
@pytest.mark.parametrize('value', [True, False, float('nan'), float('inf'), -1., '1', None])
def test_timing_and_advantage_settings_reject_boolean_nonfinite_and_invalid_values(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        valid_config(tmp_path, **{field: value}).validate()


@pytest.mark.parametrize('value', [True, False, -1, 2**32, 1.5, float('nan'), float('inf'), '42', None])
def test_training_seed_requires_supported_integer_range(tmp_path, value):
    with pytest.raises(ValueError, match='training_seed'):
        valid_config(tmp_path, training_seed=value).validate()


@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_execution_observations_requires_a_real_boolean(tmp_path, value):
    with pytest.raises(ValueError, match='execution_observations'):
        valid_config(tmp_path, execution_observations=value).validate()


def test_v4_requires_account_features_and_gae_only_requires_gae(tmp_path):
    with pytest.raises(ValueError, match='requires account_observations'):
        valid_config(tmp_path, account_observations=False).validate()
    with pytest.raises(ValueError, match='group_advantage_coef=0 requires use_gae'):
        valid_config(tmp_path, group_advantage_coef=0., use_gae=False).validate()
    valid_config(tmp_path, group_advantage_coef=0., use_gae=True).validate()
    valid_config(tmp_path, account_observations=False, execution_observations=False).validate()


def test_experiment_settings_roundtrip_through_colab_json(tmp_path):
    config = valid_config(tmp_path, decision_interval_seconds=.75, episode_duration_seconds=90.,
                          group_advantage_coef=0., lambda_gae=.99, training_seed=123)
    path = tmp_path / 'colab_config.json'
    config.save_to_file(path)
    restored = training.TrainingConfig(path)
    restored.validate()
    assert restored.__dict__ == config.__dict__


def capture_cli_config(monkeypatch, arguments):
    """Run real argument parsing/checkpoint loading, stopping before data or GPU work."""
    class ConfigurationCaptured(BaseException):
        pass

    captured = {}

    def capture(config):
        captured.update(config.__dict__)
        raise ConfigurationCaptured

    monkeypatch.setattr(sys, 'argv', ['train_xlstm.py', *arguments])
    monkeypatch.setattr(training.TrainingConfig, 'validate', capture)
    with pytest.raises(ConfigurationCaptured):
        training.main()
    return captured


def test_cli_forwards_every_new_experiment_option_and_overrides_json(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'execution_observations': False, 'group_advantage_coef': 1.,
                                'training_seed': 42}), encoding='utf-8')
    result = capture_cli_config(monkeypatch, [
        '--config', str(path), '--execution_observations', '--account_observations',
        '--decision_interval_seconds', '.5', '--episode_duration_seconds', '240',
        '--group_advantage_coef', '0', '--lambda_gae', '.99', '--training_seed', '123',
    ])
    assert result['execution_observations'] and result['account_observations']
    assert result['decision_interval_seconds'] == .5 and result['episode_duration_seconds'] == 240.
    assert result['group_advantage_coef'] == 0. and result['lambda_gae'] == .99
    assert result['training_seed'] == 123


def test_cli_can_explicitly_select_legacy_observations(monkeypatch):
    result = capture_cli_config(monkeypatch, ['--no-execution_observations', '--no-account_observations'])
    assert result['execution_observations'] is result['account_observations'] is False


@pytest.mark.parametrize('version', [2, 3, 4])
def test_cli_checkpoint_restores_schema_and_recorded_experiment_conditions(tmp_path, monkeypatch, version):
    schema = ObservationBuilder(['price'], account_observations=version >= 3,
                                execution_observations=version == 4).schema
    saved = ({'decision_interval_seconds': 1.25, 'episode_duration_seconds': 120.,
              'group_advantage_coef': .25, 'training_seed': 77, 'lambda_gae': .99,
              'liquidation_max_steps': 37} if version == 4 else {})
    path = tmp_path / f'v{version}.pt'
    torch.save({'policy_state_dict': {}, 'observation_schema': schema,
                'extra_state': {'training_config': saved}}, path)
    result = capture_cli_config(monkeypatch, ['--load_policy', str(path), '--resume'])
    assert result['account_observations'] is (version >= 3)
    assert result['execution_observations'] is (version == 4)
    assert result['decision_interval_seconds'] == saved.get('decision_interval_seconds', 0.)
    assert result['episode_duration_seconds'] == saved.get('episode_duration_seconds', 0.)
    assert result['group_advantage_coef'] == saved.get('group_advantage_coef', 1.)
    assert result['training_seed'] == saved.get('training_seed', 42)
    assert result['lambda_gae'] == saved.get('lambda_gae', .95)
    assert result['liquidation_max_steps'] == saved.get('liquidation_max_steps', 0)


def test_explicit_cli_experiment_overrides_are_not_lost_during_checkpoint_loading(tmp_path, monkeypatch):
    path = tmp_path / 'v4.pt'
    schema = ObservationBuilder(['price'], account_observations=True, execution_observations=True).schema
    torch.save({'policy_state_dict': {}, 'observation_schema': schema,
                'extra_state': {'training_config': {'decision_interval_seconds': 1.,
                    'episode_duration_seconds': 300., 'group_advantage_coef': 1.,
                    'training_seed': 77, 'lambda_gae': .95}}}, path)
    result = capture_cli_config(monkeypatch, [
        '--load_policy', str(path), '--decision_interval_seconds', '2',
        '--episode_duration_seconds', '180', '--group_advantage_coef', '0',
        '--training_seed', '99', '--lambda_gae', '1',
    ])
    assert result['decision_interval_seconds'] == 2. and result['episode_duration_seconds'] == 180.
    assert result['group_advantage_coef'] == 0. and result['training_seed'] == 99
    assert result['lambda_gae'] == 1.


def test_environment_factory_preserves_execution_timing_costs_and_seed(tmp_path, monkeypatch):
    calls = []

    def environment(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(training, 'GRPOScalpingEnvXLSTM', environment)
    config = valid_config(tmp_path, decision_interval_seconds=.75, episode_duration_seconds=120.,
                          liquidation_max_steps=37)
    first = training.create_environment(config, 'cpu', ['20260101'], seed=123)
    second = training.create_environment(config, 'cpu', ['20260101'], seed=123)
    settings = calls[0]
    assert settings['execution_observations'] and settings['account_observations']
    assert settings['decision_interval_seconds'] == .75 and settings['episode_duration_seconds'] == 120.
    assert settings['liquidation_max_steps'] == 37
    assert settings['max_episode_steps'] == config.episode_steps
    assert settings['execution_config'] == config.execution_config
    assert settings['transaction_cost_rate'] == config.transaction_cost_rate
    assert settings['sell_tax_rate'] == config.sell_tax_rate
    assert settings['allowed_dates'] == ['20260101']
    np.testing.assert_array_equal(first.np_random.integers(10000, size=8),
                                  second.np_random.integers(10000, size=8))


def legacy_best_checkpoint():
    config = {'output_dir': '/run', 'extracted_dir': '/data'}
    splits = {'train': ['20250101'], 'validation': ['20260101'], 'test': ['20270101']}
    schema = {'version': 3}
    signature = evaluation_signature(config, splits, schema)
    old_signature = copy.deepcopy(signature)
    for field in ('liquidation_max_steps', 'decision_interval_seconds', 'episode_duration_seconds'):
        old_signature['settings'].pop(field)
    checkpoint = {'policy_state_dict': {'weight': torch.ones(1)}, 'observation_schema': schema,
                  'extra_state': {'date_splits': splits, 'training_config': config,
                                  'evaluation_signature': old_signature}}
    return checkpoint, signature


def test_old_best_signature_remains_compatible_when_time_options_are_disabled():
    checkpoint, signature = legacy_best_checkpoint()
    before = copy.deepcopy(checkpoint['extra_state'])
    assert signature['settings']['decision_interval_seconds'] == signature['settings']['episode_duration_seconds'] == 0.
    assert compatible_resume_best(checkpoint, checkpoint, signature)
    assert checkpoint['extra_state'] == before


@pytest.mark.parametrize('changed', [{'decision_interval_seconds': 1.}, {'episode_duration_seconds': 300.}])
def test_changed_policy_time_conditions_make_prior_best_ineligible(changed):
    checkpoint, signature = legacy_best_checkpoint()
    current = evaluation_signature({**checkpoint['extra_state']['training_config'], **changed},
                                   signature['date_splits'], signature['observation_schema'])
    assert not compatible_resume_best(checkpoint, checkpoint, current)


@pytest.mark.parametrize('field', ['decision_interval_seconds', 'episode_duration_seconds'])
def test_missing_signature_field_cannot_hide_recorded_nonzero_time_setting(field):
    checkpoint, signature = legacy_best_checkpoint()
    checkpoint['extra_state']['training_config'][field] = 1.
    assert not compatible_resume_best(checkpoint, checkpoint, signature)
