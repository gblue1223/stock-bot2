"""Policy safeguards and no-trade budgets cross CLI/resume boundaries explicitly."""
import json

import pytest
import torch

import ai_trader.grpo.train_xlstm as training
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.training_control import TrainingControlState
from lib.observations import ObservationBuilder
from test_profit_experiment_config import valid_config, capture_cli_config


def test_fresh_config_enables_checks_and_leaves_no_trade_limit_opt_in(tmp_path):
    config = valid_config(tmp_path)
    config.validate()
    assert config.policy_update_checks is True
    assert config.rollout_logprob_tolerance == .001 and config.kl_probe_samples == 32
    assert config.no_trade_max_validations == 0


@pytest.mark.parametrize('field,values', [
    ('policy_update_checks', [0, 1, 'true', None]),
    ('no_trade_max_validations', [True, -1, 1.5, '4', None]),
    ('kl_probe_samples', [False, 0, -1, 1.5, '32', None]),
    ('rollout_logprob_tolerance', [True, -1., float('nan'), float('inf'), '0.1', None]),
])
def test_new_safeguard_config_validation_rejects_invalid_types_and_bounds(tmp_path, field, values):
    for value in values:
        with pytest.raises(ValueError, match=field):
            valid_config(tmp_path, **{field: value}).validate()


def test_cli_overrides_safeguard_json_and_can_disable_checks(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'policy_update_checks': True, 'rollout_logprob_tolerance': .001,
                               'kl_probe_samples': 32, 'no_trade_max_validations': 4}), encoding='utf-8')
    config = capture_cli_config(monkeypatch, ['--config', str(path), '--no-policy_update_checks',
                                            '--rollout_logprob_tolerance', '.002', '--kl_probe_samples', '64',
                                            '--no_trade_max_validations', '0'])
    assert config['policy_update_checks'] is False
    assert config['rollout_logprob_tolerance'] == .002 and config['kl_probe_samples'] == 64
    assert config['no_trade_max_validations'] == 0


def test_old_checkpoint_defaults_disable_checks_and_explicit_cli_enables_them(tmp_path, monkeypatch):
    path = tmp_path / 'old.pt'
    torch.save({'policy_state_dict': {}, 'observation_schema': ObservationBuilder(['price']).schema}, path)
    restored = capture_cli_config(monkeypatch, ['--load_policy', str(path), '--resume'])
    assert restored['policy_update_checks'] is False
    assert restored['no_trade_max_validations'] == 0
    assert restored['rollout_logprob_tolerance'] == .001 and restored['kl_probe_samples'] == 32
    enabled = capture_cli_config(monkeypatch, ['--load_policy', str(path), '--resume',
                                              '--policy_update_checks', '--no_trade_max_validations', '4',
                                              '--rollout_logprob_tolerance', '.002', '--kl_probe_samples', '16'])
    assert enabled['policy_update_checks'] is True and enabled['no_trade_max_validations'] == 4
    assert enabled['rollout_logprob_tolerance'] == .002 and enabled['kl_probe_samples'] == 16


def test_saved_checkpoint_safeguards_roundtrip_through_cli_loading(tmp_path, monkeypatch):
    saved = {'policy_update_checks': True, 'rollout_logprob_tolerance': .002,
             'kl_probe_samples': 12, 'no_trade_max_validations': 5}
    path = tmp_path / 'checks.pt'
    torch.save({'policy_state_dict': {}, 'observation_schema': ObservationBuilder(['price']).schema,
                'config': saved}, path)
    restored = capture_cli_config(monkeypatch, ['--load_policy', str(path), '--resume'])
    assert {key: restored[key] for key in saved} == saved


@pytest.mark.parametrize('changed_limit,changed_signature', [(False, False), (True, False), (False, True)])
def test_resume_updates_effective_config_and_resets_only_control_condition_changes(tmp_path, changed_limit, changed_signature):
    original = GRPOTrainer(torch.nn.Linear(1, 1), object(), no_trade_max_validations=4,
                           policy_update_checks=False, kl_probe_samples=32,
                           evaluation_callback=lambda _: {})
    original.training_control_state = TrainingControlState(
        validation_count=2, consecutive_no_trade_validations=2, last_validation_iteration=10)
    signature = {'settings': {'execution_action_mask': False}}
    path = tmp_path / 'resume.pt'
    original.save_checkpoint(str(path), 10, extra_state={'evaluation_signature': signature})
    checkpoint = torch.load(path, weights_only=True)
    restored = GRPOTrainer(torch.nn.Linear(1, 1), object(), evaluation_callback=lambda _: {})
    config = valid_config(tmp_path)
    overrides = {'policy_update_checks': True, 'rollout_logprob_tolerance': .002, 'kl_probe_samples': 16}
    if changed_limit:
        overrides['no_trade_max_validations'] = 5
    current_signature = {'settings': {'execution_action_mask': changed_signature}}
    assert training.restore_resume_progress(restored, config, checkpoint, overrides, current_signature) == 10
    assert config.policy_update_checks is restored.policy_update_checks is True
    assert config.rollout_logprob_tolerance == restored.rollout_logprob_tolerance == .002
    assert config.kl_probe_samples == restored.kl_probe_samples == 16
    assert config.no_trade_max_validations == restored.no_trade_max_validations == (5 if changed_limit else 4)
    expected = 0 if changed_limit or changed_signature else 2
    assert restored.training_control_state.consecutive_no_trade_validations == expected
