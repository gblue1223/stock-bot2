"""Check invocation-only LR overrides and capture isolation from training output."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from ai_trader.grpo import train_xlstm as training
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.runtime_precision import precision_metadata
from lib.observations import ObservationBuilder
from test_profit_experiment_config import valid_config, capture_cli_config


@pytest.mark.parametrize('value', [0, -1, True, '1e-5', float('nan'), float('inf')])
def test_resume_lr_rejects_nonpositive_or_nonfinite_values(tmp_path, value):
    with pytest.raises(ValueError, match='resume_lr'):
        valid_config(tmp_path, resume=True, load_policy='checkpoint.pt', resume_lr=value).validate()


def test_resume_lr_requires_resume_and_defaults_to_saved_lr(tmp_path):
    assert valid_config(tmp_path).resume_lr is None
    valid_config(tmp_path, resume=True, load_policy='checkpoint.pt', resume_lr=1e-5).validate()
    with pytest.raises(ValueError, match='resume_lr'):
        valid_config(tmp_path, resume=False, load_policy='checkpoint.pt', resume_lr=1e-5).validate()


@pytest.mark.parametrize('field', ['resume_lr', 'capture_update_bundle'])
def test_execution_requests_are_not_replayed_from_old_json_or_checkpoint(tmp_path, monkeypatch, field):
    saved_value = 1e-5 if field == 'resume_lr' else str(tmp_path / 'old_bundle.pt')
    config_path = tmp_path / 'saved_config.json'
    config_path.write_text(json.dumps({field: saved_value}), encoding='utf-8')
    assert getattr(training.TrainingConfig(config_path), field) is None
    checkpoint_path = tmp_path / 'checkpoint.pt'
    torch.save({'policy_state_dict': {}, 'observation_schema': ObservationBuilder(['price']).schema,
                'extra_state': {'training_config': {field: saved_value}}}, checkpoint_path)
    captured = capture_cli_config(monkeypatch, ['--config', str(config_path), '--load_policy', str(checkpoint_path), '--resume'])
    assert captured[field] is None
    explicit = capture_cli_config(monkeypatch, ['--config', str(config_path), '--load_policy', str(checkpoint_path),
                                               '--resume', '--' + field, str(saved_value)])
    assert explicit[field] == saved_value


@pytest.mark.parametrize('resume,load_policy,path', [(False, None, 'new.pt'), (True, None, 'new.pt'),
                                                  (True, 'checkpoint.pt', ''), (True, 'checkpoint.pt', 3)])
def test_capture_requires_restored_optimizer_and_a_path(tmp_path, resume, load_policy, path):
    with pytest.raises(ValueError, match='capture_update_bundle'):
        valid_config(tmp_path, resume=resume, load_policy=load_policy, capture_update_bundle=path).validate()


@pytest.mark.parametrize('requested_lr', [None, 1e-5])
def test_resume_override_keeps_adam_moments_and_no_trade_history(tmp_path, requested_lr):
    original = GRPOTrainer(torch.nn.Linear(1, 1), object(), learning_rate=3e-5,
                           evaluation_callback=lambda _: {'mean_net_return': 0.})
    original.policy(torch.ones(1, 1)).sum().backward()
    original.optimizer.step()
    original.training_control_state.validation_count = 2
    original.training_control_state.no_trade_evidence_count = 2
    original.training_control_state.no_trade_streak = 1
    original.training_control_state.previous_buy_probability = .3
    original.training_control_state.last_validation_iteration = 10
    signature = {'same': 'conditions'}
    checkpoint_path = tmp_path / 'resume.pt'
    original.save_checkpoint(str(checkpoint_path), 10, extra_state={'evaluation_signature': signature})
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    restored = GRPOTrainer(torch.nn.Linear(1, 1), object(), learning_rate=.1,
                           evaluation_callback=lambda _: {'mean_net_return': 0.})
    config = valid_config(tmp_path, resume=True, load_policy=str(checkpoint_path), resume_lr=requested_lr)
    training.restore_resume_progress(restored, config, checkpoint, {}, signature)
    assert restored.learning_rate == config.lr == (requested_lr or 3e-5)
    assert all(group['lr'] == config.lr for group in restored.optimizer.param_groups)
    assert restored.training_control_state.no_trade_streak == 1
    for key, state in checkpoint['optimizer_state_dict']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(restored.optimizer.state_dict()['state'][key][name], value)


def test_capture_rejects_existing_path_before_data_or_precision_changes(tmp_path, monkeypatch):
    path = tmp_path / 'existing.pt'
    path.write_bytes(b'preserve')
    config = valid_config(tmp_path, resume=True, load_policy='missing.pt')
    config_path = tmp_path / 'config.json'
    config.save_to_file(str(config_path))
    monkeypatch.setattr(sys, 'argv', ['train_xlstm.py', '--config', str(config_path),
                                     '--capture_update_bundle', str(path)])
    monkeypatch.setattr(training, 'prepare_date_splits', lambda _: pytest.fail('Data must not be opened'))
    precision = precision_metadata()
    assert not training.main()
    assert path.read_bytes() == b'preserve' and precision_metadata() == precision


def test_capture_main_uses_next_group_without_validation_training_or_model_writes(tmp_path, monkeypatch):
    from ai_trader.grpo import update_diagnostic

    model_dir = tmp_path / 'existing_model'
    model_dir.mkdir()
    sentinel = model_dir / 'date_splits.json'
    sentinel.write_text('unchanged', encoding='utf-8')
    schema = ObservationBuilder(['price'], seq_len=4).schema
    dates = {'train': ['20260101'], 'validation': ['20260102'], 'test': ['20260103']}
    config = valid_config(tmp_path, output_dir=str(model_dir), num_workers=1, seq_len=4,
                          features=1, episode_steps=2, account_observations=False,
                          execution_observations=False, execution_action_mask=False,
                          device='cpu', resume=True, total_timesteps=1)
    source = GRPOTrainer(torch.nn.Linear(1, 1), object(), learning_rate=3e-5,
                         observation_schema=schema, evaluation_callback=lambda _: {})
    source.policy(torch.ones(1, 1)).sum().backward()
    source.optimizer.step()
    source.total_timesteps = 123
    source.num_updates = 19
    checkpoint_path = tmp_path / 'checkpoint_iter19.pt'
    source.save_checkpoint(str(checkpoint_path), 19, extra_state={'date_splits': dates,
                           'training_config': dict(config.__dict__)})
    config.load_policy = str(checkpoint_path)
    config_path = tmp_path / 'capture_config.json'
    config.save_to_file(str(config_path))
    capture_path = tmp_path / 'captured_update.pt'
    calls = []

    class Environment:
        observation_schema = schema
        observation_space = SimpleNamespace(shape=(4, 16))

        def set_transaction_cost_rate(self, value):
            pass

        def close(self):
            calls.append('closed')

    def environment(_config, _device, allowed_dates, seed=None):
        assert allowed_dates == dates['train']
        return Environment()

    def collect(trainer, count):
        assert count == 16 and trainer.writer is None
        assert trainer.num_updates == 19 and trainer.total_timesteps == 123
        assert precision_metadata()['matmul_fp32_precision'] == 'ieee'
        calls.append('collect')
        return [{'number': number} for number in range(count)]

    monkeypatch.setattr(training, 'prepare_date_splits', lambda _: dates)
    monkeypatch.setattr(training, 'create_environment', environment)
    monkeypatch.setattr(training, 'create_policy', lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(GRPOTrainer, 'collect_rollouts', collect)
    monkeypatch.setattr(GRPOTrainer, 'group_episodes', lambda self, episodes: {1: episodes[8:], 0: episodes[:8]})
    monkeypatch.setattr(GRPOTrainer, 'compute_group_relative_advantages', lambda self, grouped: {
        key: [episode['number'] for episode in values] for key, values in grouped.items()})
    monkeypatch.setattr(GRPOTrainer, 'train', lambda *args, **kwargs: pytest.fail('Capture must not train'))
    monkeypatch.setattr(training, 'evaluate_policy', lambda *args, **kwargs: pytest.fail('Capture must not evaluate'))

    def save(path, *, trainer, episodes, advantages, training_config):
        assert [episode['number'] for episode in episodes] == advantages == list(range(16))
        assert trainer.num_updates == 19 and training_config['lr'] == 3e-5
        assert trainer.optimizer.state_dict()['state']
        Path(path).write_bytes(b'captured')
        return str(path)

    monkeypatch.setattr(update_diagnostic, 'save_update_bundle', save)
    monkeypatch.setattr(sys, 'argv', ['train_xlstm.py', '--config', str(config_path),
                                     '--capture_update_bundle', str(capture_path)])
    precision = precision_metadata()
    assert training.main()
    assert capture_path.read_bytes() == b'captured'
    assert sentinel.read_text(encoding='utf-8') == 'unchanged'
    assert list(model_dir.iterdir()) == [sentinel]
    assert calls == ['collect', 'closed']
    assert precision_metadata() == precision
