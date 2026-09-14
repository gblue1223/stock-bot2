import copy
import json
import random

import numpy as np
import pytest
import torch

from ai_trader.grpo.gpu_runtime import PhaseMeasurement
from ai_trader.grpo.gpu_tuning import sample_rollouts, select_trial, tune_update, validate_gpu_tuning
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.policies.scalping_policy_xlstm import mLSTMLayer
from test_policy_update_checks import CheckedPolicy, episode, assert_state_equal


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_checkpoint_off_preserves_outputs_and_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(12)
    torch.set_num_threads(1)
    checked = mLSTMLayer(3, 4, num_layers=2, checkpoint_segments=3).to(device)
    direct = copy.deepcopy(checked)
    direct.checkpoint_segments = 0
    x = torch.randn(2, 9, 3, device=device)
    a, _ = checked(x)
    b, _ = direct(x)
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    for left, right in zip(checked.parameters(), direct.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_update_trials_preserve_weights_adam_rng_masks_and_counters(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    policy = CheckedPolicy().to(device)
    trainer = GRPOTrainer(policy, object(), device=device, learning_rate=1e-5, batch_size=4,
                          num_epochs=2, policy_update_checks=True)
    source = episode(policy, n=16)
    trainer.update_policy([source], [np.zeros(16)])  # Populate Adam before testing resume preservation.
    source = episode(policy, n=16)
    source['metadata'] = {'entry_pattern': {'policy_credit': np.where(source['actions'] == 1, .2, 0).tolist()}}
    before = copy.deepcopy((policy.state_dict(), trainer.optimizer.state_dict(), source))
    np_rng, py_rng, cpu_rng = np.random.get_state(), random.getstate(), torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state() if device == 'cuda' else None
    counters, training_mode = (trainer.num_updates, trainer.total_timesteps), policy.training
    report = tune_update(trainer, [source], [np.zeros(16)],
                         {'batch_sizes': [4, 8], 'samples': 16, 'warmup_samples': 4, 'checkpoint_segments': [0]})
    assert all(trial['status'] == 'ok' for trial in report['trials'])
    assert [trial['optimizer_steps'] for trial in report['trials']] == [8, 4]
    assert_state_equal(before[0], policy.state_dict())
    assert_state_equal(before[1], trainer.optimizer.state_dict())
    for key in ('states', 'actions', 'log_probs', 'dones', 'action_masks'):
        np.testing.assert_array_equal(before[2][key], source[key])
    assert (trainer.num_updates, trainer.total_timesteps) == counters
    assert policy.training == training_mode and trainer.batch_size == 4
    current_np = np.random.get_state()
    assert current_np[0] == np_rng[0] and current_np[2:] == np_rng[2:]
    np.testing.assert_array_equal(current_np[1], np_rng[1])
    assert random.getstate() == py_rng
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if device == 'cuda':
        assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
        assert all(trial['peak_allocated_bytes'] > 0 for trial in report['trials'])
    json.dumps(report, allow_nan=False)


def test_prefix_does_not_mutate_episode_and_keeps_buy_credit_alignment():
    policy = CheckedPolicy()
    source = episode(policy, n=6)
    source['metadata'] = {'entry_pattern': {'policy_credit': [0, 1, 0, 0, -1, 0]}}
    sampled, _ = sample_rollouts([source], [np.arange(6)], 2)
    assert sampled[0]['metadata']['entry_pattern']['policy_credit'] == [0, 1]
    assert sampled[0]['dones'][-1] and not source['dones'][1]
    sampled[0]['states'][:] = 999
    assert (source['states'] != 999).all()


def test_selection_rejects_fast_failed_candidates():
    trials = [{'status': status, 'samples_per_second': speed} for status, speed in
              [('oom', 100), ('numerical_check', 100), ('incomplete_update', 100),
               ('memory_budget', 100), ('ok', 2), ('ok', 3)]]
    assert select_trial(trials) is trials[-1]
    with pytest.raises(RuntimeError, match='No GPU tuning candidate'):
        select_trial(trials[:4])


@pytest.mark.parametrize('error_type', [torch.cuda.OutOfMemoryError, FloatingPointError])
def test_failed_candidates_restore_rng_and_leave_original_weights(monkeypatch, error_type):
    policy = CheckedPolicy()
    trainer = GRPOTrainer(policy, object(), batch_size=2, num_epochs=1)
    source = episode(policy, n=4)
    before = copy.deepcopy(policy.state_dict())
    rng = torch.get_rng_state()
    numpy_rng = np.random.get_state()

    def failure(*args):
        torch.rand(2)
        np.random.rand(2)
        raise error_type('test candidate failure')

    monkeypatch.setattr(GRPOTrainer, 'update_policy', failure)
    with pytest.raises(RuntimeError, match='No GPU tuning candidate'):
        tune_update(trainer, [source], [np.zeros(4)],
                    {'batch_sizes': [2], 'samples': 4, 'warmup_samples': 2, 'checkpoint_segments': [0]})
    assert torch.equal(rng, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert_state_equal(before, policy.state_dict())
    assert trainer.num_updates == trainer.total_timesteps == 0


@pytest.mark.parametrize('settings', [{'samples': 0}, {'samples': True}, {'batch_sizes': []},
    {'batch_sizes': [0]}, {'checkpoint_segments': [-1]}, {'max_memory_fraction': float('nan')},
    {'max_memory_fraction': 1}, {'unknown': 1}])
def test_invalid_tuning_settings(settings):
    with pytest.raises(ValueError):
        validate_gpu_tuning(settings)


def test_phase_telemetry_written_for_cpu_without_fake_gpu_memory(tmp_path):
    trainer = GRPOTrainer(CheckedPolicy(), object())
    with PhaseMeasurement('cpu') as measurement:
        torch.ones(10).sum()
    phases = {'update': measurement.throughput(10)}
    trainer._record_runtime_phases(phases, 2, str(tmp_path / 'checkpoint{}.pt'))
    report = json.loads((tmp_path / 'diagnostics/runtime/iteration_000002.json').read_text())
    assert report['phases']['update']['samples'] == 10
    assert report['phases']['update']['samples_per_second'] > 0
    assert 'peak_allocated_bytes' not in report['phases']['update']


def test_tuning_application_persists_actual_settings_without_progress_change(monkeypatch, tmp_path):
    trainer = GRPOTrainer(CheckedPolicy(), object(), gpu_tuning={})
    trainer.device = 'cuda'  # Selection application is mocked; no CUDA operations here.
    trainer.extra_checkpoint_state = {'training_config': {'batch_size': 4, 'checkpoint_segments': 16}}
    report = {'selected': {'batch_size': 8, 'checkpoint_segments': 0}}
    monkeypatch.setattr('ai_trader.grpo.grpo.tune_update', lambda *a: copy.deepcopy(report))
    trainer._tune_gpu_runtime([], [], str(tmp_path / 'checkpoint{}.pt'))
    assert trainer.batch_size == 8 and trainer.num_updates == trainer.total_timesteps == 0
    assert trainer.extra_checkpoint_state['training_config'] == report['selected']
    assert json.loads((tmp_path / 'gpu_tuning.json').read_text()) == report
    monkeypatch.setattr('ai_trader.grpo.grpo.tune_update', lambda *a: pytest.fail('Tuned twice'))
    trainer._tune_gpu_runtime([], [], str(tmp_path / 'checkpoint{}.pt'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_xlstm_cuda_tuning_checks_real_model_and_checkpoint_off(tmp_path):
    from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
    torch.set_num_threads(1)
    policy = GRPOPolicyE2EXLSTM(obs_dim=17, cnn_channels=4, rnn_hidden_dim=4,
                               fc_hidden_dim=8, checkpoint_segments=2, execution_action_mask=True).cuda()
    states = torch.zeros(8, 16, 17, device='cuda')
    states[:, :, :2] = torch.randn(8, 16, 2, device='cuda')
    actions = torch.arange(8, device='cuda') % 3
    masks = torch.ones(8, 3, device='cuda', dtype=torch.bool)
    policy.eval()
    with torch.no_grad():
        logs, _, values, _ = policy.evaluate_actions_with_distribution(states, actions, masks)
    source = {'states': states.cpu().numpy(), 'actions': actions.cpu().numpy(),
              'action_masks': masks.cpu().numpy(), 'log_probs': logs.cpu().numpy(), 'values': values.cpu().numpy(),
              'rewards': np.linspace(-.01, .02, 8, dtype=np.float32), 'dones': np.arange(8) == 7}
    trainer = GRPOTrainer(policy, object(), device='cuda', learning_rate=1e-6, batch_size=2,
                          num_epochs=1, policy_update_checks=True,
                          gpu_tuning={'batch_sizes': [2, 4], 'samples': 8, 'warmup_samples': 2,
                                      'checkpoint_segments': [2, 0]})
    trainer.extra_checkpoint_state['training_config'] = {'batch_size': 2, 'checkpoint_segments': 2}
    weights = copy.deepcopy(policy.state_dict())
    trainer._tune_gpu_runtime([source], [np.zeros(8)], str(tmp_path / 'checkpoint{}.pt'))
    report = json.loads((tmp_path / 'gpu_tuning.json').read_text())
    assert len(report['trials']) == 3
    assert all(trial['status'] == 'ok' for trial in report['trials'])
    assert any(trial['checkpoint_segments'] == 0 for trial in report['trials'])
    assert report['selected'] == trainer.extra_checkpoint_state['training_config']
    assert trainer.policy.xlstm.checkpoint_segments == report['selected']['checkpoint_segments']
    assert trainer.num_updates == trainer.total_timesteps == 0
    assert_state_equal(weights, policy.state_dict())
