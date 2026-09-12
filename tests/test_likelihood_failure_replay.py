"""A likelihood failure preserves only its batch and supports bounded safe replay."""
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from ai_trader.grpo.diagnose_likelihood import main, run_diagnostics
from ai_trader.grpo.likelihood_failure import save_likelihood_failure
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from ai_trader.grpo.policy_update_checks import (RolloutLikelihoodMismatch, check_rollout_likelihood,
                                                exact_kl_from_log_probs)
from ai_trader.grpo.runtime_precision import precision_metadata


@pytest.fixture(autouse=True)
def bounded_threads():
    torch.set_num_threads(1)


def failure_fixture(samples=12, device='cpu'):
    torch.manual_seed(23)
    policy = GRPOPolicyE2EXLSTM(obs_dim=63, cnn_channels=4, rnn_hidden_dim=4, fc_hidden_dim=8,
                                checkpoint_segments=2, execution_action_mask=True).to(device).eval()
    assert torch.count_nonzero(policy.policy_head.weight) > 0
    states = torch.randn(samples, 8, 63)
    masks = torch.tensor([[True, False, False], [True, True, False], [True, False, True]] * ((samples + 2) // 3))[:samples]
    actions = torch.where(masks[:, 2], 2, torch.where(masks[:, 1], 1, 0))
    with torch.no_grad():
        old, _, _ = policy.evaluate_actions(states.to(device), actions.to(device), action_masks=masks.to(device))
    old = old.cpu()
    old[6] += .05  # A known mismatch in the second update minibatch.
    with pytest.raises(RolloutLikelihoodMismatch) as caught:
        check_rollout_likelihood(policy, states, actions, old, masks, batch_size=4, tolerance=.001, device=device)
    return policy, caught.value, states, actions, masks, old


def save_fixture(path, *, samples=12, device='cpu'):
    policy, error, *original = failure_fixture(samples, device)
    save_likelihood_failure(path, policy=policy, failure=error,
                            observation_schema={'version': 4, 'obs_dim': 63},
                            training_config={'num_workers': 2, 'batch_size': 4},
                            context={'iteration': 20, 'rollout_batch_size': 2, 'update_batch_size': 4})
    return policy, error, original


def test_structured_failure_has_independent_batch_storage_and_bounded_serialization():
    policy, error, states, actions, masks, old = failure_fixture(samples=128)
    assert isinstance(error, ValueError)
    assert error.offset == 4 and error.update_batch_size == 4 and error.total_samples == 128
    assert error.max_abs_error == pytest.approx(.05, abs=1e-6)
    for name, source in (('states', states), ('actions', actions), ('action_masks', masks), ('old_log_probs', old)):
        saved = error.batch[name]
        assert saved.device.type == 'cpu' and not saved.requires_grad and saved.grad_fn is None
        assert saved.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        assert saved.untyped_storage().nbytes() == saved.numel() * saved.element_size()
        torch.testing.assert_close(saved, source[4:8])
    encoded = io.BytesIO()
    torch.save(error.batch, encoded)
    with zipfile.ZipFile(encoded) as archive:
        stored_bytes = sum(item.file_size for item in archive.infolist() if '/data/' in item.filename)
    expected_bytes = sum(value.numel() * value.element_size() for value in error.batch.values() if value is not None)
    assert stored_bytes == expected_bytes
    assert stored_bytes < states.untyped_storage().nbytes() / 16
    assert not policy.training


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_actual_xlstm_bundle_replays_same_saved_inputs_under_four_conditions(tmp_path, monkeypatch, device):
    path = tmp_path / 'failure.pt'
    policy, error, _ = save_fixture(path, device=device)
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    assert bundle['policy_config']['kwargs']['obs_dim'] == 63
    assert bundle['policy_config']['kwargs']['checkpoint_segments'] == 2
    assert bundle['failure']['offset'] == error.offset
    assert bundle['context']['original_rollout_batch_reconstructed'] is False
    assert bundle['runtime']['policy_device'].startswith(device)
    for name, parameter in policy.state_dict().items():
        torch.testing.assert_close(bundle['policy_state_dict'][name], parameter.cpu(), rtol=0, atol=0)
    original_load = torch.load
    loads = []

    def safe_load(*args, **kwargs):
        loads.append(kwargs.get('weights_only'))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', safe_load)
    monkeypatch.setattr(torch.optim, 'Adam', lambda *args, **kwargs: pytest.fail('Replay must not create an optimizer'))
    original_precision = precision_metadata()
    report = run_diagnostics(path, device=device)
    assert loads == [True]
    assert report['precision_restored'] and precision_metadata() == original_precision
    assert report['sample_count'] == 4 and report['rollout_batch_size'] == 2 and report['update_batch_size'] == 4
    assert report['original_rollout_batch_reconstructed'] is False
    assert 'original rollout batch' in report['batch_reconstruction_notice']
    assert len(report['variants']) == len(report['comparisons']) == 4
    assert all(item['status'] == 'ok' for item in report['variants'])
    assert all(item['vs_saved_rollout']['max_abs_error'] > .01 for item in report['variants'])
    assert all(item['vs_saved_update']['max_abs_error'] < .001 for item in report['variants'])
    if device == 'cpu':
        assert report['tf32_hardware_effect_testable'] is False
        assert 'CPU cannot reproduce CUDA TF32' in report['limitation']


def test_cli_writes_json_report_and_does_not_modify_failure_bundle(tmp_path):
    bundle, output = tmp_path / 'failure.pt', tmp_path / 'report.json'
    save_fixture(bundle)
    before = bundle.read_bytes()
    assert main(['--bundle', str(bundle), '--output', str(output), '--device', 'cpu']) == 0
    assert bundle.read_bytes() == before
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['original_rollout_batch_reconstructed'] is False
    assert report['precision_restored']
    assert all(variant['gradient_enabled'] == (variant['mode'] == 'train_grad') for variant in report['variants'])


def test_cli_rejects_output_that_resolves_to_the_failure_bundle(tmp_path):
    bundle = tmp_path / 'failure.pt'
    save_fixture(bundle)
    before = bundle.read_bytes()
    with pytest.raises(ValueError, match='must differ'):
        main(['--bundle', str(bundle), '--output', str(tmp_path / '.' / 'failure.pt'), '--device', 'cpu'])
    assert bundle.read_bytes() == before


def test_cli_restores_precision_even_when_setting_a_diagnostic_mode_fails(tmp_path, monkeypatch):
    from ai_trader.grpo import runtime_precision

    bundle = tmp_path / 'failure.pt'
    save_fixture(bundle)
    original = precision_metadata()
    setter = runtime_precision.set_tf32

    def fail_after_setting(enabled):
        setter(enabled)
        raise RuntimeError('diagnostic mode failed')

    monkeypatch.setattr(runtime_precision, 'set_tf32', fail_after_setting)
    with pytest.raises(RuntimeError, match='diagnostic mode failed'):
        run_diagnostics(bundle, device='cpu')
    assert precision_metadata() == original


def test_failed_atomic_save_preserves_destination_and_removes_only_its_temp_file(tmp_path, monkeypatch):
    policy, failure, *_ = failure_fixture()
    destination = tmp_path / 'existing.pt'
    destination.write_bytes(b'keep this artifact')

    def incomplete_save(payload, handle):
        handle.write(b'incomplete')
        raise OSError('disk write failed')

    monkeypatch.setattr(torch, 'save', incomplete_save)
    with pytest.raises(OSError, match='disk write failed'):
        save_likelihood_failure(destination, policy=policy, failure=failure)
    assert destination.read_bytes() == b'keep this artifact'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['existing.pt']


@pytest.mark.parametrize('mutation', [lambda bundle: bundle['policy_config'].update(module='untrusted.module'),
                                    lambda bundle: bundle['policy_config']['kwargs'].update(obs_dim=64),
                                    lambda bundle: bundle['batch'].update(actions=torch.zeros(4, dtype=torch.bool)),
                                    lambda bundle: bundle['batch'].update(states=torch.zeros(4, 8, 63, dtype=torch.float64))])
def test_cli_rejects_malformed_reconstruction_without_importing_arbitrary_classes(tmp_path, mutation):
    path = tmp_path / 'failure.pt'
    save_fixture(path)
    bundle = torch.load(path, weights_only=True)
    mutation(bundle)
    torch.save(bundle, path)
    with pytest.raises(ValueError):
        run_diagnostics(path, device='cpu')


@pytest.mark.parametrize('current', [torch.tensor([[float('nan'), 0.]]), torch.tensor([[float('inf'), 0.]]),
                                    torch.tensor([[.2, .3]]), torch.tensor([[0., -float('inf')]])])
def test_exact_kl_table_helper_rejects_invalid_probabilities_or_lost_support(current):
    reference = torch.log(torch.tensor([[.7, .3]]))
    with pytest.raises((ValueError, FloatingPointError)):
        exact_kl_from_log_probs(reference, current)


def test_exact_kl_table_helper_handles_zero_reference_support_and_matches_direction():
    reference = torch.tensor([[np.log(.8), np.log(.2), -np.inf]], dtype=torch.float64)
    current = torch.tensor([[np.log(.4), np.log(.6), -np.inf]], dtype=torch.float64)
    result = exact_kl_from_log_probs(reference, current)
    assert result['mean_kl'] == pytest.approx(.8 * np.log(2) + .2 * np.log(1 / 3))
    assert result['per_state_kl'].device.type == 'cpu'
