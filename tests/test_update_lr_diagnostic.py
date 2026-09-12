"""One frozen rollout, independent Adam states and identical shuffles across LR variants."""
import hashlib
import io
import json
import random
import zipfile

import numpy as np
import pytest
import torch

from ai_trader.grpo.diagnose_update_lr import main, run_diagnostics, write_report_exclusive
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from ai_trader.grpo.policy_update_checks import RolloutLikelihoodMismatch, clone_state_to_cpu
from ai_trader.grpo.runtime_precision import preserve_precision, precision_metadata, set_tf32
from ai_trader.grpo.update_diagnostic import (atomic_save_exclusive, bounded_rollout_batches,
                                             prepare_rollouts, save_update_bundle, snapshot_rng)


@pytest.fixture(autouse=True)
def fp32_runtime():
    torch.set_num_threads(1)
    with preserve_precision():
        set_tf32(False)
        yield


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def fixed_rollout(device='cpu'):
    random.seed(24)
    np.random.seed(25)
    torch.manual_seed(26)
    policy = GRPOPolicyE2EXLSTM(obs_dim=63, cnn_channels=4, rnn_hidden_dim=4, fc_hidden_dim=8,
                                checkpoint_segments=2, execution_action_mask=True).to(device)
    trainer = GRPOTrainer(policy, env=None, device=device, batch_size=4, num_epochs=2,
                          learning_rate=3e-5, kl_target=.1, policy_update_checks=True, kl_probe_samples=4,
                          observation_schema={'version': 4, 'obs_dim': 63})
    # Populate every Adam moment and step counter with a real gradient update.
    states = torch.randn(8, 8, 63)
    masks = torch.tensor([[True, False, False], [True, True, False], [True, False, True],
                          [True, True, True]] * 2)
    actions = torch.tensor([0, 1, 2, 1, 0, 1, 2, 2])
    chosen, entropy, values = policy.evaluate_actions(states.to(device), actions.to(device), masks.to(device))
    (-chosen.mean() + values.square().mean() - .01 * entropy.mean()).backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    assert trainer.optimizer.state and any(state['exp_avg'].count_nonzero() for state in trainer.optimizer.state.values())
    assert torch.count_nonzero(policy.policy_head.weight) > 0
    policy.eval()
    with torch.no_grad():
        chosen, _, values = policy.evaluate_actions(states.to(device), actions.to(device), masks.to(device))
    rewards = np.array([.1, -.03, .05, -.02, .04, -.1, .02, .03], dtype=np.float64)
    episodes, advantages = [], []
    # The first episode is deliberately not a multiple of the update batch size.
    for start, end in ((0, 3), (3, 8)):
        dones = np.zeros(end - start, dtype=bool)
        dones[-1] = True
        episodes.append({'states': states.numpy()[start:end], 'actions': actions.numpy()[start:end],
                         'rewards': rewards[start:end], 'dones': dones,
                         'log_probs': chosen.cpu().numpy()[start:end],
                         'values': values.cpu().numpy().reshape(-1)[start:end],
                         'action_masks': masks.numpy()[start:end], 'unneeded_metadata': object()})
        advantages.append(np.full(end - start, .25 if start == 0 else -.15, dtype=np.float64))
    trainer.num_updates, trainer.total_timesteps = 19, 5600
    return trainer, episodes, advantages


def capture(path, device='cpu'):
    trainer, episodes, advantages = fixed_rollout(device)
    save_update_bundle(path, trainer=trainer, episodes=episodes, advantages=advantages,
                       training_config={'lr': 3e-5, 'seed': 25})
    return trainer, episodes, advantages


def test_capture_preserves_exact_inputs_optimizer_rng_and_bounded_storage(tmp_path):
    trainer, episodes, advantages = fixed_rollout()
    before_rng = snapshot_rng()
    before_optimizer = clone_state_to_cpu(trainer.optimizer.state_dict())
    before_policy = clone_state_to_cpu(trainer.policy.state_dict())
    path = tmp_path / 'fixed.pt'
    saved = save_update_bundle(path, trainer=trainer, episodes=episodes, advantages=advantages, training_config={})
    assert saved == str(path.resolve())
    assert not trainer.policy.training
    assert_tree_equal(before_rng, snapshot_rng())
    assert_tree_equal(before_optimizer, trainer.optimizer.state_dict())
    assert_tree_equal(before_policy, trainer.policy.state_dict())
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    assert bundle['format'] == 'grpo_fixed_rollout_update'
    assert bundle['context']['sample_count'] == 8
    assert bundle['reference_log_probs'].shape == (8, 3)
    assert bundle['likelihood_check']['max_abs_error'] < 1e-5
    assert bundle['progress'] == {'num_updates': 19, 'total_timesteps': 5600}
    assert_tree_equal(before_optimizer, bundle['optimizer_state_dict'])
    for episode, original, advantage, original_advantage in zip(bundle['episodes'], episodes, bundle['advantages'], advantages):
        assert 'unneeded_metadata' not in episode
        for key, value in episode.items():
            assert value.device.type == 'cpu'
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
            np.testing.assert_array_equal(value.numpy(), original[key])
        assert episode['rewards'].dtype == torch.float64
        assert advantage.dtype == torch.float64
        np.testing.assert_array_equal(advantage, original_advantage)
    payload = io.BytesIO()
    torch.save(bundle['episodes'], payload)
    with zipfile.ZipFile(payload) as archive:
        actual = sum(item.file_size for item in archive.infolist() if '/data/' in item.filename)
    assert actual == bundle['tensor_bytes']['episodes']
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_update_bundle(path, trainer=trainer, episodes=episodes, advantages=advantages, training_config={})
    assert path.read_bytes() == original_bytes


def test_precheck_batches_cross_episode_boundaries_and_checks_full_rollout(tmp_path, monkeypatch):
    trainer, episodes, advantages = fixed_rollout()
    saved, _ = prepare_rollouts(episodes, advantages, action_dim=3, masked=True)
    batches = list(bounded_rollout_batches(saved, 4))
    assert [offset for offset, _ in batches] == [0, 4]
    assert [len(batch['states']) for _, batch in batches] == [4, 4]
    torch.testing.assert_close(batches[0][1]['states'][3], saved[1]['states'][0])
    episodes[1]['log_probs'][-1] += .02
    original_rng = snapshot_rng()
    with pytest.raises(RolloutLikelihoodMismatch) as caught:
        save_update_bundle(tmp_path / 'bad.pt', trainer=trainer, episodes=episodes,
                           advantages=advantages, training_config={})
    assert caught.value.offset == 4 and caught.value.total_samples == 8
    assert not (tmp_path / 'bad.pt').exists()
    assert_tree_equal(original_rng, snapshot_rng())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_real_xlstm_variants_have_independent_adam_rng_and_same_shuffle(tmp_path, monkeypatch, device):
    path = tmp_path / 'fixed.pt'
    capture(path, device)
    original_bytes = hashlib.sha256(path.read_bytes()).hexdigest()
    original_update = GRPOTrainer.update_policy
    original_permutation = np.random.permutation
    inputs, orders, outputs = [], [], []

    def recording_permutation(size):
        result = original_permutation(size)
        orders[-1].append(result.copy())
        return result

    def recording_update(trainer, episodes, advantages):
        inputs.append((clone_state_to_cpu(trainer.policy.state_dict()),
                       clone_state_to_cpu(trainer.optimizer.state_dict()), snapshot_rng()))
        orders.append([])
        result = original_update(trainer, episodes, advantages)
        outputs.append((clone_state_to_cpu(trainer.policy.state_dict()),
                        clone_state_to_cpu(trainer.optimizer.state_dict())))
        return result

    monkeypatch.setattr(np.random, 'permutation', recording_permutation)
    monkeypatch.setattr(GRPOTrainer, 'update_policy', recording_update)
    set_tf32(True)
    initial_precision, initial_rng = precision_metadata(), snapshot_rng()
    report = run_diagnostics(path, learning_rates=[3e-5, 3e-6, 3e-5], device=device)
    assert report['all_variants_succeeded']
    assert report['precision_restored'] and report['rng_restored']
    assert precision_metadata() == initial_precision
    assert_tree_equal(initial_rng, snapshot_rng())
    assert report['model_exported'] is False and report['holdout_evaluated'] is False
    assert len(inputs) == 3
    for index in (1, 2):
        assert_tree_equal(inputs[0][0], inputs[index][0])
        assert_tree_equal(inputs[0][1]['state'], inputs[index][1]['state'])
        assert_tree_equal(inputs[0][2], inputs[index][2])
        for first, current in zip(orders[0], orders[index]):
            np.testing.assert_array_equal(first, current)
    # Duplicate LR after a different variant must reproduce weights and Adam exactly.
    assert_tree_equal(outputs[0], outputs[2])
    assert report['variants'][0]['full_rollout_kl'] == report['variants'][2]['full_rollout_kl']
    assert report['variants'][0]['full_rollout_kl']['mean_kl'] > report['variants'][1]['full_rollout_kl']['mean_kl']
    for variant in report['variants']:
        assert variant['metrics']['optimizer_accepted_steps'] > 0
        assert variant['full_rollout_kl']['num_samples'] == 8
        assert variant['full_rollout_kl']['quantiles']['p99'] <= variant['full_rollout_kl']['max_kl']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_bytes


def test_failed_variant_restores_rng_precision_and_next_variant_starts_clean(tmp_path, monkeypatch):
    path = tmp_path / 'fixed.pt'
    capture(path)
    original_update = GRPOTrainer.update_policy
    calls = []

    def fail_first(trainer, episodes, advantages):
        calls.append(snapshot_rng())
        if len(calls) == 1:
            random.random()
            np.random.random()
            torch.rand(3)
            raise FloatingPointError('intentional post-update failure fixture')
        return original_update(trainer, episodes, advantages)

    monkeypatch.setattr(GRPOTrainer, 'update_policy', fail_first)
    before_rng, before_precision = snapshot_rng(), precision_metadata()
    report = run_diagnostics(path, learning_rates=[3e-5, 3e-5], device='cpu')
    assert report['variants'][0]['status'] == 'error'
    assert report['variants'][1]['status'] == 'ok'
    assert_tree_equal(calls[0], calls[1])
    assert_tree_equal(before_rng, snapshot_rng())
    assert before_precision == precision_metadata()


def test_cli_safe_load_json_and_exclusive_outputs(tmp_path, monkeypatch):
    path = tmp_path / 'fixed.pt'
    capture(path)
    original_load = torch.load
    loads = []

    def safe_load(*args, **kwargs):
        loads.append(kwargs.get('weights_only'))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', safe_load)
    output = tmp_path / 'report.json'
    assert main(['--bundle', str(path), '--output', str(output), '--learning-rates', '0.000003', '--device', 'cpu']) == 0
    assert loads == [True]
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['all_variants_succeeded'] and report['rng_restored']
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main(['--bundle', str(path), '--output', str(path)])
    with pytest.raises(SystemExit):
        main(['--bundle', str(path), '--output', str(output)])
    with pytest.raises(FileExistsError):
        write_report_exclusive(output, {'replacement': True})
    assert output.read_bytes() == original


@pytest.mark.parametrize('rates', [[], [True], [0.], [-1.], [float('nan')], [float('inf')]])
def test_bad_rates_rejected_before_loading(rates):
    with pytest.raises(ValueError):
        run_diagnostics('does_not_exist.pt', learning_rates=rates)


def test_exclusive_publish_rejects_racing_destination(tmp_path, monkeypatch):
    from ai_trader.grpo import update_diagnostic

    destination = tmp_path / 'bundle.pt'
    original_publish = update_diagnostic._publish_exclusive

    def race(temporary, target):
        target.write_bytes(b'other writer')
        original_publish(temporary, target)

    monkeypatch.setattr(update_diagnostic, '_publish_exclusive', race)
    with pytest.raises(FileExistsError):
        atomic_save_exclusive(destination, {'input': torch.ones(3)})
    assert destination.read_bytes() == b'other writer'
    assert list(tmp_path.glob('*.tmp')) == []
