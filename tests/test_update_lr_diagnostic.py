"""One frozen rollout, independent Adam states and identical shuffles across LR variants."""
import hashlib
import io
import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from ai_trader.grpo.diagnose_update_lr import _guard_reason, main, run_diagnostics, write_report_exclusive
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


@pytest.mark.parametrize('metrics,reason', [
    ({'pre_update_kl_early_stopped': 1, 'pre_update_kl_uses_exact': 1}, 'pre_step_exact_kl'),
    ({'pre_update_kl_early_stopped': 1, 'pre_update_kl_uses_exact': 0}, 'pre_step_sampled_kl'),
    # Older diagnostics calculated exact KL only after the sampled guard stopped.
    ({'pre_update_kl_early_stopped': 1, 'pre_update_exact_kl_checked': 1,
      'policy_update_checks_enabled': 1}, 'pre_step_sampled_kl'),
    ({}, 'completed_configured_epochs'),
    ({'pre_update_kl_early_stopped': 0, 'pre_update_kl_uses_exact': 1}, 'completed_configured_epochs'),
    ({'post_update_kl_early_stopped': 1, 'pre_update_kl_early_stopped': 1,
      'pre_update_kl_uses_exact': 1}, 'post_step_exact_kl_rollback'),
])
def test_guard_reason_distinguishes_active_exact_guard_from_legacy_diagnostic_only(metrics, reason):
    assert _guard_reason(metrics) == reason


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


def export_fixture(tmp_path, *, kl_target=None):
    from ai_trader.grpo.train_xlstm import TrainingConfig
    from ai_trader.grpo.update_diagnostic import TRAINER_FIELDS
    from lib.observations import ObservationBuilder

    trainer, episodes, advantages = fixed_rollout('cpu')
    if kl_target is not None:
        trainer.kl_target = kl_target
    config = TrainingConfig()
    config.seq_len, config.features = 8, 27
    config.cnn_channels, config.rnn_hidden_dim, config.hidden_dim = 4, 4, 8
    config.checkpoint_segments = 2
    config.rolling_window_size, config.rolling_min_samples = 4, 2
    for name in TRAINER_FIELDS:
        setattr(config, {'learning_rate': 'lr', 'clip_epsilon': 'clip'}.get(name, name), getattr(trainer, name))
    schema = ObservationBuilder([f'feature_{index}' for index in range(27)], seq_len=8,
                                rolling_window_size=4, rolling_min_samples=2,
                                account_observations=True, execution_observations=True).schema
    trainer.observation_schema = schema
    dates = {'train': ['20240101', '20240102'], 'validation': ['20240103', '20240104'],
             'test': ['20240105', '20240106']}
    source = {'iteration': 19, 'num_updates': 19, 'total_timesteps': trainer.total_timesteps,
              'policy_state_dict': clone_state_to_cpu(trainer.policy.state_dict()),
              'optimizer_state_dict': clone_state_to_cpu(trainer.optimizer.state_dict()),
              'observation_schema': schema, 'config': {name: getattr(trainer, name) for name in TRAINER_FIELDS},
              'best_validation_return': 999., 'training_control_state': {'old_score': 999.},
              'extra_state': {'training_config': deepcopy(config.__dict__), 'date_splits': dates,
                              'best_validation_metrics': {'mean_net_return': 999.},
                              'validation_return': 999.}}
    source_path = tmp_path / 'source.pt'
    torch.save(source, source_path)
    trainer.total_timesteps += sum(len(episode['states']) for episode in episodes)
    bundle_path = tmp_path / 'fixed.pt'
    save_update_bundle(bundle_path, trainer=trainer, episodes=episodes, advantages=advantages,
                       training_config=deepcopy(config.__dict__))
    return source_path, bundle_path, source


def test_exported_candidates_are_actual_final_policies_with_clean_evaluation_lineage(tmp_path, monkeypatch):
    source_path, bundle_path, source = export_fixture(tmp_path)
    initial_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (source_path, bundle_path)]
    original_update = GRPOTrainer.update_policy
    actual_final_states = []

    def record_final(trainer, episodes, advantages):
        result = original_update(trainer, episodes, advantages)
        actual_final_states.append(clone_state_to_cpu(trainer.policy.state_dict()))
        return result

    monkeypatch.setattr(GRPOTrainer, 'update_policy', record_final)
    export_dir = tmp_path / 'candidates'
    report = run_diagnostics(bundle_path, learning_rates=[3e-5, 1e-5, 3e-6], device='cpu',
                              source_checkpoint=source_path, export_dir=export_dir)
    assert report['all_variants_succeeded'] and report['model_exported']
    assert report['holdout_evaluated'] is False
    assert report['precision_restored'] and report['rng_restored']
    assert report['source_checkpoint'] == str(source_path.resolve())
    assert report['source_checkpoint_sha256'] == initial_hashes[0]
    assert report['bundle_sha256'] == initial_hashes[1]
    for index, variant in enumerate(report['variants']):
        path = Path(variant['candidate_checkpoint'])
        assert path.parent == export_dir.resolve() and path.name.startswith(f'lr_{index:02d}_')
        assert variant['candidate_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        candidate = torch.load(path, map_location='cpu', weights_only=True)
        assert candidate['evaluation_only'] is True and candidate['resumable'] is False
        assert candidate['checkpoint_kind'] == 'fixed_rollout_evaluation_candidate'
        assert_tree_equal(candidate['policy_state_dict'], actual_final_states[index])
        assert candidate['num_updates'] == candidate['iteration'] == 20
        assert candidate['total_timesteps'] == source['total_timesteps'] + 8
        assert candidate['config']['learning_rate'] == variant['learning_rate']
        assert candidate['extra_state']['training_config']['lr'] == variant['learning_rate']
        assert candidate['observation_schema'] == source['observation_schema']
        assert candidate['extra_state']['date_splits'] == source['extra_state']['date_splits']
        assert set(candidate['extra_state']) == {'training_config', 'date_splits', 'export_provenance'}
        assert not {'best_validation_return', 'training_control_state', 'optimizer_state_dict'} & candidate.keys()
        provenance = candidate['extra_state']['export_provenance']
        assert provenance['source_checkpoint_sha256'] == initial_hashes[0]
        assert provenance['bundle_sha256'] == initial_hashes[1]
        assert provenance['optimizer_accepted_steps'] == variant['metrics']['optimizer_accepted_steps']
        assert provenance['inherited_validation_scores'] is False
        assert provenance['candidate_validation_performed'] is False
        assert not torch.equal(candidate['policy_state_dict']['policy_head.weight'], source['policy_state_dict']['policy_head.weight'])
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (source_path, bundle_path)] == initial_hashes


def test_exported_candidate_can_be_read_by_existing_validation_and_cannot_resume(tmp_path, monkeypatch):
    from ai_trader.grpo import diagnose_xlstm
    from ai_trader.grpo.train_xlstm import TrainingConfig, create_policy

    source_path, bundle_path, source = export_fixture(tmp_path)
    report = run_diagnostics(bundle_path, learning_rates=[1e-5], device='cpu',
                              source_checkpoint=source_path, export_dir=tmp_path / 'candidates')
    path = report['variants'][0]['candidate_checkpoint']
    candidate = torch.load(path, weights_only=True, map_location='cpu')
    observed = {}
    fake_env = SimpleNamespace(observation_space=SimpleNamespace(shape=(8, 63)),
                               observation_schema=source['observation_schema'],
                               valid_keys=[('fixture', 'path', date) for date in source['extra_state']['date_splits']['validation']],
                               close=lambda: observed.update(closed=True))

    def environment(config, device, allowed_dates):
        observed.update(lr=config.lr, dates=allowed_dates)
        return fake_env

    def evaluate(policy, env, **kwargs):
        assert_tree_equal(policy.state_dict(), candidate['policy_state_dict'])
        return {'num_episodes': kwargs['num_episodes'], 'mean_net_return': 0.0}

    monkeypatch.setattr(diagnose_xlstm, '_training_factories', lambda: (TrainingConfig, environment, create_policy))
    monkeypatch.setattr(diagnose_xlstm, 'evaluate_policy', evaluate)
    validation = diagnose_xlstm.run_diagnostics(path, split='validation', episodes=2, device='cpu', extracted_dir=tmp_path)
    assert observed['closed'] and observed['lr'] == 1e-5
    assert observed['dates'] == source['extra_state']['date_splits']['validation']
    assert validation['checkpoint_iteration'] == 20
    trainer, _, _ = fixed_rollout('cpu')
    with pytest.raises(ValueError, match='optimizer_state_dict'):
        trainer.restore_training_progress(candidate)


def test_export_collision_is_rejected_before_any_update(tmp_path, monkeypatch):
    source_path, bundle_path, _ = export_fixture(tmp_path)
    directory = tmp_path / 'candidates'
    directory.mkdir()
    collision = directory / 'lr_01_1e-05.pt'
    collision.write_bytes(b'existing evidence')
    monkeypatch.setattr(GRPOTrainer, 'update_policy', lambda *args: pytest.fail('Update must not start on a destination collision'))
    with pytest.raises(FileExistsError, match='already exists'):
        run_diagnostics(bundle_path, learning_rates=[3e-5, 1e-5], device='cpu',
                          source_checkpoint=source_path, export_dir=directory)
    assert list(directory.iterdir()) == [collision]
    assert collision.read_bytes() == b'existing evidence'


def test_failed_update_or_full_kl_does_not_export_that_variant(tmp_path, monkeypatch):
    from ai_trader.grpo import diagnose_update_lr

    source_path, bundle_path, _ = export_fixture(tmp_path)
    original_update, original_kl = GRPOTrainer.update_policy, diagnose_update_lr._full_kl
    updates, checks = [], []

    def update(trainer, episodes, advantages):
        updates.append(trainer.learning_rate)
        if len(updates) == 1:
            raise FloatingPointError('controlled update failure')
        return original_update(trainer, episodes, advantages)

    def full_kl(*args, **kwargs):
        checks.append(True)
        if len(checks) == 1:
            raise FloatingPointError('controlled full rollout KL failure')
        return original_kl(*args, **kwargs)

    monkeypatch.setattr(GRPOTrainer, 'update_policy', update)
    monkeypatch.setattr(diagnose_update_lr, '_full_kl', full_kl)
    directory = tmp_path / 'candidates'
    report = run_diagnostics(bundle_path, learning_rates=[3e-5, 1e-5, 3e-6], device='cpu',
                              source_checkpoint=source_path, export_dir=directory)
    assert [variant['status'] for variant in report['variants']] == ['error', 'error', 'ok']
    assert report['model_exported'] and not report['all_variants_succeeded']
    assert all(variant['candidate_checkpoint'] is None for variant in report['variants'][:2])
    assert [path.name for path in directory.iterdir()] == ['lr_02_3e-06.pt']


def test_wrong_export_source_is_rejected_before_updates(tmp_path, monkeypatch):
    source_path, bundle_path, source = export_fixture(tmp_path)
    source['policy_state_dict']['policy_head.weight'].add_(1)
    torch.save(source, source_path)
    monkeypatch.setattr(GRPOTrainer, 'update_policy', lambda *args: pytest.fail('Wrong source must fail before update'))
    with pytest.raises(ValueError, match='policy_state_dict'):
        run_diagnostics(bundle_path, device='cpu', source_checkpoint=source_path, export_dir=tmp_path / 'candidates')
    assert not (tmp_path / 'candidates').exists()


def test_export_options_are_paired_and_cli_emits_candidate_paths(tmp_path):
    source_path, bundle_path, _ = export_fixture(tmp_path)
    with pytest.raises(ValueError, match='provided together'):
        run_diagnostics(bundle_path, device='cpu', source_checkpoint=source_path)
    with pytest.raises(ValueError, match='provided together'):
        run_diagnostics(bundle_path, device='cpu', export_dir=tmp_path / 'candidates')
    with pytest.raises(SystemExit):
        main(['--bundle', str(bundle_path), '--output', str(tmp_path / 'missing.json'), '--export-dir', str(tmp_path / 'candidates')])
    output = tmp_path / 'report.json'
    assert main(['--bundle', str(bundle_path), '--output', str(output), '--device', 'cpu',
                  '--learning-rates', '0.00001', '--source-checkpoint', str(source_path),
                  '--export-dir', str(tmp_path / 'candidates')]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['model_exported'] and Path(report['variants'][0]['candidate_checkpoint']).is_file()


def test_poststep_rejection_exports_the_restored_final_weights_and_records_zero_accepted_steps(tmp_path):
    source_path, bundle_path, source = export_fixture(tmp_path, kl_target=1e-6)
    report = run_diagnostics(bundle_path, learning_rates=[.1], device='cpu',
                              source_checkpoint=source_path, export_dir=tmp_path / 'candidates')
    variant = report['variants'][0]
    assert variant['status'] == 'ok' and variant['guard_reason'] == 'post_step_exact_kl_rollback'
    assert variant['metrics']['optimizer_rejected_steps'] == 1
    assert variant['metrics']['optimizer_accepted_steps'] == 0
    candidate = torch.load(variant['candidate_checkpoint'], map_location='cpu', weights_only=True)
    assert_tree_equal(candidate['policy_state_dict'], source['policy_state_dict'])
    assert candidate['extra_state']['export_provenance']['optimizer_accepted_steps'] == 0
    assert candidate['extra_state']['export_provenance']['optimizer_rejected_steps'] == 1
    assert candidate['extra_state']['export_provenance']['completed_update_calls'] == 1
