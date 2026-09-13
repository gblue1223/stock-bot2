"""Compare one frozen GRPO update; optionally export candidates for separate validation."""
import argparse
from copy import deepcopy
import gc
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import time

import torch

from .diagnose_likelihood import _reconstruct_policy
from .likelihood_failure import portable_metadata, runtime_metadata
from .policy_update_checks import clone_state_to_cpu, exact_kl_from_log_probs, exact_policy_kl
from .runtime_precision import precision_metadata, restore_precision, set_tf32
from .update_diagnostic import (BUNDLE_FORMAT, BUNDLE_VERSION, TRAINER_FIELDS, bounded_rollout_batches,
                                atomic_save_exclusive, prepare_rollouts, restore_rng, snapshot_rng,
                                verify_rollout_likelihood)


DEFAULT_LEARNING_RATES = (3e-5, 1e-5, 3e-6)
logger = logging.getLogger(__name__)


def _rates(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('At least one learning rate is required')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Learning rates must be finite positive numbers')
    return [float(value) for value in values]


def _load_bundle(path):
    bundle = torch.load(Path(path).resolve(), map_location='cpu', weights_only=True)
    if (not isinstance(bundle, dict) or bundle.get('format') != BUNDLE_FORMAT
            or bundle.get('format_version') != BUNDLE_VERSION):
        raise ValueError('Unsupported fixed update bundle format/version')
    params = bundle.get('trainer_hyperparameters')
    if not isinstance(params, dict) or set(params) != set(TRAINER_FIELDS):
        raise ValueError('Incomplete saved trainer hyperparameters')
    for key, value in params.items():
        if key in ('use_gae', 'policy_update_checks'):
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be boolean')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite and numeric')
    for key in ('episodes_per_group', 'num_groups', 'batch_size', 'num_epochs', 'kl_probe_samples'):
        if not isinstance(params[key], int) or params[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not params['policy_update_checks']:
        raise ValueError('Saved policy_update_checks must be enabled')
    if bundle.get('optimizer_class') != 'torch.optim.Adam' or not isinstance(bundle.get('optimizer_state_dict'), dict):
        raise ValueError('A saved Adam state is required')
    if not isinstance(bundle.get('rng_state'), dict):
        raise ValueError('A saved RNG state is required')
    config = bundle.get('policy_config', {}).get('kwargs', {})
    episodes, advantages = prepare_rollouts(
        bundle.get('episodes'), bundle.get('advantages'), action_dim=config.get('action_dim', 3),
        masked=config.get('execution_action_mask', False))
    bundle['episodes'], bundle['advantages'] = episodes, advantages
    reference = bundle.get('reference_log_probs')
    samples = sum(len(episode['states']) for episode in episodes)
    if (not isinstance(reference, torch.Tensor) or reference.shape != (samples, config.get('action_dim'))):
        raise ValueError('Saved full-rollout reference table is missing or misaligned')
    exact_kl_from_log_probs(reference, reference)
    context = bundle.get('context', {})
    if context.get('sample_count') != samples or context.get('reference_batch_size') != params['batch_size']:
        raise ValueError('Saved rollout batching metadata is inconsistent')
    return bundle


def _policy_for_bundle(bundle, device):
    # Reuse the likelihood replay's explicit architecture allowlist and strict load.
    return _reconstruct_policy({**bundle, 'batch': {'states': bundle['episodes'][0]['states']}}, device)


def _full_kl(policy, episodes, reference, *, batch_size, device, threshold):
    per_state = []
    for offset, batch in bounded_rollout_batches(episodes, batch_size):
        result = exact_policy_kl(
            policy, batch['states'], batch['actions'], reference[offset:offset + len(batch['states'])],
            batch.get('action_masks'), batch_size=batch_size, device=device)
        per_state.append(result['per_state_kl'])
    values = torch.cat(per_state).double()
    return {'mean_kl': float(values.mean()), 'max_kl': float(values.max()),
            'num_samples': len(values), 'threshold': float(threshold),
            'samples_over_threshold': int((values > threshold).sum()),
            'fraction_over_threshold': float((values > threshold).double().mean()),
            'mean_over_threshold': bool(values.mean() > threshold),
            'quantiles': {f'p{int(q * 100)}': float(torch.quantile(values, q)) for q in (.5, .9, .95, .99)}}


def _fingerprint(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _guard_reason(metrics):
    if metrics.get('post_update_kl_early_stopped'):
        return 'post_step_exact_kl_rollback'
    if metrics.get('pre_update_kl_early_stopped'):
        if metrics.get('pre_update_kl_uses_exact'):
            return 'pre_step_exact_kl'
        return 'pre_step_sampled_kl'
    return 'completed_configured_epochs'


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_exports(bundle_path, source_checkpoint, export_dir, rates):
    """Validate every destination and source before any optimizer update starts."""
    from .local_check_inputs import validate_bundle_source

    directory = Path(export_dir).expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f'Candidate export directory is not a directory: {directory}')
    paths = [directory / f'lr_{index:02d}_{rate:.12g}.pt' for index, rate in enumerate(rates)]
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f'Candidate checkpoint already exists: {path}')
    source_path = Path(source_checkpoint).expanduser().resolve()
    verification = validate_bundle_source(bundle_path, source_path)
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    # Retain source metadata only; each variant already owns its policy/Adam copy.
    metadata = {
        'training_config': portable_metadata(source['extra_state']['training_config']),
        'date_splits': deepcopy(verification['date_splits']),
        'observation_schema': deepcopy(source['observation_schema']),
        'iteration': source['iteration'], 'num_updates': source['num_updates'],
        'total_timesteps': source['total_timesteps'],
    }
    provenance = {
        'source_checkpoint': str(source_path), 'source_checkpoint_sha256': _sha256_file(source_path),
        'bundle': str(Path(bundle_path).resolve()), 'bundle_sha256': _sha256_file(bundle_path),
        'export_dir': str(directory), 'source_verification': verification,
    }
    return paths, metadata, provenance


def _export_candidate(path, *, trainer, bundle, source, provenance, variant, index):
    """Save the actual accepted/rolled-back final policy without inherited scores."""
    config = deepcopy(source['training_config'])
    hyperparameters = {name: getattr(trainer, name) for name in TRAINER_FIELDS}
    for name, value in hyperparameters.items():
        config[{'learning_rate': 'lr', 'clip_epsilon': 'clip'}.get(name, name)] = value
    # Invocation requests are not properties of the resulting evaluation policy.
    config['resume_lr'] = None
    config['capture_update_bundle'] = None
    calls = trainer.num_updates - bundle['progress']['num_updates']
    if calls != 1:
        raise ValueError('Candidate export requires exactly one completed diagnostic update call')
    export_provenance = {
        'source_checkpoint': provenance['source_checkpoint'],
        'source_checkpoint_sha256': provenance['source_checkpoint_sha256'],
        'bundle': provenance['bundle'], 'bundle_sha256': provenance['bundle_sha256'],
        'variant_index': index, 'learning_rate': trainer.learning_rate,
        'source_iteration': source['iteration'], 'source_num_updates': source['num_updates'],
        'source_total_timesteps': source['total_timesteps'],
        'bundle_num_updates': bundle['progress']['num_updates'],
        'bundle_total_timesteps': bundle['progress']['total_timesteps'],
        'completed_update_calls': calls, 'guard_reason': variant['guard_reason'],
        'optimizer_accepted_steps': variant['metrics']['optimizer_accepted_steps'],
        'optimizer_attempted_steps': variant['metrics']['optimizer_attempted_steps'],
        'optimizer_rejected_steps': variant['metrics']['optimizer_rejected_steps'],
        'full_rollout_kl': variant['full_rollout_kl'], 'runtime': runtime_metadata(trainer.policy),
        'candidate_validation_performed': False, 'inherited_validation_scores': False,
    }
    payload = {
        'checkpoint_kind': 'fixed_rollout_evaluation_candidate', 'format_version': 1,
        'evaluation_only': True, 'resumable': False,
        'iteration': source['iteration'] + calls, 'num_updates': trainer.num_updates,
        'total_timesteps': trainer.total_timesteps,
        'policy_state_dict': clone_state_to_cpu(getattr(trainer.policy, 'module', trainer.policy).state_dict()),
        'observation_schema': deepcopy(source['observation_schema']),
        'config': portable_metadata(hyperparameters),
        'extra_state': {'training_config': portable_metadata(config),
                        'date_splits': deepcopy(source['date_splits']),
                        'export_provenance': portable_metadata(export_provenance)},
    }
    # No source optimizer/control/best/validation state is copied into an evaluation candidate.
    saved = atomic_save_exclusive(path, payload)
    logger.info('Evaluation-only candidate saved: %s (LR %.9g, accepted optimizer steps %d)',
                saved, trainer.learning_rate, export_provenance['optimizer_accepted_steps'])
    return saved


def run_diagnostics(bundle_path, *, learning_rates=DEFAULT_LEARNING_RATES, device='auto',
                    source_checkpoint=None, export_dir=None):
    """Replay independently; restore the caller's precision and RNG even on failure."""
    from .grpo import GRPOTrainer

    rates = _rates(learning_rates)
    if (source_checkpoint is None) != (export_dir is None):
        raise ValueError('source_checkpoint and export_dir must be provided together')
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    selected_device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if selected_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu')
    original_precision, original_rng = precision_metadata(), snapshot_rng()
    policy = trainer = None
    report = None
    try:
        set_tf32(False)
        export_paths = source = export_provenance = None
        if export_dir is not None:
            export_paths, source, export_provenance = _prepare_exports(bundle_path, source_checkpoint, export_dir, rates)
        bundle = _load_bundle(bundle_path)
        parameters = bundle['trainer_hyperparameters']
        numpy_episodes = [{key: value.numpy() if isinstance(value, torch.Tensor) else deepcopy(value)
                           for key, value in episode.items()} for episode in bundle['episodes']]
        numpy_advantages = [tensor.numpy() for tensor in bundle['advantages']]
        replay_rng = bundle['rng_state']
        cuda_rng_notice = None
        if selected_device == 'cuda' and not replay_rng['torch_cuda']:
            # CPU captures have no CUDA state; pick one fixed state for every replay.
            replay_rng = {**replay_rng, 'torch_cuda': original_rng['torch_cuda']}
            cuda_rng_notice = 'Capture has no CUDA RNG; all variants share the diagnostic starting CUDA RNG.'
        report = {
            'format': 'grpo_learning_rate_comparison', 'format_version': 1,
            'bundle': str(Path(bundle_path).resolve()), 'bundle_file_bytes': Path(bundle_path).stat().st_size,
            'purpose': 'Update stability diagnostic only; no return evaluation or deployability claim.',
            'model_exported': False, 'holdout_evaluated': False, 'device': selected_device,
            'learning_rates': rates, 'sample_count': bundle['context']['sample_count'],
            'episode_count': len(bundle['episodes']), 'batch_size': parameters['batch_size'],
            'saved_progress': bundle['progress'], 'saved_runtime': bundle['runtime'],
            'saved_trainer_hyperparameters': parameters, 'tensor_bytes': bundle['tensor_bytes'],
            'capture_likelihood_check': bundle['likelihood_check'],
            'same_initial_policy_and_adam': True, 'same_initial_rng': True,
            'initial_rng_sha256': _fingerprint(replay_rng), 'cuda_rng_notice': cuda_rng_notice,
            'minibatch_order': 'Same NumPy RNG restored immediately before each update; early stopping may truncate the shared order.',
            'full_rollout_reference': 'Saved policy weights recomputed once in this FP32 runtime, in original update input order.',
            'cross_runtime_notice': 'Different hardware or PyTorch versions can change numeric results; compare variants within this run.',
            'variants': [],
        }
        if export_provenance is not None:
            report.update(export_provenance)
            report['purpose'] = ('Update stability comparison with evaluation-only policy candidates for separate validation; '
                                 'candidate export is not a deployment or profitability qualification.')
        reference = None
        for index, rate in enumerate(rates):
            logger.info('Replaying fixed update at LR %.9g (%d samples, batch size %d)',
                        rate, report['sample_count'], report['batch_size'])
            variant_started = time.perf_counter()
            variant = {'learning_rate': rate, 'status': 'error', 'metrics': None, 'full_rollout_kl': None}
            if export_paths is not None:
                variant.update(candidate_checkpoint=None, candidate_sha256=None)
            try:
                policy = _policy_for_bundle(bundle, selected_device)
                trainer = GRPOTrainer(policy=policy, env=None, device=selected_device,
                                      observation_schema=bundle['observation_schema'], **parameters)
                # Adam.load_state_dict can alias CPU tensors, including step
                # counters. Give every variant its own independent moment state.
                trainer.optimizer.load_state_dict(clone_state_to_cpu(bundle['optimizer_state_dict']))
                trainer.total_timesteps = bundle['progress']['total_timesteps']
                trainer.num_updates = bundle['progress']['num_updates']
                trainer.set_learning_rate(rate)
                if reference is None:
                    report['diagnostic_runtime'] = runtime_metadata(policy)
                    baseline = verify_rollout_likelihood(
                        policy, bundle['episodes'], batch_size=parameters['batch_size'],
                        tolerance=parameters['rollout_logprob_tolerance'], device=selected_device)
                    reference = baseline.pop('reference_log_probs')
                    report['replay_likelihood_check'] = baseline
                    alignment = exact_kl_from_log_probs(bundle['reference_log_probs'], reference)
                    report['capture_to_replay_reference_kl'] = {key: value for key, value in alignment.items()
                                                                if key != 'per_state_kl'}
                # Constructors and the baseline probe consume RNG in some policies.
                # Restore only after all setup, immediately before the actual update.
                restore_rng(replay_rng, restore_cuda=selected_device == 'cuda')
                update_started = time.perf_counter()
                metrics = trainer.update_policy(numpy_episodes, numpy_advantages)
                variant['update_seconds'] = time.perf_counter() - update_started
                variant['metrics'] = portable_metadata(metrics)
                variant['guard_reason'] = _guard_reason(metrics)
                variant['status'] = 'ok'
                kl_started = time.perf_counter()
                variant['full_rollout_kl'] = _full_kl(
                    policy, bundle['episodes'], reference, batch_size=parameters['batch_size'],
                    device=selected_device, threshold=parameters['kl_target'] * 1.5)
                variant['full_rollout_kl_seconds'] = time.perf_counter() - kl_started
                if export_paths is not None:
                    saved_candidate = _export_candidate(
                        export_paths[index], trainer=trainer, bundle=bundle, source=source,
                        provenance=export_provenance, variant=variant, index=index)
                    variant['candidate_checkpoint'] = saved_candidate
                    report['model_exported'] = True
                    variant['candidate_sha256'] = _sha256_file(saved_candidate)
            except Exception as exc:
                variant['status'] = 'error'
                variant['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                variant['guard_reason'] = 'exception'
            finally:
                variant['total_seconds'] = time.perf_counter() - variant_started
                report['variants'].append(variant)
                logger.info('LR %.9g diagnostic %s in %.1fs: guard=%s, accepted=%s, full-rollout mean KL=%s',
                            rate, variant['status'], variant['total_seconds'], variant.get('guard_reason'),
                            (variant.get('metrics') or {}).get('optimizer_accepted_steps'),
                            (variant.get('full_rollout_kl') or {}).get('mean_kl'))
                # Never retain multiple policy/Adam allocations on the accelerator.
                trainer = policy = None
                gc.collect()
        report['all_variants_succeeded'] = all(variant['status'] == 'ok' for variant in report['variants'])
    finally:
        trainer = policy = None
        restore_precision(original_precision)
        restore_rng(original_rng)
    report['precision_restored'] = precision_metadata() == original_precision
    report['rng_restored'] = _fingerprint(snapshot_rng()) == _fingerprint(original_rng)
    return report


def write_report_exclusive(path, report):
    """An exclusively created report also works on Drive filesystems without hard links."""
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Encode completely before publishing. Never truncate a prior report or bundle.
    with destination.open('xb') as handle:
        try:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            destination.unlink()
            raise
    return str(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--learning-rates', nargs='+', type=float, default=list(DEFAULT_LEARNING_RATES))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--source-checkpoint', help='Original training checkpoint required to verify exported candidate lineage')
    parser.add_argument('--export-dir', help='Optional directory for new evaluation-only candidate checkpoints')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if Path(args.bundle).resolve() == Path(args.output).resolve():
        parser.error('--output must differ from the input bundle')
    if Path(args.output).exists():
        parser.error('--output already exists; choose a new report path')
    if bool(args.source_checkpoint) != bool(args.export_dir):
        parser.error('--source-checkpoint and --export-dir must be supplied together')
    if args.source_checkpoint and Path(args.source_checkpoint).resolve() == Path(args.output).resolve():
        parser.error('--output must differ from the source checkpoint')
    report = run_diagnostics(args.bundle, learning_rates=args.learning_rates, device=args.device,
                             source_checkpoint=args.source_checkpoint, export_dir=args.export_dir)
    path = write_report_exclusive(args.output, report)
    print(f'Learning-rate update diagnostic report: {path}')
    return 0 if report['all_variants_succeeded'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
