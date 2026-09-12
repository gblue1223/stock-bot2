"""Compare one frozen GRPO update at different learning rates; never export a model."""
import argparse
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
                                prepare_rollouts, restore_rng, snapshot_rng, verify_rollout_likelihood)


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
        return 'pre_step_sampled_kl'
    return 'completed_configured_epochs'


def run_diagnostics(bundle_path, *, learning_rates=DEFAULT_LEARNING_RATES, device='auto'):
    """Replay independently; restore the caller's precision and RNG even on failure."""
    from .grpo import GRPOTrainer

    rates = _rates(learning_rates)
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
        bundle = _load_bundle(bundle_path)
        parameters = bundle['trainer_hyperparameters']
        numpy_episodes = [{key: tensor.numpy() for key, tensor in episode.items()} for episode in bundle['episodes']]
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
        reference = None
        for rate in rates:
            logger.info('Replaying fixed update at LR %.9g (%d samples, batch size %d)',
                        rate, report['sample_count'], report['batch_size'])
            variant_started = time.perf_counter()
            variant = {'learning_rate': rate, 'status': 'error', 'metrics': None, 'full_rollout_kl': None}
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if Path(args.bundle).resolve() == Path(args.output).resolve():
        parser.error('--output must differ from the input bundle')
    if Path(args.output).exists():
        parser.error('--output already exists; choose a new report path')
    report = run_diagnostics(args.bundle, learning_rates=args.learning_rates, device=args.device)
    path = write_report_exclusive(args.output, report)
    print(f'Learning-rate update diagnostic report: {path}')
    return 0 if report['all_variants_succeeded'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
