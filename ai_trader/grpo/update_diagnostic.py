"""Portable, immutable input to one GRPO update, with bounded verification forwards."""
from datetime import datetime, timezone
import ctypes
import errno
import logging
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch

from .likelihood_failure import portable_metadata, policy_reconstruction_config, runtime_metadata
from .policy_update_checks import (RolloutLikelihoodMismatch, check_rollout_likelihood,
                                   clone_state_to_cpu)


logger = logging.getLogger(__name__)
BUNDLE_FORMAT = 'grpo_fixed_rollout_update'
BUNDLE_VERSION = 1
TRAINER_FIELDS = ('episodes_per_group', 'num_groups', 'learning_rate', 'gamma', 'lambda_gae',
                  'clip_epsilon', 'kl_target', 'entropy_coef', 'value_coef', 'max_grad_norm',
                  'batch_size', 'use_gae', 'num_epochs', 'group_advantage_coef',
                  'policy_update_checks', 'rollout_logprob_tolerance', 'kl_probe_samples')
EPISODE_FIELDS = ('states', 'actions', 'rewards', 'dones', 'log_probs', 'values', 'action_masks')


def snapshot_rng():
    """Use only tensors and builtins so weights_only=True can load every RNG state."""
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {'generator': numpy_state[0], 'keys': torch.from_numpy(numpy_state[1].astype(np.int64)),
                  'position': int(numpy_state[2]), 'has_gauss': int(numpy_state[3]),
                  'cached_gaussian': float(numpy_state[4])},
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                      if torch.cuda.is_available() else [],
    }


def restore_rng(state, *, restore_cuda=True):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state['generator'], numpy_state['keys'].numpy().astype(np.uint32),
                         numpy_state['position'], numpy_state['has_gauss'], numpy_state['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if restore_cuda and state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('Saved CUDA RNG device count differs from this runtime; use --device cpu')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _bounded_cpu_tensor(value, *, dtype=None):
    tensor = torch.as_tensor(value, dtype=dtype).detach()
    # A small view can otherwise serialize an entire unrelated backing storage.
    if (tensor.device.type != 'cpu' or not tensor.is_contiguous() or tensor.storage_offset()
            or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
        tensor = tensor.to(device='cpu', copy=True).contiguous()
    return tensor


def prepare_rollouts(episodes, advantages, *, action_dim, masked):
    """Trim unused episode metadata; retain every update input and its numeric precision."""
    if not isinstance(episodes, (list, tuple)) or not episodes or len(episodes) != len(advantages):
        raise ValueError('Expected a nonempty rollout and one advantage array per episode')
    result, saved_advantages = [], []
    observation_shape = None
    for episode, advantage in zip(episodes, advantages):
        if not isinstance(episode, dict) or any(key not in episode for key in EPISODE_FIELDS[:6]):
            raise ValueError('Update bundles require cached states/actions/rewards/dones/log_probs/values')
        states = _bounded_cpu_tensor(episode['states'], dtype=torch.float32)
        if states.ndim != 3 or any(size < 1 for size in states.shape) or not torch.isfinite(states).all():
            raise ValueError('Rollout states must be finite nonempty N-by-sequence-by-feature observations')
        if observation_shape is None:
            observation_shape = states.shape[1:]
        if states.shape[1:] != observation_shape:
            raise ValueError('All rollout observation shapes must match')
        saved = {'states': states}
        size = len(states)
        for key in EPISODE_FIELDS[1:6]:
            tensor = _bounded_cpu_tensor(episode[key])
            if tensor.shape != (size,) or not torch.isfinite(tensor).all():
                raise ValueError(f'Rollout {key} must be finite and aligned with states')
            if key == 'actions':
                if tensor.dtype == torch.bool or tensor.is_floating_point():
                    raise ValueError('Rollout actions must be integer indices')
                tensor = tensor.to(torch.int64)
                if (tensor < 0).any() or (tensor >= action_dim).any():
                    raise ValueError('Rollout action index is out of range')
            elif key == 'dones':
                if not ((tensor == 0) | (tensor == 1)).all():
                    raise ValueError('Rollout dones must be boolean or zero/one')
                tensor = tensor.to(torch.bool)
            elif not tensor.is_floating_point():
                raise ValueError(f'Rollout {key} must use a floating point dtype')
            saved[key] = tensor
        masks = episode.get('action_masks')
        if masked:
            if masks is None:
                raise ValueError('Execution-masked policies require cached masks in every episode')
            masks = _bounded_cpu_tensor(masks)
            if masks.dtype != torch.bool or masks.shape != (size, action_dim) or not masks.any(-1).all():
                raise ValueError('Cached action masks must be aligned nonempty boolean masks')
            if not masks.gather(1, saved['actions'][:, None]).all():
                raise ValueError('Cached action is disabled by its cached mask')
            saved['action_masks'] = masks
        elif masks is not None:
            raise ValueError('Unmasked policy cannot capture masked rollout inputs')
        adv = _bounded_cpu_tensor(advantage)
        if adv.shape != (size,) or not adv.is_floating_point() or not torch.isfinite(adv).all():
            raise ValueError('Group advantages must be finite floating point arrays aligned with episodes')
        result.append(saved)
        saved_advantages.append(adv)
    return result, saved_advantages


def bounded_rollout_batches(episodes, batch_size):
    """Match concatenate(episodes) batching without concatenating all observations."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    pending, pending_size, offset = [], 0, 0
    keys = ('states', 'actions', 'log_probs')
    if 'action_masks' in episodes[0]:
        keys += ('action_masks',)
    for episode in episodes:
        start = 0
        while start < len(episode['states']):
            count = min(batch_size - pending_size, len(episode['states']) - start)
            pending.append({key: episode[key][start:start + count] for key in keys})
            start += count
            pending_size += count
            if pending_size == batch_size:
                yield offset, {key: (pending[0][key] if len(pending) == 1 else
                                     torch.cat([part[key] for part in pending])) for key in keys}
                offset += pending_size
                pending, pending_size = [], 0
    if pending_size:
        yield offset, {key: (pending[0][key] if len(pending) == 1 else
                            torch.cat([part[key] for part in pending])) for key in keys}


def verify_rollout_likelihood(policy, episodes, *, batch_size, tolerance, device):
    tables, max_error, error_sum, samples = [], 0.0, 0.0, 0
    total_samples = sum(len(episode['states']) for episode in episodes)
    for offset, batch in bounded_rollout_batches(episodes, batch_size):
        try:
            result = check_rollout_likelihood(
                policy, batch['states'], batch['actions'], batch['log_probs'], batch.get('action_masks'),
                batch_size=batch_size, tolerance=tolerance, device=device)
        except RolloutLikelihoodMismatch as failure:
            failure.offset += offset
            failure.total_samples = total_samples
            raise
        tables.append(result['reference_log_probs'])
        max_error = max(max_error, result['max_abs_error'])
        error_sum += result['mean_abs_error'] * result['num_samples']
        samples += result['num_samples']
    return {'max_abs_error': max_error, 'mean_abs_error': error_sum / samples, 'num_samples': samples,
            'reference_log_probs': torch.cat(tables)}


def _publish_exclusive(temporary, destination):
    if os.name == 'nt':
        # Windows rename is atomic and refuses an existing destination.
        os.rename(temporary, destination)
        return
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, 'renameat2', None)
        if rename is not None:
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            if rename(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1) == 0:
                return
            error = ctypes.get_errno()
            if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
                raise OSError(error, os.strerror(error), str(destination))
    # An exclusive hard link publishes a complete file, even if rename flags are unavailable.
    try:
        os.link(temporary, destination)
    except OSError as exc:
        if exc.errno in (errno.ENOTSUP, errno.EPERM, errno.ENOSYS):
            raise OSError('This filesystem cannot publish a bundle atomically without overwrite; '
                          'capture to a local path such as /content/update_bundle.pt') from exc
        raise
    temporary.unlink()


def atomic_save_exclusive(path, payload):
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f'Update bundle already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{destination.name}.', suffix='.tmp',
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_exclusive(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(destination)


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def save_update_bundle(path, *, trainer, episodes, advantages, training_config):
    """Save one frozen update's inputs after checking every cached chosen likelihood.

    The caller must supply episodes and advantages in the exact flattened group
    order it would pass to update_policy. No optimizer or environment is advanced.
    """
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f'Update bundle already exists: {destination}')
    policy = getattr(trainer.policy, 'module', trainer.policy)
    if type(trainer.optimizer) is not torch.optim.Adam:
        raise ValueError('Update capture requires the trainer Adam optimizer')
    config = policy_reconstruction_config(policy)
    if not config['supported_by_replay_cli']:
        raise ValueError('Update capture supports only GRPOPolicyE2EXLSTM policies')
    hyperparameters = portable_metadata({field: getattr(trainer, field) for field in TRAINER_FIELDS})
    if not hyperparameters['policy_update_checks']:
        raise ValueError('Update capture requires policy_update_checks=True to retain all update safeguards')
    runtime = runtime_metadata(policy)
    precision = runtime['precision']
    if any(precision[key] != 'ieee' for key in
           ('matmul_fp32_precision', 'cudnn_conv_fp32_precision', 'cudnn_rnn_fp32_precision')):
        raise ValueError('Update capture requires FP32 with TF32 disabled')
    if runtime['policy_parameter_dtypes'] != ['torch.float32'] or runtime['autocast_cuda_enabled'] or runtime['autocast_cpu_enabled']:
        raise ValueError('Update capture requires float32 weights and disabled autocast')
    saved_episodes, saved_advantages = prepare_rollouts(
        episodes, advantages, action_dim=policy.action_dim, masked=policy.execution_action_mask)
    rng = snapshot_rng()
    try:
        likelihood = verify_rollout_likelihood(
            policy, saved_episodes, batch_size=trainer.batch_size,
            tolerance=trainer.rollout_logprob_tolerance, device=trainer.device)
        reference = likelihood.pop('reference_log_probs')
        payload = {
            'format': BUNDLE_FORMAT, 'format_version': BUNDLE_VERSION,
            'saved_at_utc': datetime.now(timezone.utc).isoformat(),
            'policy_config': config, 'policy_state_dict': clone_state_to_cpu(policy.state_dict()),
            'optimizer_class': 'torch.optim.Adam', 'optimizer_state_dict': clone_state_to_cpu(trainer.optimizer.state_dict()),
            'trainer_hyperparameters': hyperparameters,
            'training_config': portable_metadata(training_config or {}),
            'observation_schema': portable_metadata(trainer.observation_schema),
            'runtime': runtime, 'rng_state': rng,
            'progress': {'num_updates': int(trainer.num_updates), 'total_timesteps': int(trainer.total_timesteps)},
            'episodes': saved_episodes, 'advantages': saved_advantages,
            'reference_log_probs': reference, 'likelihood_check': likelihood,
            'context': {'episode_count': len(saved_episodes), 'sample_count': len(reference),
                        'flattening_order': 'caller supplied, identical to update_policy input',
                        'reference_batch_size': trainer.batch_size,
                        'purpose': 'Learning-rate update stability comparison; not a return or holdout evaluation'},
        }
        payload['tensor_bytes'] = {key: _tensor_bytes(payload[key]) for key in
                                   ('episodes', 'advantages', 'policy_state_dict', 'optimizer_state_dict',
                                    'reference_log_probs', 'rng_state')}
        payload['tensor_bytes']['total'] = sum(payload['tensor_bytes'].values())
        saved_path = atomic_save_exclusive(destination, payload)
        logger.info('Fixed update bundle saved: %s | episodes=%d, samples=%d, tensor bytes=%d, file bytes=%d, '
                    'likelihood max error=%.9g', saved_path, len(saved_episodes), len(reference),
                    payload['tensor_bytes']['total'], destination.stat().st_size, likelihood['max_abs_error'])
        return saved_path
    finally:
        restore_rng(rng)
