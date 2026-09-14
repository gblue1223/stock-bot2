"""Bounded throughput comparison on copies of a real rollout and optimizer.

Only runtime settings are selected. Trial weights, RNG draws and counters never
enter training. Truncated trial returns are not performance/profit estimates.
"""
import copy
import gc
import logging
import math
import random

import numpy as np
import torch

from .gpu_runtime import PhaseMeasurement
from .policy_update_checks import clone_state_to_cpu, RolloutLikelihoodMismatch

logger = logging.getLogger(__name__)


def validate_gpu_tuning(settings):
    if settings is None:
        return None
    defaults = {'batch_sizes': None, 'checkpoint_segments': [16], 'samples': 256,
                'warmup_samples': 8, 'max_memory_fraction': .8}
    if not isinstance(settings, dict) or set(settings) - set(defaults):
        raise ValueError('Invalid gpu_tuning settings')
    result = {**defaults, **copy.deepcopy(settings)}
    for key in ('samples', 'warmup_samples'):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f'gpu_tuning.{key} must be a positive integer')
    for key, minimum in (('batch_sizes', 1), ('checkpoint_segments', 0)):
        values = result[key]
        if key == 'batch_sizes' and values is None:
            continue
        if (not isinstance(values, list) or not values or len(values) > 8
                or any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values)):
            raise ValueError(f'gpu_tuning.{key} must contain 1-8 integers >= {minimum}')
        result[key] = list(dict.fromkeys(values))
    fraction = result['max_memory_fraction']
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not .1 <= fraction <= .95:
        raise ValueError('gpu_tuning.max_memory_fraction must be in [0.1, 0.95]')
    return result


def sample_rollouts(episodes, advantages, limit):
    """Keep cached masks/log likelihoods/entry credits aligned in a bounded prefix."""
    sampled, grouped = [], []
    for episode, advantage in zip(episodes, advantages):
        count = min(limit, len(episode['actions']))
        if count <= 0:
            break
        item = {key: np.array(episode[key][:count], copy=True) for key in
                ('states', 'actions', 'log_probs', 'rewards', 'dones', 'values', 'action_masks') if key in episode}
        item['dones'][-1] = True
        item['metadata'] = {}
        pattern = episode.get('metadata', {}).get('entry_pattern')
        if pattern is not None:
            item['metadata']['entry_pattern'] = {'policy_credit': list(pattern['policy_credit'][:count])}
            if 'config' in pattern:
                item['metadata']['entry_pattern']['config'] = copy.deepcopy(pattern['config'])
        sampled.append(item)
        grouped.append(np.array(advantage[:count], copy=True))
        limit -= count
    if not sampled:
        raise ValueError('GPU tuning requires a nonempty rollout')
    return sampled, grouped


def select_trial(trials):
    valid = [trial for trial in trials if trial['status'] == 'ok']
    if not valid:
        raise RuntimeError('No GPU tuning candidate passed memory, likelihood and full-update checks')
    return max(valid, key=lambda trial: trial['samples_per_second'])


def _trial(trainer, episodes, advantages, batch_size, segments, memory_fraction, optimizer_state):
    from .grpo import GRPOTrainer
    probe = None
    try:
        policy = copy.deepcopy(trainer.policy)
        policy.zero_grad(set_to_none=True)
        if hasattr(policy, 'xlstm'):
            policy.xlstm.checkpoint_segments = segments
        names = ('learning_rate', 'gamma', 'lambda_gae', 'group_advantage_coef', 'clip_epsilon',
                 'kl_target', 'entropy_coef', 'value_coef', 'max_grad_norm', 'use_gae', 'num_epochs',
                 'policy_update_checks', 'rollout_logprob_tolerance', 'kl_probe_samples')
        probe = GRPOTrainer(policy, object(), device=trainer.device, batch_size=batch_size,
                            **{name: getattr(trainer, name) for name in names})
        # load_state_dict may reuse CPU tensors: give each trial its own Adam state.
        probe.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        del policy
        with PhaseMeasurement(trainer.device) as measurement:
            metrics = probe.update_policy(episodes, advantages)
        count = sum(len(ep['actions']) for ep in episodes)
        result = dict(measurement.throughput(count * trainer.num_epochs), batch_size=batch_size,
                      checkpoint_segments=segments, status='ok', optimizer_steps=metrics['optimizer_steps'])
        expected = trainer.num_epochs * math.ceil(count / batch_size)
        if metrics['optimizer_steps'] != expected or metrics['kl_early_stopped']:
            result['status'] = 'incomplete_update'
        if result.get('peak_reserved_bytes', 0) > memory_fraction * result.get('device_total_bytes', float('inf')):
            result['status'] = 'memory_budget'
        return result
    except torch.cuda.OutOfMemoryError:
        return {'batch_size': batch_size, 'checkpoint_segments': segments, 'status': 'oom'}
    except (RolloutLikelihoodMismatch, FloatingPointError) as error:
        return {'batch_size': batch_size, 'checkpoint_segments': segments, 'status': 'numerical_check',
                'error': str(error)}
    finally:
        del probe
        gc.collect()
        if torch.device(trainer.device).type == 'cuda':
            torch.cuda.empty_cache()


def tune_update(trainer, episodes, advantages, settings):
    settings = validate_gpu_tuning(settings)
    device = torch.device(trainer.device)
    cuda = device.type == 'cuda'
    total = torch.cuda.get_device_properties(device).total_memory if cuda else 0
    candidates = settings['batch_sizes'] or ([32, 64, 128] if total >= 70 * 1024**3 else [16, 32, 64])
    sampled, grouped = sample_rollouts(episodes, advantages, settings['samples'])
    count = sum(len(ep['actions']) for ep in sampled)
    candidates = sorted({min(trainer.batch_size, count), *(b for b in candidates if b <= count)})
    baseline_segments = getattr(getattr(trainer.policy, 'xlstm', None), 'checkpoint_segments', 0)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    optimizer_state = clone_state_to_cpu(trainer.optimizer.state_dict())
    trials = []
    try:
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if cuda else []
        with torch.random.fork_rng(devices=devices):
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if cuda else None

            def run(batch, segments, data=sampled, relative=grouped):
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                torch.set_rng_state(cpu_rng)
                if cuda:
                    torch.cuda.set_rng_state(cuda_rng, device)
                    torch.cuda.empty_cache()
                logger.info('GPU tuning candidate: batch=%d, checkpoint_segments=%d, samples=%d',
                            batch, segments, sum(len(ep['actions']) for ep in data))
                return _trial(trainer, data, relative, batch, segments,
                              settings['max_memory_fraction'], optimizer_state)

            warm, warm_adv = sample_rollouts(sampled, grouped, settings['warmup_samples'])
            run(min(trainer.batch_size, settings['warmup_samples']), baseline_segments, warm, warm_adv)
            for batch in candidates:
                trials.append(run(batch, baseline_segments))
                logger.info('GPU tuning result: %s', trials[-1])
            best = select_trial(trials)
            # Compare recomputation only at the winning batch, avoiding a large grid.
            for segments in settings['checkpoint_segments']:
                if segments != baseline_segments:
                    trials.append(run(best['batch_size'], segments))
                    logger.info('GPU tuning result: %s', trials[-1])
            selected = select_trial(trials)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return {'device': str(device), 'device_name': torch.cuda.get_device_name(device) if cuda else 'cpu',
            'settings': settings, 'samples': count, 'trials': trials,
            'selected': {key: selected[key] for key in ('batch_size', 'checkpoint_segments')},
            'selection_basis': 'full-update throughput on truncated rollout copies; not profitability',
            'batch_size_changes_optimizer_step_count': True}
