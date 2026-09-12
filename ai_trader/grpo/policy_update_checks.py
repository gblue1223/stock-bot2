"""Bounded checks of rollout likelihoods and the policy after an optimizer step."""
import copy

import numpy as np
import torch


def validate_update_check_settings(enabled, tolerance, probe_samples):
    if not isinstance(enabled, bool):
        raise ValueError('policy_update_checks must be a bool')
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, (int, float, np.integer, np.floating))
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('rollout_logprob_tolerance must be finite and nonnegative')
    if (isinstance(probe_samples, (bool, np.bool_)) or not isinstance(probe_samples, (int, np.integer))
            or probe_samples < 1):
        raise ValueError('kl_probe_samples must be a positive integer')
    return enabled, float(tolerance), int(probe_samples)


def clone_state_to_cpu(value):
    """Copy parameters, buffers and Adam moments without a second GPU model."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def snapshot_optimizer_step(policy, optimizer):
    return clone_state_to_cpu(policy.state_dict()), clone_state_to_cpu(optimizer.state_dict())


def restore_optimizer_step(policy, optimizer, snapshot):
    policy.load_state_dict(snapshot[0], strict=True)
    optimizer.load_state_dict(snapshot[1])
    optimizer.zero_grad(set_to_none=True)


def _device(policy, device):
    if device is not None:
        return device
    parameter = next(policy.parameters(), None)
    return parameter.device if parameter is not None else 'cpu'


def _cpu_inputs(states, actions, action_masks, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    states = torch.as_tensor(states, device='cpu').float()
    actions = torch.as_tensor(actions, device='cpu')
    if (not len(states) or actions.shape != (len(states),) or actions.dtype == torch.bool
            or not torch.isfinite(actions).all() or torch.any(actions != actions.long())):
        raise ValueError('Expected nonempty observations and aligned integer actions')
    masks = None if action_masks is None else torch.as_tensor(action_masks, device='cpu')
    if masks is not None and (masks.dtype != torch.bool or masks.ndim != 2
                              or len(masks) != len(states) or not masks.any(-1).all()):
        raise ValueError('Expected cached boolean action masks with an allowed action per state')
    return states, actions.long(), masks


def evaluate_action_distribution(policy, states, actions, masks=None):
    method = getattr(policy, 'evaluate_actions_with_distribution', None)
    if not callable(method):
        raise ValueError('policy_update_checks requires evaluate_actions_with_distribution on the policy')
    kwargs = {'action_masks': masks} if masks is not None else {}
    chosen, entropy, values, log_probabilities = method(states, actions, **kwargs)
    if (chosen.shape != actions.shape or log_probabilities.ndim != 2
            or len(log_probabilities) != len(actions) or torch.any(actions < 0)
            or torch.any(actions >= log_probabilities.shape[1])):
        raise ValueError('Invalid action distribution shape in policy update check')
    if (not torch.isfinite(chosen).all() or torch.isnan(log_probabilities).any()
            or torch.isposinf(log_probabilities).any()
            or not torch.isfinite(entropy).all() or not torch.isfinite(values).all()):
        raise FloatingPointError('Nonfinite policy output in policy update check')
    probabilities = log_probabilities.double().exp()
    if not torch.allclose(probabilities.sum(-1), torch.ones(len(actions), dtype=torch.float64,
                                                          device=actions.device), atol=1e-6, rtol=1e-6):
        raise ValueError('Policy update check requires normalized all-action log probabilities')
    gathered = log_probabilities.gather(1, actions[:, None]).squeeze(1)
    if not torch.allclose(chosen, gathered, atol=1e-6, rtol=1e-6):
        raise ValueError('Selected log probability disagrees with the all-action distribution')
    if masks is not None:
        if masks.shape != log_probabilities.shape or not masks.gather(1, actions[:, None]).all():
            raise ValueError('Recorded action is forbidden by its cached action_masks')
        if torch.any(probabilities.masked_select(~masks) != 0):
            raise ValueError('Policy assigns probability to forbidden cached actions')
    return chosen, entropy, values, log_probabilities


def check_rollout_likelihood(policy, states, actions, old_log_probs, action_masks=None, *,
                             batch_size=32, tolerance=1e-3, device=None):
    """Recompute all rollout likelihoods using the actual training/gradient path.

    Returns a tiny CPU N-by-actions reference table; observation batches alone
    cross the device boundary. No gradients or optimizer changes are performed.
    """
    validate_update_check_settings(True, tolerance, 1)
    states, actions, masks = _cpu_inputs(states, actions, action_masks, batch_size)
    old = torch.as_tensor(old_log_probs, device='cpu').float()
    if old.shape != actions.shape or not torch.isfinite(old).all():
        raise ValueError('Cached rollout log probabilities must be finite and aligned')
    was_training = policy.training
    saved_buffers = clone_state_to_cpu(dict(policy.named_buffers()))
    references, differences = [], []
    try:
        policy.train(True)
        with torch.enable_grad():
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                chosen, entropy, values, distribution = evaluate_action_distribution(
                    policy, states[start:end].to(_device(policy, device)),
                    actions[start:end].to(_device(policy, device)),
                    masks[start:end].to(_device(policy, device)) if masks is not None else None)
                difference = (chosen.detach().cpu() - old[start:end]).abs()
                differences.append(difference)
                references.append(distribution.detach().cpu())
                maximum = float(difference.max())
                if maximum > tolerance:
                    raise ValueError(f'Rollout likelihood mismatch before optimizer: max_abs_error={maximum:.8g} '
                                     f'> tolerance={tolerance:.8g}; check train/rollout modes, batches and cached masks')
                del chosen, entropy, values, distribution
        if any(not torch.equal(buffer.detach().cpu(), saved_buffers[name])
               for name, buffer in policy.named_buffers()):
            raise ValueError('Training forwards mutate policy buffers; rollout likelihoods are not stationary')
    finally:
        with torch.no_grad():
            for name, buffer in policy.named_buffers():
                buffer.copy_(saved_buffers[name])
        policy.train(was_training)
    errors = torch.cat(differences)
    return {'max_abs_error': float(errors.max()), 'mean_abs_error': float(errors.mean()),
            'num_samples': len(states), 'reference_log_probs': torch.cat(references)}


def exact_policy_kl(policy, states, actions, reference_log_probs, action_masks=None, *, batch_size=32, device=None):
    """All-action KL(reference || current), including only reference-supported actions."""
    states, actions, masks = _cpu_inputs(states, actions, action_masks, batch_size)
    reference = torch.as_tensor(reference_log_probs, device='cpu').double()
    if reference.ndim != 2 or len(reference) != len(states) or torch.isnan(reference).any() or torch.isposinf(reference).any():
        raise ValueError('Invalid fixed reference log probabilities')
    if not torch.allclose(reference.exp().sum(-1), torch.ones(len(states), dtype=torch.float64),
                          atol=1e-6, rtol=1e-6):
        raise ValueError('Fixed reference log probabilities must be normalized')
    was_training = policy.training
    rows = []
    try:
        policy.train(True)
        with torch.enable_grad():
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                chosen, entropy, values, distribution = evaluate_action_distribution(
                    policy, states[start:end].to(_device(policy, device)),
                    actions[start:end].to(_device(policy, device)),
                    masks[start:end].to(_device(policy, device)) if masks is not None else None)
                current = distribution.detach().cpu().double()
                baseline = reference[start:end]
                if current.shape != baseline.shape:
                    raise ValueError('Fixed reference and current action distributions differ in shape')
                probability = baseline.exp()
                terms = torch.where(probability > 0, probability * (baseline - current), 0.)
                divergence = terms.sum(-1)
                if not torch.isfinite(divergence).all():
                    raise FloatingPointError('Nonfinite exact post-update KL')
                rows.append(divergence.clamp_min(0))  # Remove floating-point roundoff below zero.
                del chosen, entropy, values, distribution
    finally:
        policy.train(was_training)
    per_state = torch.cat(rows)
    return {'mean_kl': float(per_state.mean()), 'max_kl': float(per_state.max()),
            'num_samples': len(states), 'per_state_kl': per_state}
