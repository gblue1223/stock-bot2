"""Recheck a saved likelihood failure under TF32 on/off without training.

Example: python -m ai_trader.grpo.diagnose_likelihood --bundle diagnostics/failure.pt
         --output diagnostics/replay.json --device cuda
"""
import argparse
from pathlib import Path

import torch

from .likelihood_failure import (BUNDLE_FORMAT, BUNDLE_VERSION, REGROUPING_NOTICE,
                                  runtime_metadata, write_json_report)
from .policy_update_checks import evaluate_action_distribution


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _load_bundle(path):
    # The CLI never imports serialized objects or falls back to unrestricted pickle.
    bundle = torch.load(Path(path).resolve(), map_location='cpu', weights_only=True)
    if (not isinstance(bundle, dict) or bundle.get('format') != BUNDLE_FORMAT
            or bundle.get('format_version') != BUNDLE_VERSION):
        raise ValueError('Unsupported likelihood failure bundle format/version')
    batch, failure = bundle.get('batch'), bundle.get('failure')
    if not isinstance(batch, dict) or not isinstance(failure, dict):
        raise ValueError('Failure bundle is missing its batch or failure metadata')
    states = batch.get('states')
    if (not isinstance(states, torch.Tensor) or states.dtype != torch.float32
            or states.ndim != 3 or any(size < 1 for size in states.shape) or not torch.isfinite(states).all()):
        raise ValueError('Saved observations must be finite nonempty float32 N-by-sequence-by-feature tensors')
    size = len(states)
    actions = batch.get('actions')
    if not isinstance(actions, torch.Tensor) or actions.dtype != torch.int64 or actions.shape != (size,):
        raise ValueError('Saved actions must be an aligned int64 tensor')
    for name in ('old_log_probs', 'recomputed_log_probs'):
        value = batch.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (size,) or not torch.isfinite(value).all():
            raise ValueError(f'Saved {name} must be finite and aligned')
    masks = batch.get('action_masks')
    if masks is not None and (not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
                              or masks.ndim != 2 or len(masks) != size or not masks.any(-1).all()):
        raise ValueError('Saved action masks must be aligned boolean tensors')
    if failure.get('saved_batch_size') != size or size > _positive_integer('update_batch_size', failure.get('update_batch_size')):
        raise ValueError('Saved batch exceeds or contradicts its failed update batch size')
    return bundle


def _reconstruct_policy(bundle, device):
    from .policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM

    config = bundle.get('policy_config')
    if (not isinstance(config, dict) or config.get('class_name') != 'GRPOPolicyE2EXLSTM'
            or config.get('module') != GRPOPolicyE2EXLSTM.__module__):
        raise ValueError('Replay CLI supports only the repository GRPOPolicyE2EXLSTM policy')
    kwargs = config.get('kwargs')
    expected = {'obs_dim', 'cnn_channels', 'rnn_hidden_dim', 'fc_hidden_dim', 'action_dim',
                'checkpoint_segments', 'max_stages', 'execution_action_mask'}
    if not isinstance(kwargs, dict) or set(kwargs) != expected:
        raise ValueError('Incomplete or unsupported policy reconstruction configuration')
    for name in expected - {'execution_action_mask'}:
        _positive_integer(name, kwargs[name])
    if not isinstance(kwargs['execution_action_mask'], bool):
        raise ValueError('Saved execution_action_mask must be boolean')
    if kwargs['obs_dim'] != bundle['batch']['states'].shape[-1]:
        raise ValueError('Saved observation width does not match the policy')
    weights = bundle.get('policy_state_dict')
    if (not isinstance(weights, dict) or not weights
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                   for value in weights.values())):
        raise ValueError('Saved policy weights must be a finite tensor state dictionary')
    # Constructor initialization is discarded immediately; preserve the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        policy = GRPOPolicyE2EXLSTM(**kwargs)
    policy.load_state_dict(weights, strict=True)
    return policy.to(device)


def _chosen_log_probs(policy, batch, *, training, batch_size, device):
    policy.train(training)
    results = []
    with torch.set_grad_enabled(training):
        for offset in range(0, len(batch['states']), batch_size):
            end = offset + batch_size
            masks = batch['action_masks']
            chosen, entropy, values, distribution = evaluate_action_distribution(
                policy, batch['states'][offset:end].to(device), batch['actions'][offset:end].to(device),
                masks[offset:end].to(device) if masks is not None else None)
            results.append(chosen.detach().cpu())
            del chosen, entropy, values, distribution
    return torch.cat(results)


def _difference(left, right, tolerance):
    errors = (left.double() - right.double()).abs()
    return {'max_abs_error': float(errors.max()), 'mean_abs_error': float(errors.mean()),
            'samples_over_tolerance': int((errors > tolerance).sum()),
            'within_tolerance': bool(torch.all(errors <= tolerance))}


def run_diagnostics(bundle_path, *, device='auto', rollout_batch_size=None, update_batch_size=None):
    from .runtime_precision import precision_metadata, restore_precision, set_tf32

    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    selected_device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if selected_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu')
    bundle = _load_bundle(bundle_path)
    policy = _reconstruct_policy(bundle, selected_device)
    batch = bundle['batch']
    saved_context = bundle.get('context', {})
    if not isinstance(saved_context, dict):
        raise ValueError('Invalid saved context')
    rollout_source = 'explicit_cli' if rollout_batch_size is not None else 'saved_context'
    if rollout_batch_size is None:
        rollout_batch_size = saved_context.get('rollout_batch_size')
        if rollout_batch_size is None:
            rollout_batch_size = min(8, len(batch['states']))
            rollout_source = 'assumed_default_not_recorded'
    if update_batch_size is None:
        update_batch_size = bundle['failure']['update_batch_size']
    rollout_batch_size = _positive_integer('rollout_batch_size', rollout_batch_size)
    update_batch_size = _positive_integer('update_batch_size', update_batch_size)
    tolerance = bundle['failure'].get('tolerance')
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not torch.isfinite(torch.tensor(tolerance)) or tolerance < 0):
        raise ValueError('Saved likelihood tolerance must be finite and nonnegative')
    original_precision = precision_metadata()
    report = {
        'bundle': str(Path(bundle_path).resolve()), 'device': selected_device,
        'original_rollout_batch_reconstructed': False, 'batch_reconstruction_notice': REGROUPING_NOTICE,
        'sample_count': len(batch['states']), 'rollout_batch_size': rollout_batch_size,
        'rollout_batch_size_source': rollout_source, 'update_batch_size': update_batch_size,
        'saved_failure': bundle['failure'], 'saved_runtime': bundle.get('runtime'),
        'saved_context': saved_context, 'diagnostic_runtime': runtime_metadata(policy),
        'tf32_hardware_effect_testable': selected_device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8,
        'limitation': ('CPU cannot reproduce CUDA TF32 effects.' if selected_device == 'cpu' else
                       'TF32 permission may be ignored by hardware or environment overrides; '
                       'these regrouped saved observations do not recreate the original rollout batches.'),
        'variants': [], 'comparisons': {},
    }
    outputs = {}
    try:
        for enabled in (True, False):
            set_tf32(enabled)
            for mode, training, size in (('eval_no_grad', False, rollout_batch_size),
                                         ('train_grad', True, update_batch_size)):
                label = f'tf32_{"on" if enabled else "off"}/{mode}'
                variant = {'label': label, 'tf32_requested': enabled, 'mode': mode,
                           'gradient_enabled': training, 'batch_size': size, 'precision': precision_metadata()}
                # Restore the identical saved weights and buffers before every pass.
                policy.load_state_dict(bundle['policy_state_dict'], strict=True)
                try:
                    logs = _chosen_log_probs(policy, batch, training=training, batch_size=size, device=selected_device)
                    outputs[label] = logs
                    variant.update(status='ok', recomputed_log_probs=logs.tolist(),
                                   vs_saved_rollout=_difference(logs, batch['old_log_probs'], tolerance),
                                   vs_saved_update=_difference(logs, batch['recomputed_log_probs'], tolerance))
                except Exception as error:
                    variant.update(status='error', error_type=type(error).__name__, error=str(error))
                report['variants'].append(variant)
        for enabled in ('on', 'off'):
            left, right = f'tf32_{enabled}/eval_no_grad', f'tf32_{enabled}/train_grad'
            if left in outputs and right in outputs:
                report['comparisons'][f'tf32_{enabled}_eval_vs_train'] = _difference(outputs[left], outputs[right], tolerance)
        for mode in ('eval_no_grad', 'train_grad'):
            left, right = f'tf32_on/{mode}', f'tf32_off/{mode}'
            if left in outputs and right in outputs:
                report['comparisons'][f'{mode}_tf32_on_vs_off'] = _difference(outputs[left], outputs[right], tolerance)
    finally:
        restore_precision(original_precision)
    report['precision_restored'] = precision_metadata() == original_precision
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, help='Structured likelihood failure .pt bundle')
    parser.add_argument('--output', required=True, help='JSON diagnostic report destination')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--rollout-batch-size', type=int)
    parser.add_argument('--update-batch-size', type=int)
    args = parser.parse_args(argv)
    if Path(args.bundle).resolve() == Path(args.output).resolve():
        raise ValueError('--output must differ from the input --bundle path')
    report = run_diagnostics(args.bundle, device=args.device, rollout_batch_size=args.rollout_batch_size,
                             update_batch_size=args.update_batch_size)
    destination = write_json_report(args.output, report)
    print(f'Likelihood replay report saved: {destination}')
    print(REGROUPING_NOTICE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
