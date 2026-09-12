"""Portable, bounded evidence for a failed pre-update likelihood check."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .policy_update_checks import RolloutLikelihoodMismatch, clone_state_to_cpu


BUNDLE_FORMAT = 'grpo_rollout_likelihood_failure'
BUNDLE_VERSION = 1
REGROUPING_NOTICE = ('Only the failed update minibatch is saved. Its original rollout batch companions '
                     'and ordering are not preserved. Diagnostic rollout-sized batches regroup these '
                     'saved observations and do not reconstruct the original rollout batches.')


def portable_metadata(value):
    """Keep the bundle compatible with torch.load(weights_only=True)."""
    if value is None or isinstance(value, (bool, str)):
        return str(value) if isinstance(value, str) else value
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError('Failure bundle metadata must be finite')
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('Failure bundle metadata keys must be strings')
        return {str(key): portable_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable_metadata(item) for item in value]
    raise ValueError(f'Unsupported failure bundle metadata type: {type(value).__name__}')


def policy_reconstruction_config(policy):
    base_policy = getattr(policy, 'module', policy)
    keys = ('obs_dim', 'cnn_channels', 'rnn_hidden_dim', 'fc_hidden_dim', 'action_dim',
            'max_stages', 'execution_action_mask')
    kwargs = {name: getattr(base_policy, name) for name in keys if hasattr(base_policy, name)}
    if hasattr(base_policy, 'xlstm') and hasattr(base_policy.xlstm, 'checkpoint_segments'):
        kwargs['checkpoint_segments'] = base_policy.xlstm.checkpoint_segments
    return {'class_name': base_policy.__class__.__name__,
            'module': base_policy.__class__.__module__,
            'kwargs': portable_metadata(kwargs),
            'supported_by_replay_cli': base_policy.__class__.__name__ == 'GRPOPolicyE2EXLSTM'}


def runtime_metadata(policy):
    from .runtime_precision import precision_metadata

    parameter = next(policy.parameters(), None)
    device = parameter.device if parameter is not None else torch.device('cpu')
    metadata = {
        'torch_version': str(torch.__version__), 'cuda_build_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(), 'cuda_available': bool(torch.cuda.is_available()),
        'policy_device': str(device), 'policy_parameter_dtypes': sorted({str(p.dtype) for p in policy.parameters()}),
        'autocast_cuda_enabled': bool(torch.is_autocast_enabled('cuda')),
        'autocast_cpu_enabled': bool(torch.is_autocast_enabled('cpu')),
        'precision': precision_metadata(),
    }
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        metadata.update(gpu_name=properties.name, gpu_compute_capability=[properties.major, properties.minor],
                        gpu_total_memory_bytes=int(properties.total_memory))
    else:
        metadata['gpu_name'] = None
    return portable_metadata(metadata)


def atomic_write(path, writer):
    """Publish a complete artifact; only our temporary file is removed on failure."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{destination.name}.', suffix='.tmp',
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(destination)


def save_likelihood_failure(path, *, policy, failure, observation_schema=None, training_config=None, context=None):
    """Save this policy and the first failed minibatch; never the whole rollout."""
    if not isinstance(failure, RolloutLikelihoodMismatch):
        raise TypeError('Expected a structured RolloutLikelihoodMismatch')
    base_policy = getattr(policy, 'module', policy)
    payload = {
        'format': BUNDLE_FORMAT, 'format_version': BUNDLE_VERSION,
        'saved_at_utc': datetime.now(timezone.utc).isoformat(),
        'policy_state_dict': clone_state_to_cpu(base_policy.state_dict()),
        'policy_config': policy_reconstruction_config(base_policy),
        'observation_schema': portable_metadata(observation_schema),
        'training_config': portable_metadata(training_config or {}),
        'runtime': runtime_metadata(base_policy),
        'context': {**portable_metadata(context or {}), 'original_rollout_batch_reconstructed': False,
                    'batch_reconstruction_notice': REGROUPING_NOTICE},
        'failure': {'message': str(failure), 'offset': failure.offset,
                    'update_batch_size': failure.update_batch_size, 'saved_batch_size': len(failure.batch['states']),
                    'total_rollout_samples': failure.total_samples, 'tolerance': failure.tolerance,
                    'max_abs_error': failure.max_abs_error,
                    'input_dtype': str(failure.batch['states'].dtype)},
        # These tensors already own bounded detached CPU storage. Reusing them
        # avoids an unnecessary second observation-batch copy during serialization.
        'batch': failure.batch,
    }
    return atomic_write(path, lambda handle: torch.save(payload, handle))


def write_json_report(path, report):
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    return atomic_write(path, lambda handle: handle.write(encoded))
