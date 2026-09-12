"""One TF32 API per PyTorch version, shared by training and replay diagnostics.

These values describe backend settings, not whether a particular GPU instruction
actually used TF32. Hardware support and environment overrides also matter.
"""
import os
from contextlib import contextmanager

import torch


def _uses_precision_api():
    return hasattr(torch.backends.cuda.matmul, 'fp32_precision')


def _new_api_backends():
    backends = torch.backends
    cudnn = backends.cudnn
    nodes = (backends, backends.cuda.matmul, cudnn,
             getattr(cudnn, 'conv', None), getattr(cudnn, 'rnn', None))
    if any(node is None or not hasattr(node, 'fp32_precision') for node in nodes):
        raise RuntimeError('Incomplete PyTorch fp32_precision API; refusing to mix TF32 APIs')
    return nodes


def set_tf32(enabled: bool) -> dict:
    """Set CUDA matmul and cuDNN TF32 permission; preserve other runtime choices."""
    if not isinstance(enabled, bool):
        raise ValueError('TF32 enabled must be a bool')
    if _uses_precision_api():
        global_backend, matmul, cudnn, convolution, recurrent = _new_api_backends()
        value = 'tf32' if enabled else 'ieee'
        # Explicit operator settings avoid inherited settings from earlier cells.
        global_backend.fp32_precision = 'ieee'
        matmul.fp32_precision = value
        cudnn.fp32_precision = 'ieee'
        convolution.fp32_precision = value
        recurrent.fp32_precision = value
    else:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
    return precision_metadata()


def precision_metadata() -> dict:
    """Read JSON-compatible settings without switching APIs or changing state."""
    if _uses_precision_api():
        global_backend, matmul, cudnn, convolution, recurrent = _new_api_backends()
        settings = dict(
            api='fp32_precision', global_fp32_precision=global_backend.fp32_precision,
            matmul_fp32_precision=matmul.fp32_precision,
            cudnn_fp32_precision=cudnn.fp32_precision,
            cudnn_conv_fp32_precision=convolution.fp32_precision,
            cudnn_rnn_fp32_precision=recurrent.fp32_precision,
            float32_matmul_precision=None,
        )
    else:
        matmul = bool(torch.backends.cuda.matmul.allow_tf32)
        cudnn = bool(torch.backends.cudnn.allow_tf32)
        settings = dict(
            api='allow_tf32', global_fp32_precision=None,
            matmul_fp32_precision='tf32' if matmul else 'ieee',
            cudnn_fp32_precision='tf32' if cudnn else 'ieee',
            cudnn_conv_fp32_precision='tf32' if cudnn else 'ieee',
            cudnn_rnn_fp32_precision='tf32' if cudnn else 'ieee',
            float32_matmul_precision=torch.get_float32_matmul_precision(),
        )
    return dict(
        settings, torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        num_threads=int(torch.get_num_threads()), num_interop_threads=int(torch.get_num_interop_threads()),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        environment_overrides={key: os.environ.get(key) for key in
            ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')},
    )


def snapshot_precision() -> dict:
    """Capture the TF32 settings, including legacy high/medium matmul precision."""
    return precision_metadata()


def restore_precision(snapshot: dict) -> None:
    """Restore a snapshot from this runtime without mixing precision API families."""
    expected_api = 'fp32_precision' if _uses_precision_api() else 'allow_tf32'
    if snapshot.get('api') != expected_api:
        raise ValueError('Cannot restore precision settings from a different PyTorch API')
    if expected_api == 'fp32_precision':
        nodes = _new_api_backends()
        fields = ('global_fp32_precision', 'matmul_fp32_precision', 'cudnn_fp32_precision',
                  'cudnn_conv_fp32_precision', 'cudnn_rnn_fp32_precision')
        values = [snapshot[field] for field in fields]
        for node, value in zip(nodes, values):
            node.fp32_precision = value
    else:
        # Restoring only allow_tf32 would turn the previous 'medium' into 'high'.
        precision = snapshot['float32_matmul_precision']
        cudnn_tf32 = snapshot['cudnn_fp32_precision'] == 'tf32'
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@contextmanager
def preserve_precision():
    """Restore precision on success or failure; other runtime settings are untouched."""
    snapshot = snapshot_precision()
    try:
        yield snapshot
    finally:
        restore_precision(snapshot)
