"""Exercise both precision API families without requiring an Ampere GPU."""
import json
from types import SimpleNamespace

import pytest
import torch

from ai_trader.grpo import runtime_precision


class PrecisionNode:
    def __init__(self, precision='ieee', **extra):
        self.fp32_precision = precision
        self.__dict__.update(extra)

    @property
    def allow_tf32(self):
        raise AssertionError('The new precision API must never read the legacy API')

    @allow_tf32.setter
    def allow_tf32(self, value):
        raise AssertionError('The new precision API must never write the legacy API')


class LegacyMatmul:
    def __init__(self, precision='medium'):
        self.precision = precision

    @property
    def allow_tf32(self):
        return self.precision != 'highest'

    @allow_tf32.setter
    def allow_tf32(self, value):
        self.precision = 'high' if value else 'highest'


def fake_torch(new_api):
    if new_api:
        matmul = PrecisionNode('ieee')
        cudnn = PrecisionNode('tf32', conv=PrecisionNode('ieee'), rnn=PrecisionNode('tf32'),
                              benchmark=True, deterministic=True)
        backends = PrecisionNode('tf32', cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn)

        def forbidden(*args):
            raise AssertionError('Do not mix legacy float32_matmul_precision with the new API')

        get_precision = set_precision = forbidden
    else:
        matmul = LegacyMatmul()
        cudnn = SimpleNamespace(allow_tf32=False, benchmark=True, deterministic=True)
        backends = SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn)
        get_precision = lambda: matmul.precision
        set_precision = lambda value: setattr(matmul, 'precision', value)
    return SimpleNamespace(
        backends=backends, __version__='2.11.fake' if new_api else '2.8.fake',
        version=SimpleNamespace(cuda=None), get_num_threads=lambda: 7,
        get_num_interop_threads=lambda: 2, are_deterministic_algorithms_enabled=lambda: True,
        get_float32_matmul_precision=get_precision, set_float32_matmul_precision=set_precision,
    )


@pytest.mark.parametrize('new_api', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_set_tf32_uses_one_api_and_preserves_other_runtime_settings(monkeypatch, new_api, enabled):
    fake = fake_torch(new_api)
    monkeypatch.setattr(runtime_precision, 'torch', fake)
    metadata = runtime_precision.set_tf32(enabled)
    expected = 'tf32' if enabled else 'ieee'
    assert metadata['api'] == ('fp32_precision' if new_api else 'allow_tf32')
    assert metadata['matmul_fp32_precision'] == expected
    assert metadata['cudnn_conv_fp32_precision'] == metadata['cudnn_rnn_fp32_precision'] == expected
    assert metadata['num_threads'] == 7 and metadata['num_interop_threads'] == 2
    assert metadata['cudnn_benchmark'] and metadata['cudnn_deterministic']
    assert metadata['deterministic_algorithms']
    assert json.loads(json.dumps(metadata)) == metadata


@pytest.mark.parametrize('new_api', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_precision_context_restores_all_precisions_even_on_failure(monkeypatch, new_api, fails):
    monkeypatch.setattr(runtime_precision, 'torch', fake_torch(new_api))
    before = runtime_precision.precision_metadata()
    try:
        with runtime_precision.preserve_precision():
            runtime_precision.set_tf32(False)
            runtime_precision.set_tf32(True)
            if fails:
                raise RuntimeError('diagnostic failed')
    except RuntimeError as error:
        assert fails and str(error) == 'diagnostic failed'
    assert runtime_precision.precision_metadata() == before
    if not new_api:
        assert before['float32_matmul_precision'] == 'medium'


def test_partial_new_api_fails_before_mutation_instead_of_using_legacy(monkeypatch):
    fake = fake_torch(True)
    del fake.backends.cudnn.rnn
    monkeypatch.setattr(runtime_precision, 'torch', fake)
    with pytest.raises(RuntimeError, match='Incomplete'):
        runtime_precision.set_tf32(False)
    assert fake.backends.fp32_precision == 'tf32'
    assert fake.backends.cudnn.fp32_precision == 'tf32'


@pytest.mark.parametrize('value', [None, 0, 1, 'false'])
def test_set_tf32_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match='bool'):
        runtime_precision.set_tf32(value)


def test_snapshot_from_another_api_is_rejected_before_mutation(monkeypatch):
    monkeypatch.setattr(runtime_precision, 'torch', fake_torch(False))
    before = runtime_precision.precision_metadata()
    with pytest.raises(ValueError, match='different PyTorch API'):
        runtime_precision.restore_precision({'api': 'fp32_precision'})
    assert runtime_precision.precision_metadata() == before


def test_metadata_records_environment_overrides_without_applying_them(monkeypatch):
    monkeypatch.setattr(runtime_precision, 'torch', fake_torch(False))
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0')
    monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '1')
    metadata = runtime_precision.precision_metadata()
    assert metadata['environment_overrides'] == {
        'NVIDIA_TF32_OVERRIDE': '0', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1'}
    assert metadata['float32_matmul_precision'] == 'medium'


def test_tf32_off_cpu_forward_preserves_thread_and_precision_settings():
    before = runtime_precision.precision_metadata()
    with runtime_precision.preserve_precision():
        metadata = runtime_precision.set_tf32(False)
        assert metadata['matmul_fp32_precision'] == 'ieee'
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4).requires_grad_()
        (values @ values.T).sum().backward()
        assert torch.isfinite(values.grad).all()
        for key in ('num_threads', 'num_interop_threads', 'cudnn_benchmark',
                    'cudnn_deterministic', 'deterministic_algorithms'):
            assert metadata[key] == before[key]
    assert runtime_precision.precision_metadata() == before
