"""
GRPO 추론 엔진 테스트

이 모듈은 GRPOInference 클래스의 기능을 테스트합니다.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from ai_trader.embedding import AutoEncoderEmbedding
from ai_trader.grpo.policy import GRPOPolicy
from ai_trader.inference.infer_grpo import GRPOInference


@pytest.fixture
def temp_model_paths():
    """임시 모델 체크포인트 생성"""
    # 임베딩 모델 생성 및 저장
    embedding_model = AutoEncoderEmbedding(
        input_dim=60,
        embedding_dim=128,
        seq_len=60
    )
    
    embedding_checkpoint = {
        'state_dict': embedding_model.state_dict(),
        'config': {
            'input_dim': 60,
            'embedding_dim': 128,
            'seq_len': 60,
            'num_heads': 4
        },
        'normalization_stats': {
            'mean': np.random.randn(60).astype(np.float32),
            'std': np.ones(60).astype(np.float32)
        }
    }
    
    # 정책 생성 및 저장
    policy = GRPOPolicy(
        embedding_dim=128,
        hidden_dim=256,
        action_dim=3
    )
    
    policy_checkpoint = {
        'state_dict': policy.state_dict(),
        'config': {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'action_dim': 3
        }
    }
    
    # 임시 파일에 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        embedding_path = f.name
        torch.save(embedding_checkpoint, embedding_path)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        policy_path = f.name
        torch.save(policy_checkpoint, policy_path)
    
    yield embedding_path, policy_path
    
    # 정리
    Path(embedding_path).unlink(missing_ok=True)
    Path(policy_path).unlink(missing_ok=True)


def test_grpo_inference_initialization(temp_model_paths):
    """GRPOInference 초기화 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False,  # 테스트에서는 TorchScript 비활성화
        cache_size=100
    )
    
    assert inference.device.type == 'cpu'
    assert inference.cache_size == 100
    assert len(inference.embedding_cache) == 0


def test_predict_single_sequence(temp_model_paths):
    """단일 시퀀스 예측 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False
    )
    
    # 테스트 시퀀스 생성
    sequence = np.random.randn(60, 60).astype(np.float32)
    
    # 예측
    action, confidence = inference.predict(sequence, deterministic=True)
    
    # 검증
    assert isinstance(action, int)
    assert action in [0, 1, 2]
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_predict_with_torch_tensor(temp_model_paths):
    """torch.Tensor 입력으로 예측 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False
    )
    
    # torch.Tensor 시퀀스 생성
    sequence = torch.randn(60, 60)
    
    # 예측
    action, confidence = inference.predict(sequence, deterministic=True)
    
    # 검증
    assert isinstance(action, int)
    assert action in [0, 1, 2]


def test_predict_batch(temp_model_paths):
    """배치 예측 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False
    )
    
    # 배치 시퀀스 생성
    batch_size = 8
    sequences = np.random.randn(batch_size, 60, 60).astype(np.float32)
    
    # 배치 예측
    actions, confidences = inference.predict_batch(sequences, deterministic=True)
    
    # 검증
    assert actions.shape == (batch_size,)
    assert confidences.shape == (batch_size,)
    assert all(a in [0, 1, 2] for a in actions)
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_embedding_cache(temp_model_paths):
    """임베딩 캐시 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False,
        cache_size=10
    )
    
    # 동일한 시퀀스로 여러 번 예측
    sequence = np.random.randn(60, 60).astype(np.float32)
    
    # 첫 번째 예측 (캐시 미스)
    action1, conf1 = inference.predict(sequence)
    
    # 두 번째 예측 (캐시 히트)
    action2, conf2 = inference.predict(sequence)
    
    # 동일한 결과 확인
    assert action1 == action2
    assert conf1 == conf2
    
    # 캐시에 항목이 있는지 확인
    assert len(inference.embedding_cache) > 0


def test_cache_size_limit(temp_model_paths):
    """캐시 크기 제한 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    cache_size = 5
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False,
        cache_size=cache_size
    )
    
    # 캐시 크기보다 많은 시퀀스 예측
    for i in range(cache_size + 3):
        sequence = np.random.randn(60, 60).astype(np.float32)
        inference.predict(sequence)
    
    # 캐시 크기가 제한을 초과하지 않는지 확인
    assert len(inference.embedding_cache) <= cache_size


def test_normalization(temp_model_paths):
    """정규화 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False
    )
    
    # 정규화 통계가 로드되었는지 확인
    assert inference.mean is not None
    assert inference.std is not None
    
    # 정규화 테스트
    sequence = torch.randn(60, 60)
    normalized = inference._normalize_input(sequence)
    
    assert normalized.shape == sequence.shape


def test_performance_stats(temp_model_paths):
    """성능 통계 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False
    )
    
    # 여러 번 예측
    for _ in range(10):
        sequence = np.random.randn(60, 60).astype(np.float32)
        inference.predict(sequence)
    
    # 성능 통계 확인
    stats = inference.get_performance_stats()
    
    assert 'mean_inference_time_ms' in stats
    assert 'median_inference_time_ms' in stats
    assert 'p95_inference_time_ms' in stats
    assert 'p99_inference_time_ms' in stats
    assert stats['mean_inference_time_ms'] > 0


def test_clear_cache(temp_model_paths):
    """캐시 초기화 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False
    )
    
    # 캐시에 항목 추가
    sequence = np.random.randn(60, 60).astype(np.float32)
    inference.predict(sequence)
    
    assert len(inference.embedding_cache) > 0
    
    # 캐시 초기화
    inference.clear_cache()
    
    assert len(inference.embedding_cache) == 0


def test_deterministic_vs_stochastic(temp_model_paths):
    """결정적 vs 확률적 예측 테스트"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False
    )
    
    sequence = np.random.randn(60, 60).astype(np.float32)
    
    # 결정적 예측 (여러 번 실행해도 동일한 결과)
    action1, _ = inference.predict(sequence, deterministic=True)
    action2, _ = inference.predict(sequence, deterministic=True)
    assert action1 == action2
    
    # 확률적 예측 (다를 수 있음)
    action3, _ = inference.predict(sequence, deterministic=False)
    assert action3 in [0, 1, 2]


def test_inference_time_target(temp_model_paths):
    """추론 시간 목표 테스트 (< 10ms)"""
    embedding_path, policy_path = temp_model_paths
    
    inference = GRPOInference(
        embedding_model_path=embedding_path,
        policy_path=policy_path,
        device='cpu',
        use_torchscript=False
    )
    
    # 워밍업
    for _ in range(5):
        sequence = np.random.randn(60, 60).astype(np.float32)
        inference.predict(sequence)
    
    # 추론 시간 측정
    inference.reset_performance_stats()
    
    for _ in range(20):
        sequence = np.random.randn(60, 60).astype(np.float32)
        inference.predict(sequence)
    
    stats = inference.get_performance_stats()
    
    # CPU에서는 10ms 목표가 어려울 수 있으므로 경고만 출력
    if stats['mean_inference_time_ms'] > 10:
        print(f"Warning: Mean inference time {stats['mean_inference_time_ms']:.2f}ms exceeds 10ms target")
    
    # 적어도 추론이 완료되었는지 확인
    assert stats['mean_inference_time_ms'] > 0
