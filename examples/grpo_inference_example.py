"""
GRPO 추론 엔진 사용 예제

이 스크립트는 GRPOInference 클래스를 사용하여
실시간 매매 결정을 내리는 방법을 보여줍니다.
"""

import numpy as np
import torch
from pathlib import Path

from ai_trader.inference import GRPOInference


def main():
    """GRPO 추론 엔진 사용 예제"""
    
    # 모델 경로 설정
    embedding_model_path = "models/embedding/checkpoint_epoch50.pt"
    policy_path = "models/grpo_scalping/checkpoint_step1000000.pt"
    
    # 경로 확인
    if not Path(embedding_model_path).exists():
        print(f"Error: Embedding model not found at {embedding_model_path}")
        print("Please train the embedding model first using:")
        print("  python examples/autoencoder_training_example.py")
        return
    
    if not Path(policy_path).exists():
        print(f"Error: Policy not found at {policy_path}")
        print("Please train the GRPO policy first using:")
        print("  python -m ai_trader.grpo.train_grpo")
        return
    
    # 추론 엔진 초기화
    print("Initializing GRPO inference engine...")
    inference = GRPOInference(
        embedding_model_path=embedding_model_path,
        policy_path=policy_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_torchscript=True,  # TorchScript 컴파일로 속도 향상
        cache_size=1000  # 임베딩 캐시 크기
    )
    print("Inference engine initialized successfully!")
    
    # 예제 1: 단일 시퀀스 예측
    print("\n=== Example 1: Single Sequence Prediction ===")
    
    # 테스트 시퀀스 생성 (실제로는 실시간 데이터를 사용)
    # 형태: (seq_len, input_dim) = (60, 60)
    sequence = np.random.randn(60, 60).astype(np.float32)
    
    # 예측 (결정적 모드)
    action, confidence = inference.predict(sequence, deterministic=True)
    
    action_names = {0: "보유", 1: "매수", 2: "매도"}
    print(f"Action: {action_names[action]} (confidence: {confidence:.4f})")
    
    # 예제 2: 배치 예측
    print("\n=== Example 2: Batch Prediction ===")
    
    # 여러 종목의 시퀀스 생성
    batch_size = 5
    sequences = np.random.randn(batch_size, 60, 60).astype(np.float32)
    
    # 배치 예측
    actions, confidences = inference.predict_batch(sequences, deterministic=True)
    
    print(f"Batch predictions for {batch_size} stocks:")
    for i, (action, confidence) in enumerate(zip(actions, confidences)):
        print(f"  Stock {i+1}: {action_names[action]} (confidence: {confidence:.4f})")
    
    # 예제 3: 확률적 샘플링
    print("\n=== Example 3: Stochastic Sampling ===")
    
    sequence = np.random.randn(60, 60).astype(np.float32)
    
    # 여러 번 예측하여 확률 분포 확인
    print("Running 10 stochastic predictions:")
    action_counts = {0: 0, 1: 0, 2: 0}
    
    for _ in range(10):
        action, confidence = inference.predict(sequence, deterministic=False)
        action_counts[action] += 1
    
    print("Action distribution:")
    for action, count in action_counts.items():
        print(f"  {action_names[action]}: {count}/10")
    
    # 예제 4: 성능 통계
    print("\n=== Example 4: Performance Statistics ===")
    
    # 여러 번 예측하여 성능 측정
    for _ in range(100):
        sequence = np.random.randn(60, 60).astype(np.float32)
        inference.predict(sequence)
    
    stats = inference.get_performance_stats()
    
    print("Inference performance:")
    print(f"  Mean inference time: {stats['mean_inference_time_ms']:.2f} ms")
    print(f"  Median inference time: {stats['median_inference_time_ms']:.2f} ms")
    print(f"  P95 inference time: {stats['p95_inference_time_ms']:.2f} ms")
    print(f"  P99 inference time: {stats['p99_inference_time_ms']:.2f} ms")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    
    # 목표 지연 시간 확인
    if stats['mean_inference_time_ms'] < 10:
        print("\n✓ Target latency (<10ms) achieved!")
    else:
        print(f"\n⚠ Target latency not achieved (mean: {stats['mean_inference_time_ms']:.2f}ms)")
    
    # 예제 5: 캐시 관리
    print("\n=== Example 5: Cache Management ===")
    
    print(f"Current cache size: {len(inference.embedding_cache)}")
    
    # 캐시 초기화
    inference.clear_cache()
    print(f"Cache cleared. New size: {len(inference.embedding_cache)}")
    
    # 성능 통계 초기화
    inference.reset_performance_stats()
    print("Performance statistics reset.")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
