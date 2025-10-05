"""
GRPO 추론 예제

이 스크립트는 GRPOInference 클래스의 predict() 메서드 사용법을 보여줍니다.
"""

import numpy as np
import torch
from pathlib import Path

from ai_trader.inference.infer_grpo import GRPOInference


def main():
    """GRPO 추론 예제 실행"""
    
    # 모델 경로 설정
    embedding_model_path = "models/embedding/checkpoint_epoch50.pt"
    policy_path = "models/grpo_scalping/checkpoint_step1000000.pt"
    
    # 경로 확인
    if not Path(embedding_model_path).exists():
        print(f"Error: Embedding model not found at {embedding_model_path}")
        print("Please train the embedding model first using:")
        print("  python -m ai_trader.embedding.train_embedding")
        return
    
    if not Path(policy_path).exists():
        print(f"Error: Policy not found at {policy_path}")
        print("Please train the GRPO policy first using:")
        print("  python -m ai_trader.grpo.train_grpo")
        return
    
    # GRPOInference 초기화
    print("Initializing GRPOInference...")
    inference = GRPOInference(
        embedding_model_path=embedding_model_path,
        policy_path=policy_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_torchscript=True,
        cache_size=1000
    )
    print("Initialization complete!")
    
    # 예제 1: 단일 시퀀스 예측 (numpy array)
    print("\n=== Example 1: Single sequence prediction (numpy) ===")
    sequence_np = np.random.randn(60, 60).astype(np.float32)
    
    action, confidence = inference.predict(sequence_np, deterministic=True)
    
    action_names = {0: "보유 (Hold)", 1: "매수 (Buy)", 2: "매도 (Sell)"}
    print(f"Action: {action} ({action_names[action]})")
    print(f"Confidence: {confidence:.4f}")
    
    # 예제 2: 단일 시퀀스 예측 (torch tensor)
    print("\n=== Example 2: Single sequence prediction (torch) ===")
    sequence_torch = torch.randn(60, 60)
    
    action, confidence = inference.predict(sequence_torch, deterministic=True)
    print(f"Action: {action} ({action_names[action]})")
    print(f"Confidence: {confidence:.4f}")
    
    # 예제 3: 확률적 예측
    print("\n=== Example 3: Stochastic prediction ===")
    for i in range(5):
        action, confidence = inference.predict(sequence_np, deterministic=False)
        print(f"  Trial {i+1}: Action={action} ({action_names[action]}), Confidence={confidence:.4f}")
    
    # 예제 4: 배치 예측
    print("\n=== Example 4: Batch prediction ===")
    batch_size = 8
    sequences_batch = np.random.randn(batch_size, 60, 60).astype(np.float32)
    
    actions, confidences = inference.predict_batch(sequences_batch, deterministic=True)
    
    print(f"Batch size: {batch_size}")
    for i in range(batch_size):
        print(f"  Sample {i+1}: Action={actions[i]} ({action_names[actions[i]]}), Confidence={confidences[i]:.4f}")
    
    # 예제 5: 캐시 효과 확인
    print("\n=== Example 5: Cache performance ===")
    
    # 동일한 시퀀스로 여러 번 예측
    print("Predicting same sequence 10 times...")
    for _ in range(10):
        inference.predict(sequence_np, deterministic=True)
    
    stats = inference.get_performance_stats()
    print(f"Mean inference time: {stats['mean_inference_time_ms']:.2f} ms")
    print(f"Median inference time: {stats['median_inference_time_ms']:.2f} ms")
    print(f"P95 inference time: {stats['p95_inference_time_ms']:.2f} ms")
    print(f"P99 inference time: {stats['p99_inference_time_ms']:.2f} ms")
    print(f"Cache size: {stats['cache_size']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    
    # 예제 6: 성능 벤치마크
    print("\n=== Example 6: Performance benchmark ===")
    inference.reset_performance_stats()
    
    num_predictions = 100
    print(f"Running {num_predictions} predictions...")
    
    for _ in range(num_predictions):
        sequence = np.random.randn(60, 60).astype(np.float32)
        inference.predict(sequence, deterministic=True)
    
    stats = inference.get_performance_stats()
    print(f"Mean inference time: {stats['mean_inference_time_ms']:.2f} ms")
    print(f"Median inference time: {stats['median_inference_time_ms']:.2f} ms")
    print(f"P95 inference time: {stats['p95_inference_time_ms']:.2f} ms")
    print(f"P99 inference time: {stats['p99_inference_time_ms']:.2f} ms")
    
    # 목표 지연 시간 확인
    if stats['mean_inference_time_ms'] < 10:
        print(f"✓ Target latency achieved! ({stats['mean_inference_time_ms']:.2f} ms < 10 ms)")
    else:
        print(f"✗ Target latency not achieved ({stats['mean_inference_time_ms']:.2f} ms > 10 ms)")
        print("  Consider using GPU or TorchScript compilation for better performance")


if __name__ == "__main__":
    main()
