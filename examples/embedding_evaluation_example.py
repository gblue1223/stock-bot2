"""
임베딩 평가 메트릭 사용 예제

이 예제는 임베딩 모델의 품질을 평가하는 방법을 보여줍니다.
"""

import numpy as np
from ai_trader.embedding.evaluation import (
    compute_silhouette_score,
    compute_temporal_coherence,
    evaluate_embedding_quality
)


def example_silhouette_score():
    """Silhouette score 계산 예제"""
    print("=" * 80)
    print("Silhouette Score 예제")
    print("=" * 80)
    
    # 두 개의 잘 분리된 클러스터 생성
    cluster1 = np.random.randn(100, 128) + np.array([5, 0] + [0] * 126)
    cluster2 = np.random.randn(100, 128) + np.array([-5, 0] + [0] * 126)
    embeddings = np.vstack([cluster1, cluster2])
    
    # 종목 코드 (삼성전자, SK하이닉스)
    stock_codes = ['005930'] * 100 + ['000660'] * 100
    
    # Silhouette score 계산
    score = compute_silhouette_score(embeddings, stock_codes)
    
    print(f"임베딩 개수: {len(embeddings)}")
    print(f"종목 수: {len(set(stock_codes))}")
    print(f"Silhouette Score: {score:.4f}")
    print()
    
    # 해석
    if score > 0.5:
        print("✓ 클러스터가 잘 분리되어 있습니다 (좋음)")
    elif score > 0.25:
        print("○ 클러스터가 어느 정도 분리되어 있습니다 (보통)")
    else:
        print("✗ 클러스터가 겹쳐있습니다 (개선 필요)")
    
    print()


def example_temporal_coherence():
    """Temporal coherence 계산 예제"""
    print("=" * 80)
    print("Temporal Coherence 예제")
    print("=" * 80)
    
    # 시간에 따라 천천히 변하는 임베딩 생성
    embeddings = []
    for i in range(200):
        # 천천히 변하는 패턴
        emb = np.sin(i / 20) * np.ones(128) + np.random.randn(128) * 0.1
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    
    # 타임스탬프 (초 단위)
    timestamps = list(range(200))
    
    # Temporal coherence 계산 (다양한 윈도우 크기)
    print(f"임베딩 개수: {len(embeddings)}")
    print()
    
    for window_size in [1, 5, 10]:
        score = compute_temporal_coherence(embeddings, timestamps, window_size)
        print(f"Window size {window_size}: {score:.4f}")
    
    print()
    
    # 해석
    score = compute_temporal_coherence(embeddings, timestamps, window_size=1)
    if score > 0.8:
        print("✓ 시간적으로 매우 일관성이 있습니다 (좋음)")
    elif score > 0.6:
        print("○ 시간적으로 어느 정도 일관성이 있습니다 (보통)")
    else:
        print("✗ 시간적 일관성이 부족합니다 (개선 필요)")
    
    print()


def example_comprehensive_evaluation():
    """종합 평가 예제"""
    print("=" * 80)
    print("종합 평가 예제")
    print("=" * 80)
    
    # 실제 시나리오를 시뮬레이션
    # 3개 종목, 각 100개 샘플
    embeddings = []
    stock_codes = []
    timestamps = []
    
    for stock_idx, stock_code in enumerate(['005930', '000660', '035420']):
        for t in range(100):
            # 종목별로 다른 중심을 가지되, 시간에 따라 변하는 임베딩
            center = np.array([stock_idx * 3, 0] + [0] * 126)
            temporal_shift = np.sin(t / 10) * 0.5
            noise = np.random.randn(128) * 0.2
            
            emb = center + temporal_shift + noise
            embeddings.append(emb)
            stock_codes.append(stock_code)
            timestamps.append(stock_idx * 100 + t)
    
    embeddings = np.array(embeddings)
    
    # 종합 평가
    metrics = evaluate_embedding_quality(
        embeddings=embeddings,
        stock_codes=stock_codes,
        timestamps=timestamps,
        window_size=1
    )
    
    # 결과 출력
    print(f"샘플 수: {metrics['num_samples']}")
    print(f"종목 수: {metrics['num_stocks']}")
    print(f"임베딩 차원: {metrics['embedding_dim']}")
    print()
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Temporal Coherence: {metrics['temporal_coherence']:.4f}")
    print()
    
    # 종합 평가
    silhouette_ok = metrics['silhouette_score'] > 0.3
    temporal_ok = metrics['temporal_coherence'] > 0.6
    
    if silhouette_ok and temporal_ok:
        print("✓ 임베딩 품질이 우수합니다!")
    elif silhouette_ok or temporal_ok:
        print("○ 임베딩 품질이 보통입니다. 일부 개선이 필요합니다.")
    else:
        print("✗ 임베딩 품질이 낮습니다. 모델 재훈련을 고려하세요.")
    
    print()


def example_real_world_usage():
    """실제 사용 시나리오 예제"""
    print("=" * 80)
    print("실제 사용 시나리오")
    print("=" * 80)
    
    print("""
실제 사용 방법:

1. 훈련 중 검증:
   - 매 에포크마다 검증 세트에서 메트릭 계산
   - TensorBoard에 로깅하여 훈련 진행 상황 모니터링
   
2. 모델 선택:
   - 여러 체크포인트 중 최고 성능 모델 선택
   - Silhouette score와 Temporal coherence를 모두 고려
   
3. 최종 평가:
   - 테스트 세트에서 최종 성능 평가
   - 프로덕션 배포 전 품질 확인

평가 기준:
- Silhouette Score > 0.3: 종목별 클러스터링이 양호
- Temporal Coherence > 0.6: 시간적 일관성이 양호
- 두 메트릭 모두 만족 시 프로덕션 배포 가능

CLI 사용 예제:
    python -m ai_trader.embedding.evaluate_embedding \\
        --checkpoint models/embedding/checkpoint_epoch50.pt \\
        --db "datasets_norm_all.duckdb" \\
        --table datasets \\
        --split test \\
        --device cuda \\
        --output evaluation_results.json
    """)


def main():
    """메인 함수"""
    print("\n")
    print("*" * 80)
    print("임베딩 평가 메트릭 사용 예제")
    print("*" * 80)
    print("\n")
    
    # 예제 실행
    example_silhouette_score()
    example_temporal_coherence()
    example_comprehensive_evaluation()
    example_real_world_usage()
    
    print("*" * 80)
    print("예제 완료!")
    print("*" * 80)


if __name__ == '__main__':
    main()
