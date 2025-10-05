"""
GRPO 에이전트 평가 예제

이 스크립트는 GRPO 에이전트의 성능을 평가하는 방법을 보여줍니다.
요구사항 7.2에 따라 승률, 거래당 평균 수익, 최대 낙폭, 샤프 비율, 평균 보유 시간을 계산합니다.
"""

import numpy as np
from ai_trader.grpo.evaluation import GRPOEvaluationMetrics, evaluate_grpo_agent


def create_sample_episodes():
    """샘플 에피소드 데이터 생성"""
    np.random.seed(42)
    
    episodes = []
    
    # 10개의 에피소드 생성
    for episode_idx in range(10):
        # 각 에피소드는 5-15개의 거래를 포함
        num_trades = np.random.randint(5, 16)
        
        trades = []
        for trade_idx in range(num_trades):
            # 수익률: 평균 0.005, 표준편차 0.02
            profit_rate = np.random.normal(0.005, 0.02)
            
            # 거래 비용 차감 (왕복 0.43%)
            reward = profit_rate - 0.0043
            
            # 보유 시간: 5-30초
            holding_time = np.random.uniform(5.0, 30.0)
            
            # 진입/청산 가격 (더미 값)
            entry_price = 100.0
            exit_price = entry_price * (1 + profit_rate)
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'holding_time': holding_time,
                'profit_rate': profit_rate,
                'reward': reward
            })
        
        # 에피소드 메타데이터
        total_return = sum(t['reward'] for t in trades)
        
        episode = {
            'trades': trades,
            'total_return': total_return,
            'num_trades': num_trades,
            'episode_reward': total_return,
            'episode_steps': num_trades * 10  # 더미 값
        }
        
        episodes.append(episode)
    
    return episodes


def example_basic_evaluation():
    """기본 평가 예제"""
    print("\n" + "=" * 70)
    print("예제 1: 기본 평가")
    print("=" * 70)
    
    # 샘플 에피소드 생성
    episodes = create_sample_episodes()
    
    # 평가 수행 (결과 출력)
    metrics = evaluate_grpo_agent(episodes, print_results=True)
    
    # 개별 메트릭 접근
    print(f"\n개별 메트릭 접근:")
    print(f"  - 승률이 50% 이상인가? {metrics['win_rate'] >= 0.5}")
    print(f"  - 샤프 비율이 1.0 이상인가? {metrics['sharpe_ratio'] >= 1.0}")
    print(f"  - 최대 낙폭이 10% 미만인가? {metrics['max_drawdown'] < 0.1}")
    print(f"  - 평균 보유 시간이 30초 미만인가? {metrics['avg_holding_time'] < 30.0}")


def example_incremental_evaluation():
    """점진적 평가 예제 (에피소드를 하나씩 추가)"""
    print("\n" + "=" * 70)
    print("예제 2: 점진적 평가")
    print("=" * 70)
    
    # 평가기 초기화
    evaluator = GRPOEvaluationMetrics()
    
    # 에피소드를 하나씩 추가하면서 평가
    episodes = create_sample_episodes()
    
    for i, episode in enumerate(episodes[:5], 1):
        evaluator.add_episode(episode)
        
        # 중간 평가
        metrics = evaluator.compute_metrics()
        
        print(f"\n에피소드 {i} 추가 후:")
        print(f"  총 에피소드: {metrics['total_episodes']}")
        print(f"  총 거래: {metrics['total_trades']}")
        print(f"  승률: {metrics['win_rate']:.2%}")
        print(f"  평균 수익: {metrics['avg_profit_per_trade']:.4f}")


def example_comparison():
    """여러 에이전트 비교 예제"""
    print("\n" + "=" * 70)
    print("예제 3: 여러 에이전트 비교")
    print("=" * 70)
    
    # 에이전트 A: 보수적 전략 (낮은 수익, 낮은 위험)
    np.random.seed(42)
    episodes_a = []
    for _ in range(10):
        trades = []
        for _ in range(np.random.randint(3, 8)):
            profit_rate = np.random.normal(0.003, 0.01)  # 낮은 변동성
            reward = profit_rate - 0.0043
            holding_time = np.random.uniform(10.0, 20.0)
            
            trades.append({
                'reward': reward,
                'holding_time': holding_time,
                'profit_rate': profit_rate
            })
        
        episodes_a.append({
            'trades': trades,
            'total_return': sum(t['reward'] for t in trades),
            'num_trades': len(trades)
        })
    
    # 에이전트 B: 공격적 전략 (높은 수익, 높은 위험)
    np.random.seed(43)
    episodes_b = []
    for _ in range(10):
        trades = []
        for _ in range(np.random.randint(8, 16)):
            profit_rate = np.random.normal(0.008, 0.03)  # 높은 변동성
            reward = profit_rate - 0.0043
            holding_time = np.random.uniform(5.0, 15.0)
            
            trades.append({
                'reward': reward,
                'holding_time': holding_time,
                'profit_rate': profit_rate
            })
        
        episodes_b.append({
            'trades': trades,
            'total_return': sum(t['reward'] for t in trades),
            'num_trades': len(trades)
        })
    
    # 평가
    print("\n에이전트 A (보수적 전략):")
    metrics_a = evaluate_grpo_agent(episodes_a, print_results=False)
    
    print("\n에이전트 B (공격적 전략):")
    metrics_b = evaluate_grpo_agent(episodes_b, print_results=False)
    
    # 비교
    print("\n" + "=" * 70)
    print("비교 결과:")
    print("=" * 70)
    print(f"{'메트릭':<25} {'에이전트 A':>15} {'에이전트 B':>15} {'우수':>10}")
    print("-" * 70)
    
    comparisons = [
        ('승률', 'win_rate', '{:.2%}', 'higher'),
        ('평균 수익', 'avg_profit_per_trade', '{:.4f}', 'higher'),
        ('최대 낙폭', 'max_drawdown', '{:.2%}', 'lower'),
        ('샤프 비율', 'sharpe_ratio', '{:.4f}', 'higher'),
        ('평균 보유 시간', 'avg_holding_time', '{:.2f}s', 'lower'),
        ('총 거래', 'total_trades', '{:.0f}', 'neutral'),
    ]
    
    for label, key, fmt, better in comparisons:
        val_a = metrics_a[key]
        val_b = metrics_b[key]
        
        if better == 'higher':
            winner = 'A' if val_a > val_b else 'B' if val_b > val_a else '-'
        elif better == 'lower':
            winner = 'A' if val_a < val_b else 'B' if val_b < val_a else '-'
        else:
            winner = '-'
        
        print(f"{label:<25} {fmt.format(val_a):>15} {fmt.format(val_b):>15} {winner:>10}")


def example_performance_thresholds():
    """성능 임계값 체크 예제"""
    print("\n" + "=" * 70)
    print("예제 4: 성능 임계값 체크")
    print("=" * 70)
    
    # 샘플 에피소드 생성
    episodes = create_sample_episodes()
    
    # 평가
    metrics = evaluate_grpo_agent(episodes, print_results=False)
    
    # 성능 임계값 정의 (요구사항 7.2 참고)
    thresholds = {
        'win_rate': (0.50, '>='),  # 승률 > 50%
        'sharpe_ratio': (1.0, '>='),  # 샤프 비율 > 1.0
        'max_drawdown': (0.10, '<'),  # 최대 낙폭 < 10%
        'avg_holding_time': (30.0, '<'),  # 평균 보유 시간 < 30초
    }
    
    print("\n성능 임계값 체크:")
    print("-" * 70)
    
    all_passed = True
    
    for metric_name, (threshold, operator) in thresholds.items():
        value = metrics[metric_name]
        
        if operator == '>=':
            passed = value >= threshold
            comparison = f"{value:.4f} >= {threshold}"
        elif operator == '<':
            passed = value < threshold
            comparison = f"{value:.4f} < {threshold}"
        else:
            passed = False
            comparison = "Unknown operator"
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{metric_name:<25} {comparison:<20} {status}")
        
        if not passed:
            all_passed = False
    
    print("-" * 70)
    if all_passed:
        print("✓ 모든 성능 임계값을 통과했습니다!")
    else:
        print("✗ 일부 성능 임계값을 통과하지 못했습니다.")


def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("GRPO 에이전트 평가 예제")
    print("=" * 70)
    
    # 예제 실행
    example_basic_evaluation()
    example_incremental_evaluation()
    example_comparison()
    example_performance_thresholds()
    
    print("\n" + "=" * 70)
    print("모든 예제 완료!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
