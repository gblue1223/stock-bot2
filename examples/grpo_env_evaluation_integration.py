"""
GRPO 환경과 평가 메트릭 통합 예제

이 스크립트는 GRPO 환경에서 에피소드를 실행하고
평가 메트릭을 계산하는 방법을 보여줍니다.
"""

import numpy as np
import torch
import torch.nn as nn
from ai_trader.grpo import GRPOScalpingEnv, GRPOEvaluationMetrics


class DummyEmbeddingModel(nn.Module):
    """테스트용 더미 임베딩 모델"""
    
    def __init__(self, input_dim=60, seq_len=60, embedding_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        # 간단한 선형 레이어
        self.fc = nn.Linear(input_dim * seq_len, embedding_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, embedding_dim)
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # (batch, seq_len * input_dim)
        return self.fc(x)


def run_random_policy_episodes(env, num_episodes=5):
    """
    랜덤 정책으로 에피소드 실행
    
    Args:
        env: GRPO 환경
        num_episodes: 실행할 에피소드 수
        
    Returns:
        에피소드 메타데이터 리스트
    """
    episodes = []
    
    print(f"\n랜덤 정책으로 {num_episodes}개 에피소드 실행 중...")
    
    for episode_idx in range(num_episodes):
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            # 랜덤 행동 선택
            action = env.action_space.sample()
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
        
        # 에피소드 메타데이터 수집
        episode_metadata = info.get('episode', {})
        episodes.append(episode_metadata)
        
        print(f"  에피소드 {episode_idx + 1}: "
              f"수익={episode_metadata.get('total_return', 0.0):.4f}, "
              f"거래={episode_metadata.get('num_trades', 0)}, "
              f"승률={episode_metadata.get('win_rate', 0.0):.2%}")
    
    return episodes


def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("GRPO 환경과 평가 메트릭 통합 예제")
    print("=" * 70)
    
    # 더미 임베딩 모델 생성
    print("\n1. 더미 임베딩 모델 생성...")
    embedding_model = DummyEmbeddingModel(
        input_dim=60,
        seq_len=60,
        embedding_dim=128
    )
    embedding_model.eval()
    
    # GRPO 환경 생성
    print("2. GRPO 환경 생성...")
    
    # 실제 데이터베이스 경로 (존재하지 않으면 에러 발생)
    db_path = r"C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb"
    
    try:
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=db_path,
            table_name='datasets',
            seq_len=60,
            embedding_dim=128,
            transaction_cost_rate=0.00215,
            quick_exit_threshold=1.5,
            quick_exit_penalty=0.01,
            device='cpu'
        )
        
        # 랜덤 정책으로 에피소드 실행
        print("\n3. 랜덤 정책으로 에피소드 실행...")
        episodes = run_random_policy_episodes(env, num_episodes=5)
        
        # 평가 메트릭 계산
        print("\n4. 평가 메트릭 계산...")
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 결과 출력
        evaluator.print_metrics(metrics)
        
        # 환경 종료
        env.close()
        
        print("\n통합 테스트 완료!")
        
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\n경고: 환경 초기화 실패: {e}")
        print("이 예제는 실제 데이터베이스가 필요합니다.")
        print("\n대신 시뮬레이션된 데이터로 평가 메트릭 계산을 시연합니다...")
        
        # 시뮬레이션된 에피소드 데이터 생성
        simulated_episodes = []
        np.random.seed(42)
        
        for _ in range(5):
            num_trades = np.random.randint(5, 15)
            trades = []
            
            for _ in range(num_trades):
                profit_rate = np.random.normal(0.005, 0.02)
                reward = profit_rate - 0.0043
                holding_time = np.random.uniform(5.0, 30.0)
                
                trades.append({
                    'reward': reward,
                    'holding_time': holding_time,
                    'profit_rate': profit_rate
                })
            
            simulated_episodes.append({
                'trades': trades,
                'total_return': sum(t['reward'] for t in trades),
                'num_trades': num_trades
            })
        
        # 평가 메트릭 계산
        evaluator = GRPOEvaluationMetrics(simulated_episodes)
        metrics = evaluator.compute_metrics()
        
        # 결과 출력
        evaluator.print_metrics(metrics)
        
        print("\n시뮬레이션 완료!")
    
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
