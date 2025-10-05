"""
GRPO TensorBoard 로깅 예제

Task 7.6: TensorBoard 로깅 구현 데모

이 예제는 GRPO 훈련 중 TensorBoard에 로깅되는 메트릭을 보여줍니다.

실행 방법:
    python examples/grpo_tensorboard_logging_example.py

TensorBoard 실행:
    tensorboard --logdir=models/grpo_example/tensorboard_logs
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class DemoPolicy(nn.Module):
    """데모용 정책 네트워크"""
    
    def __init__(self, embedding_dim=128, hidden_dim=256, action_dim=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """행동 확률 분포 반환"""
        h = torch.relu(self.fc1(x))
        logits = self.fc2(h)
        return torch.softmax(logits, dim=-1)
    
    def evaluate_actions(self, states, actions):
        """행동 평가"""
        h = torch.relu(self.fc1(states))
        logits = self.fc2(h)
        probs = torch.softmax(logits, dim=-1)
        
        # 로그 확률
        log_probs = torch.log(probs + 1e-8)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # 엔트로피
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # 가치
        values = self.value_head(h).squeeze(-1)
        
        return action_log_probs, entropy, values


def create_demo_episodes(num_episodes=8, num_steps=20):
    """데모용 에피소드 생성"""
    episodes = []
    
    for i in range(num_episodes):
        # 시뮬레이션된 에피소드 데이터
        episode = {
            'states': np.random.randn(num_steps, 128).astype(np.float32),
            'actions': np.random.randint(0, 3, num_steps),
            'rewards': np.random.randn(num_steps).astype(np.float32) * 0.01,
            'next_states': np.random.randn(num_steps, 128).astype(np.float32),
            'dones': np.zeros(num_steps, dtype=bool),
            'log_probs': np.random.randn(num_steps).astype(np.float32),
            'metadata': {
                'episode_reward': np.random.randn() * 0.1,
                'episode_steps': num_steps,
                'win_rate': 0.5 + np.random.randn() * 0.1,
                'avg_holding_time': 15 + np.random.randn() * 5,
                'quick_exit_violations': np.random.randint(0, 3),
                'num_trades': np.random.randint(3, 8),
                'sharpe_ratio': 1.0 + np.random.randn() * 0.5
            }
        }
        episodes.append(episode)
    
    return episodes


def demo_tensorboard_logging():
    """TensorBoard 로깅 데모"""
    print("=" * 70)
    print("GRPO TensorBoard 로깅 데모")
    print("=" * 70)
    print()
    
    # 출력 디렉토리 생성
    output_dir = Path("models/grpo_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard 로그 디렉토리
    tensorboard_dir = output_dir / "tensorboard_logs"
    
    print(f"TensorBoard 로그 디렉토리: {tensorboard_dir}")
    print()
    
    # 정책 생성
    policy = DemoPolicy(embedding_dim=128, hidden_dim=256, action_dim=3)
    
    # TensorBoard writer 생성
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    print("시뮬레이션 시작...")
    print()
    
    # 여러 반복 시뮬레이션
    num_iterations = 10
    
    for iteration in range(num_iterations):
        # 데모 에피소드 생성
        episodes = create_demo_episodes(num_episodes=8, num_steps=20)
        
        # 에피소드 그룹화 (2개 그룹)
        grouped_episodes = {
            0: episodes[:4],
            1: episodes[4:]
        }
        
        # 각 에피소드에 그룹 정보 추가
        for ep in episodes[:4]:
            ep['group_id'] = 0
        for ep in episodes[4:]:
            ep['group_id'] = 1
        
        # ===== 전체 메트릭 계산 및 로깅 =====
        mean_reward = np.mean([ep['metadata']['episode_reward'] for ep in episodes])
        mean_win_rate = np.mean([ep['metadata']['win_rate'] for ep in episodes])
        mean_holding_time = np.mean([ep['metadata']['avg_holding_time'] for ep in episodes])
        total_violations = sum([ep['metadata']['quick_exit_violations'] for ep in episodes])
        
        writer.add_scalar('train/mean_reward', mean_reward, iteration)
        writer.add_scalar('train/win_rate', mean_win_rate, iteration)
        writer.add_scalar('train/avg_holding_time', mean_holding_time, iteration)
        writer.add_scalar('train/quick_exit_violations_total', total_violations, iteration)
        
        # 정책 메트릭 (시뮬레이션)
        policy_entropy = 1.0 - iteration * 0.05  # 점진적 감소
        kl_divergence = 0.01 + np.random.randn() * 0.002
        
        writer.add_scalar('train/entropy', policy_entropy, iteration)
        writer.add_scalar('train/kl_divergence', kl_divergence, iteration)
        
        # ===== 그룹 수준 메트릭 계산 및 로깅 =====
        for group_id, group_episodes in grouped_episodes.items():
            group_rewards = [ep['metadata']['episode_reward'] for ep in group_episodes]
            group_mean_reward = np.mean(group_rewards)
            group_win_rate = np.mean([ep['metadata']['win_rate'] for ep in group_episodes])
            group_violations = sum([ep['metadata']['quick_exit_violations'] for ep in group_episodes])
            
            # 그룹별 정책 엔트로피 (시뮬레이션)
            group_entropy = policy_entropy + np.random.randn() * 0.1
            
            # 그룹별 KL 발산 (시뮬레이션)
            group_kl = kl_divergence + np.random.randn() * 0.001
            
            writer.add_scalar(f'group_{group_id}/mean_return', group_mean_reward, iteration)
            writer.add_scalar(f'group_{group_id}/win_rate', group_win_rate, iteration)
            writer.add_scalar(f'group_{group_id}/quick_exit_violations', group_violations, iteration)
            writer.add_scalar(f'group_{group_id}/policy_entropy', group_entropy, iteration)
            writer.add_scalar(f'group_{group_id}/kl_divergence', group_kl, iteration)
        
        # 진행 상황 출력
        if (iteration + 1) % 2 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}:")
            print(f"  평균 보상: {mean_reward:.4f}")
            print(f"  승률: {mean_win_rate:.2%}")
            print(f"  정책 엔트로피: {policy_entropy:.4f}")
            print(f"  KL 발산: {kl_divergence:.6f}")
            print(f"  빠른 손절 룰 위반: {total_violations}")
            print()
    
    writer.close()
    
    print("=" * 70)
    print("시뮬레이션 완료!")
    print()
    print("TensorBoard로 결과 확인:")
    print(f"  tensorboard --logdir={tensorboard_dir}")
    print()
    print("로깅된 메트릭:")
    print()
    print("전체 메트릭:")
    print("  - train/mean_reward: 평균 보상")
    print("  - train/win_rate: 승률")
    print("  - train/avg_holding_time: 평균 보유 시간")
    print("  - train/quick_exit_violations_total: 빠른 손절 룰 위반 횟수")
    print("  - train/entropy: 정책 엔트로피")
    print("  - train/kl_divergence: KL 발산")
    print()
    print("그룹 수준 메트릭 (각 그룹별):")
    print("  - group_X/mean_return: 그룹 평균 수익")
    print("  - group_X/win_rate: 그룹 승률")
    print("  - group_X/quick_exit_violations: 그룹 빠른 손절 룰 위반")
    print("  - group_X/policy_entropy: 그룹 정책 엔트로피")
    print("  - group_X/kl_divergence: 그룹 KL 발산")
    print("=" * 70)


if __name__ == '__main__':
    demo_tensorboard_logging()
