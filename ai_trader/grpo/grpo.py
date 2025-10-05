"""
GRPO (Group Relative Policy Optimization) 알고리즘 구현

이 모듈은 그룹 상대 어드밴티지를 사용하여 정책을 최적화하는 GRPO 훈련기를 정의합니다.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .env import GRPOScalpingEnv

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    Group Relative Policy Optimization 훈련기
    
    핵심 아이디어:
    - 에피소드를 시장 상황별로 그룹화 (변동성, 추세)
    - 그룹 내 상대 성능을 기반으로 어드밴티지 계산
    - PPO 스타일 클리핑으로 정책 업데이트
    
    그룹화 기준:
    - 변동성 분위수 (낮음/중간/높음/매우높음)
    - 추세 방향 (상승/하락/횡보)
    
    Args:
        policy: GRPO 정책 네트워크
        env: GRPO 스캘핑 환경
        episodes_per_group: 그룹당 에피소드 수 (기본값: 12)
        num_groups: 그룹 수 (기본값: 4)
        learning_rate: 학습률 (기본값: 3e-4)
        gamma: 할인 계수 (기본값: 0.99)
        clip_epsilon: PPO 클리핑 파라미터 (기본값: 0.2)
        kl_target: KL 발산 목표값 (기본값: 0.01)
        entropy_coef: 엔트로피 계수 (기본값: 0.01)
        value_coef: 가치 손실 계수 (기본값: 0.5)
        max_grad_norm: 그래디언트 클리핑 최대값 (기본값: 0.5)
        device: 디바이스 ('cpu' 또는 'cuda')
        tensorboard_log_dir: TensorBoard 로그 디렉토리 (선택사항)
    """
    
    def __init__(
        self,
        policy: nn.Module,
        env: GRPOScalpingEnv,
        episodes_per_group: int = 12,
        num_groups: int = 4,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        kl_target: float = 0.01,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
        tensorboard_log_dir: Optional[str] = None
    ):
        self.policy = policy
        self.env = env
        self.device = device
        
        # 정책을 디바이스로 이동
        self.policy.to(device)
        
        # 하이퍼파라미터
        self.episodes_per_group = episodes_per_group
        self.num_groups = num_groups
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.kl_target = kl_target
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer 초기화
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # TensorBoard writer 초기화
        self.writer = None
        if tensorboard_log_dir:
            self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            logger.info(f"TensorBoard logging to: {tensorboard_log_dir}")
        
        # 훈련 상태
        self.total_timesteps = 0
        self.num_updates = 0
        
        # 참조 정책 (KL 발산 계산용)
        self.reference_policy = None
        
        logger.info(f"GRPOTrainer initialized with episodes_per_group={episodes_per_group}, "
                   f"num_groups={num_groups}, lr={learning_rate}, "
                   f"clip_epsilon={clip_epsilon}, device={device}")
    
    def collect_rollouts(self, num_episodes: int) -> List[Dict[str, Any]]:
        """
        현재 정책으로 롤아웃 수집
        
        요구사항 4.1에 따라 현재 정책을 사용하여 여러 에피소드를 실행하고
        상태, 행동, 보상, 다음 상태를 수집합니다.
        
        Args:
            num_episodes: 수집할 에피소드 수
            
        Returns:
            에피소드 데이터 리스트
            각 에피소드는 다음을 포함:
            - states: 상태 리스트
            - actions: 행동 리스트
            - rewards: 보상 리스트
            - next_states: 다음 상태 리스트
            - dones: 종료 플래그 리스트
            - log_probs: 로그 확률 리스트
            - metadata: 에피소드 메타데이터
        """
        episodes = []
        
        logger.info(f"Collecting {num_episodes} rollouts...")
        
        for episode_idx in range(num_episodes):
            # 에피소드 데이터 초기화
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            log_probs = []
            
            # 환경 리셋
            state, info = self.env.reset()
            
            episode_reward = 0.0
            episode_steps = 0
            
            # 에피소드 실행
            while True:
                # 상태를 텐서로 변환
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                
                # 정책에서 행동 샘플링 (stochastic mode)
                with torch.no_grad():
                    action, log_prob = self._sample_action(state_tensor)
                
                # 환경에서 스텝 실행
                next_state, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                
                # 데이터 저장
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                log_probs.append(log_prob)
                
                episode_reward += reward
                episode_steps += 1
                self.total_timesteps += 1
                
                # 다음 상태로 이동
                state = next_state
                
                if done:
                    break
            
            # 에피소드 메타데이터
            episode_metadata = step_info.get('episode', {})
            episode_metadata['episode_reward'] = episode_reward
            episode_metadata['episode_steps'] = episode_steps
            
            # 에피소드 데이터 저장
            episode_data = {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'next_states': np.array(next_states),
                'dones': np.array(dones),
                'log_probs': np.array(log_probs),
                'metadata': episode_metadata
            }
            
            episodes.append(episode_data)
            
            logger.debug(f"Episode {episode_idx + 1}/{num_episodes}: "
                        f"reward={episode_reward:.4f}, steps={episode_steps}, "
                        f"trades={episode_metadata.get('num_trades', 0)}")
        
        logger.info(f"Collected {num_episodes} rollouts, total timesteps: {self.total_timesteps}")
        
        return episodes
    
    def _sample_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        정책에서 행동 샘플링
        
        Args:
            state: 상태 텐서 (1, embedding_dim)
            
        Returns:
            (action, log_prob) 튜플
        """
        # 정책에서 행동 확률 분포 얻기
        # 정책이 구현되면 policy.get_action() 메서드 사용
        # 현재는 placeholder로 랜덤 행동 반환
        action = self.env.action_space.sample()
        log_prob = 0.0  # placeholder
        
        return action, log_prob
    
    def group_episodes(self, episodes: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        에피소드를 시장 상황별로 그룹화
        
        요구사항 4.2에 따라 시장 체제 지표(변동성, 추세 강도)를 기반으로
        에피소드를 그룹화합니다.
        
        Args:
            episodes: 에피소드 데이터 리스트
            
        Returns:
            그룹 ID를 키로 하는 에피소드 딕셔너리
        """
        # Placeholder: 실제 구현은 task 7.3에서 수행
        # 현재는 단순히 순차적으로 그룹 할당
        grouped_episodes = {}
        
        for idx, episode in enumerate(episodes):
            group_id = idx % self.num_groups
            
            if group_id not in grouped_episodes:
                grouped_episodes[group_id] = []
            
            grouped_episodes[group_id].append(episode)
        
        logger.debug(f"Grouped {len(episodes)} episodes into {len(grouped_episodes)} groups")
        
        return grouped_episodes
    
    def compute_group_relative_advantages(
        self,
        grouped_episodes: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[int, List[np.ndarray]]:
        """
        그룹 상대 어드밴티지 계산
        
        요구사항 4.3에 따라 각 그룹의 평균 수익을 계산하고
        그룹 내 상대 어드밴티지를 계산합니다.
        
        Args:
            grouped_episodes: 그룹화된 에피소드
            
        Returns:
            그룹 ID를 키로 하는 어드밴티지 리스트 딕셔너리
        """
        # Placeholder: 실제 구현은 task 7.4에서 수행
        group_advantages = {}
        
        for group_id, group_episodes in grouped_episodes.items():
            advantages = []
            
            for episode in group_episodes:
                # 단순히 보상을 어드밴티지로 사용 (placeholder)
                advantage = episode['rewards']
                advantages.append(advantage)
            
            group_advantages[group_id] = advantages
        
        return group_advantages
    
    def update_policy(
        self,
        episodes: List[Dict[str, Any]],
        advantages: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        정책 업데이트
        
        요구사항 4.4, 4.5에 따라 PPO 스타일 클리핑된 목적 함수와
        KL 발산 제약을 사용하여 정책을 업데이트합니다.
        
        Args:
            episodes: 에피소드 데이터 리스트
            advantages: 어드밴티지 리스트
            
        Returns:
            업데이트 메트릭 딕셔너리
        """
        # Placeholder: 실제 구현은 task 7.5에서 수행
        update_metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'kl_divergence': 0.0,
            'clip_fraction': 0.0
        }
        
        self.num_updates += 1
        
        return update_metrics
    
    def train(
        self,
        total_episodes: int,
        checkpoint_interval: int = 100,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        GRPO 훈련 실행
        
        Args:
            total_episodes: 총 훈련 에피소드 수
            checkpoint_interval: 체크포인트 저장 간격
            checkpoint_path: 체크포인트 저장 경로
            
        Returns:
            훈련 메트릭 딕셔너리
        """
        logger.info(f"Starting GRPO training for {total_episodes} episodes...")
        
        num_iterations = total_episodes // (self.episodes_per_group * self.num_groups)
        
        for iteration in range(num_iterations):
            # 1. 롤아웃 수집
            num_episodes = self.episodes_per_group * self.num_groups
            episodes = self.collect_rollouts(num_episodes)
            
            # 2. 에피소드 그룹화
            grouped_episodes = self.group_episodes(episodes)
            
            # 3. 그룹 상대 어드밴티지 계산
            group_advantages = self.compute_group_relative_advantages(grouped_episodes)
            
            # 4. 정책 업데이트
            # 모든 에피소드와 어드밴티지를 평탄화
            all_episodes = []
            all_advantages = []
            for group_id in sorted(grouped_episodes.keys()):
                all_episodes.extend(grouped_episodes[group_id])
                all_advantages.extend(group_advantages[group_id])
            
            update_metrics = self.update_policy(all_episodes, all_advantages)
            
            # 5. 메트릭 로깅
            if self.writer:
                self._log_metrics(iteration, episodes, grouped_episodes, update_metrics)
            
            # 6. 체크포인트 저장
            if checkpoint_path and (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_path, iteration)
            
            logger.info(f"Iteration {iteration + 1}/{num_iterations}: "
                       f"timesteps={self.total_timesteps}, "
                       f"policy_loss={update_metrics['policy_loss']:.4f}")
        
        logger.info("GRPO training completed!")
        
        # 최종 메트릭 반환
        final_metrics = {
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates
        }
        
        return final_metrics
    
    def _log_metrics(
        self,
        iteration: int,
        episodes: List[Dict[str, Any]],
        grouped_episodes: Dict[int, List[Dict[str, Any]]],
        update_metrics: Dict[str, float]
    ):
        """
        TensorBoard에 메트릭 로깅
        
        요구사항 4.6에 따라 그룹 수준 메트릭과 전체 메트릭을 로깅합니다.
        """
        # 전체 메트릭
        mean_reward = np.mean([ep['metadata']['episode_reward'] for ep in episodes])
        mean_steps = np.mean([ep['metadata']['episode_steps'] for ep in episodes])
        
        # 승률 계산
        win_rates = [ep['metadata'].get('win_rate', 0.0) for ep in episodes]
        mean_win_rate = np.mean(win_rates)
        
        # 평균 보유 시간
        avg_holding_times = [ep['metadata'].get('avg_holding_time', 0.0) for ep in episodes]
        mean_holding_time = np.mean(avg_holding_times)
        
        # 빠른 손절 룰 위반 횟수
        quick_exit_violations = [ep['metadata'].get('quick_exit_violations', 0) for ep in episodes]
        total_violations = sum(quick_exit_violations)
        
        # 전체 메트릭 로깅
        self.writer.add_scalar('train/mean_reward', mean_reward, iteration)
        self.writer.add_scalar('train/mean_steps', mean_steps, iteration)
        self.writer.add_scalar('train/win_rate', mean_win_rate, iteration)
        self.writer.add_scalar('train/avg_holding_time', mean_holding_time, iteration)
        self.writer.add_scalar('train/quick_exit_violations', total_violations, iteration)
        
        # 정책 업데이트 메트릭
        for key, value in update_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, iteration)
        
        # 그룹 수준 메트릭
        for group_id, group_episodes in grouped_episodes.items():
            group_rewards = [ep['metadata']['episode_reward'] for ep in group_episodes]
            group_mean_reward = np.mean(group_rewards)
            
            self.writer.add_scalar(f'group_{group_id}/mean_return', group_mean_reward, iteration)
        
        logger.debug(f"Logged metrics for iteration {iteration}")
    
    def save_checkpoint(self, checkpoint_path: str, iteration: int):
        """
        체크포인트 저장
        
        Args:
            checkpoint_path: 체크포인트 저장 경로
            iteration: 현재 반복 횟수
        """
        checkpoint = {
            'iteration': iteration,
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'episodes_per_group': self.episodes_per_group,
                'num_groups': self.num_groups,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'clip_epsilon': self.clip_epsilon,
                'kl_target': self.kl_target,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'max_grad_norm': self.max_grad_norm
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.num_updates = checkpoint['num_updates']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, "
                   f"timesteps={self.total_timesteps}, updates={self.num_updates}")
    
    def close(self):
        """훈련기 종료"""
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard writer closed")
        
        self.env.close()
        logger.info("Environment closed")
