"""
GRPO (Group Relative Policy Optimization) 알고리즘 구현

이 모듈은 그룹 상대 어드밴티지를 사용하여 정책을 최적화하는 GRPO 훈련기를 정의합니다.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans

from .environments import GRPOScalpingEnv
import concurrent.futures  # ✅ 병렬 처리용 추가

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
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
        kl_target: float = 0.01,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        device: str = 'cpu',
        tensorboard_log_dir: Optional[str] = None,
        use_gae: bool = True,
        num_epochs: int = 4,
        observation_schema: Optional[Dict[str, Any]] = None,
        evaluation_callback: Optional[Callable[[nn.Module], Dict[str, Any]]] = None,
        evaluation_interval: int = 1,
        selection_require_liquidation: bool = False,
        diagnostics_interval: int = 5,
        diagnostics_max_samples: int = 256,
    ):
        self.policy = policy
        
        # 환경 처리: 리스트 또는 단일 인스턴스
        if isinstance(env, list):
            self.envs = env
            self.env = env[0]  # 참조용 (첫 번째 환경)
        else:
            self.envs = [env]
            self.env = env
            
        self.device = device
        
        # 정책을 디바이스로 이동
        self.policy.to(device)
        
        # 하이퍼파라미터
        self.episodes_per_group = episodes_per_group
        self.num_groups = num_groups
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.kl_target = kl_target
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.num_epochs = num_epochs
        self.observation_schema = observation_schema
        self.evaluation_callback = evaluation_callback
        self.evaluation_interval = max(1, evaluation_interval)
        self.selection_require_liquidation = selection_require_liquidation
        self.diagnostics_interval = max(1, int(diagnostics_interval))
        self.diagnostics_max_samples = max(1, int(diagnostics_max_samples))
        if selection_require_liquidation and evaluation_callback is None:
            raise ValueError("Liquidation-aware selection requires a validation callback")
        logger.info(f"🤖 GRPOTrainer initialized: use_gae={self.use_gae}, num_epochs={self.num_epochs}, batch_size={self.batch_size}")
        
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
        
        # 외부 상태 (체크포인트에 함께 저장됨, 예: curriculum cost rate)
        self.extra_checkpoint_state = {}
        
        # 참조 정책 (KL 발산 계산용)
        self.reference_policy = None
        
        logger.info(f"GRPOTrainer initialized with episodes_per_group={episodes_per_group}, "
                   f"num_groups={num_groups}, lr={learning_rate}, "
                   f"clip_epsilon={clip_epsilon}, device={device}")
    
    def collect_rollouts(self, num_episodes: int) -> List[Dict[str, Any]]:
        """
        벡터화된 환경(SubprocVecEnv)을 이용한 배치 롤아웃 수집 (GIL 우회 및 GPU Batch Inference)
        """
        logger.info(f"Collecting {num_episodes} rollouts using Vectorized Environments...")
        self.policy.eval()
        
        # self.env가 VecEnv라고 가정 (SubprocVecEnv)
        vec_env = self.env
        num_envs = getattr(vec_env, 'num_envs', 1)
        
        episodes_collected = []
        
        # 각 환경별 임시 저장소
        current_states = [[] for _ in range(num_envs)]
        current_actions = [[] for _ in range(num_envs)]
        current_rewards = [[] for _ in range(num_envs)]
        current_dones = [[] for _ in range(num_envs)]
        current_log_probs = [[] for _ in range(num_envs)]
        current_values = [[] for _ in range(num_envs)]
        
        # 최초 Reset
        obs, infos = vec_env.reset()
        start_infos = [dict(info) for info in infos]
        start_market = [np.asarray(info.get('market_context', self._initial_market_indicators(state)))
                        for state, info in zip(obs, infos)]
        
        episodes_done = 0
        
        while episodes_done < num_episodes:
            # 1. 상태를 하나의 텐서로 배치화 (N, obs_dim)
            if hasattr(obs, 'shape') and len(obs.shape) > 1:
                obs_batch = np.array(obs, dtype=np.float32)
            else:
                obs_batch = np.stack(obs).astype(np.float32)
            
            with torch.no_grad():
                states_tensor = torch.from_numpy(obs_batch).float().to(self.device)
                
                # 2. 정책에서 행동 샘플링 (Batched Inference)
                # DataParallel 클래스 등으로 래핑된 경우 unwrap
                base_policy = getattr(self.policy, 'module', self.policy)
                
                rollout_values = None
                if hasattr(base_policy, 'get_action_with_value'):
                    actions, log_probs, rollout_values = base_policy.get_action_with_value(
                        states_tensor, deterministic=False)
                    actions = actions.detach().cpu().numpy()
                    log_probs = log_probs.detach().cpu().numpy()
                    rollout_values = rollout_values.detach().cpu().numpy().reshape(-1)
                elif hasattr(base_policy, 'get_action'):
                    try:
                        actions, log_probs = base_policy.get_action(states_tensor, deterministic=False)
                        if isinstance(actions, torch.Tensor):
                            actions = actions.cpu().numpy()
                        if isinstance(log_probs, torch.Tensor):
                            log_probs = log_probs.cpu().numpy()
                    except Exception as e:
                        # Fallback: Loop if policy doesn't support batched get_action yet
                        actions_list = []
                        log_probs_list = []
                        for i in range(num_envs):
                            a, lp = self.policy.get_action(states_tensor[i:i+1], deterministic=False)
                            actions_list.append(a.item() if isinstance(a, torch.Tensor) else a)
                            log_probs_list.append(lp.item() if isinstance(lp, torch.Tensor) else lp)
                        actions = np.array(actions_list)
                        log_probs = np.array(log_probs_list)
                else:
                    logger.error(f"Policy {type(base_policy)} missing get_action. Falling back to zeros.")
                    actions = np.array([0]*num_envs, dtype=np.int64)
                    log_probs = np.array([0.0]*num_envs, dtype=np.float32)
                    
            # 3. 환경 스텝 (멀티프로세스로 분산 전송 및 대기)
            # Ensure actions is a clean numpy int array before sending over pipes
            actions_clean = np.array(actions, dtype=np.int32)
            next_obs_raw, rewards, dones, step_infos = vec_env.step(actions_clean)
            
            # 여기서 넘어온 obs도 object 배열일 수 있으므로 변환
            if hasattr(next_obs_raw, 'shape') and len(next_obs_raw.shape) > 1:
                next_obs = np.array(next_obs_raw, dtype=np.float32)
            else:
                next_obs = np.stack(next_obs_raw).astype(np.float32)
            
            # 4. 데이터 기록 및 에피소드 종료 처리
            for i in range(num_envs):
                if episodes_done >= num_episodes:
                    break # 더 이상 수집 불필요
                    
                current_states[i].append(obs[i])
                current_actions[i].append(actions[i])
                current_rewards[i].append(rewards[i])
                
                current_dones[i].append(dones[i])
                current_log_probs[i].append(log_probs[i])
                if rollout_values is not None:
                    current_values[i].append(rollout_values[i])
                
                if dones[i]:
                    # 에피소드 종료
                    ep_rewards = current_rewards[i]
                    ep_steps = len(ep_rewards)
                    
                    # 메타데이터 추출 (자동 리셋 시 이전 info는 reset_info나 단계 info에 있음)
                    ep_info = step_infos[i].get('episode', {}).copy()
                    for k in ['num_trades', 'win_rate', 'sharpe_ratio', 'avg_holding_time', 'quick_exit_violations']:
                        if k in step_infos[i]:
                            ep_info[k] = step_infos[i][k]
                            
                    ep_reward = sum(ep_rewards)
                    if 'total_return' in ep_info:
                        ep_reward = float(ep_info['total_return'])
                        
                    ep_info['episode_reward'] = ep_reward
                    ep_info['episode_steps'] = ep_steps
                    ep_info['initial_market_indicators'] = start_market[i].tolist()
                    ep_info['episode_start'] = start_infos[i]
                    
                    episode_data = {
                        'states': np.array(current_states[i]),
                        'actions': np.array(current_actions[i]),
                        'rewards': np.array(current_rewards[i]),
                        'dones': np.array(current_dones[i]),
                        'log_probs': np.array(current_log_probs[i]),
                        'metadata': ep_info
                    }
                    if len(current_values[i]) == ep_steps:
                        episode_data['values'] = np.asarray(current_values[i], dtype=np.float32)
                    episodes_collected.append(episode_data)
                    episodes_done += 1
                    
                    self.total_timesteps += ep_steps
                    
                    # 버퍼 비우기
                    current_states[i] = []
                    current_actions[i] = []
                    current_rewards[i] = []
                    current_dones[i] = []
                    current_log_probs[i] = []
                    current_values[i] = []
                    start_infos[i] = dict(step_infos[i].get('reset_info', {}))
                    start_market[i] = np.asarray(start_infos[i].get(
                        'market_context', self._initial_market_indicators(next_obs[i])))
            
            obs = next_obs
            
        logger.info(f"Collected {len(episodes_collected)} rollouts (Vectorized), total timesteps: {self.total_timesteps}")
        return episodes_collected
    
    def _sample_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        정책에서 행동 샘플링
        
        Args:
            state: 상태 텐서 (1, embedding_dim)
            
        Returns:
            (action, log_prob) 튜플
        """
        # 정책에서 행동 확률 분포 얻기
        # 정책이 get_action 메서드를 가지고 있는지 확인
        if hasattr(self.policy, 'get_action'):
            # 정책의 get_action 메서드 사용 (stochastic mode)
            action, log_prob = self.policy.get_action(state, deterministic=False)
            
            # 텐서를 스칼라로 변환
            if isinstance(action, torch.Tensor):
                action = action.item()
            if isinstance(log_prob, torch.Tensor):
                log_prob = log_prob.item()
        else:
            # 정책이 구현되지 않은 경우 랜덤 행동 반환
            # 이는 정책 네트워크가 구현되기 전의 임시 동작입니다
            action = self.env.action_space.sample()
            # 균등 분포 가정: log(1/3) = -log(3)
            log_prob = -np.log(self.env.action_space.n)
            
            logger.warning("Policy does not have get_action method, using random actions")
        
        return action, log_prob
    
    @staticmethod
    def _initial_market_indicators(state: np.ndarray) -> np.ndarray:
        """Regime features observed before any action, excluding position fields."""
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 2 or not len(state):
            raise ValueError("Expected an initial sequence observation")
        market = state[:, :-15] if state.shape[1] > 15 else state
        changes = np.diff(market, axis=0)
        return np.array([
            np.std(changes) if changes.size else 0.0,
            np.mean(market[-1] - market[0]),
            np.mean(np.ptp(market, axis=0)),
        ])

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
        if len(episodes) == 0:
            logger.warning("No episodes to group")
            return {}
        
        # 1. 각 에피소드에서 시장 체제 지표 추출
        market_indicators = []
        
        for episode in episodes:
            indicators = episode.get('metadata', {}).get('initial_market_indicators')
            if indicators is None:
                indicators = self._initial_market_indicators(episode['states'][0])
            indicators = np.asarray(indicators, dtype=np.float64)
            if indicators.shape != (3,) or not np.isfinite(indicators).all():
                raise ValueError("Invalid pre-action market regime indicators")
            market_indicators.append(indicators)
        
        market_indicators = np.array(market_indicators)
        
        # 2. K-means 클러스터링으로 그룹화
        # num_groups가 에피소드 수보다 많으면 조정
        n_clusters = min(self.num_groups, max(1, len(episodes) // 2),
                         len(np.unique(market_indicators, axis=0)))
        
        if n_clusters < 2:
            # 클러스터링이 불가능한 경우 모든 에피소드를 그룹 0에 할당
            logger.warning(f"Not enough episodes for clustering (n={len(episodes)}), assigning all to group 0")
            return {0: episodes}
        
        try:
            # K-means 클러스터링 수행
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            group_labels = kmeans.fit_predict(market_indicators)
            
            # K-means 붕괴 감지: 실제 그룹 수가 요청보다 적으면 폴백
            actual_groups = len(set(group_labels))
            if actual_groups < n_clusters:
                logger.warning(f"K-means collapsed to {actual_groups} groups (requested {n_clusters}), falling back to rule-based grouping")
                group_labels = self._rule_based_grouping(market_indicators)
            else:
                # 클러스터 중심 로깅
                logger.debug(f"K-means cluster centers:\n{kmeans.cluster_centers_}")
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}, falling back to rule-based grouping")
            # K-means 실패 시 규칙 기반 그룹화로 대체
            group_labels = self._rule_based_grouping(market_indicators)
        
        # 3. 그룹별로 에피소드 분류
        grouped_episodes = {}
        
        for idx, (episode, group_id) in enumerate(zip(episodes, group_labels)):
            group_id = int(group_id)
            
            if group_id not in grouped_episodes:
                grouped_episodes[group_id] = []
            
            # 에피소드에 그룹 정보 추가
            episode['group_id'] = group_id
            episode['market_indicators'] = market_indicators[idx]
            
            grouped_episodes[group_id].append(episode)
        
        # 그룹 통계 로깅
        for group_id, group_episodes in grouped_episodes.items():
            group_size = len(group_episodes)
            group_mean_reward = np.mean([ep['metadata']['episode_reward'] for ep in group_episodes])
            group_indicators = np.mean([ep['market_indicators'] for ep in group_episodes], axis=0)
            
            logger.debug(f"Group {group_id}: size={group_size}, "
                        f"mean_reward={group_mean_reward:.4f}, "
                        f"volatility={group_indicators[0]:.4f}, "
                        f"trend={group_indicators[1]:.4f}, "
                        f"range={group_indicators[2]:.4f}")
        
        logger.info(f"Grouped {len(episodes)} episodes into {len(grouped_episodes)} groups using K-means")
        
        return grouped_episodes
    
    def _rule_based_grouping(self, market_indicators: np.ndarray) -> np.ndarray:
        """
        규칙 기반 그룹화 (K-means 대체)
        
        변동성과 추세 강도를 기준으로 에피소드를 그룹화합니다.
        
        그룹화 규칙:
        - 그룹 0: 낮은 변동성, 횡보 (volatility < 25%, |trend| < 25%)
        - 그룹 1: 낮은 변동성, 추세 (volatility < 25%, |trend| >= 25%)
        - 그룹 2: 높은 변동성, 횡보 (volatility >= 25%, |trend| < 25%)
        - 그룹 3: 높은 변동성, 추세 (volatility >= 25%, |trend| >= 25%)
        
        Args:
            market_indicators: 시장 지표 배열 (n_episodes, 3)
            
        Returns:
            그룹 레이블 배열 (n_episodes,)
        """
        volatilities = market_indicators[:, 0]
        trends = market_indicators[:, 1]
        
        # 변동성과 추세의 분위수 계산
        volatility_threshold = np.percentile(volatilities, 50)  # 중앙값
        trend_threshold = np.percentile(np.abs(trends), 50)  # 절대값의 중앙값
        
        # 그룹 레이블 초기화
        group_labels = np.zeros(len(market_indicators), dtype=int)
        
        for i in range(len(market_indicators)):
            vol = volatilities[i]
            trend = trends[i]
            
            # 변동성 기준
            high_volatility = vol >= volatility_threshold
            
            # 추세 기준
            strong_trend = np.abs(trend) >= trend_threshold
            
            # 그룹 할당
            if not high_volatility and not strong_trend:
                group_labels[i] = 0  # 낮은 변동성, 횡보
            elif not high_volatility and strong_trend:
                group_labels[i] = 1  # 낮은 변동성, 추세
            elif high_volatility and not strong_trend:
                group_labels[i] = 2  # 높은 변동성, 횡보
            else:
                group_labels[i] = 3  # 높은 변동성, 추세
        
        # num_groups에 맞게 조정
        if self.num_groups < 4:
            group_labels = group_labels % self.num_groups
        
        logger.debug(f"Rule-based grouping: volatility_threshold={volatility_threshold:.4f}, "
                    f"trend_threshold={trend_threshold:.4f}")
        
        return group_labels
    
    def compute_group_relative_advantages(
        self,
        grouped_episodes: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[int, List[np.ndarray]]:
        """
        그룹 상대 어드밴티지 계산
        
        요구사항 4.3에 따라 각 그룹의 평균 수익을 계산하고
        그룹 내 상대 어드밴티지를 계산합니다.
        
        그룹 상대 어드밴티지는 다음과 같이 계산됩니다:
        A_i = R_i - mean(R_group)
        
        여기서:
        - A_i: 에피소드 i의 어드밴티지
        - R_i: 에피소드 i의 총 수익 (에피소드 보상의 합)
        - mean(R_group): 그룹 내 모든 에피소드의 평균 수익
        
        Args:
            grouped_episodes: 그룹화된 에피소드
            
        Returns:
            그룹 ID를 키로 하는 어드밴티지 리스트 딕셔너리
            각 어드밴티지는 에피소드의 각 타임스텝에 대한 어드밴티지 배열
        """
        group_advantages = {}
        
        for group_id, group_episodes in grouped_episodes.items():
            # 1. 그룹 내 각 에피소드의 총 수익 계산
            episode_returns = []
            
            for episode in group_episodes:
                # 에피소드의 총 수익 = 모든 보상의 합
                total_return = np.sum(episode['rewards'])
                episode_returns.append(total_return)
            
            # 2. 그룹의 평균 수익 계산
            mean_group_return = np.mean(episode_returns)
            
            # 3. 각 에피소드의 그룹 상대 어드밴티지 계산
            advantages = []
            
            for episode_idx, episode in enumerate(group_episodes):
                # 에피소드의 총 수익
                episode_return = episode_returns[episode_idx]
                
                # 그룹 상대 어드밴티지: A_i = R_i - mean(R_group)
                relative_advantage = episode_return - mean_group_return
                
                # 에피소드의 각 타임스텝에 어드밴티지 할당 (시간 할인 적용)
                # 에피소드 마지막(결과 결정 시점)에 가까울수록 더 큰 어드밴티지를 할당합니다.
                num_steps = len(episode['rewards'])
                advantage_array = np.zeros(num_steps, dtype=np.float32)
                for t in range(num_steps):
                    # t가 마지막 스텝에 가까울수록 (num_steps - 1 - t가 0에 가까울수록) 할인율이 1에 수렴
                    advantage_array[t] = relative_advantage * (self.gamma ** (num_steps - 1 - t))
                
                advantages.append(advantage_array)
                
                # 에피소드 메타데이터에 어드밴티지 정보 추가
                episode['relative_advantage'] = relative_advantage
                episode['group_mean_return'] = mean_group_return
            
            group_advantages[group_id] = advantages
            
            # 그룹 통계 로깅
            logger.debug(f"Group {group_id}: mean_return={mean_group_return:.4f}, "
                        f"num_episodes={len(group_episodes)}, "
                        f"return_std={np.std(episode_returns):.4f}, "
                        f"return_min={np.min(episode_returns):.4f}, "
                        f"return_max={np.max(episode_returns):.4f}")
        
        logger.info(f"Computed group relative advantages for {len(grouped_episodes)} groups")
        
        return group_advantages
    
    def _episode_tensor_batches(self, states: np.ndarray, actions: np.ndarray):
        """Slice the CPU replay before moving each bounded batch to the device."""
        if len(states) != len(actions):
            raise ValueError("Episode states/actions lengths must match")
        for start in range(0, len(states), self.batch_size):
            end = min(start + self.batch_size, len(states))
            yield (
                torch.from_numpy(states[start:end]).float().to(self.device),
                torch.from_numpy(actions[start:end]).long().to(self.device),
            )

    def update_policy(
        self,
        episodes: List[Dict[str, Any]],
        advantages: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        정책 업데이트
        
        요구사항 4.4, 4.5에 따라 PPO 스타일 클리핑된 목적 함수와
        KL 발산 제약을 사용하여 정책을 업데이트합니다.
        
        PPO 클리핑 목적 함수:
        L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
        
        여기서:
        - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) (확률 비율)
        - A_t: 어드밴티지
        - ε: 클리핑 파라미터 (self.clip_epsilon)
        
        KL 발산 제약:
        - 참조 정책과의 KL 발산이 목표값(self.kl_target)을 초과하면 조기 종료
        
        Args:
            episodes: 에피소드 데이터 리스트
            advantages: 어드밴티지 리스트
            
        Returns:
            업데이트 메트릭 딕셔너리
        """
        # 1. 데이터 준비
        self.policy.train()
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        all_old_values = []
        gae_started = time.perf_counter()
        
        if self.use_gae:
            # 에피소드별로 값 함수 추정 후 GAE 계산 및 그룹 상대 어드밴티지 결합
            for episode, group_adv in zip(episodes, advantages):
                states = episode['states']
                actions = episode['actions']
                old_log_probs = episode['log_probs']
                rewards = episode['rewards']
                dones = episode['dones']
                
                if 'values' in episode:
                    values_ep = np.asarray(episode['values'], dtype=np.float32)
                    if values_ep.shape != (len(rewards),) or not np.isfinite(values_ep).all():
                        raise ValueError('Invalid cached rollout values')
                else:
                    # Both device transfer and forward are bounded by batch_size.
                    # Keep value estimates on CPU as well; do not retain an entire
                    # episode's observation tensor during the later PPO update.
                    with torch.no_grad():
                        values_ep_list = []
                        for batch_s, batch_a in self._episode_tensor_batches(states, actions):
                            _, _, v = self.policy.evaluate_actions(batch_s, batch_a)
                            values_ep_list.append(v.reshape(-1).cpu().numpy())
                        values_ep = np.concatenate(values_ep_list)
                
                # GAE 계산 (타임스텝별 세부 기여도 평가)
                adv_ep, ret_ep = self._compute_gae(rewards, dones, values_ep)
                
                # 하이브리드 어드밴티지 적용: GAE + 그룹 상대 어드밴티지 (A안)
                # GAE 어드밴티지와 스케일을 맞추기 위해 그룹 상대 어드밴티지를 step 수로 나누어 per-step 스케일로 만듭니다.
                # A_hybrid(t) = A_gae(t) + alpha * (relative_advantage / num_steps)
                alpha = 1.0  # 기여도 조정 계수
                num_steps = len(rewards)
                relative_adv_step = group_adv / num_steps
                adv_hybrid = adv_ep + alpha * relative_adv_step
                
                # The critic predicts actual discounted NAV rewards, not the
                # group-relative policy baseline (which changes with the batch).
                
                # 축적
                all_states.append(states)
                all_actions.append(actions)
                all_old_log_probs.append(old_log_probs)
                all_advantages.append(adv_hybrid)
                all_returns.append(ret_ep)
                all_old_values.append(values_ep)
        else:
            for episode, advantage in zip(episodes, advantages):
                states = episode['states']
                actions = episode['actions']
                old_log_probs = episode['log_probs']
                rewards = episode['rewards']
                
                # 리턴 계산 (할인된 누적 보상)
                returns = self._compute_returns(rewards)
                
                # 데이터 추가
                all_states.append(states)
                all_actions.append(actions)
                all_old_log_probs.append(old_log_probs)
                all_advantages.append(advantage)
                all_returns.append(returns)
        
        # 배열로 변환
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_old_log_probs = np.concatenate(all_old_log_probs, axis=0)
        all_advantages = np.concatenate(all_advantages, axis=0)
        all_returns = np.concatenate(all_returns, axis=0)
        
        # 어드밴티지 정규화
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # VRAM 보호를 위해 전체 버퍼는 CPU 메모리에 유지합니다.
        states_tensor = torch.from_numpy(all_states).float()
        actions_tensor = torch.from_numpy(all_actions).long()
        old_log_probs_tensor = torch.from_numpy(all_old_log_probs).float()
        advantages_tensor = torch.from_numpy(all_advantages).float()
        returns_tensor = torch.from_numpy(all_returns).float()
        
        # The frozen rollout likelihood is already stored with every action.
        # No second model or reference forward is needed for sampled KL.
        self.reference_policy = None
        preparation_seconds = time.perf_counter() - gae_started
        optimization_started = time.perf_counter()
        
        # 3. 여러 에포크 동안 정책 업데이트 (PPO의 multiple epochs)
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        num_samples = len(all_states)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_divergence = 0.0
        total_clip_fraction = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        early_stopped = False
        checked_kl = 0.0
        
        for epoch in range(num_epochs):
            # 데이터 셔플
            indices = np.random.permutation(num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # 미니 배치 데이터 슬라이싱 후 GPU 메모리로 이동 (VRAM 폭발 방지)
                batch_states = states_tensor[batch_indices].to(self.device)
                batch_actions = actions_tensor[batch_indices].long().to(self.device)
                batch_old_log_probs = old_log_probs_tensor[batch_indices].to(self.device)
                batch_advantages = advantages_tensor[batch_indices].to(self.device)
                batch_returns = returns_tensor[batch_indices].to(self.device)
                
                # 4. 정책 평가
                log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # 5. PPO 클리핑 목적 함수 계산
                # 확률 비율: r_t = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                log_ratio = log_probs - batch_old_log_probs
                kl_divergence = ((ratio - 1) - log_ratio).mean()
                checked_kl = float(kl_divergence.detach())
                if not np.isfinite(checked_kl):
                    raise FloatingPointError('Nonfinite policy KL; update aborted')
                # Test the policy that would receive this step, without diluting
                # a recent jump with earlier epochs' near-zero KL estimates.
                if checked_kl > self.kl_target * 1.5:
                    early_stopped = True
                    logger.warning('KL guard stopped before optimizer step: %.6f > %.6f',
                                   checked_kl, self.kl_target * 1.5)
                    break
                
                # 클리핑되지 않은 목적 함수
                surr1 = ratio * batch_advantages
                
                # 클리핑된 목적 함수
                ratio_clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                )
                surr2 = ratio_clipped * batch_advantages
                
                # 최소값 선택 (보수적 정책 업데이트)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 6. 가치 손실 계산 (MSE)
                if self.use_gae:
                    value_loss = F.mse_loss(values, batch_returns)
                else:
                    value_loss = torch.tensor(0.0, device=self.device)
                
                # 7. 엔트로피 보너스 (탐험 장려)
                entropy_loss = -entropy.mean()
                
                # 8. 총 손실 계산
                if self.use_gae:
                    loss = (
                        policy_loss +
                        self.value_coef * value_loss +
                        self.entropy_coef * entropy_loss
                    )
                else:
                    # GRPO 모드에서는 value loss를 제외합니다.
                    loss = (
                        policy_loss +
                        self.entropy_coef * entropy_loss
                    )
                
                # 9. 그래디언트 업데이트
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss; update aborted')
                self.optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm, error_if_nonfinite=True
                )
                
                self.optimizer.step()
                
                # GRU 가중치 연속 메모리 보장 (optimizer.step이 가중치를 비연속적으로 만들 수 있음)
                if hasattr(self.policy, 'gru'):
                    self.policy.gru.flatten_parameters()
                
                # 10. KL 발산 계산 (조기 종료 체크)
                with torch.no_grad():
                    # KL 발산: KL(π_old || π_new)
                    log_ratio = log_probs - batch_old_log_probs
                    kl_divergence = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    
                    # 클리핑 비율 계산
                    clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | 
                                   (ratio > 1.0 + self.clip_epsilon)).float().mean()
                
                # 메트릭 누적
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl_divergence += kl_divergence.item()
                total_clip_fraction += clip_fraction.item()
                total_grad_norm += float(grad_norm)
                num_batches += 1
            
            # KL 발산이 목표값을 초과하면 조기 종료
            if early_stopped:
                break
        
        # 평균 메트릭 계산
        update_metrics = {
            'policy_loss': total_policy_loss / max(1, num_batches),
            'value_loss': total_value_loss / max(1, num_batches),
            'entropy': total_entropy / max(1, num_batches),
            'kl_divergence': total_kl_divergence / max(1, num_batches),
            'clip_fraction': total_clip_fraction / max(1, num_batches),
            'grad_norm_before_clip': total_grad_norm / max(1, num_batches),
            'optimizer_steps': num_batches,
            'kl_early_stopped': int(early_stopped),
            'last_checked_kl': checked_kl,
            'preparation_seconds': preparation_seconds,
            'optimization_seconds': time.perf_counter() - optimization_started,
            'return_target_mean': float(all_returns.mean()),
            'return_target_std': float(all_returns.std()),
        }
        if all_old_values:
            old_values = np.concatenate(all_old_values)
            variance = float(all_returns.var())
            update_metrics.update({
                'rollout_value_mean': float(old_values.mean()),
                'rollout_value_std': float(old_values.std()),
                'explained_variance_available': int(variance > 1e-12),
                'explained_variance_rollout': (
                    float(1 - (all_returns - old_values).var() / variance)
                    if variance > 1e-12 else 0.0),
            })
        
        self.num_updates += 1
        
        logger.debug(f"Policy updated: policy_loss={update_metrics['policy_loss']:.4f}, "
                    f"value_loss={update_metrics['value_loss']:.4f}, "
                    f"entropy={update_metrics['entropy']:.4f}, "
                    f"kl_div={update_metrics['kl_divergence']:.6f}, "
                    f"clip_frac={update_metrics['clip_fraction']:.4f}")
        
        return update_metrics
    
    def _compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        """
        할인된 누적 보상 계산
        
        R_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
        
        Args:
            rewards: 보상 배열 (T,)
            
        Returns:
            리턴 배열 (T,)
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        
        # 역순으로 계산
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GAE(Generalized Advantage Estimation) 계산
        Args:
            rewards: 보상 배열 (T,)
            dones: 종료 플래그 배열 (T,) - 각 스텝이 에피소드 종료인지 여부
            values: 값 함수 추정 배열 (T,) - 각 상태의 V(s_t)
        Returns:
            (advantages, returns) 튜플. advantages와 returns는 길이 T.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        gamma = self.gamma
        lam = self.lambda_gae
        
        for t in reversed(range(T)):
            # done이면 다음 상태의 값은 0으로 부트스트랩
            next_non_terminal = 0.0 if dones[t] else 1.0
            if t < T - 1:
                next_value = values[t + 1] * next_non_terminal
            else:
                next_value = 0.0  # 마지막 스텝의 다음 값은 0 (에피소드 종료 가정)
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages.astype(np.float32), returns.astype(np.float32)
    
    def _record_validation_metrics(self, metrics, iteration, checkpoint_path=None, label=None):
        """Persist the current evaluation even when it ties or loses to best."""
        if checkpoint_path:
            directory = os.path.join(os.path.dirname(checkpoint_path.format(iteration)), 'validation')
            os.makedirs(directory, exist_ok=True)
            suffix = f'_{label}' if label else ''
            path = os.path.join(directory, f'iteration_{iteration:06d}{suffix}.json')
            payload = {'iteration': iteration, 'total_timesteps': self.total_timesteps,
                       'metrics': metrics}
            temporary = path + '.tmp'
            with open(temporary, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False,
                          default=lambda value: value.tolist() if isinstance(value, np.ndarray)
                          else value.item() if isinstance(value, np.generic) else str(value))
            os.replace(temporary, path)
        if self.writer:
            def scalars(prefix, values):
                for name, value in values.items():
                    if isinstance(value, dict):
                        scalars(f'{prefix}/{name}', value)
                    elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                        self.writer.add_scalar(f'{prefix}/{name}', float(value), max(0, iteration - 1))
            scalars(f'validation_{label}' if label else 'validation', metrics)
            self.writer.flush()

    def _validation_selection_score(self, metrics):
        """Rank realized validation returns; incomplete liquidation is ineligible."""
        value = float(metrics['mean_net_return'])
        if not np.isfinite(value):
            raise ValueError("Validation mean_net_return must be finite")
        if self.selection_require_liquidation:
            try:
                incomplete = float(metrics['incomplete_liquidation_episodes'])
                residual = float(metrics['max_open_quantity'])
            except (KeyError, TypeError, ValueError):
                logger.warning("Validation candidate excluded: missing/invalid liquidation diagnostics")
                return None
            if not (np.isfinite(incomplete) and np.isfinite(residual)
                    and incomplete == 0 and residual == 0):
                logger.warning("Validation candidate excluded: incomplete_liquidation_episodes=%s, "
                               "max_open_quantity=%s", incomplete, residual)
                return None
        return value

    def _seed_resume_best(self, checkpoint_path, start_iteration, candidates):
        """Protect imported weights' validated baseline before any new update."""
        if self.evaluation_callback is None:
            raise ValueError("Initial best preservation requires a validation callback")

        best_metrics = self.evaluation_callback(self.policy)
        best_score = self._validation_selection_score(best_metrics)
        self._record_validation_metrics(best_metrics, start_iteration, checkpoint_path, 'initial_source')
        best_checkpoint = None
        if candidates:
            import copy
            original_state = copy.deepcopy(self.policy.state_dict())
            try:
                for candidate_index, candidate in enumerate(candidates):
                    if candidate.get('observation_schema') != self.observation_schema:
                        raise ValueError("Resume best checkpoint observation schema mismatch")
                    self.policy.load_state_dict(candidate['policy_state_dict'], strict=True)
                    metrics = self.evaluation_callback(self.policy)
                    value = self._validation_selection_score(metrics)
                    self._record_validation_metrics(metrics, start_iteration, checkpoint_path,
                                                    f'initial_candidate_{candidate_index}')
                    if value is not None and (best_score is None or value >= best_score):
                        best_score, best_metrics, best_checkpoint = value, metrics, candidate
            finally:
                # Training still continues from the requested source checkpoint,
                # including its optimizer state, regardless of which best wins.
                self.policy.load_state_dict(original_state, strict=True)
        if best_score is None:
            logger.warning("No imported checkpoint satisfies validation liquidation requirements")
            return float('-inf')
        if checkpoint_path:
            destination = os.path.join(os.path.dirname(checkpoint_path.format(0)), 'checkpoint_best.pt')
            extra = dict(self.extra_checkpoint_state)
            extra.update({'best_validation_return': best_score,
                          'validation_metrics': best_metrics,
                          'selection_metric': 'validation.mean_net_return',
                          'selection_require_liquidation': self.selection_require_liquidation})
            if best_checkpoint is None:
                self.save_checkpoint(destination, start_iteration, extra_state=extra)
            else:
                selected = dict(best_checkpoint)
                selected['extra_state'] = extra
                torch.save(selected, destination)
        logger.info("Preserved initial validation baseline: %.6f%%", best_score)
        return best_score

    def train(
        self,
        total_episodes: int,
        checkpoint_interval: int = 100,
        checkpoint_path: Optional[str] = None,
        on_iteration_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        start_iteration: int = 0,
        max_timesteps: Optional[int] = None,
        revert_to_best_patience: int = 0,
        resume: bool = False,
        resume_best_checkpoints: Optional[List[Dict[str, Any]]] = None,
        preserve_initial_policy: bool = False,
    ) -> Dict[str, Any]:
        """
        GRPO 훈련 실행
        
        Args:
            total_episodes: Episode count, or initial estimate when max_timesteps is set.
            checkpoint_interval: 체크포인트 저장 간격
            checkpoint_path: 체크포인트 저장 경로 (format string with {} for iteration)
            on_iteration_end: 매 반복 종료 시 호출될 콜백 함수 (iteration, metrics) -> None
            start_iteration: 시작 반복 횟수 (재개 시 사용)
            max_timesteps: 최대 훈련 타임스텝 수 (실제 누적 타임스텝 기준 조기 종료)
            preserve_initial_policy: Validate loaded fine-tuning weights before updating.
            
        Returns:
            훈련 메트릭 딕셔너리
        """
        import time
        
        logger.info(f"Starting GRPO training for {total_episodes} episodes...")
        
        if total_episodes <= 0:
            raise ValueError("total_episodes must be positive")
        if max_timesteps is not None and max_timesteps <= 0:
            raise ValueError("max_timesteps must be positive")
        episodes_per_iteration = self.episodes_per_group * self.num_groups
        num_iterations = start_iteration + int(np.ceil(total_episodes / episodes_per_iteration))
        logger.info(f"Total iterations: {num_iterations} (Starting from {start_iteration})")
        
        start_time = time.time()
        iteration_times = []
        
        # Model selection uses the current weights on validation paths only.
        best_validation_return = float('-inf')
        if resume or preserve_initial_policy:
            best_validation_return = self._seed_resume_best(
                checkpoint_path, start_iteration,
                (resume_best_checkpoints or []) if resume else [])
        no_improve_count = 0
        validation_metrics = None

        iteration = start_iteration
        last_completed_iteration = start_iteration
        episodes_trained = 0
        budget_start_timesteps = self.total_timesteps
        while True:
            iteration_start_time = time.time()
            phase_started = time.perf_counter()
            phase_seconds = {}
            # Episode lengths can be shorter than the configured maximum. A
            # timestep budget must use observed steps, not a fixed episode cap.
            if max_timesteps is not None:
                remaining_steps = max_timesteps - self.total_timesteps
                if remaining_steps <= 0:
                    break
                if episodes_trained:
                    mean_steps = (self.total_timesteps - budget_start_timesteps) / episodes_trained
                    remaining_episodes = int(np.ceil(remaining_steps / mean_steps))
                else:
                    remaining_episodes = total_episodes
            else:
                remaining_episodes = total_episodes - episodes_trained
                if remaining_episodes <= 0:
                    break
            num_episodes = min(episodes_per_iteration, remaining_episodes)
            previous_timesteps = self.total_timesteps
            episodes = self.collect_rollouts(num_episodes)
            phase_seconds['rollout'] = time.perf_counter() - phase_started
            if not episodes:
                raise RuntimeError("Rollout collection returned no completed episodes")
            if max_timesteps is not None and self.total_timesteps <= previous_timesteps:
                raise RuntimeError("Rollout collection made no timestep progress")
            episodes_trained += len(episodes)
            if max_timesteps is not None:
                mean_steps = (self.total_timesteps - budget_start_timesteps) / episodes_trained
                remaining_episodes = int(np.ceil(max(0, max_timesteps - self.total_timesteps) / mean_steps))
                num_iterations = iteration + 1 + int(np.ceil(remaining_episodes / episodes_per_iteration))
            
            # 2. 에피소드 그룹화
            phase_started = time.perf_counter()
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
            
            phase_seconds['grouping'] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            update_metrics = self.update_policy(all_episodes, all_advantages)
            phase_seconds['update'] = time.perf_counter() - phase_started
            last_completed_iteration = iteration + 1
            
            # 5. 메트릭 로깅
            phase_started = time.perf_counter()
            if self.writer:
                self._log_metrics(iteration, episodes, grouped_episodes, update_metrics)
            phase_seconds['diagnostics'] = time.perf_counter() - phase_started
            
            # 6. 체크포인트 저장
            phase_started = time.perf_counter()
            if checkpoint_path and (iteration + 1) % checkpoint_interval == 0:
                # Format checkpoint path with iteration number
                formatted_checkpoint_path = checkpoint_path.format(iteration + 1)
                self.save_checkpoint(
                    formatted_checkpoint_path, iteration + 1,
                    extra_state=self.extra_checkpoint_state if self.extra_checkpoint_state else None
                )
            phase_seconds['checkpoint'] = time.perf_counter() - phase_started
            
            # 평균 보상 및 추가 메트릭 계산
            mean_reward = np.mean([ep['metadata']['episode_reward'] for ep in episodes])
            
            # 추가 메트릭 계산
            mean_win_rate = np.mean([ep['metadata'].get('win_rate', 0.0) for ep in episodes])
            mean_trades = np.mean([ep['metadata'].get('num_trades', 0) for ep in episodes])
            mean_sharpe = np.mean([ep['metadata'].get('sharpe_ratio', 0.0) for ep in episodes])
            mean_holding_time = np.mean([ep['metadata'].get('avg_holding_time', 0.0) for ep in episodes])
            
            validation_metrics = None
            if (self.evaluation_callback is not None and
                    ((iteration + 1) % self.evaluation_interval == 0 or
                     iteration + 1 == num_iterations or
                     (max_timesteps and self.total_timesteps >= max_timesteps))):
                phase_started = time.perf_counter()
                logger.info('Starting validation for iteration %d...', iteration + 1)
                validation_metrics = self.evaluation_callback(self.policy)
                score = self._validation_selection_score(validation_metrics)
                self._record_validation_metrics(validation_metrics, iteration + 1, checkpoint_path)
                reported_return = float(validation_metrics['mean_net_return'])
                if self.writer:
                    self.writer.add_scalar('validation/mean_net_return', reported_return, iteration)
                    self.writer.add_scalar('validation/selection_eligible', int(score is not None), iteration)
                logger.info("Validation net return: %.6f%% (current weights; selection eligible=%s)",
                            reported_return, score is not None)
                if score is not None and score > best_validation_return:
                    best_validation_return = score
                    no_improve_count = 0
                    if checkpoint_path:
                        selected_path = os.path.join(
                            os.path.dirname(checkpoint_path.format(0)), 'checkpoint_best.pt')
                        extra = dict(self.extra_checkpoint_state)
                        extra.update({'best_validation_return': score,
                                      'validation_metrics': validation_metrics,
                                      'selection_metric': 'validation.mean_net_return',
                                      'selection_require_liquidation': self.selection_require_liquidation})
                        self.save_checkpoint(selected_path, iteration + 1, extra_state=extra)
                else:
                    no_improve_count += 1
                if (checkpoint_path and np.isfinite(best_validation_return) and revert_to_best_patience > 0 and
                        no_improve_count >= revert_to_best_patience):
                    selected_path = os.path.join(
                        os.path.dirname(checkpoint_path.format(0)), 'checkpoint_best.pt')
                    if os.path.exists(selected_path):
                        # Reverting parameters must not rewind the actual training budget.
                        steps, updates = self.total_timesteps, self.num_updates
                        self.load_checkpoint(selected_path)
                        self.total_timesteps, self.num_updates = steps, updates
                        no_improve_count = 0
                phase_seconds['validation'] = time.perf_counter() - phase_started

            if self.writer:
                for name, seconds in phase_seconds.items():
                    self.writer.add_scalar(f'timing/{name}_seconds', seconds, iteration)
                self.writer.flush()
            logger.info('Phase seconds: %s', ', '.join(
                f'{name}={seconds:.1f}' for name, seconds in phase_seconds.items()))
            if on_iteration_end:
                metrics_summary = {
                    'mean_reward': mean_reward,
                    'mean_win_rate': mean_win_rate,
                    'mean_trades': mean_trades,
                    'mean_sharpe': mean_sharpe,
                    'mean_holding_time': mean_holding_time,
                    'validation': validation_metrics,
                    'iteration': iteration + 1,
                    'total_iterations': num_iterations
                }
                on_iteration_end(iteration + 1, metrics_summary)
            
            # 7. 진행률 및 예상 시간 계산
            iteration_elapsed = time.time() - iteration_start_time
            iteration_times.append(iteration_elapsed)
            
            # 최근 10개 iteration의 평균 시간으로 예상 시간 계산
            recent_times = iteration_times[-10:]
            avg_iteration_time = np.mean(recent_times)
            remaining_iterations = num_iterations - (iteration + 1)
            estimated_remaining_time = avg_iteration_time * remaining_iterations
            
            # 진행률 계산
            progress_pct = (min(100.0, self.total_timesteps / max_timesteps * 100)
                            if max_timesteps is not None else (iteration + 1) / num_iterations * 100)
            
            # 시간 포맷팅
            elapsed_time = time.time() - start_time
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(estimated_remaining_time)
            
            logger.info(f"Iteration {iteration + 1}/{num_iterations} ({progress_pct:.1f}%) | "
                       f"Timesteps: {self.total_timesteps} | "
                       f"Mean Reward: {mean_reward:.4f} | "
                       f"FillWin: {mean_win_rate:.1%} | "
                       f"SellFills: {mean_trades:.1f} | "
                       f"Sharpe: {mean_sharpe:.2f} | "
                       f"FillHold: {mean_holding_time:.1f}s | "
                       f"Policy Loss: {update_metrics['policy_loss']:.4f} | "
                       f"Elapsed: {elapsed_str} | "
                       f"ETA: {remaining_str}")
            
            # 조기 종료 체크: 실제 경과 타임스텝 수 기준
            if max_timesteps and self.total_timesteps >= max_timesteps:
                logger.info(f"[EARLY STOP] Stopped because total_timesteps ({self.total_timesteps:,}) >= max_timesteps ({max_timesteps:,})")
                if checkpoint_path:
                    formatted_checkpoint_path = checkpoint_path.format(iteration + 1)
                    self.save_checkpoint(
                        formatted_checkpoint_path, iteration + 1,
                        extra_state=self.extra_checkpoint_state if self.extra_checkpoint_state else None
                    )
                break
            iteration += 1

        if self.selection_require_liquidation and not np.isfinite(best_validation_return):
            if checkpoint_path and last_completed_iteration > start_iteration:
                self.save_checkpoint(checkpoint_path.format(last_completed_iteration), last_completed_iteration,
                                     extra_state=self.extra_checkpoint_state or None)
            raise RuntimeError(
                "No eligible validation checkpoint: every candidate has incomplete liquidation, "
                "open inventory, or missing liquidation diagnostics. Iteration checkpoints are retained "
                "for diagnosis; an existing best checkpoint is not selected.")
        
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
        
        요구사항 4.6, 6.6에 따라 그룹 수준 메트릭과 전체 메트릭을 로깅합니다.
        
        그룹 수준 메트릭:
        - 그룹당 평균 수익
        - 그룹당 정책 엔트로피
        - 그룹당 KL 발산
        
        전체 메트릭:
        - 평균 보상
        - 승률
        - 평균 보유 시간
        - 빠른 손절 룰 위반 횟수
        - 정책 엔트로피
        - KL 발산
        """
        # ===== 전체 메트릭 =====
        mean_reward = np.mean([ep['metadata']['episode_reward'] for ep in episodes])
        mean_steps = np.mean([ep['metadata']['episode_steps'] for ep in episodes])
        
        # 승률 계산
        win_rates = [ep['metadata'].get('win_rate', 0.0) for ep in episodes]
        mean_win_rate = np.mean(win_rates) if win_rates else 0.0
        
        # 평균 보유 시간
        avg_holding_times = [ep['metadata'].get('avg_holding_time', 0.0) for ep in episodes]
        mean_holding_time = np.mean(avg_holding_times) if avg_holding_times else 0.0
        
        # 빠른 손절 룰 위반 횟수
        quick_exit_violations = [ep['metadata'].get('quick_exit_violations', 0) for ep in episodes]
        total_violations = sum(quick_exit_violations)
        mean_violations = np.mean(quick_exit_violations) if quick_exit_violations else 0.0
        
        # 거래 횟수
        num_trades = [ep['metadata'].get('num_trades', 0) for ep in episodes]
        mean_trades = np.mean(num_trades) if num_trades else 0.0
        
        # 샤프 비율
        sharpe_ratios = [ep['metadata'].get('sharpe_ratio', 0.0) for ep in episodes]
        mean_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0.0
        
        # 전체 메트릭 로깅
        self.writer.add_scalar('train/mean_reward', mean_reward, iteration)
        self.writer.add_scalar('train/mean_steps', mean_steps, iteration)
        self.writer.add_scalar('train/win_rate', mean_win_rate, iteration)
        self.writer.add_scalar('train/avg_holding_time', mean_holding_time, iteration)
        self.writer.add_scalar('train/quick_exit_violations_total', total_violations, iteration)
        self.writer.add_scalar('train/quick_exit_violations_mean', mean_violations, iteration)
        self.writer.add_scalar('train/mean_trades', mean_trades, iteration)
        self.writer.add_scalar('train/mean_sharpe_ratio', mean_sharpe, iteration)
        # Legacy num_trades counts FIFO sell-fill fragments, not round trips.
        for name in ('round_trip_count', 'round_trip_win_rate', 'quantity_weighted_holding_time',
                     'fill_count', 'fill_win_rate', 'fill_avg_holding_time',
                     'total_entry_fees', 'total_exit_fees', 'total_fees', 'gross_realized_pnl',
                     'realized_net_pnl', 'liquidation_steps', 'liquidation_seconds',
                     'open_quantity'):
            values = [ep['metadata'][name] for ep in episodes if name in ep['metadata']]
            if values:
                self.writer.add_scalar(f'train/{name}', float(np.mean(values)), iteration)
        liquidation = [ep['metadata']['liquidation_complete'] for ep in episodes
                       if 'liquidation_complete' in ep['metadata']]
        if liquidation:
            self.writer.add_scalar('train/incomplete_liquidation_fraction',
                                   float(np.mean(np.logical_not(liquidation))), iteration)
        
        # 정책 업데이트 메트릭 (정책 엔트로피, KL 발산 포함)
        for key, value in update_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, iteration)
        
        # One deterministic sample over the full rollout, bounded even when
        # there are more episodes than the diagnostic sample budget.
        diagnostic_indices = {}
        lengths = [len(ep['states']) for ep in episodes]
        total_states = sum(lengths)
        positions = (np.linspace(0, total_states - 1,
                                 min(self.diagnostics_max_samples, total_states), dtype=int)
                     if total_states else np.array([], dtype=int))
        offset = 0
        for ep, length in zip(episodes, lengths):
            diagnostic_indices[id(ep)] = positions[(positions >= offset) & (positions < offset + length)] - offset
            offset += length

        # ===== 그룹 수준 메트릭 =====
        for group_id, group_episodes in grouped_episodes.items():
            # 그룹 평균 수익
            group_rewards = [ep['metadata']['episode_reward'] for ep in group_episodes]
            group_mean_reward = np.mean(group_rewards)
            group_std_reward = np.std(group_rewards)
            
            # 그룹 승률
            group_win_rates = [ep['metadata'].get('win_rate', 0.0) for ep in group_episodes]
            group_mean_win_rate = np.mean(group_win_rates) if group_win_rates else 0.0
            
            # 그룹 평균 보유 시간
            group_holding_times = [ep['metadata'].get('avg_holding_time', 0.0) for ep in group_episodes]
            group_mean_holding_time = np.mean(group_holding_times) if group_holding_times else 0.0
            
            # 그룹 빠른 손절 룰 위반 횟수
            group_violations = [ep['metadata'].get('quick_exit_violations', 0) for ep in group_episodes]
            group_total_violations = sum(group_violations)
            
            # 그룹 거래 횟수
            group_trades = [ep['metadata'].get('num_trades', 0) for ep in group_episodes]
            group_mean_trades = np.mean(group_trades) if group_trades else 0.0
            
            # 그룹 샤프 비율
            group_sharpe = [ep['metadata'].get('sharpe_ratio', 0.0) for ep in group_episodes]
            group_mean_sharpe = np.mean(group_sharpe) if group_sharpe else 0.0
            
            # Preserve the original mean-of-episode-means aggregation while
            # limiting logging transfers/forwards to the training mini-batch size.
            group_entropies = []
            group_kl_divergences = []
            diagnostic_samples = 0
            run_policy_diagnostics = (iteration == 0 or
                                      (iteration + 1) % self.diagnostics_interval == 0)
            with torch.no_grad():
                for ep in group_episodes if run_policy_diagnostics else []:
                    entropy_sum = 0.0
                    kl_sum = 0.0
                    sample_count = 0
                    indices = diagnostic_indices[id(ep)]
                    if not len(indices):
                        continue
                    selected_states = ep['states'][indices]
                    selected_actions = ep['actions'][indices]
                    selected_old_logs = ep['log_probs'][indices]
                    offset = 0
                    for states_batch, actions_batch in self._episode_tensor_batches(
                            selected_states, selected_actions):
                        current_log_probs, entropy, _ = self.policy.evaluate_actions(
                            states_batch, actions_batch)
                        entropy_sum += entropy.sum().item()
                        sample_count += len(states_batch)
                        if self.reference_policy is not None:
                            ref_log_probs, _, _ = self.reference_policy.evaluate_actions(
                                states_batch, actions_batch)
                            kl_sum += (ref_log_probs - current_log_probs).sum().item()
                        else:
                            old_logs = torch.as_tensor(
                                selected_old_logs[offset:offset + len(states_batch)],
                                dtype=current_log_probs.dtype, device=current_log_probs.device)
                            log_ratio = current_log_probs - old_logs
                            kl_sum += (torch.expm1(log_ratio) - log_ratio).sum().item()
                        offset += len(states_batch)
                    if sample_count:
                        group_entropies.append(entropy_sum / sample_count)
                        group_kl_divergences.append(kl_sum / sample_count)
                        diagnostic_samples += sample_count
            group_mean_entropy = np.mean(group_entropies) if group_entropies else 0.0
            group_mean_kl = np.mean(group_kl_divergences) if group_kl_divergences else 0.0

            # 그룹 메트릭 로깅
            self.writer.add_scalar(f'group_{group_id}/mean_return', group_mean_reward, iteration)
            self.writer.add_scalar(f'group_{group_id}/std_return', group_std_reward, iteration)
            self.writer.add_scalar(f'group_{group_id}/win_rate', group_mean_win_rate, iteration)
            self.writer.add_scalar(f'group_{group_id}/avg_holding_time', group_mean_holding_time, iteration)
            self.writer.add_scalar(f'group_{group_id}/quick_exit_violations', group_total_violations, iteration)
            self.writer.add_scalar(f'group_{group_id}/mean_trades', group_mean_trades, iteration)
            self.writer.add_scalar(f'group_{group_id}/sharpe_ratio', group_mean_sharpe, iteration)
            if run_policy_diagnostics and diagnostic_samples:
                self.writer.add_scalar(f'group_{group_id}/policy_entropy', group_mean_entropy, iteration)
                self.writer.add_scalar(f'group_{group_id}/kl_divergence', group_mean_kl, iteration)
                self.writer.add_scalar(f'group_{group_id}/diagnostic_samples', diagnostic_samples, iteration)
                self.writer.add_scalar(f'group_{group_id}/diagnostic_sampling_fraction',
                                       diagnostic_samples / sum(len(ep['states']) for ep in group_episodes), iteration)
            
            # 그룹 크기
            self.writer.add_scalar(f'group_{group_id}/size', len(group_episodes), iteration)
        
        # 로깅 완료 메시지
        logger.debug(f"Logged metrics for iteration {iteration}: "
                    f"mean_reward={mean_reward:.4f}, win_rate={mean_win_rate:.4f}, "
                    f"violations={total_violations}, entropy={update_metrics.get('entropy', 0.0):.4f}, "
                    f"kl_div={update_metrics.get('kl_divergence', 0.0):.6f}")
    
    def save_checkpoint(self, checkpoint_path: str, iteration: int, extra_state: Optional[Dict[str, Any]] = None):
        """
        체크포인트 저장
        
        Args:
            checkpoint_path: 체크포인트 저장 경로
            iteration: 현재 반복 횟수
            extra_state: 추가 상태 (예: curriculum cost rate 등)
        """
        # 정책 타입 및 설정 정보 추출
        policy_class_name = self.policy.__class__.__name__
        policy_config = {
            'policy_type': policy_class_name,
            'action_dim': self.policy.action_dim if hasattr(self.policy, 'action_dim') else 3,
            'hidden_dim': getattr(self.policy, 'fc_hidden_dim', getattr(self.policy, 'hidden_dim', 128)),
            'obs_dim': getattr(self.policy, 'obs_dim', None),
            'cnn_channels': getattr(self.policy, 'cnn_channels', None),
            'rnn_hidden_dim': getattr(self.policy, 'rnn_hidden_dim', None),
        }
        
        # GRPOPolicy의 경우 embedding_dim 저장
        if hasattr(self.policy, 'embedding_dim'):
            policy_config['embedding_dim'] = self.policy.embedding_dim
        
        checkpoint = {
            'iteration': iteration,
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
            'policy_state_dict': self.policy.state_dict(),
            'observation_schema': self.observation_schema,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'episodes_per_group': self.episodes_per_group,
                'num_groups': self.num_groups,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'lambda_gae': self.lambda_gae,
                'clip_epsilon': self.clip_epsilon,
                'kl_target': self.kl_target,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'max_grad_norm': self.max_grad_norm,
                'selection_require_liquidation': self.selection_require_liquidation,
                **policy_config  # 정책 설정 병합
            }
        }
        
        # 추가 상태 저장 (curriculum cost rate 등)
        if extra_state:
            checkpoint['extra_state'] = extra_state
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        if extra_state:
            logger.debug(f"  Extra state saved: {extra_state}")
    
    def restore_training_progress(self, checkpoint: Dict[str, Any]) -> int:
        """Restore optimizer/counters only for an explicitly requested resume."""
        required = ('optimizer_state_dict', 'total_timesteps', 'num_updates', 'iteration')
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(f"Resume requires a full training checkpoint; missing {missing}")
        for name in ('total_timesteps', 'num_updates', 'iteration'):
            value = checkpoint[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid checkpoint {name}: expected a nonnegative integer")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.num_updates = checkpoint['num_updates']
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        return checkpoint['iteration']

    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
            
        Returns:
            extra_state: 체크포인트에 저장된 추가 상태 (없으면 None)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if checkpoint.get('observation_schema') != self.observation_schema:
            raise ValueError('Checkpoint observation schema does not match the training environment')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.restore_training_progress(checkpoint)
        
        extra_state = checkpoint.get('extra_state', None)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, "
                   f"timesteps={self.total_timesteps}, updates={self.num_updates}")
        if extra_state:
            logger.info(f"  Extra state restored: {extra_state}")
        
        return extra_state
    
    def _format_time(self, seconds: float) -> str:
        """
        시간을 읽기 쉬운 형식으로 포맷팅
        
        Args:
            seconds: 초 단위 시간
            
        Returns:
            포맷팅된 시간 문자열 (예: "1h 23m 45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def close(self):
        """훈련기 종료"""
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard writer closed")
        
        self.env.close()
        logger.info("Environment closed")
