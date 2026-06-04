"""
GRPO (Group Relative Policy Optimization) 알고리즘 구현

이 모듈은 그룹 상대 어드밴티지를 사용하여 정책을 최적화하는 GRPO 훈련기를 정의합니다.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
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
        use_gae: bool = True
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
        
        # self.env가 VecEnv라고 가정 (SubprocVecEnv)
        vec_env = self.env
        num_envs = getattr(vec_env, 'num_envs', 1)
        
        episodes_collected = []
        
        # 각 환경별 임시 저장소
        current_states = [[] for _ in range(num_envs)]
        current_actions = [[] for _ in range(num_envs)]
        current_rewards = [[] for _ in range(num_envs)]
        current_next_states = [[] for _ in range(num_envs)]
        current_dones = [[] for _ in range(num_envs)]
        current_log_probs = [[] for _ in range(num_envs)]
        
        # 최초 Reset
        obs, infos = vec_env.reset()
        
        episodes_done = 0
        
        while episodes_done < num_episodes:
            # 1. 상태를 하나의 텐서로 배치화 (N, obs_dim)
            if hasattr(obs, 'shape') and len(obs.shape) > 1:
                obs_batch = np.array(obs, dtype=np.float32)
            else:
                obs_batch = np.stack(obs).astype(np.float32)
            
            with torch.no_grad():
                states_tensor = torch.from_numpy(obs_batch).float().to(self.device, non_blocking=True)
                
                # 2. 정책에서 행동 샘플링 (Batched Inference under BF16 autocast)
                # DataParallel 클래스 등으로 래핑된 경우 unwrap
                base_policy = getattr(self.policy, 'module', self.policy)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if hasattr(base_policy, 'get_action'):
                        try:
                            actions, log_probs = base_policy.get_action(states_tensor, deterministic=False)
                            if isinstance(actions, torch.Tensor):
                                actions = actions.cpu().numpy()
                            if isinstance(log_probs, torch.Tensor):
                                log_probs = log_probs.float().cpu().numpy()
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
                
                # SubprocVecEnv는 done=True시 자동 reset하며 next_obs는 새 에피소드의 시작을 담습니다.
                # 실제 터미널 상태는 info['terminal_observation']에 있을 수 있습니다.
                if dones[i]:
                    actual_next_obs = step_infos[i].get('terminal_observation', next_obs[i])
                    
                    # terminal_observation도 형변환
                    if isinstance(actual_next_obs, (list, tuple)):
                        actual_next_obs = np.array(actual_next_obs, dtype=np.float32)
                    elif hasattr(actual_next_obs, 'astype'):
                        actual_next_obs = actual_next_obs.astype(np.float32)
                else:
                    actual_next_obs = next_obs[i]
                    
                current_next_states[i].append(actual_next_obs)
                current_dones[i].append(dones[i])
                current_log_probs[i].append(log_probs[i])
                
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
                    
                    episode_data = {
                        'states': np.array(current_states[i]),
                        'actions': np.array(current_actions[i]),
                        'rewards': np.array(current_rewards[i]),
                        'next_states': np.array(current_next_states[i]),
                        'dones': np.array(current_dones[i]),
                        'log_probs': np.array(current_log_probs[i]),
                        'metadata': ep_info
                    }
                    episodes_collected.append(episode_data)
                    episodes_done += 1
                    
                    self.total_timesteps += ep_steps
                    
                    # 버퍼 비우기
                    current_states[i] = []
                    current_actions[i] = []
                    current_rewards[i] = []
                    current_next_states[i] = []
                    current_dones[i] = []
                    current_log_probs[i] = []
            
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
            # 에피소드의 보상 시퀀스에서 변동성과 추세 강도 계산
            rewards = episode['rewards']
            
            # 변동성: 보상의 표준편차
            volatility = np.std(rewards) if len(rewards) > 1 else 0.0
            
            # 추세 강도: 보상의 평균 (양수면 상승, 음수면 하락)
            trend_strength = np.mean(rewards)
            
            # 추가 지표: 보상의 범위 (최대 - 최소)
            reward_range = np.max(rewards) - np.min(rewards) if len(rewards) > 0 else 0.0
            
            # 지표를 벡터로 저장
            indicators = np.array([volatility, trend_strength, reward_range])
            market_indicators.append(indicators)
        
        market_indicators = np.array(market_indicators)
        
        # 2. K-means 클러스터링으로 그룹화
        # num_groups가 에피소드 수보다 많으면 조정
        n_clusters = min(self.num_groups, len(episodes))
        
        if n_clusters < 2:
            # 클러스터링이 불가능한 경우 모든 에피소드를 그룹 0에 할당
            logger.warning(f"Not enough episodes for clustering (n={len(episodes)}), assigning all to group 0")
            return {0: episodes}
        
        try:
            # K-means 클러스터링 수행
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            group_labels = kmeans.fit_predict(market_indicators)
            
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
                
                # 에피소드의 각 타임스텝에 동일한 어드밴티지 할당
                # 이는 에피소드 전체의 성과를 각 행동에 귀속시키는 방식입니다
                num_steps = len(episode['rewards'])
                advantage_array = np.full(num_steps, relative_advantage, dtype=np.float32)
                
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
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        if self.use_gae:
            # 에피소드별로 값 함수 추정 후 GAE 계산
            for episode in episodes:
                states = episode['states']
                actions = episode['actions']
                old_log_probs = episode['log_probs']
                rewards = episode['rewards']
                dones = episode['dones']
                
                # 텐서화
                states_tensor_ep = torch.from_numpy(states).float().to(self.device)
                actions_tensor_ep = torch.from_numpy(actions).long().to(self.device)
                
                with torch.no_grad():
                    # 현재 정책에서 값 함수 추정 (evaluate_actions가 values 반환)
                    # 메모리 부족(CUDA illegal memory access) 방지를 위해 미니 배치 단위로 평가
                    values_ep_list = []
                    for start_i in range(0, len(states_tensor_ep), self.batch_size):
                        end_i = min(start_i + self.batch_size, len(states_tensor_ep))
                        batch_s = states_tensor_ep[start_i:end_i]
                        batch_a = actions_tensor_ep[start_i:end_i]
                        _, _, v = self.policy.evaluate_actions(batch_s, batch_a)
                        values_ep_list.append(v)
                    values_ep = torch.cat(values_ep_list, dim=0)
                    values_ep = values_ep.squeeze(-1).cpu().numpy() if values_ep.dim() > 1 else values_ep.cpu().numpy()
                
                # GAE 계산
                adv_ep, ret_ep = self._compute_gae(rewards, dones, values_ep)
                
                # 축적
                all_states.append(states)
                all_actions.append(actions)
                all_old_log_probs.append(old_log_probs)
                all_advantages.append(adv_ep)
                all_returns.append(ret_ep)
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
        
        # 2. 참조 정책 저장 (KL 발산 계산용)
        if self.reference_policy is None:
            # 첫 업데이트 시 참조 정책 초기화
            # 정책 클래스에 따라 다른 초기화 방법 사용
            policy_class = type(self.policy)
            
            # 정책 클래스별 초기화 파라미터 결정
            # 순서 중요: 더 구체적인 속성부터 체크
            if hasattr(self.policy, 'embedding_dim'):
                # 기존 GRPOPolicy (마지막에 체크)
                self.reference_policy = policy_class(
                    embedding_dim=self.policy.embedding_dim,
                    hidden_dim=self.policy.hidden_dim,
                    action_dim=self.policy.action_dim
                ).to(self.device)
            else:
                # 기본값으로 현재 정책을 복사
                import copy
                self.reference_policy = copy.deepcopy(self.policy)
        
        # 현재 정책을 참조 정책으로 복사
        self.reference_policy.load_state_dict(self.policy.state_dict())
        # GRU 가중치를 연속 메모리 블록으로 재정렬 (cuDNN 역전파 안정성)
        if hasattr(self.reference_policy, 'gru'):
            self.reference_policy.gru.flatten_parameters()
        self.reference_policy.eval()
        
        # 3. 여러 에포크 동안 정책 업데이트 (PPO의 multiple epochs)
        num_epochs = 4  # PPO 표준 설정
        batch_size = self.batch_size
        num_samples = len(all_states)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_divergence = 0.0
        total_clip_fraction = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # 데이터 셔플
            indices = np.random.permutation(num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # 미니 배치 데이터 슬라이싱 후 GPU 메모리로 이동 (VRAM 폭발 방지)
                batch_states = states_tensor[batch_indices].to(self.device, non_blocking=True)
                batch_actions = actions_tensor[batch_indices].long().to(self.device, non_blocking=True)
                batch_old_log_probs = old_log_probs_tensor[batch_indices].to(self.device, non_blocking=True)
                batch_advantages = advantages_tensor[batch_indices].to(self.device, non_blocking=True)
                batch_returns = returns_tensor[batch_indices].to(self.device, non_blocking=True)
                
                # Autocast policy evaluation and loss computation to BF16 for speed/memory efficiency
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # 4. 정책 평가
                    log_probs, entropy, values = self.policy.evaluate_actions(
                        batch_states, batch_actions
                    )
                    
                    # 수치적 안정성을 위해 실수형 변환
                    log_probs = log_probs.float()
                    entropy = entropy.float()
                    values = values.float()
                    
                    # 5. PPO 클리핑 목적 함수 계산
                    # 확률 비율: r_t = π_θ(a|s) / π_θ_old(a|s)
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    
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
                    value_loss = F.mse_loss(values, batch_returns)
                    
                    # 7. 엔트로피 보너스 (탐험 장려)
                    entropy_loss = -entropy.mean()
                    
                    # 8. 총 손실 계산
                    loss = (
                        policy_loss +
                        self.value_coef * value_loss +
                        self.entropy_coef * entropy_loss
                    )
                
                # 9. 그래디언트 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # GRU 가중치 연속 메모리 보장 (optimizer.step이 가중치를 비연속적으로 만들 수 있음)
                if hasattr(self.policy, 'gru'):
                    self.policy.gru.flatten_parameters()
                
                # 10. KL 발산 계산 (조기 종료 체크)
                with torch.no_grad():
                    # Autocast reference policy evaluation
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        ref_log_probs, _, _ = self.reference_policy.evaluate_actions(
                            batch_states, batch_actions
                        )
                        ref_log_probs = ref_log_probs.float()
                    
                    # KL 발산: KL(π_old || π_new)
                    kl_divergence = (batch_old_log_probs - log_probs).mean()
                    
                    # 클리핑 비율 계산
                    clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | 
                                   (ratio > 1.0 + self.clip_epsilon)).float().mean()
                
                # 메트릭 누적
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl_divergence += kl_divergence.item()
                total_clip_fraction += clip_fraction.item()
                num_batches += 1
            
            # KL 발산이 목표값을 초과하면 조기 종료
            avg_kl = total_kl_divergence / num_batches
            if avg_kl > self.kl_target * 1.5:
                logger.warning(f"Early stopping at epoch {epoch + 1}/{num_epochs} "
                             f"due to high KL divergence: {avg_kl:.6f} > {self.kl_target * 1.5:.6f}")
                break
        
        # 평균 메트릭 계산
        update_metrics = {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'entropy': total_entropy / num_batches,
            'kl_divergence': total_kl_divergence / num_batches,
            'clip_fraction': total_clip_fraction / num_batches
        }
        
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
    
    def train(
        self,
        total_episodes: int,
        checkpoint_interval: int = 100,
        checkpoint_path: Optional[str] = None,
        on_iteration_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        start_iteration: int = 0
    ) -> Dict[str, Any]:
        """
        GRPO 훈련 실행
        
        Args:
            total_episodes: 총 훈련 에피소드 수
            checkpoint_interval: 체크포인트 저장 간격
            checkpoint_path: 체크포인트 저장 경로 (format string with {} for iteration)
            on_iteration_end: 매 반복 종료 시 호출될 콜백 함수 (iteration, metrics) -> None
            start_iteration: 시작 반복 횟수 (재개 시 사용)
            
        Returns:
            훈련 메트릭 딕셔너리
        """
        import time
        
        logger.info(f"Starting GRPO training for {total_episodes} episodes...")
        
        num_iterations = total_episodes // (self.episodes_per_group * self.num_groups)
        logger.info(f"Total iterations: {num_iterations} (Starting from {start_iteration})")
        
        start_time = time.time()
        iteration_times = []
        
        for iteration in range(start_iteration, num_iterations):
            iteration_start_time = time.time()
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
                # Format checkpoint path with iteration number
                formatted_checkpoint_path = checkpoint_path.format(iteration + 1)
                self.save_checkpoint(
                    formatted_checkpoint_path, iteration + 1,
                    extra_state=self.extra_checkpoint_state if self.extra_checkpoint_state else None
                )
            
            # 평균 보상 및 추가 메트릭 계산
            mean_reward = np.mean([ep['metadata']['episode_reward'] for ep in episodes])
            
            # 추가 메트릭 계산
            mean_win_rate = np.mean([ep['metadata'].get('win_rate', 0.0) for ep in episodes])
            mean_trades = np.mean([ep['metadata'].get('num_trades', 0) for ep in episodes])
            mean_sharpe = np.mean([ep['metadata'].get('sharpe_ratio', 0.0) for ep in episodes])
            mean_holding_time = np.mean([ep['metadata'].get('avg_holding_time', 0.0) for ep in episodes])
            
            # 콜백 호출 (Curriculum Learning 등)
            if on_iteration_end:
                metrics_summary = {
                    'mean_reward': mean_reward,
                    'mean_win_rate': mean_win_rate,
                    'mean_trades': mean_trades,
                    'mean_sharpe': mean_sharpe,
                    'mean_holding_time': mean_holding_time,
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
            progress_pct = (iteration + 1) / num_iterations * 100
            
            # 시간 포맷팅
            elapsed_time = time.time() - start_time
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(estimated_remaining_time)
            
            logger.info(f"Iteration {iteration + 1}/{num_iterations} ({progress_pct:.1f}%) | "
                       f"Timesteps: {self.total_timesteps} | "
                       f"Mean Reward: {mean_reward:.4f} | "
                       f"Win Rate: {mean_win_rate:.1%} | "
                       f"Trades: {mean_trades:.0f} | "
                       f"Sharpe: {mean_sharpe:.2f} | "
                       f"AvgHold: {mean_holding_time:.1f}s | "
                       f"Policy Loss: {update_metrics['policy_loss']:.4f} | "
                       f"Elapsed: {elapsed_str} | "
                       f"ETA: {remaining_str}")
        
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
        
        # 정책 업데이트 메트릭 (정책 엔트로피, KL 발산 포함)
        for key, value in update_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, iteration)
        
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
            
            # 그룹별 정책 엔트로피 계산
            # 각 에피소드의 상태에서 정책 엔트로피를 계산
            group_entropies = []
            for ep in group_episodes:
                states = ep['states']
                states_tensor = torch.from_numpy(states).float().to(self.device)
                
                with torch.no_grad():
                    # 정책에서 행동 확률 분포 얻기
                    if hasattr(self.policy, 'forward'):
                        result = self.policy(states_tensor)
                        if isinstance(result, tuple):
                            action_logits, _ = result  # (action_logits, state_value) 튜플
                            action_probs = torch.softmax(action_logits, dim=-1)
                        else:
                            action_probs = result
                        
                        # 엔트로피 계산: -sum(p * log(p))
                        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
                        group_entropies.append(entropy.item())
            
            group_mean_entropy = np.mean(group_entropies) if group_entropies else 0.0
            
            # 그룹별 KL 발산 계산
            # 참조 정책과 현재 정책 간의 KL 발산
            group_kl_divergences = []
            if self.reference_policy is not None:
                for ep in group_episodes:
                    states = ep['states']
                    actions = ep['actions']
                    states_tensor = torch.from_numpy(states).float().to(self.device)
                    actions_tensor = torch.from_numpy(actions).long().to(self.device)
                    
                    with torch.no_grad():
                        # 현재 정책의 로그 확률
                        if hasattr(self.policy, 'evaluate_actions'):
                            current_log_probs, _, _ = self.policy.evaluate_actions(
                                states_tensor, actions_tensor
                            )
                            
                            # 참조 정책의 로그 확률
                            ref_log_probs, _, _ = self.reference_policy.evaluate_actions(
                                states_tensor, actions_tensor
                            )
                            
                            # KL 발산: KL(π_ref || π_current)
                            kl_div = (ref_log_probs - current_log_probs).mean()
                            group_kl_divergences.append(kl_div.item())
            
            group_mean_kl = np.mean(group_kl_divergences) if group_kl_divergences else 0.0
            
            # 그룹 메트릭 로깅
            self.writer.add_scalar(f'group_{group_id}/mean_return', group_mean_reward, iteration)
            self.writer.add_scalar(f'group_{group_id}/std_return', group_std_reward, iteration)
            self.writer.add_scalar(f'group_{group_id}/win_rate', group_mean_win_rate, iteration)
            self.writer.add_scalar(f'group_{group_id}/avg_holding_time', group_mean_holding_time, iteration)
            self.writer.add_scalar(f'group_{group_id}/quick_exit_violations', group_total_violations, iteration)
            self.writer.add_scalar(f'group_{group_id}/mean_trades', group_mean_trades, iteration)
            self.writer.add_scalar(f'group_{group_id}/sharpe_ratio', group_mean_sharpe, iteration)
            self.writer.add_scalar(f'group_{group_id}/policy_entropy', group_mean_entropy, iteration)
            self.writer.add_scalar(f'group_{group_id}/kl_divergence', group_mean_kl, iteration)
            
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
        # 만약 모델이 컴파일된 상태라면 실제 원본 모델 추출
        actual_policy = self.policy._orig_mod if hasattr(self.policy, '_orig_mod') else self.policy
        
        # 정책 타입 및 설정 정보 추출
        policy_class_name = actual_policy.__class__.__name__
        policy_config = {
            'policy_type': policy_class_name,
            'action_dim': actual_policy.action_dim if hasattr(actual_policy, 'action_dim') else 3,
            'hidden_dim': actual_policy.hidden_dim if hasattr(actual_policy, 'hidden_dim') else 128,
        }
        
        # GRPOPolicy의 경우 embedding_dim 저장
        if hasattr(actual_policy, 'embedding_dim'):
            policy_config['embedding_dim'] = actual_policy.embedding_dim
        
        checkpoint = {
            'iteration': iteration,
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
            'policy_state_dict': actual_policy.state_dict(),
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
                'max_grad_norm': self.max_grad_norm,
                **policy_config  # 정책 설정 병합
            }
        }
        
        # 추가 상태 저장 (curriculum cost rate 등)
        if extra_state:
            checkpoint['extra_state'] = extra_state
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        if extra_state:
            logger.info(f"  Extra state saved: {extra_state}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
            
        Returns:
            extra_state: 체크포인트에 저장된 추가 상태 (없으면 None)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.num_updates = checkpoint['num_updates']
        
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
