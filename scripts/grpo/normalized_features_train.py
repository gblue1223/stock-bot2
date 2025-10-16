#!/usr/bin/env python3
"""
정규화된 특징 기반 GRPO 훈련 - 정규화 데이터 최적화

핵심 아이디어:
1. 정규화된 데이터의 특성을 활용
2. 통계적 패턴과 시간적 변화 포착
3. 상대적 위치와 추세 분석
4. 기술적 지표 대신 정규화 특징 조합 사용
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.grpo import GRPOTrainer
import duckdb

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NormalizedFeatureEnv:
    """
    정규화된 특징 기반 환경
    
    정규화된 데이터에서 유용한 패턴을 추출하여 거래 신호로 활용
    """
    
    def __init__(
        self,
        db_path: str,
        table_name: str = 'datasets',
        seq_len: int = 30,  # 짧은 시퀀스로 효율성 증대
        expected_features: int = 28,
        transaction_cost_rate: float = 0.00215,
        max_episode_steps: int = 150,
        device: str = 'cpu'
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.seq_len = seq_len
        self.expected_features = expected_features
        self.transaction_cost_rate = transaction_cost_rate
        self.max_episode_steps = max_episode_steps
        self.device = device
        
        # 행동 공간: 0=보유, 1=매수, 2=매도
        self.action_space = type('ActionSpace', (), {'n': 3, 'sample': lambda: np.random.randint(3)})()
        
        # 관측 공간: 정규화된 특징 조합
        # 현재값(10) + 단기변화(5) + 중기변화(3) + 변동성(5) + 상대위치(3) + 추세(3) = 29
        self.feature_dim = 29
        self.observation_space = type('ObservationSpace', (), {'shape': (self.feature_dim,)})()
        
        # 데이터베이스 연결
        self.conn = None
        self._connect_db()
        
        # 에피소드 상태
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.current_price = 0.0
        self.current_time = 0.0
        
        # 현재 에피소드 데이터
        self.episode_data = None
        self.episode_features = None
        self.episode_length = 0
        
        # 에피소드 메타데이터 (통계 추적)
        self.episode_trades = []
        self.episode_rewards = []
        
        logger.info(f"NormalizedFeatureEnv initialized: seq_len={seq_len}, "
                   f"feature_dim={self.feature_dim}, expected_features={expected_features}")
    
    def _connect_db(self):
        """데이터베이스 연결"""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise RuntimeError(f"Cannot connect to database {self.db_path}: {e}")
    
    def _get_feature_columns(self) -> list:
        """특징 컬럼 목록 가져오기"""
        try:
            query = f"DESCRIBE {self.table_name}"
            columns_df = self.conn.execute(query).fetchdf()
            all_columns = columns_df['column_name'].tolist()
            column_types = columns_df['column_type'].tolist()
            
            # 메타데이터 컬럼 제외
            exclude_columns = {'날짜', '종목코드', '시간', '종목명', '번호'}
            
            # 숫자형 컬럼만 선택
            feature_columns = []
            for col, col_type in zip(all_columns, column_types):
                if col not in exclude_columns:
                    if any(numeric_type in col_type.upper() for numeric_type in ['DOUBLE', 'FLOAT', 'INTEGER', 'BIGINT', 'DECIMAL']):
                        feature_columns.append(col)
            
            # 정확히 expected_features 개수만 사용
            if len(feature_columns) >= self.expected_features:
                feature_columns = feature_columns[:self.expected_features]
            else:
                raise RuntimeError(f"Insufficient features: found {len(feature_columns)}, expected {self.expected_features}")
            
            logger.info(f"Selected {len(feature_columns)} normalized feature columns")
            return feature_columns
        except Exception as e:
            logger.error(f"Failed to get feature columns: {e}")
            raise RuntimeError(f"Cannot get feature columns: {e}")
    
    def _sample_episode_start(self):
        """에피소드 시작 지점 샘플링"""
        feature_cols = self._get_feature_columns()
        min_required = self.seq_len + self.max_episode_steps + 30
        
        try:
            # 랜덤 종목 및 날짜 선택
            query = f"""
                SELECT DISTINCT 종목코드, 날짜, COUNT(*) as count
                FROM {self.table_name}
                GROUP BY 종목코드, 날짜
                HAVING COUNT(*) >= ?
                ORDER BY RANDOM()
                LIMIT 1
            """
            result = self.conn.execute(query, [min_required]).fetchdf()
            
            if len(result) == 0:
                raise RuntimeError(f"No data found with minimum {min_required} samples per stock/date")
            
            stock_code = str(result['종목코드'].iloc[0])
            date = int(result['날짜'].iloc[0])
            
            # 정규화된 특징 데이터 로드
            query = f"""
                SELECT {', '.join(feature_cols)}
                FROM {self.table_name}
                WHERE 종목코드 = ? AND 날짜 = ?
                ORDER BY 시간
            """
            df = self.conn.execute(query, [stock_code, date]).fetchdf()
            
            if len(df) >= min_required:
                features = df[feature_cols].values.astype(np.float32)
                
                # 랜덤 시작 지점 선택
                max_start_idx = len(features) - min_required
                if max_start_idx > 0:
                    start_idx = np.random.randint(0, max_start_idx)
                    end_idx = start_idx + min_required
                    features = features[start_idx:end_idx]
                
                logger.debug(f"Sampled episode: stock={stock_code}, date={date}, length={len(features)}")
                return features
            else:
                raise RuntimeError(f"Insufficient data: {len(df)} < {min_required}")
                
        except Exception as e:
            logger.error(f"Failed to sample episode start: {e}")
            raise RuntimeError(f"Cannot sample episode start: {e}")
    
    def _extract_normalized_features(self, normalized_data):
        """정규화된 데이터에서 유용한 특징 추출"""
        seq_len, n_features = normalized_data.shape
        
        # 각 타임스텝에서 추출할 특징들
        features_per_step = []
        
        for i in range(seq_len):
            step_features = []
            
            # 1. 현재 시점의 주요 특징 (첫 10개)
            current_main = normalized_data[i, :10]
            step_features.extend(current_main)
            
            # 2. 단기 변화율 (최근 5스텝)
            if i >= 5:
                recent_change = normalized_data[i, :5] - normalized_data[i-5, :5]
                step_features.extend(recent_change)
            else:
                step_features.extend([0.0] * 5)
            
            # 3. 중기 변화율 (최근 10스텝)
            if i >= 10:
                medium_change = normalized_data[i, :3] - normalized_data[i-10, :3]
                step_features.extend(medium_change)
            else:
                step_features.extend([0.0] * 3)
            
            # 4. 최근 변동성 (최근 10스텝의 표준편차)
            if i >= 10:
                recent_volatility = np.std(normalized_data[i-10:i+1, :5], axis=0)
                step_features.extend(recent_volatility)
            else:
                step_features.extend([0.0] * 5)
            
            # 5. 상대적 위치 (최근 20스텝 내에서의 위치)
            if i >= 20:
                recent_data = normalized_data[i-20:i+1, :3]
                for j in range(3):
                    feature_data = recent_data[:, j]
                    min_val, max_val = np.min(feature_data), np.max(feature_data)
                    if max_val != min_val:
                        relative_pos = (normalized_data[i, j] - min_val) / (max_val - min_val)
                    else:
                        relative_pos = 0.5
                    step_features.append(relative_pos)
            else:
                step_features.extend([0.5] * 3)
            
            # 6. 추세 강도 (선형 회귀 기울기)
            if i >= 10:
                x = np.arange(10)
                for j in range(3):
                    y = normalized_data[i-9:i+1, j]
                    try:
                        slope = np.polyfit(x, y, 1)[0]
                        step_features.append(slope)
                    except:
                        step_features.append(0.0)
            else:
                step_features.extend([0.0] * 3)
            
            features_per_step.append(step_features)
        
        # 배열로 변환
        result = np.array(features_per_step, dtype=np.float32)
        
        # NaN 및 무한대 처리
        result = np.nan_to_num(result, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return result
    
    def _get_current_observation(self):
        """현재 관측값 반환"""
        if self.current_step < len(self.episode_features):
            observation = self.episode_features[self.current_step].copy()
        else:
            observation = np.zeros(self.feature_dim, dtype=np.float32)
        
        return observation
    
    def reset(self, seed=None, options=None):
        """환경 리셋"""
        # 에피소드 데이터 샘플링
        self.episode_data = self._sample_episode_start()
        self.episode_length = len(self.episode_data)
        
        # 정규화된 특징 추출
        self.episode_features = self._extract_normalized_features(self.episode_data)
        
        # 에피소드 상태 초기화
        self.current_step = 25  # 충분한 히스토리 확보 후 시작
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = 0.0
        
        # 현재 가격 및 시간 (첫 번째 특징을 가격으로 사용)
        self.current_price = float(self.episode_data[self.current_step, 0])
        self.current_time = float(self.current_step)  # 스텝을 시간으로 사용
        
        # 에피소드 메타데이터 초기화
        self.episode_trades = []
        self.episode_rewards = []
        
        # 초기 관측값
        observation = self._get_current_observation()
        
        info = {'step': self.current_step}
        
        return observation, info
    
    def step(self, action):
        """행동 실행"""
        reward = 0.0
        terminated = False
        truncated = False
        
        # 행동 실행
        if action == 1:  # 매수
            if self.position == 0:
                self.position = 1
                self.entry_price = self.current_price
                self.entry_time = self.current_time
        
        elif action == 2:  # 매도
            if self.position == 1:
                # 보유 시간 계산
                holding_time = self.current_time - self.entry_time
                
                # 수익률 계산 (정규화된 값이므로 차이로 계산)
                profit = self.current_price - self.entry_price
                
                # 정규화된 데이터에서의 보상 계산
                current_features = self.episode_features[self.current_step]
                
                # 개선된 수익 보상 (더 세밀한 구간)
                if profit > 0.01:  # 0.05 → 0.01 (더 낮은 임계값)
                    reward = 5.0 * profit  # 수익에 비례한 보상
                elif profit > 0.005:
                    reward = 3.0 * profit
                elif profit > 0:
                    reward = 1.0 * profit
                elif profit > -0.005:
                    reward = 2.0 * profit  # 작은 손실은 덜 페널티
                elif profit > -0.01:
                    reward = 3.0 * profit
                else:
                    reward = 5.0 * profit  # 큰 손실은 강한 페널티
                
                # 특징 기반 보너스
                # 단기 변화율이 양수이고 실제로 수익이 났을 때 보너스
                short_term_changes = current_features[10:15]  # 단기 변화율
                if profit > 0 and np.mean(short_term_changes) > 0:
                    reward += 0.5
                
                # 추세가 상승이고 수익이 났을 때 보너스
                trend_features = current_features[26:29]  # 추세 특징
                if profit > 0 and np.mean(trend_features) > 0:
                    reward += 0.3
                
                # 변동성이 높은 상황에서 올바른 방향 거래 시 보너스
                volatility_features = current_features[18:23]  # 변동성 특징
                if profit > 0 and np.mean(volatility_features) > 0.5:
                    reward += 0.2
                
                # 거래비용 차감
                reward -= self.transaction_cost_rate
                
                # 거래 기록 저장
                trade_info = {
                    'entry_price': self.entry_price,
                    'exit_price': self.current_price,
                    'profit': profit,
                    'holding_time': holding_time,
                    'reward': reward
                }
                self.episode_trades.append(trade_info)
                
                # 포지션 청산
                self.position = 0
                self.entry_price = 0.0
                self.entry_time = 0.0
        
        # 보상 기록
        self.episode_rewards.append(reward)
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 에피소드 종료 체크
        if (self.current_step >= min(self.episode_length - 1, self.max_episode_steps + 25) or
            self.current_step >= len(self.episode_features) - 1):
            terminated = True
        
        # 에피소드 종료 시 포지션 강제 청산
        if terminated and self.position == 1:
            holding_time = self.current_time - self.entry_time
            profit = self.current_price - self.entry_price
            
            if profit > 0:
                reward += 0.5
            else:
                reward -= 0.5
            
            # 강제 청산 거래 기록
            trade_info = {
                'entry_price': self.entry_price,
                'exit_price': self.current_price,
                'profit': profit,
                'holding_time': holding_time,
                'reward': reward,
                'forced_liquidation': True
            }
            self.episode_trades.append(trade_info)
            self.episode_rewards.append(reward)
        
        # 현재 가격 및 시간 업데이트
        if not terminated and self.current_step < len(self.episode_data):
            self.current_price = float(self.episode_data[self.current_step, 0])
            self.current_time = float(self.current_step)
        
        # 다음 관측값
        if not terminated:
            observation = self._get_current_observation()
        else:
            observation = np.zeros(self.feature_dim, dtype=np.float32)
        
        # 정보 구성
        info = {
            'step': self.current_step,
            'position': self.position,
            'current_price': self.current_price
        }
        
        # 에피소드 종료 시 메타데이터 추가
        if terminated:
            episode_metadata = self._calculate_episode_metadata()
            info['episode'] = episode_metadata
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_episode_metadata(self):
        """에피소드 종료 시 메타데이터 계산"""
        # 총 수익
        total_return = sum(self.episode_rewards)
        
        # 거래 횟수
        num_trades = len(self.episode_trades)
        
        # 평균 보유 시간
        if self.episode_trades:
            avg_holding_time = np.mean([t['holding_time'] for t in self.episode_trades])
        else:
            avg_holding_time = 0.0
        
        # 샤프 비율 계산
        if len(self.episode_rewards) > 1:
            mean_reward = np.mean(self.episode_rewards)
            std_reward = np.std(self.episode_rewards)
            
            if std_reward > 0:
                sharpe_ratio = mean_reward / std_reward
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # 승률 계산
        if self.episode_trades:
            win_rate = np.mean([1 if t['profit'] > 0 else 0 for t in self.episode_trades])
        else:
            win_rate = 0.0
        
        # 평균 거래당 수익
        if self.episode_trades:
            avg_profit_per_trade = np.mean([t['profit'] for t in self.episode_trades])
        else:
            avg_profit_per_trade = 0.0
        
        metadata = {
            'total_return': float(total_return),
            'num_trades': int(num_trades),
            'avg_holding_time': float(avg_holding_time),
            'sharpe_ratio': float(sharpe_ratio),
            'win_rate': float(win_rate),
            'avg_profit_per_trade': float(avg_profit_per_trade),
            'episode_reward': float(total_return),
            'episode_steps': int(self.current_step),
            'trades': self.episode_trades
        }
        
        logger.info(f"Episode finished: total_return={total_return:.4f}, "
                   f"num_trades={num_trades}, "
                   f"avg_holding_time={avg_holding_time:.2f}s, "
                   f"sharpe_ratio={sharpe_ratio:.4f}, "
                   f"win_rate={win_rate:.2%}")
        
        return metadata
    
    def close(self):
        """환경 종료"""
        if self.conn:
            self.conn.close()


class NormalizedFeaturePolicy(nn.Module):
    """
    정규화된 특징 기반 정책 네트워크
    """
    
    def __init__(
        self,
        feature_dim: int = 29,
        hidden_dim: int = 64,
        action_dim: int = 3
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        # GRPO 훈련기 호환성을 위한 속성
        self.embedding_dim = feature_dim
        
        # 특징 처리 레이어
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 정책 헤드 (행동 확률)
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # 가치 헤드 (상태 가치)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        logger.info(f"NormalizedFeaturePolicy initialized: feature_dim={feature_dim}, "
                   f"hidden_dim={hidden_dim}, action_dim={action_dim}")
    
    def forward(self, state):
        """정책 네트워크 forward pass"""
        # 특징 처리
        x = self.feature_processor(state)
        
        # 정책 헤드: 행동 로짓
        action_logits = self.policy_head(x)
        
        # 가치 헤드: 상태 가치
        state_value = self.value_head(x)
        
        return action_logits, state_value
    
    def get_action(self, state, deterministic=False):
        """정책에서 행동 샘플링"""
        from torch.distributions import Categorical
        
        # Forward pass
        action_logits, _ = self.forward(state)
        
        # 행동 확률 분포 생성
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if deterministic:
            # 최대 확률 행동 선택
            action = torch.argmax(action_probs, dim=-1)
        else:
            # 확률적 샘플링
            action = dist.sample()
        
        # 로그 확률 계산
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(self, states, actions):
        """주어진 상태와 행동에 대한 로그 확률, 엔트로피, 가치 계산"""
        from torch.distributions import Categorical
        
        # Forward pass
        action_logits, state_values = self.forward(states)
        
        # 행동 확률 분포 생성
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # 로그 확률 계산
        log_probs = dist.log_prob(actions)
        
        # 엔트로피 계산
        entropy = dist.entropy()
        
        # 가치를 1D로 변환
        values = state_values.squeeze(-1)
        
        return log_probs, entropy, values


def main():
    """정규화된 특징 기반 GRPO 훈련"""
    
    logger.info("📊 Starting NORMALIZED FEATURES GRPO Training...")
    logger.info("🎯 정규화된 데이터 최적화 특징 학습")
    logger.info("=" * 60)
    
    # 경로 설정
    db_path = r"C:\Users\user\Workspace\datasets@20251013\datasets_norm_all.duckdb"
    output_dir = "models/grpo_normalized_features"
    
    # 경로 확인
    logger.info("🔍 Checking files...")
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found: {db_path}")
        return False
    
    logger.info("✅ Database found")
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Using device: {device}")
    
    try:
        # 1. 정규화된 특징 환경 생성
        logger.info("📊 Creating NORMALIZED FEATURES environment...")
        env = NormalizedFeatureEnv(
            db_path=db_path,
            table_name='datasets',
            seq_len=30,                  # 효율적인 시퀀스 길이
            expected_features=28,
            transaction_cost_rate=0.00215,  # 현실적 거래비용 유지
            max_episode_steps=150,       # 120 → 150 (더 긴 에피소드)
            device=device
        )
        
        logger.info("✅ NORMALIZED FEATURES environment created")
        logger.info("📊 Normalized features:")
        logger.info("  - Current values (10 features)")
        logger.info("  - Short-term changes (5 features)")
        logger.info("  - Medium-term changes (3 features)")
        logger.info("  - Volatility patterns (5 features)")
        logger.info("  - Relative positions (3 features)")
        logger.info("  - Trend slopes (3 features)")
        logger.info(f"  - Total feature dim: {env.feature_dim}")
        
        # 2. 정규화된 특징 정책 네트워크
        logger.info("🧠 Creating NORMALIZED FEATURES policy...")
        
        policy = NormalizedFeaturePolicy(
            feature_dim=env.feature_dim,
            hidden_dim=64,
            action_dim=3
        )
        
        policy.to(device)
        logger.info("✅ NORMALIZED FEATURES policy created")
        
        # 3. 훈련기 설정
        logger.info("📊 Creating trainer...")
        
        os.makedirs(output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
        
        trainer = GRPOTrainer(
            policy=policy,
            env=env,
            episodes_per_group=6,        # 더 많은 샘플
            num_groups=3,                # 세밀한 그룹화
            learning_rate=0.0005,        # 0.0003 → 0.0005 (더 빠른 학습)
            gamma=0.99,
            clip_epsilon=0.2,
            kl_target=0.01,
            entropy_coef=0.05,           # 0.02 → 0.05 (더 많은 탐험)
            value_coef=0.5,
            max_grad_norm=0.5,
            device=device,
            tensorboard_log_dir=tensorboard_dir
        )
        
        logger.info("✅ Trainer created")
        
        # 4. 훈련 설정 (증가된 훈련량)
        logger.info("📊 Starting NORMALIZED FEATURES training...")
        logger.info("=" * 60)
        
        # 증가된 훈련량 (2500 → 10000)
        total_timesteps = 10000          # 4배 증가
        episodes_per_iteration = 6 * 3   # 18 episodes per iteration (증가)
        total_episodes = (total_timesteps // 25) * episodes_per_iteration
        checkpoint_interval = 10         # 더 자주 저장
        
        logger.info(f"📊 NORMALIZED FEATURES Configuration (Extended):")
        logger.info(f"  Total Timesteps: {total_timesteps} (4x increased)")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Feature Dimension: {env.feature_dim}")
        logger.info(f"  Sequence Length: {env.seq_len}")
        logger.info(f"  Learning Rate: 0.0005")
        logger.info(f"  Expected Time: 45-60 minutes")
        
        # 체크포인트 경로
        checkpoint_path = os.path.join(output_dir, 'checkpoints', 'checkpoint_iter{}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # 훈련 시작
        start_time = time.time()
        
        final_metrics = trainer.train(
            total_episodes=total_episodes,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path
        )
        
        training_time = time.time() - start_time
        
        # 5. 결과 출력
        logger.info("=" * 60)
        logger.info("🎉 NORMALIZED FEATURES TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        logger.info(f"📊 Results:")
        logger.info(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Total Timesteps: {final_metrics['total_timesteps']}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(output_dir, 'normalized_features_model.pt')
        trainer.save_checkpoint(final_model_path, final_metrics['num_updates'])
        logger.info(f"  Final Model: {final_model_path}")
        
        logger.info("=" * 60)
        logger.info("🎊 NORMALIZED FEATURES completed!")
        logger.info("📈 Expected improvements:")
        logger.info("  - Optimized for normalized data")
        logger.info("  - Statistical pattern recognition")
        logger.info("  - Temporal change detection")
        logger.info("  - Relative position analysis")
        logger.info(f"📊 TensorBoard: tensorboard --logdir {tensorboard_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return False
    
    finally:
        # 정리
        if 'env' in locals():
            env.close()
        
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    success = main()
    if success:
        print("\n🎉 정규화된 특징 훈련이 완료되었습니다!")
        print("📊 정규화 데이터에 최적화된 학습을 했습니다!")
    else:
        print("\n❌ 훈련이 실패했습니다.")
    
    sys.exit(0 if success else 1)