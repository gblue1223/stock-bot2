#!/usr/bin/env python3
"""
직접 특징 사용 GRPO 훈련 - 임베딩 우회

핵심 아이디어:
1. 임베딩 사용하지 않음
2. 원본 특징을 직접 정책에 입력
3. 단순하지만 효과적인 접근법
4. 거래 관련성 문제 해결
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
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


class DirectFeatureEnv:
    """
    직접 특징 사용 환경
    
    임베딩 없이 원본 특징을 직접 사용하는 단순한 환경
    """
    
    def __init__(
        self,
        db_path: str,
        table_name: str = 'datasets',
        seq_len: int = 60,
        expected_features: int = 28,
        transaction_cost_rate: float = 0.00215,
        max_episode_steps: int = 200,
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
        
        # 관측 공간: 평탄화된 시퀀스
        obs_dim = seq_len * expected_features
        self.observation_space = type('ObservationSpace', (), {'shape': (obs_dim,)})()
        
        # 데이터베이스 연결
        self.conn = None
        self._connect_db()
        
        # 에피소드 상태
        self.current_step = 0
        self.position = 0  # 0: 포지션 없음, 1: 매수 포지션
        self.entry_price = 0.0
        self.current_price = 0.0
        
        # 현재 에피소드 데이터
        self.episode_data = None
        self.episode_length = 0
        
        logger.info(f"DirectFeatureEnv initialized: seq_len={seq_len}, "
                   f"features={expected_features}, obs_dim={obs_dim}")
    
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
            
            logger.info(f"Selected {len(feature_columns)} feature columns")
            return feature_columns
        except Exception as e:
            logger.error(f"Failed to get feature columns: {e}")
            raise RuntimeError(f"Cannot get feature columns: {e}")
    
    def _sample_episode_start(self):
        """에피소드 시작 지점 샘플링"""
        feature_cols = self._get_feature_columns()
        
        min_required = self.seq_len + self.max_episode_steps + 10
        
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
            
            # 해당 종목/날짜의 데이터 로드
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
    
    def _get_current_observation(self):
        """현재 관측값 반환 (평탄화된 시퀀스)"""
        # 현재 스텝에서 seq_len만큼의 시퀀스 추출
        start_idx = max(0, self.current_step - self.seq_len + 1)
        end_idx = self.current_step + 1
        
        sequence = self.episode_data[start_idx:end_idx]
        
        # 시퀀스가 seq_len보다 짧으면 패딩
        if len(sequence) < self.seq_len:
            padding = np.zeros((self.seq_len - len(sequence), sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([padding, sequence])
        
        # 평탄화: (seq_len, features) -> (seq_len * features,)
        observation = sequence.flatten()
        
        return observation
    
    def reset(self, seed=None, options=None):
        """환경 리셋"""
        # 에피소드 데이터 샘플링
        self.episode_data = self._sample_episode_start()
        self.episode_length = len(self.episode_data)
        
        # 에피소드 상태 초기화
        self.current_step = self.seq_len - 1  # 최소 seq_len만큼의 히스토리 필요
        self.position = 0
        self.entry_price = 0.0
        
        # 현재 가격 (첫 번째 특징이 가격이라고 가정)
        self.current_price = float(self.episode_data[self.current_step, 0])
        
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
        
        elif action == 2:  # 매도
            if self.position == 1:
                # 수익률 계산
                profit_rate = (self.current_price - self.entry_price) / self.entry_price
                
                # 단순한 보상: 수익이면 +1, 손실이면 -1
                if profit_rate > 0.001:  # 0.1% 이상 수익
                    reward = 1.0
                elif profit_rate < -0.001:  # 0.1% 이상 손실
                    reward = -1.0
                else:
                    reward = 0.0
                
                # 거래비용 차감
                reward -= self.transaction_cost_rate
                
                # 포지션 청산
                self.position = 0
                self.entry_price = 0.0
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 에피소드 종료 체크
        if self.current_step >= min(self.episode_length - 1, self.max_episode_steps):
            terminated = True
        
        # 에피소드 종료 시 포지션 강제 청산
        if terminated and self.position == 1:
            profit_rate = (self.current_price - self.entry_price) / self.entry_price
            if profit_rate > 0:
                reward += 0.5  # 작은 보상
            else:
                reward -= 0.5  # 작은 페널티
        
        # 현재 가격 업데이트
        if not terminated:
            self.current_price = float(self.episode_data[self.current_step, 0])
        
        # 다음 관측값
        if not terminated:
            observation = self._get_current_observation()
        else:
            observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        info = {
            'step': self.current_step,
            'position': self.position,
            'current_price': self.current_price
        }
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """환경 종료"""
        if self.conn:
            self.conn.close()


class DirectFeaturePolicy(nn.Module):
    """
    직접 특징 사용 정책 네트워크
    
    임베딩 없이 평탄화된 시퀀스를 직접 입력으로 사용
    """
    
    def __init__(
        self,
        input_dim: int,  # seq_len * features
        hidden_dim: int = 128,
        action_dim: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        # GRPO 훈련기 호환성을 위한 속성
        self.embedding_dim = input_dim
        
        # 입력 차원 축소
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 공유 특징 추출 레이어
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 정책 헤드 (행동 확률)
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # 가치 헤드 (상태 가치)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        logger.info(f"DirectFeaturePolicy initialized: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, action_dim={action_dim}")
    
    def forward(self, state):
        """정책 네트워크 forward pass"""
        # 입력 차원 축소
        x = F.relu(self.input_projection(state))
        
        # 공유 특징 추출
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
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
    """직접 특징 사용 GRPO 훈련"""
    
    logger.info("🎯 Starting DIRECT FEATURES GRPO Training...")
    logger.info("🔧 임베딩 우회 - 원본 특징 직접 사용")
    logger.info("=" * 60)
    
    # 경로 설정
    db_path = r"C:\Users\user\Workspace\datasets@20251013\datasets_norm_all.duckdb"
    output_dir = "models/grpo_direct_features"
    
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
        # 1. 직접 특징 환경 생성
        logger.info("🎯 Creating DIRECT FEATURES environment...")
        env = DirectFeatureEnv(
            db_path=db_path,
            table_name='datasets',
            seq_len=30,                  # 더 짧은 시퀀스 (계산 효율성)
            expected_features=28,
            transaction_cost_rate=0.00215,
            max_episode_steps=100,       # 짧은 에피소드
            device=device
        )
        
        logger.info("✅ DIRECT FEATURES environment created")
        logger.info("🎯 Direct features settings:")
        logger.info("  - No embedding (direct raw features)")
        logger.info("  - Sequence length: 30 (efficient)")
        logger.info("  - Transaction cost: 0.1% (low)")
        logger.info("  - Episode steps: 100 (short)")
        logger.info(f"  - Observation dim: {env.observation_space.shape[0]}")
        
        # 2. 직접 특징 정책 네트워크
        logger.info("🧠 Creating DIRECT FEATURES policy...")
        
        input_dim = env.observation_space.shape[0]  # seq_len * features
        policy = DirectFeaturePolicy(
            input_dim=input_dim,
            hidden_dim=64,   # 작은 네트워크
            action_dim=3
        )
        
        policy.to(device)
        logger.info("✅ DIRECT FEATURES policy created")
        
        # 3. 보수적 훈련기 설정
        logger.info("🎯 Creating CONSERVATIVE trainer...")
        
        os.makedirs(output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
        
        trainer = GRPOTrainer(
            policy=policy,
            env=env,
            episodes_per_group=4,        # 작은 그룹
            num_groups=2,                # 단순한 그룹화
            learning_rate=0.0003,        # 보수적 학습률
            gamma=0.99,                  # 표준 할인율
            clip_epsilon=0.2,            # 표준 클리핑
            kl_target=0.01,              # 표준 KL
            entropy_coef=0.05,           # 적당한 탐험
            value_coef=0.5,              # 가치 함수 중시
            max_grad_norm=0.5,           # 안정적 그래디언트
            device=device,
            tensorboard_log_dir=tensorboard_dir
        )
        
        logger.info("✅ CONSERVATIVE trainer created")
        
        # 4. 훈련 설정
        logger.info("🎯 Starting DIRECT FEATURES training...")
        logger.info("=" * 60)
        
        # 적당한 규모
        total_timesteps = 2000           # 적당한 규모
        episodes_per_iteration = 4 * 2   # 8 episodes per iteration
        total_episodes = (total_timesteps // 25) * episodes_per_iteration
        checkpoint_interval = 5          # 자주 저장
        
        logger.info(f"📊 DIRECT FEATURES Configuration:")
        logger.info(f"  Total Timesteps: {total_timesteps}")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Episodes per Group: 4")
        logger.info(f"  Input Dimension: {input_dim}")
        logger.info(f"  Learning Rate: 0.0003")
        logger.info(f"  Expected Time: 15-25 minutes")
        
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
        logger.info("🎉 DIRECT FEATURES TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        logger.info(f"📊 Results:")
        logger.info(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Total Timesteps: {final_metrics['total_timesteps']}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(output_dir, 'direct_features_model.pt')
        trainer.save_checkpoint(final_model_path, final_metrics['num_updates'])
        logger.info(f"  Final Model: {final_model_path}")
        
        logger.info("=" * 60)
        logger.info("🎊 DIRECT FEATURES completed!")
        logger.info("📈 Expected improvements:")
        logger.info("  - No embedding quality issues")
        logger.info("  - Direct trading signal access")
        logger.info("  - Faster convergence")
        logger.info("  - Better trading relevance")
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
        print("\n🎉 직접 특징 훈련이 완료되었습니다!")
        print("📈 임베딩 문제를 우회했습니다!")
    else:
        print("\n❌ 훈련이 실패했습니다.")
    
    sys.exit(0 if success else 1)