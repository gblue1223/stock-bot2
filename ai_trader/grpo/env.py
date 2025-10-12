"""
GRPO 스캘핑 환경

초단위 스캘핑을 위한 Gymnasium 환경을 정의합니다.
임베딩 모델을 사용하여 관측값을 생성하고, 스캘핑 특화 보상 구조를 제공합니다.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import duckdb

logger = logging.getLogger(__name__)


class GRPOScalpingEnv(gym.Env):
    """
    스캘핑을 위한 GRPO 훈련 환경
    
    관측 공간: 임베딩 벡터 (embedding_dim,)
    행동 공간: Discrete(3) - 0: 보유, 1: 매수, 2: 매도
    
    보상 구조:
    - 수익: (청산가 - 진입가) / 진입가 - 거래비용
    - 거래비용: 수수료(0.015%) + 세금(0.2%) = 0.215% (매수/매도 각각 적용)
    - 총 거래비용: 0.43% (왕복)
    - 양의 보상 조건: 수익률 > 0.43%
    - 빠른 손절 룰 위반: -0.01 페널티 (시간 임계값 설정 가능, 기본값 1.5초)
    - 장기 보유 페널티: -0.001 * (보유시간 - 60초)
    
    Args:
        embedding_model: 훈련된 임베딩 모델
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명 (기본값: 'datasets')
        seq_len: 시퀀스 길이 (기본값: 60)
        embedding_dim: 임베딩 차원 (기본값: 128)
        transaction_cost_rate: 거래 비용 비율 (기본값: 0.00215 = 0.215%)
        quick_exit_threshold: 빠른 손절 시간 임계값 (초, 기본값: 1.5)
        quick_exit_penalty: 빠른 손절 룰 위반 페널티 (기본값: 0.01)
        max_holding_time: 최대 보유 시간 (초, 기본값: 60)
        holding_penalty_rate: 장기 보유 페널티 비율 (기본값: 0.001)
        device: 디바이스 ('cpu' 또는 'cuda')
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        embedding_model: torch.nn.Module,
        db_path: str,
        table_name: str = 'datasets',
        seq_len: int = 60,
        embedding_dim: int = 128,
        transaction_cost_rate: float = 0.00215,
        quick_exit_threshold: float = 1.5,
        quick_exit_penalty: float = 0.01,
        max_holding_time: float = 60.0,
        holding_penalty_rate: float = 0.001,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.embedding_model = embedding_model
        self.embedding_model.eval()  # 추론 모드
        self.device = device
        self.embedding_model.to(device)
        
        self.db_path = db_path
        self.table_name = table_name
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        # 거래 비용 설정
        self.transaction_cost_rate = transaction_cost_rate
        self.round_trip_cost = transaction_cost_rate * 2  # 왕복 거래비용: 0.43%
        
        # 빠른 손절 룰 설정
        self.quick_exit_threshold = quick_exit_threshold
        self.quick_exit_penalty = quick_exit_penalty
        
        # 보유 시간 페널티 설정
        self.max_holding_time = max_holding_time
        self.holding_penalty_rate = holding_penalty_rate
        
        # 관측 공간: 임베딩 벡터
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(embedding_dim,),
            dtype=np.float32
        )
        
        # 행동 공간: 0=보유, 1=매수, 2=매도
        self.action_space = spaces.Discrete(3)
        
        # 데이터베이스 연결
        self.conn = None
        self._connect_db()
        
        # 에피소드 상태
        self.current_step = 0
        self.position = 0  # 0: 포지션 없음, 1: 매수 포지션
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.current_price = 0.0
        self.current_time = 0.0
        
        # 에피소드 메타데이터
        self.episode_trades = []
        self.episode_rewards = []
        self.quick_exit_violations = 0
        
        # 현재 에피소드 데이터
        self.episode_data = None
        self.episode_metadata = None
        self.episode_length = 0
        
        logger.info(f"GRPOScalpingEnv initialized with embedding_dim={embedding_dim}, "
                   f"quick_exit_threshold={quick_exit_threshold}s, "
                   f"transaction_cost={transaction_cost_rate*100:.3f}%")
    
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
            
            # 메타데이터 컬럼 및 문자열 컬럼 제외
            exclude_columns = {'날짜', '종목코드', '시간', '종목명'}  # 시간 컬럼도 제외
            
            # 숫자형 컬럼만 선택
            feature_columns = []
            for col, col_type in zip(all_columns, column_types):
                if col not in exclude_columns:
                    # 숫자형 타입만 포함 (VARCHAR, TEXT 등 문자열 타입 제외)
                    if any(numeric_type in col_type.upper() for numeric_type in ['DOUBLE', 'FLOAT', 'INTEGER', 'BIGINT', 'DECIMAL']):
                        feature_columns.append(col)
                    else:
                        logger.debug(f"Excluding non-numeric column: {col} (type: {col_type})")
            
            logger.info(f"Selected {len(feature_columns)} numeric feature columns")
            return feature_columns
        except Exception as e:
            logger.error(f"Failed to get feature columns: {e}")
            raise RuntimeError(f"Cannot get feature columns: {e}")
    
    def _sample_episode_start(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        에피소드 시작 지점 샘플링
        
        다양한 시장 상황을 가진 시작 지점을 DuckDB에서 샘플링합니다.
        
        Returns:
            (data, metadata) 튜플
            - data: (episode_length, n_features)
            - metadata: (episode_length, 3) - [종목코드, 날짜, 시간]
        """
        feature_cols = self._get_feature_columns()
        
        max_attempts = 10  # 최대 시도 횟수 제한
        attempt = 0
        
        while attempt < max_attempts:
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
                min_required = self.seq_len + 100
                result = self.conn.execute(query, [min_required]).fetchdf()
                
                if len(result) == 0:
                    # 충분한 데이터가 있는 조합이 없으면 요구사항을 낮춤
                    logger.warning(f"No stock/date combination with {min_required} samples found, trying with lower requirement")
                    min_required = self.seq_len + 10
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
                available_count = int(result['count'].iloc[0])
                
                # 해당 종목/날짜의 데이터 로드
                query = f"""
                    SELECT 종목코드, 날짜, 시간, {', '.join(feature_cols)}
                    FROM {self.table_name}
                    WHERE 종목코드 = ? AND 날짜 = ?
                    ORDER BY 시간
                """
                df = self.conn.execute(query, [stock_code, date]).fetchdf()
                
                if len(df) >= self.seq_len + 10:  # 최소 요구사항 충족
                    # 메타데이터와 특징 분리
                    metadata = df[['종목코드', '날짜', '시간']].values
                    features = df[feature_cols].values.astype(np.float32)
                    
                    logger.debug(f"Sampled episode: stock={stock_code}, date={date}, length={len(features)}")
                    
                    return features, metadata
                else:
                    logger.warning(f"Insufficient data for stock={stock_code}, date={date}: {len(df)} samples")
                    attempt += 1
                    continue
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to sample episode start: {e}")
                attempt += 1
                if attempt >= max_attempts:
                    raise RuntimeError(f"Cannot sample episode start after {max_attempts} attempts: {e}")
                continue
        
        raise RuntimeError(f"Cannot sample episode start after {max_attempts} attempts")
    
    def _get_embedding(self, sequence: np.ndarray) -> np.ndarray:
        """
        시퀀스를 임베딩으로 변환
        
        Args:
            sequence: (seq_len, n_features)
            
        Returns:
            임베딩 벡터 (embedding_dim,)
        """
        with torch.no_grad():
            # (seq_len, n_features) -> (1, seq_len, n_features)
            seq_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
            
            # 임베딩 생성 - MaskedAutoEncoder는 (reconstruction, embedding, mask) 튜플을 반환
            result = self.embedding_model(seq_tensor)
            
            if isinstance(result, tuple):
                # MaskedAutoEncoder의 경우: (reconstruction, embedding, mask)
                _, embedding, _ = result
            else:
                # 일반 AutoEncoder의 경우: embedding만 반환
                embedding = result
            
            # (1, embedding_dim) -> (embedding_dim,)
            embedding = embedding.squeeze(0).cpu().numpy()
            
        return embedding
    
    def _get_current_observation(self) -> np.ndarray:
        """
        현재 관측값 (임베딩) 반환
        
        Returns:
            임베딩 벡터 (embedding_dim,)
        """
        # 현재 스텝에서 seq_len만큼의 시퀀스 추출
        start_idx = max(0, self.current_step - self.seq_len + 1)
        end_idx = self.current_step + 1
        
        sequence = self.episode_data[start_idx:end_idx]
        
        # 시퀀스가 seq_len보다 짧으면 패딩
        if len(sequence) < self.seq_len:
            padding = np.zeros((self.seq_len - len(sequence), sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([padding, sequence])
        
        # 임베딩 생성
        embedding = self._get_embedding(sequence)
        
        return embedding
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        환경 리셋
        
        Args:
            seed: 랜덤 시드
            options: 추가 옵션
            
        Returns:
            (observation, info) 튜플
        """
        super().reset(seed=seed)
        
        # 에피소드 데이터 샘플링
        self.episode_data, self.episode_metadata = self._sample_episode_start()
        self.episode_length = len(self.episode_data)
        
        # 에피소드 상태 초기화
        self.current_step = self.seq_len - 1  # 최소 seq_len만큼의 히스토리 필요
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = 0.0
        
        # 현재 가격 및 시간 (메타데이터의 시간 컬럼 사용)
        self.current_price = self._get_current_price()
        self.current_time = float(self.episode_metadata[self.current_step, 2])
        
        # 에피소드 메타데이터 초기화
        self.episode_trades = []
        self.episode_rewards = []
        self.quick_exit_violations = 0
        
        # 초기 관측값
        observation = self._get_current_observation()
        
        info = {
            'stock_code': str(self.episode_metadata[self.current_step, 0]),
            'date': str(self.episode_metadata[self.current_step, 1]),
            'time': self.current_time
        }
        
        return observation, info
    
    def _get_current_price(self) -> float:
        """
        현재 가격 반환
        
        실제로는 '등락률' 특징을 사용하여 가격 변화를 시뮬레이션합니다.
        간단히 하기 위해 등락률을 누적하여 가격을 계산합니다.
        """
        # 등락률은 첫 번째 특징이라고 가정 (실제로는 feature_columns에서 확인 필요)
        # 여기서는 간단히 인덱스 0을 사용
        return float(self.episode_data[self.current_step, 0])
    
    def _calculate_reward(self, entry_price: float, exit_price: float, holding_time: float) -> Tuple[float, Dict[str, float]]:
        """
        보상 계산
        
        보상 구조:
        - 수익률 기반 보상: (청산가 - 진입가) / 진입가 - 거래비용
        - 거래비용: 수수료(0.015%) + 세금(0.2%) = 0.215% (매수/매도 각각 적용)
        - 총 거래비용: 0.43% (왕복)
        - 양의 보상 조건: 수익률 > 0.43%
        - 장기 보유 페널티: -0.001 * (보유시간 - 60초)
        
        Args:
            entry_price: 진입 가격
            exit_price: 청산 가격
            holding_time: 보유 시간 (초)
            
        Returns:
            (total_reward, reward_components) 튜플
            - total_reward: 총 보상
            - reward_components: 보상 구성 요소 딕셔너리
        """
        # 수익률 계산: (청산가 - 진입가) / 진입가
        profit_rate = (exit_price - entry_price) / entry_price
        
        # 거래 비용 차감 (왕복 0.43%)
        # 양의 보상을 받으려면 수익률이 0.43%를 초과해야 함
        reward = profit_rate - self.round_trip_cost
        
        # 장기 보유 페널티: -0.001 * (보유시간 - 60초)
        holding_penalty = 0.0
        if holding_time > self.max_holding_time:
            holding_penalty = self.holding_penalty_rate * (holding_time - self.max_holding_time)
            reward -= holding_penalty
        
        # 보상 구성 요소
        reward_components = {
            'profit_rate': profit_rate,
            'transaction_cost': -self.round_trip_cost,
            'holding_penalty': -holding_penalty,
            'net_reward': reward
        }
        
        return reward, reward_components
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        행동 실행
        
        Args:
            action: 0=보유, 1=매수, 2=매도
            
        Returns:
            (observation, reward, terminated, truncated, info) 튜플
        """
        reward = 0.0
        terminated = False
        truncated = False
        quick_exit_triggered = False
        
        # 행동 실행
        if action == 1:  # 매수
            if self.position == 0:
                self.position = 1
                self.entry_price = self.current_price
                self.entry_time = self.current_time
                logger.debug(f"Buy at price={self.entry_price:.4f}, time={self.entry_time}")
        
        elif action == 2:  # 매도
            if self.position == 1:
                # 보유 시간 계산
                holding_time = self.current_time - self.entry_time
                
                # 보상 계산
                reward, reward_components = self._calculate_reward(
                    self.entry_price,
                    self.current_price,
                    holding_time
                )
                
                # 거래 기록
                trade_info = {
                    'entry_price': self.entry_price,
                    'exit_price': self.current_price,
                    'holding_time': holding_time,
                    'profit_rate': reward_components['profit_rate'],
                    'reward': reward,
                    'reward_components': reward_components
                }
                self.episode_trades.append(trade_info)
                
                logger.debug(f"Sell at price={self.current_price:.4f}, "
                           f"profit_rate={reward_components['profit_rate']:.4f}, "
                           f"reward={reward:.4f}, holding_time={holding_time:.2f}s")
                
                # 포지션 청산
                self.position = 0
                self.entry_price = 0.0
                self.entry_time = 0.0
        
        elif action == 0:  # 보유
            # 빠른 손절 룰 체크: 매수 후 설정 가능한 시간 임계값 이내에 가격이 상승하지 않으면 자동 매도
            if self.position == 1:
                holding_time = self.current_time - self.entry_time
                
                # 임계값 이내이고 가격이 상승하지 않았는지 체크
                if holding_time <= self.quick_exit_threshold and self.current_price <= self.entry_price:
                    # 빠른 손절 룰 위반
                    quick_exit_triggered = True
                    self.quick_exit_violations += 1
                    
                    # 자동 매도 및 페널티 적용
                    reward, reward_components = self._calculate_reward(
                        self.entry_price,
                        self.current_price,
                        holding_time
                    )
                    
                    # 빠른 손절 룰 위반 페널티 추가
                    reward -= self.quick_exit_penalty
                    
                    # 거래 기록
                    trade_info = {
                        'entry_price': self.entry_price,
                        'exit_price': self.current_price,
                        'holding_time': holding_time,
                        'profit_rate': reward_components['profit_rate'],
                        'reward': reward,
                        'reward_components': reward_components,
                        'quick_exit_violation': True,
                        'quick_exit_penalty': self.quick_exit_penalty
                    }
                    self.episode_trades.append(trade_info)
                    
                    logger.debug(f"Quick exit rule triggered: price={self.current_price:.4f}, "
                               f"entry_price={self.entry_price:.4f}, "
                               f"holding_time={holding_time:.2f}s, "
                               f"penalty={self.quick_exit_penalty:.4f}, "
                               f"reward={reward:.4f}")
                    
                    # 포지션 청산
                    self.position = 0
                    self.entry_price = 0.0
                    self.entry_time = 0.0
        
        # 보상 기록
        self.episode_rewards.append(reward)
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 에피소드 종료 체크
        if self.current_step >= self.episode_length - 1:
            terminated = True
            
            # 포지션이 남아있으면 강제 청산
            if self.position == 1:
                holding_time = self.current_time - self.entry_time
                final_reward, final_components = self._calculate_reward(
                    self.entry_price,
                    self.current_price,
                    holding_time
                )
                self.episode_rewards.append(final_reward)
                
                # 강제 청산 거래 기록
                self.episode_trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': self.current_price,
                    'holding_time': holding_time,
                    'profit_rate': final_components['profit_rate'],
                    'reward': final_reward,
                    'reward_components': final_components,
                    'forced_liquidation': True
                })
                
                logger.debug(f"Forced liquidation: reward={final_reward:.4f}")
        
        # 현재 가격 및 시간 업데이트
        if not terminated:
            self.current_price = self._get_current_price()
            self.current_time = float(self.episode_metadata[self.current_step, 2])
        
        # 다음 관측값
        observation = self._get_current_observation() if not terminated else np.zeros(self.embedding_dim, dtype=np.float32)
        
        # 정보
        info = {
            'position': self.position,
            'current_price': self.current_price,
            'current_time': self.current_time,
            'quick_exit_violations': self.quick_exit_violations,
            'quick_exit_triggered': quick_exit_triggered
        }
        
        if terminated:
            # 에피소드 종료 시 메타데이터 추가
            episode_metadata = self._calculate_episode_metadata()
            info['episode'] = episode_metadata
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_episode_metadata(self) -> Dict[str, Any]:
        """
        에피소드 종료 시 메타데이터 계산
        
        요구사항 3.7에 따라 다음 메트릭을 계산합니다:
        - 총 수익 (total_return)
        - 거래 횟수 (num_trades)
        - 평균 보유 시간 (avg_holding_time)
        - 샤프 비율 (sharpe_ratio)
        - 빠른 손절 룰 위반 횟수 (quick_exit_violations)
        
        Returns:
            에피소드 메타데이터 딕셔너리
        """
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
        # 샤프 비율 = (평균 수익률 - 무위험 수익률) / 수익률 표준편차
        # 스캘핑의 경우 무위험 수익률은 0으로 가정
        if len(self.episode_rewards) > 1:
            mean_reward = np.mean(self.episode_rewards)
            std_reward = np.std(self.episode_rewards)
            
            # 표준편차가 0이면 샤프 비율은 0
            if std_reward > 0:
                sharpe_ratio = mean_reward / std_reward
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # 승률 계산 (추가 메트릭)
        if self.episode_trades:
            win_rate = np.mean([1 if t['reward'] > 0 else 0 for t in self.episode_trades])
        else:
            win_rate = 0.0
        
        # 평균 거래당 수익 (추가 메트릭)
        if self.episode_trades:
            avg_profit_per_trade = np.mean([t['reward'] for t in self.episode_trades])
        else:
            avg_profit_per_trade = 0.0
        
        metadata = {
            'total_return': float(total_return),
            'num_trades': int(num_trades),
            'avg_holding_time': float(avg_holding_time),
            'sharpe_ratio': float(sharpe_ratio),
            'quick_exit_violations': int(self.quick_exit_violations),
            'win_rate': float(win_rate),
            'avg_profit_per_trade': float(avg_profit_per_trade),
            'episode_length': int(self.episode_length),
            'steps_taken': int(self.current_step),
            'trades': self.episode_trades  # 개별 거래 데이터 포함 (평가용)
        }
        
        logger.info(f"Episode finished: total_return={total_return:.4f}, "
                   f"num_trades={num_trades}, "
                   f"avg_holding_time={avg_holding_time:.2f}s, "
                   f"sharpe_ratio={sharpe_ratio:.4f}, "
                   f"quick_exit_violations={self.quick_exit_violations}, "
                   f"win_rate={win_rate:.2%}")
        
        return metadata
    
    def close(self):
        """환경 종료"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
