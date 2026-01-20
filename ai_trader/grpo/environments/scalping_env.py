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
from datetime import datetime, timedelta  # ✅ 시간 계산용 추가

# ✅ RollingNormalizer import
from lib.rolling_normalization import RollingNormalizer

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
    - 빠른 손절 룰 위반: -0.01 페널티 (시간 임계값 설정 가능, 기본값 1.5초)
    
    Args:
        embedding_model: 훈련된 임베딩 모델
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명 (기본값: 'datasets')
        seq_len: 시퀀스 길이 (기본값: 60)
        embedding_dim: 임베딩 차원 (기본값: 128)
        expected_features: 예상 특징 수 (기본값: 28, 임베딩 모델과 일치해야 함)
        transaction_cost_rate: 거래 비용 비율 (기본값: 0.00215 = 0.215%)
        quick_exit_threshold: 빠른 손절 시간 임계값 (초, 기본값: 1.5)
        quick_exit_penalty: 빠른 손절 룰 위반 페널티 (기본값: 0.01)
        max_episode_steps: 에피소드당 최대 스텝 수 (기본값: None, 제한 없음)
        quick_exit_mode: 빠른 손절 룰 동작 모드 (기본값: 'penalty_only')
            - 'penalty_only': 페널티만 부여, 정책이 학습
            - 'force_close': 강제 청산 (이전 동작)
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
        expected_features: int = 28,
        transaction_cost_rate: float = 0.00215,
        quick_exit_penalty: float = 0.01,
        quick_exit_threshold: float = 1.5,
        quick_exit_mode: str = 'penalty_only',
        max_episode_steps: Optional[int] = None,
        use_raw_data: bool = True,  # ✅ 원본 데이터 사용 여부
        rolling_window_size: int = 1000,  # ✅ Rolling window 크기
        rolling_min_samples: int = 100,  # ✅ 최소 샘플 수
        base_price: float = 100000.0,  # ✅ 기준 가격 (기본값: 10만원)
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.base_price = base_price
        
        self.embedding_model = embedding_model
        self.embedding_model.eval()  # 추론 모드
        self.device = device
        self.embedding_model.to(device)
        
        self.db_path = db_path
        self.table_name = table_name
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.expected_features = expected_features
        
        # 거래 비용 설정
        self.transaction_cost_rate = transaction_cost_rate
        self.round_trip_cost = transaction_cost_rate * 2  # 왕복 거래비용: 0.43%
        
        # 빠른 손절 룰 설정
        self.quick_exit_threshold = quick_exit_threshold
        self.quick_exit_penalty = quick_exit_penalty
        self.quick_exit_mode = quick_exit_mode
        
        # 동작 모드 검증
        if quick_exit_mode not in ['penalty_only', 'force_close']:
            raise ValueError(f"Invalid quick_exit_mode: {quick_exit_mode}. "
                           f"Must be 'penalty_only' or 'force_close'")
        

        
        # 에피소드 길이 제한
        self.max_episode_steps = max_episode_steps
        
        # ✅ 정규화 설정
        self.use_raw_data = use_raw_data
        self.rolling_window_size = rolling_window_size
        self.rolling_min_samples = rolling_min_samples
        
        # ✅ RollingNormalizer 초기화 (원본 데이터 사용 시)
        if use_raw_data:
            # FEATURE_NAMES는 live_trading.py와 동일한 순서여야 함
            from lib.normalization import FEATURE_NAMES
            self.normalizer = RollingNormalizer(
                window_size=rolling_window_size,
                min_samples=rolling_min_samples,
                feature_names=FEATURE_NAMES
            )
            logger.info(f"RollingNormalizer enabled: window={rolling_window_size}, min_samples={rolling_min_samples}")
        else:
            self.normalizer = None
            logger.info("Using pre-normalized data from database")
        
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
        
        # ✅ 특징 컬럼 및 주요 인덱스 식별
        # 컬럼 순서 변경으로 인한 버그 방지를 위해 동적으로 인덱스 찾기
        self.feature_columns = self._get_feature_columns()
        
        try:
            self.return_rate_index = self.feature_columns.index('등락률')
            logger.info(f"Column Mapping Identified: '등락률' at Index {self.return_rate_index}")
        except ValueError as e:
            logger.error(f"Critical Column Missing: {e}")
            logger.error(f"Available columns: {self.feature_columns}")
            raise RuntimeError(f"Required column (등락률) missing from database features")
        
        # 에피소드 상태
        self.current_step = 0
        self.position = 0  # 0: 포지션 없음, 1: 매수 포지션
        self.entry_price = 0.0
        self.entry_price = 0.0
        self.entry_time = 0
        self.current_price = 0.0
        self.current_time = 0
        
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
                   f"quick_exit_mode={quick_exit_mode}, "
                   f"transaction_cost={transaction_cost_rate*100:.3f}%, "
                   f"use_raw_data={use_raw_data}")
        
        # ✅ 유효한 에피소드 키 캐싱
        self.valid_keys = []
        self._preload_valid_keys()

    def _preload_valid_keys(self):
        """유효한 에피소드 키(종목코드, 날짜) 미리 로드"""
        logger.info("Preloading valid episode keys from database...")
        
        # 필요한 최소 데이터 길이 계산
        if self.max_episode_steps is not None:
            min_required = self.seq_len + self.max_episode_steps + 100
        else:
            min_required = self.seq_len + 100
            
        try:
            query = f"""
                SELECT 종목코드, 날짜, COUNT(*) as cnt
                FROM {self.table_name}
                GROUP BY 종목코드, 날짜
                HAVING COUNT(*) >= ?
            """
            result = self.conn.execute(query, [min_required]).fetchdf()
            
            if len(result) == 0:
                logger.warning(f"No keys found with {min_required} samples. Trying with reduced requirement.")
                min_required = self.seq_len + 10
                result = self.conn.execute(query, [min_required]).fetchdf()
            
            if len(result) > 0:
                self.valid_keys = list(zip(result['종목코드'], result['날짜'], result['cnt']))
                logger.info(f"✅ Loaded {len(self.valid_keys)} valid episode keys.")
            else:
                raise RuntimeError(f"No valid data found in database (min samples={min_required})")
                
        except Exception as e:
            logger.error(f"Failed to preload keys: {e}")
            raise
    
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
            
            # 메타데이터 컬럼 및 문자열 컬럼 제외 (번호 컬럼 명시적 제외)
            # 메타데이터 컬럼 및 문자열 컬럼 제외 (번호 컬럼 명시적 제외)
            # ✅ '현재가' 제외: 등락률과 기준가로 가격을 계산하므로 원본 현재가는 피처에서 제외
            exclude_columns = {'날짜', '종목코드', '시간', '종목명', '번호', '현재가'}  # '번호', '현재가' 컬럼 명시적 제외
            
            # 숫자형 컬럼만 선택
            feature_columns = []
            for col, col_type in zip(all_columns, column_types):
                if col not in exclude_columns:
                    # 숫자형 타입만 포함 (VARCHAR, TEXT 등 문자열 타입 제외)
                    if any(numeric_type in col_type.upper() for numeric_type in ['DOUBLE', 'FLOAT', 'INTEGER', 'BIGINT', 'DECIMAL']):
                        feature_columns.append(col)
                    else:
                        logger.debug(f"Excluding non-numeric column: {col} (type: {col_type})")
            
            # 임베딩 모델과 호환성을 위해 정확한 수의 특징만 사용
            expected_features = self.expected_features
            
            # 상세 로그: 발견된 모든 피처 출력
            # logger.info(f"Found {len(feature_columns)} numeric features in database (after excluding metadata): {', '.join(feature_columns)}")
            
            # 정확히 28개 피처가 되도록 처리
            if len(feature_columns) == expected_features:
                logger.info(f"✅ Perfect match: {len(feature_columns)} features = {expected_features} expected")
            elif len(feature_columns) > expected_features:
                # 추가 제거가 필요한 경우
                excluded_features = feature_columns[expected_features:]
                logger.warning(f"Found {len(feature_columns)} features, but embedding model expects {expected_features}.")
                logger.warning(f"EXCLUDED additional features ({len(excluded_features)}): {', '.join(excluded_features)}")
                logger.info(f"💡 Consider retraining embedding model with {len(feature_columns)} features for better performance")
                feature_columns = feature_columns[:expected_features]
            elif len(feature_columns) < expected_features:
                logger.error(f"❌ Insufficient features: found {len(feature_columns)}, expected {expected_features}")
                logger.error(f"Available features: {', '.join(feature_columns)}")
                raise RuntimeError(f"Cannot proceed with {len(feature_columns)} features when {expected_features} are required")
            elif len(feature_columns) < expected_features:
                logger.error(f"Found only {len(feature_columns)} features, but embedding model expects {expected_features}")
                raise RuntimeError(f"Insufficient features: found {len(feature_columns)}, expected {expected_features}")
            
            # debug
            # logger.info(f"Selected {len(feature_columns)} numeric feature columns (matching embedding model): {', '.join(feature_columns)}")
            return feature_columns
        except Exception as e:
            logger.error(f"Failed to get feature columns: {e}")
            raise RuntimeError(f"Cannot get feature columns: {e}")
    
    def _sample_episode_start(self, max_attempts=100) -> Tuple[np.ndarray, np.ndarray]:
        """
        랜덤한 에피소드 시작 지점 샘플링
        
        Returns:
            features: (seq_len, n_features)
            metadata: (seq_len, 3) - [종목코드, 날짜, 시간]
        """
        # ✅ Cached columns usage
        feature_cols = self.feature_columns
        
        attempt = 0
        
        # 필요한 최소 데이터 길이 계산
        if self.max_episode_steps is not None:
            # max_episode_steps가 설정된 경우, seq_len + max_episode_steps만큼 필요
            min_required = self.seq_len + self.max_episode_steps + 100
        else:
            min_required = self.seq_len + 100
        
        while attempt < max_attempts:
            try:
                if not self.valid_keys:
                    raise RuntimeError("No valid keys available for sampling")
                
                # 캐시된 키에서 랜덤 선택
                stock_code, date, total_count = self.valid_keys[np.random.randint(0, len(self.valid_keys))]
                
                stock_code = str(stock_code)
                date = int(date)
                total_count = int(total_count)
                
                # 최적화: 필요한 데이터만 부분 로드 (OFFSET/LIMIT)
                needed_len = self.seq_len + 10
                if self.max_episode_steps is not None:
                    needed_len = self.seq_len + self.max_episode_steps
                
                # 데이터가 충분히 많고 max_episode_steps가 설정된 경우 최적화
                use_optimization = (self.max_episode_steps is not None) and (total_count >= needed_len)
                
                if use_optimization:
                    max_start_offset = total_count - needed_len
                    # 랜덤 시작 위치 결정 (전체 데이터 범위 내)
                    offset = np.random.randint(0, max_start_offset + 1)
                    
                    query = f"""
                        SELECT 종목코드, 날짜, 시간, {', '.join(feature_cols)}
                        FROM {self.table_name}
                        WHERE 종목코드 = ? AND 날짜 = ?
                        ORDER BY 시간
                        LIMIT ? OFFSET ?
                    """
                    # DuckDB에 정수형 파라미터 전달
                    df = self.conn.execute(query, [stock_code, date, needed_len, offset]).fetchdf()
                else:
                    # 전체 로드 (Fallback)
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
                    
                    # max_episode_steps가 설정된 경우, 랜덤한 시작 지점 선택
                    if self.max_episode_steps is not None:
                        # 가능한 시작 지점 범위 계산
                        max_start_idx = len(features) - (self.seq_len + self.max_episode_steps)
                        if max_start_idx > 0:
                            # 랜덤 시작 지점 선택
                            start_idx = np.random.randint(0, max_start_idx)
                            end_idx = start_idx + self.seq_len + self.max_episode_steps
                            
                            features = features[start_idx:end_idx]
                            metadata = metadata[start_idx:end_idx]
                            
                            # ✅ Bad Data Guard: 비정상적인 등락률 검사 (수정된 로직)
                            # 데이터가 이미 '누적 등락률'이므로, 값 자체가 30%를 넘는지 확인하면 됨
                            try:
                                return_rates = features[:, self.return_rate_index]
                                
                                # 원본 데이터(%)라면 100으로 나눔
                                if self.use_raw_data:
                                    return_rates = return_rates / 100.0
                                
                                # 절대값 기준 35% 초과 시 기각 (상/하한가 30% + 여유분)
                                # 누적 곱(cumprod) 불필요. 값 자체가 기준일 대비 수익률임.
                                max_abs_return = np.max(np.abs(return_rates))
                                
                                if max_abs_return > 0.35:
                                    logger.warning(f"⚠️ Bad Data (Overflow): stock={stock_code}, date={date}, max_return={max_abs_return*100:.2f}%")
                                    attempt += 1
                                    continue
                                    
                            except Exception as e:
                                logger.warning(f"Data validation failed: {e}")
                            
                            logger.debug(f"Sampled episode: stock={stock_code}, date={date}, "
                                       f"total_length={len(df)}, selected_range=[{start_idx}:{end_idx}], "
                                       f"selected_length={len(features)}")
                        else:
                            # 데이터가 충분하지 않으면 전체 사용
                            logger.debug(f"Sampled episode: stock={stock_code}, date={date}, length={len(features)}")
                    else:
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
            sequence: (seq_len, n_features) - 원본 또는 사전 정규화된 데이터
            
        Returns:
            임베딩 벡터 (embedding_dim,)
        """
        # ✅ Rolling normalization 적용 (원본 데이터 사용 시)
        if self.use_raw_data and self.normalizer is not None:
            normalized_seq = np.zeros_like(sequence, dtype=np.float32)
            for t in range(len(sequence)):
                normalized_seq[t] = self.normalizer.normalize(
                    sequence[t],
                    update=True  # 훈련 중이므로 통계 업데이트
                )
            sequence = normalized_seq
        
        with torch.no_grad():
            # (seq_len, n_features) -> (1, seq_len, n_features)
            seq_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
            
            # 임베딩 생성
            result = self.embedding_model(seq_tensor)
            
            if isinstance(result, tuple):
                if len(result) == 3:
                    # MaskedAutoEncoder의 경우: (reconstruction, embedding, mask)
                    _, embedding, _ = result
                elif len(result) == 2:
                    # AutoEncoderEmbedding의 경우: (reconstruction, embedding)
                    _, embedding = result
                else:
                    raise ValueError(f"Unexpected result length: {len(result)}")
            else:
                # embedding만 반환하는 경우
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
        
        # ✅ 에피소드마다 normalizer 리셋 (각 에피소드가 독립적인 종목/날짜)
        if self.use_raw_data and self.normalizer is not None:
            self.normalizer.reset()
            logger.debug("Normalizer reset for new episode")
        
        # 에피소드 데이터 샘플링
        self.episode_data, self.episode_metadata = self._sample_episode_start()
        self.episode_length = len(self.episode_data)
        
        # 🔧 등락률로부터 가격 계산
        self._compute_prices()
        
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
    
    def _compute_prices(self):
        """
        등락률로부터 가격 계산 (수정됨)
        
        등락률 컬럼이 '기준가 대비 누적 등락률'이므로, 
        복리 계산 없이 (1 + 등락률)을 기준가에 곱하여 바로 가격을 산출합니다.
        
        주의: 
        - 데이터베이스 컬럼 순서: 등락률, 거래량, ... ('현재가' 제외됨)
        - 원본 데이터의 등락률은 백분율(%) 단위이므로 100으로 나눠야 합니다.
        """
        # self.base_price는 __init__에서 설정됨
        
        # ✅ 올바른 로직: Price = Base * (1 + Return_Rate)
        # 루프 없이 벡터 연산으로 처리하여 폭발 원천 차단
        return_rates = self.episode_data[:, self.return_rate_index]
        
        if self.use_raw_data:
            return_rates = return_rates / 100.0
            
        # (1 + r) * base
        self.prices = self.base_price * (1.0 + return_rates)
        
        # 🔧 최소 가격 보장: 1,000원 이상 (벡터 연산)
        self.prices = np.maximum(self.prices, 1000.0)
        
        logger.debug(
            f"Computed prices: min={self.prices.min():.2f}, "
            f"max={self.prices.max():.2f}, "
            f"mean={self.prices.mean():.2f}"
        )
    
    def _get_current_price(self) -> float:
        """
        현재 가격 반환
        
        등락률로부터 계산된 실제 가격을 반환합니다.
        """
        return float(self.prices[self.current_step])
    
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
        # 🔧 Division by zero 방지
        if entry_price <= 0:
            logger.warning(
                f"Invalid entry_price: {entry_price}, setting to 1000.0"
            )
            entry_price = 1000.0
        
        # 수익률 계산: (청산가 - 진입가) / 진입가
        profit_rate = (exit_price - entry_price) / entry_price
        
        # 🔧 수익률 클리핑: ±100% (비현실적인 수익률 방지)
        profit_rate = np.clip(profit_rate, -1.0, 1.0)
        
        # 거래 비용 차감 (왕복 0.43%)
        # 양의 보상을 받으려면 수익률이 0.43%를 초과해야 함
        reward = profit_rate - self.round_trip_cost
        
        # 🔧 보상 스케일링: ×100 (학습 안정성)
        # 수익률 1% = 보상 1.0
        reward = reward * 100
        
        # ✅ 승리 보너스 추가 (수익 발생 시 추가 점수)
        # 순수익(거래비용 제외)이 0보다 크면 보너스 부여
        if profit_rate > self.round_trip_cost:
            reward += 1.0  # 승리 보너스
        
        # 🔧 장기 보유 페널티 제거 (매 스텝 페널티로 대체됨)
        # 매 스텝마다 보유 페널티가 적용되므로 여기서는 중복 제거
        # holding_penalty = 0.0
        # if holding_time > self.max_holding_time:
        #     holding_penalty = self.holding_penalty_rate * (holding_time - self.max_holding_time)
        #     reward -= holding_penalty
        
        # 보상 구성 요소
        reward_components = {
            'profit_rate': profit_rate,
            'transaction_cost': -self.round_trip_cost,
            'net_reward': reward
        }
        
        return reward, reward_components

    def set_transaction_cost_rate(self, rate: float):
        """
        거래 비용율을 동적으로 설정합니다 (Curriculum Learning용).
        
        Args:
           rate: 새로운 거래 비용율 (예: 0.00215)
        """
        self.transaction_cost_rate = rate
        # 왕복 비용(매수/매도 각각 적용된다고 가정하면 2배, or 이미 구현된 로직에 맞춤)
        # _calculate_reward logic: 수수료+세금 = 0.215%, 왕복 0.43%
        # self.round_trip_cost는 calculate_reward에서 쓰임
        self.round_trip_cost = rate * 2
        logger.info(f"Transaction cost rate updated to {rate:.6f} (Round trip: {self.round_trip_cost:.6f})")

    def _check_quick_exit_penalty_only(self, holding_time: float) -> Tuple[float, bool]:
        """
        빠른 손절 룰 체크 (penalty_only 모드)
        
        임계값을 초과하고 손실 중이면 페널티만 부여.
        강제 청산하지 않고 정책이 학습하도록 유도.
        
        Args:
            holding_time: 보유 시간 (초)
            
        Returns:
            (reward, quick_exit_triggered) 튜플
        """
        reward = 0.0
        quick_exit_triggered = False
        
        # 임계값을 초과하고 손실 중이면 페널티
        if holding_time > self.quick_exit_threshold and self.current_price < self.entry_price:
            quick_exit_triggered = True
            self.quick_exit_violations += 1
            reward = -self.quick_exit_penalty
            
            logger.debug(f"Quick exit penalty: price={self.current_price:.4f}, "
                       f"entry_price={self.entry_price:.4f}, "
                       f"holding_time={holding_time:.2f}s, "
                       f"penalty={self.quick_exit_penalty:.4f}")
        
        return reward, quick_exit_triggered
    
    def _check_quick_exit_force_close(self, holding_time: float) -> Tuple[float, bool]:
        """
        빠른 손절 룰 체크 (force_close 모드)
        
        임계값 이내에 가격이 상승하지 않으면 자동 매도.
        과도한 거래를 유발할 수 있음 (이전 동작).
        
        Args:
            holding_time: 보유 시간 (초)
            
        Returns:
            (reward, quick_exit_triggered) 튜플
        """
        reward = 0.0
        quick_exit_triggered = False
        
        # 임계값 이내이고 가격이 상승하지 않으면 강제 매도
        if holding_time <= self.quick_exit_threshold and self.current_price <= self.entry_price:
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
            
            logger.debug(f"Quick exit rule triggered (force close): price={self.current_price:.4f}, "
                       f"entry_price={self.entry_price:.4f}, "
                       f"holding_time={holding_time:.2f}s, "
                       f"penalty={self.quick_exit_penalty:.4f}, "
                       f"reward={reward:.4f}")
            
            # 포지션 청산
            self.position = 0
            self.entry_price = 0.0
            self.entry_time = 0.0
        
        return reward, quick_exit_triggered
    
    def _calculate_seconds_diff(self, start_time_val, end_time_val) -> float:
        """
        두 시간 값의 차이를 초 단위로 계산
        입력 포맷: HHMMSSmmm (9자리) + 선택적 소수점 (.0)
        예: 90000000.0 (09:00:00.000)
        """
        if str(start_time_val).startswith('0') or start_time_val == 0: return 0.0
        
        try:
            # 1. 문자열 변환 및 소수점 제거 (100000030.0 -> "100000030")
            s_str = str(start_time_val).split('.')[0]
            e_str = str(end_time_val).split('.')[0]
            
            # 2. 9자리 패딩 (090000000) - 앞자리 0이 생략된 경우 대비
            s_str = s_str.zfill(9)
            e_str = e_str.zfill(9)
            
            # 3. 파싱 (HH MM SS mmm)
            # 수동 슬라이싱이 strptime보다 빠르고 안전함
            def parse_time(t_str):
                h = int(t_str[:2])
                m = int(t_str[2:4])
                s = int(t_str[4:6])
                ms = int(t_str[6:9])
                return h * 3600 + m * 60 + s + ms / 1000.0
            
            s_seconds = parse_time(s_str)
            e_seconds = parse_time(e_str)
            
            # 4. 날짜 경계 처리 (밤 11시 -> 새벽 1시 인 경우 등을 대비)
            # 여기선 단순 차이만 계산하되, 음수면 하루(86400초) 더함
            diff = e_seconds - s_seconds
            if diff < 0:
                diff += 86400.0
                
            return diff
            
        except Exception as e:
            # 파싱 실패 시 안전장치
            # logger.warning(f"Time parsing failed: {start_time_val} -> {end_time_val} ({e})")
            return 1.0  # 기본값 1초 반환하여 에러 방지

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
        
        # 1. 행동 실행
        if action == 1:  # 매수
            if self.position == 0:
                self.position = 1
                self.entry_price = self.current_price
                self.entry_time = self.current_time
                
                # 매수 시 거래비용 차감 (Dense Reward)
                # 비용은 편도 0.215% -> 0.43%? 아니, 매수/매도 각각 0.215%가 맞음?
                # _calculate_reward에서는 self.round_trip_cost (0.43%)를 한번에 뺐음.
                # Dense Reward에서는 매수/매도 각각 반씩 뺄 수도 있고, 매수 때 다 뺄 수도 있음.
                # 편의상 매도 때 비용을 정산하던 기존 방식과 달리, 
                # 여기선 "진입/청산 비용"을 각각 부과하거나, 왕복 비용을 나눠서 부과.
                # 기존 round_trip_cost = 0.43%. Half = 0.215%.
                
                # 비용 페널티: -0.215 (스케일링 전 %, 즉 0.00215) * 100 = -0.215
                reward -= self.transaction_cost_rate * 100
                
                logger.debug(f"Buy at price={self.entry_price:.4f}, time={self.entry_time}")
            else:
                # 이미 포지션 보유 중: 불필요한 행동 페널티 (완화)
                reward -= 0.01
        
        elif action == 2:  # 매도
            if self.position == 0:
                # 포지션 없는데 매도: 불필요한 행동 페널티 (완화)
                reward -= 0.01
            elif self.position == 1:
                # 보유 시간 계산
                holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time)
                
                # 매도 시 거래비용 차감 (Dense Reward)
                reward -= self.transaction_cost_rate * 100
                
                # 거래 통계용 (Realized PnL) 계산 - 보상에는 더하지 않음 (이중 계산 방지)
                # 단, 로그용으로는 기존 함수 사용
                realized_reward, reward_components = self._calculate_reward(
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
                    'reward': realized_reward, # 기록용
                    'reward_components': reward_components
                }
                self.episode_trades.append(trade_info)
                
                logger.debug(f"Sell at price={self.current_price:.4f}, "
                           f"profit_rate={reward_components['profit_rate']:.4f}, "
                           f"realized_reward={realized_reward:.4f}")
                
                # 포지션 청산
                self.position = 0
                self.entry_price = 0.0
                self.entry_time = 0.0
        
        elif action == 0:  # 보유
            # 별도 페널티 없음 (Dense Reward가 알아서 처리)
            pass
        
        # 2. 포지션 보유에 따른 Step Reward (Dense Reward의 핵심)
        # 포지션을 들고 다음 스텝으로 넘어가면, 가격 변동분을 즉시 보상으로 반영
        if self.position == 1:
            # 다음 스텝의 가격 가져오기 (미래 참조가 아님, 시뮬레이션의 다음 상태)
            next_step_idx = self.current_step + 1
            if next_step_idx < len(self.prices):
                next_price = float(self.prices[next_step_idx])
                
                # 변동률 계산
                # (P_next - P_curr) / P_curr
                step_return = (next_price - self.current_price) / self.current_price
                
                # 스케일링 (1% = 1.0)
                step_reward = step_return * 100
                
                reward += step_reward
                
                # 빠른 손절/익절 체크 (보조)
                # Dense Reward가 있으므로 강제 청산 로직보다는 
                # 큰 손실이 누적될 때 페널티를 주는 등의 보조 장치만 유지
                
                holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time)
                if self.quick_exit_mode == 'penalty_only':
                     penalty, quick_exit_triggered = self._check_quick_exit_penalty_only(holding_time)
                     reward += penalty
        
        # 보상 기록
        self.episode_rewards.append(reward)
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 에피소드 종료 체크
        if self.current_step >= self.episode_length - 1:
            terminated = True
        
        # 최대 스텝 수 체크
        if self.max_episode_steps is not None and self.current_step >= self.max_episode_steps:
            truncated = True
        
        # 에피소드 종료 시 강제 청산 (통계용)
        if (terminated or truncated) and self.position == 1:
            holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time)
            
            # 매도 비용 지불 (Dense Reward 관점)
            final_step_penalty = -self.transaction_cost_rate * 100
            
            reward += final_step_penalty
            if len(self.episode_rewards) > 0:
                self.episode_rewards[-1] += final_step_penalty
            
            # 통계용 기록
            final_reward, final_components = self._calculate_reward(
                self.entry_price, self.current_price, holding_time
            )
            
            self.episode_trades.append({
                'entry_price': self.entry_price,
                'exit_price': self.current_price,
                'holding_time': holding_time,
                'profit_rate': final_components['profit_rate'],
                'reward': final_reward,
                'reward_components': final_components,
                'forced_liquidation': True
            })
            
            logger.debug(f"Forced liquidation")
        
        # 현재 가격 및 시간 업데이트
        if not (terminated or truncated):
            # 에피소드가 계속되는 경우에만 업데이트
            self.current_price = self._get_current_price()
            self.current_time = float(self.episode_metadata[self.current_step, 2])
        
        # 다음 관측값
        if not (terminated or truncated):
            observation = self._get_current_observation()
        else:
            observation = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # 정보
        info = {
            'position': self.position,
            'current_price': self.current_price,
            'current_time': self.current_time,
            'quick_exit_violations': self.quick_exit_violations,
            'quick_exit_triggered': quick_exit_triggered
        }
        
        if terminated or truncated:
            # 에피소드 종료 시 메타데이터 추가 (terminated, truncated 모두 포함)
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
        
        # 샤프 비율 계산 (안정화)
        # 우선 개별 거래 보상을 사용하고, 거래가 없으면 스텝 보상 사용
        sharpe_ratio = 0.0
        epsilon = 1e-6  # near-zero 분모 방지 임계값
        if self.episode_trades and len(self.episode_trades) > 1:
            trade_rewards = np.asarray([t['reward'] for t in self.episode_trades], dtype=np.float64)
            mean_reward = float(np.mean(trade_rewards))
            std_reward = float(np.std(trade_rewards))
            if np.isfinite(std_reward) and std_reward >= epsilon:
                sharpe_ratio = mean_reward / std_reward
        elif len(self.episode_rewards) > 1:
            step_rewards = np.asarray(self.episode_rewards, dtype=np.float64)
            mean_reward = float(np.mean(step_rewards))
            std_reward = float(np.std(step_rewards))
            if np.isfinite(std_reward) and std_reward >= epsilon:
                sharpe_ratio = mean_reward / std_reward
        
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
        
        logger.info(f"Episode finished: "
                   f"total_return={total_return:.4f}, "
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
