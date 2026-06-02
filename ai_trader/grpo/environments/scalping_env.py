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
        seq_len: int = 120,
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
        stop_loss_pct: float = 2.0,     # ✅ 손절 기준 (%)
        max_split_count: int = 1,       # ✅ 최대 분할 매수 횟수
        min_holding_time: float = 2.0,  # ✅ 최소 보유 시간
        max_holding_time: float = 100.0,# ✅ 최대 보유 시간
        no_trade_penalty: float = 10.0, # ✅ 거래 안할 시 패널티 (기본값: 10.0)
        max_trades_per_episode: Optional[int] = None, # ✅ 최대 거래 횟수 제한
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.base_price = base_price
        self.stop_loss_pct = stop_loss_pct
        self.max_split_count = max_split_count
        self.min_holding_time = min_holding_time
        self.max_holding_time = max_holding_time
        self.no_trade_penalty = no_trade_penalty
        self.max_trades_per_episode = max_trades_per_episode
        
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
        
        # 관측 공간: 임베딩 벡터 + 포지션 정보(3)
        # 1. Position (0 or 1)
        # 2. Position Steps (Normalized)
        # 3. Holding Time (Normalized)
        self.obs_dim = embedding_dim + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
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
        self.position = 0  # 0: 포지션 없음, 1: 보유 중 (호환성 유지)
        self.position_steps = 0  # 현재 분할 매수 단계 (0 ~ max_split_count)
        self.avg_entry_price = 0.0  # 평단가

        self.entry_time = 0
        self.current_price = 0.0
        self.current_time = 0
        
        # 리스크 관리용 상태
        self.max_price_since_entry = 0.0  # 진입 후 최고가 (본전 청산용)
        
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
                   f"stop_loss={stop_loss_pct}%, max_split={max_split_count}, "
                   f"holding_time_range=[{min_holding_time}s, {max_holding_time}s], "
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
        
        # ✅ 포지션 상태 정보 추가 (중요: 에이전트가 자신의 상태를 알아야 함)
        if self.position == 1:
            holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time)
            holding_time_norm = min(holding_time / (self.max_holding_time + 1e-6), 1.0)
            steps_norm = self.position_steps / (self.max_split_count or 1)
        else:
            holding_time_norm = 0.0
            steps_norm = 0.0
            
        extra_features = np.array([
            float(self.position),
            float(steps_norm),
            float(holding_time_norm)
        ], dtype=np.float32)
        
        # 임베딩 + 추가 정보 결합
        observation = np.concatenate([embedding, extra_features])
        
        return observation
    
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
        self.position_steps = 0
        self.avg_entry_price = 0.0
        self.entry_time = 0.0
        self.max_price_since_entry = 0.0
        
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
        if holding_time > self.quick_exit_threshold and self.current_price < self.avg_entry_price:
            quick_exit_triggered = True
            self.quick_exit_violations += 1
            reward = -self.quick_exit_penalty
            
            logger.debug(f"Quick exit penalty: price={self.current_price:.4f}, "
                       f"avg_entry_price={self.avg_entry_price:.4f}, "
                       f"holding_time={holding_time:.2f}s, "
                       f"penalty={self.quick_exit_penalty:.4f}")
        
        return reward, quick_exit_triggered
    
    def _force_close_position(self, reason: str, penalty: float = 0.0) -> float:
        """
        포지션 강제 청산 (손절, 본전청산 등)
        Returns:
            청산에 따른 보상(reward)
        """
        if self.position == 0:
            return 0.0
            
        holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time)
        
        # 매도 비용 (보유 물량 전체에 대한 비용)
        # 비용 = 요율 * 100 * (보유비중)
        weight = self.position_steps / self.max_split_count if self.max_split_count > 0 else 1.0
        
        # 매도 비용 지불
        exit_cost = self.transaction_cost_rate * 100 * weight
        
        # 거래 통계용 계산
        realized_reward, reward_components = self._calculate_reward(
            self.avg_entry_price,
            self.current_price,
            holding_time
        )
        
        # 강제 청산 페널티 적용 (실현 손익 - 비용 + 페널티)
        # 여기서 realized_reward는 단순히 (수익률 - 0.43%) * 100 형태이므로,
        # 가중치를 적용하려면 수익률 부분에도 가중치를 곱해야 함.
        # 기존 _calculate_reward는 가중치 개념이 없음.
        
        # ✅ 가중치 적용된 보상 계산 직접 수행
        profit_rate = (self.current_price - self.avg_entry_price) / self.avg_entry_price
        # 수익금 보상 = 수익률 * 100 * 가중치
        weighted_profit_reward = profit_rate * 100 * weight
        
        # 최종 보상 = 가중 수익금 - 매도비용 + 페널티 (매수 비용은 이미 지불됨)
        # 주의: _calculate_reward는 왕복 비용을 포함하고 있음.
        # 여기서는 매도 비용만 따로 빼고, 수익 부분만 가중치 적용.
        
        final_reward = weighted_profit_reward - exit_cost + penalty
        
        # 거래 기록
        trade_info = {
            'entry_price': self.avg_entry_price,
            'exit_price': self.current_price,
            'holding_time': holding_time,
            'profit_rate': profit_rate,
            'reward': final_reward, # Logged reward
            'reward_components': reward_components,
            'exit_reason': reason,
            'position_steps': self.position_steps,
            'weight': weight  # 가중치 기록
        }
        self.episode_trades.append(trade_info)
        
        logger.debug(f"Forced Exit ({reason}): price={self.current_price:.4f}, "
                   f"avg_entry={self.avg_entry_price:.4f}, "
                   f"steps={self.position_steps}, "
                   f"reward={final_reward:.4f}")
        
        # 상태 초기화
        self.position = 0
        self.position_steps = 0
        self.avg_entry_price = 0.0
        self.entry_time = 0.0
        self.max_price_since_entry = 0.0
        
        return final_reward
    
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
        
        if action == 1:  # 매수
            if self.position == 0 and self.max_trades_per_episode is not None and len(self.episode_trades) >= self.max_trades_per_episode:
                # 최대 거래 횟수 초과로 신규 진입 차단
                pass
            elif self.position_steps < self.max_split_count:
                # 분할 매수 (또는 신규 진입)
                old_steps = self.position_steps
                new_steps = old_steps + 1
                
                # 평단가 갱신 (가중 평균)
                # 이전 총액 + 현재 매수액 / 총 수량
                prev_total_value = self.avg_entry_price * old_steps
                new_total_value = prev_total_value + self.current_price
                self.avg_entry_price = new_total_value / new_steps
                
                self.position_steps = new_steps
                self.position = 1  # 1개라도 있으면 포지션 ON
                
                if old_steps == 0:
                    self.entry_time = self.current_time
                    self.max_price_since_entry = self.current_price
                
                # 매수 비용 차감 (1회분 = 1/max_split)
                # 예: 10분할이면 전체 자산의 10%만 매수했으므로 비용도 10%만 발생
                buy_weight = 1.0 / self.max_split_count
                reward -= self.transaction_cost_rate * 100 * buy_weight
                
                logger.debug(f"Buy (Step {new_steps}/{self.max_split_count}): "
                           f"price={self.current_price:.1f}, "
                           f"new_avg={self.avg_entry_price:.1f}, "
                           f"cost_weight={buy_weight:.2f}")
            else:
                # 이미 풀매수 상태: 과도한 매수 시도 페널티 (선택사항)
                pass
        
        elif action == 2:  # 매도
            if self.position == 1:
                # 보유 시간 계산
                holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time)
                
                # ✅ 최소 보유 시간 체크 (Dense Reward로 가이드)
                if holding_time < self.min_holding_time:
                    # 너무 빨리 파는 경우: 약한 페널티 부여 (-0.2점)
                    # 거래를 아예 포기하지 않도록 페널티 완화
                    early_exit_penalty = 0.2
                    reward -= early_exit_penalty
                    logger.debug(f"Early Exit Penalty applied: -{early_exit_penalty} (holding_time {holding_time:.2f}s < {self.min_holding_time}s)")
                else:
                    # 충분히 기다렸다가 파는 경우: 인내 보너스 (+0.1점)
                    patience_bonus = 0.1
                    reward += patience_bonus
                    # logger.debug(f"Patience Bonus applied: +{patience_bonus}")

                # 전량 매도 진행
                
                # 매도 가중치 (전량 매도이므로 현재 보유 비중)
                weight = self.position_steps / self.max_split_count
            
                # 매도 비용 차감 (보유 수량만큼)
                reward -= self.transaction_cost_rate * 100 * weight
                
                # 수익률 계산 (평단가 기준)
                profit_rate = (self.current_price - self.avg_entry_price) / self.avg_entry_price
                
                # 가중치가 적용된 수익 보상 시뮬레이션
                # (단순 수익률이 아니라, '내 돈이 얼마나 들어갔나'에 비례한 수익금 개념)
                weighted_profit_reward = profit_rate * 100 * weight
                reward += weighted_profit_reward
                
                # 승리/손실 보너스에도 가중치 적용
                # 풀매수 성공 시 보너스 큼, 짤짤이 성공 시 보너스 작음
                if profit_rate > self.round_trip_cost:
                    bonus = 1.5 * weight
                    reward += bonus
                    logger.debug(f"Win bonus applied: +{bonus:.2f} (weight={weight:.2f})")
                elif profit_rate < 0:
                    penalty = 0.5 * weight
                    reward -= penalty
                    logger.debug(f"Loss penalty applied: -{penalty:.2f} (weight={weight:.2f})")
                
                # 기록용 (호환성 유지)
                _, reward_components = self._calculate_reward(
                    self.avg_entry_price, self.current_price, holding_time
                )
                
                trade_info = {
                    'entry_price': self.avg_entry_price,
                    'exit_price': self.current_price,
                    'holding_time': holding_time,
                    'profit_rate': profit_rate,
                    'reward': reward,
                    'reward_components': reward_components,
                    'position_steps': self.position_steps,
                    'weight': weight
                }
                self.episode_trades.append(trade_info)
                
                logger.debug(f"Sell at price={self.current_price:.1f}, "
                           f"avg={self.avg_entry_price:.1f}, "
                           f"steps={self.position_steps}, "
                           f"profit={profit_rate*100:.2f}%, "
                           f"weighted_reward={weighted_profit_reward:.4f}")
                
                # 상태 초기화
                self.position = 0
                self.position_steps = 0
                self.avg_entry_price = 0.0
                self.entry_time = 0.0
                self.max_price_since_entry = 0.0
        
        elif action == 0:  # 보유
            pass
        
        # 2. 포지션 보유에 따른 Step Reward (Dense Reward의 핵심)
        # 포지션을 들고 다음 스텝으로 넘어가면, 가격 변동분을 즉시 보상으로 반영
        if self.position == 1 and self.avg_entry_price > 0:
            # 최고가 갱신
            self.max_price_since_entry = max(self.max_price_since_entry, self.current_price)
            
            # 다음 스텝 가격으로 변동분 보상 계산
            next_step_idx = self.current_step + 1
            if next_step_idx < len(self.prices):
                next_price = float(self.prices[next_step_idx])
                step_return = (next_price - self.current_price) / self.current_price
                
                # ✅ 분할 매수 비중에 따른 가중치 적용 (핵심)
                # 1단계만 보유 시 보상 10%, 10단계(풀매수) 보유 시 보상 100%
                weight = self.position_steps / self.max_split_count
                
                step_reward = step_return * 100 * weight
                reward += step_reward
                
                # --- 리스크 관리 (손절 & 본전청산) ---
                
                # 현재 누적 수익률 (평단가 기준)
                current_return = (self.current_price - self.avg_entry_price) / self.avg_entry_price
                
                # 최고 수익률 (진입 이후)
                max_return = (self.max_price_since_entry - self.avg_entry_price) / self.avg_entry_price
                
                # 1. 손절매 (Stop Loss)
                # 예: -2% 이하 시 손절
                stop_loss_threshold = -(self.stop_loss_pct / 100.0)
                
                # 2. 본전 청산 (Breakeven)
                # 예: 최고 수익률이 0.5% 이상이었다가, 다시 0.05% 이하로 떨어지면 청산
                breakeven_activation = 0.005  # 0.5%
                breakeven_trigger = 0.0005    # 0.05%
                
                force_exit_reason = None
                
                if current_return <= stop_loss_threshold:
                    force_exit_reason = "Stop Loss"
                    # 손절 페널티 부여
                    reward -= 1.0
                
                elif max_return >= breakeven_activation and current_return <= breakeven_trigger:
                     force_exit_reason = "Breakeven"
                     # 본전 청산은 중립적이거나 약한 보상
                     reward += 0.1
                     
                # 3. 최대 보유 시간 초과 (Time Limit)
                # 시간이 너무 지체되면 강제 청산
                holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time)
                if not force_exit_reason and holding_time >= self.max_holding_time:
                    force_exit_reason = "TimeLimit"
                    # 시간 초과 페널티 (지루하게 오래 끌면 안됨)
                    reward -= 0.5
                
                # 강제 청산 실행
                if force_exit_reason:
                    # 최소 보유 시간 체크 무시 (손절/본전청산/시간초과는 강제성이 있으므로)
                    exit_reward = self._force_close_position(force_exit_reason)
                    reward += exit_reward  # 청산 시 실현 손익 반영
                    
                    # 빠른 손절 체크 로직은 해제 (이 로직이 대체)
                    pass
                else:
                    # 기존의 '시간 경과에 따른 빠른 손절' 로직 유지 (Only if not forced closed)
                    # holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time) # 위에서 계산함
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
        
        # 에피소드 최대 거래 횟수 달성 체크 (포지션이 없을 때 조기 종료)
        if self.max_trades_per_episode is not None and len(self.episode_trades) >= self.max_trades_per_episode and self.position == 0:
            terminated = True
            
        # 최대 스텝 수 체크 (상대 스텝으로 계산)
        if self.max_episode_steps is not None and (self.current_step - (self.seq_len - 1)) >= self.max_episode_steps:
            truncated = True
        
        # 에피소드 종료 시 강제 청산 (통계용)
        # 에피소드 종료 시 강제 청산 (통계용) 및 no_trade_penalty 적용
        if terminated or truncated:
            if self.position == 1:
                holding_time = self._calculate_seconds_diff(self.entry_time, self.current_time)
                weight = self.position_steps / self.max_split_count if self.max_split_count > 0 else 1.0
                
                # 가중치 적용된 실제 매도 및 수익 정산
                profit_rate = (self.current_price - self.avg_entry_price) / self.avg_entry_price
                weighted_profit_reward = profit_rate * 100 * weight
                exit_cost = self.transaction_cost_rate * 100 * weight
                
                final_trade_reward = weighted_profit_reward - exit_cost
                
                reward += final_trade_reward
                if len(self.episode_rewards) > 0:
                    self.episode_rewards[-1] += final_trade_reward
                
                _, reward_components = self._calculate_reward(
                    self.avg_entry_price, self.current_price, holding_time
                )
                
                self.episode_trades.append({
                    'entry_price': self.avg_entry_price,
                    'exit_price': self.current_price,
                    'holding_time': holding_time,
                    'profit_rate': profit_rate,
                    'reward': final_trade_reward,
                    'reward_components': reward_components,
                    'weight': weight,
                    'forced_liquidation': True
                })
                logger.debug(f"Forced liquidation on episode end: profit={profit_rate*100:.2f}%, reward={final_trade_reward:.4f}")
            
            # 거래를 한 번도 안 한 경우 패널티 부여 (no-trade collapse 방지)
            if len(self.episode_trades) == 0:
                reward -= self.no_trade_penalty
                if len(self.episode_rewards) > 0:
                    self.episode_rewards[-1] -= self.no_trade_penalty
                logger.debug(f"No trade penalty applied: -{self.no_trade_penalty}")
        
        # 현재 가격 및 시간 업데이트
        if not (terminated or truncated):
            # 에피소드가 계속되는 경우에만 업데이트
            self.current_price = self._get_current_price()
            self.current_time = float(self.episode_metadata[self.current_step, 2])
        
        # 다음 관측값
        if not (terminated or truncated):
            observation = self._get_current_observation()
        else:
            observation = np.zeros(self.obs_dim, dtype=np.float32)
        
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
        # 총 수익 (거래별 net return의 합)
        if self.episode_trades:
            total_return = sum((t['profit_rate'] - self.round_trip_cost) * 100 * t.get('weight', 1.0) for t in self.episode_trades)
        else:
            total_return = -self.no_trade_penalty if self.no_trade_penalty > 0 else 0.0
            
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
            trade_net_profits = np.asarray([
                (t['profit_rate'] - self.round_trip_cost) * 100 * t.get('weight', 1.0)
                for t in self.episode_trades
            ], dtype=np.float64)
            mean_reward = float(np.mean(trade_net_profits))
            std_reward = float(np.std(trade_net_profits))
            if np.isfinite(std_reward) and std_reward >= epsilon:
                sharpe_ratio = mean_reward / std_reward
        elif len(self.episode_rewards) > 1:
            step_rewards = np.asarray(self.episode_rewards, dtype=np.float64)
            mean_reward = float(np.mean(step_rewards))
            std_reward = float(np.std(step_rewards))
            if np.isfinite(std_reward) and std_reward >= epsilon:
                sharpe_ratio = mean_reward / std_reward
        
        # 승률 계산 (수익률이 왕복 거래비용을 초과하는 거래 비율)
        if self.episode_trades:
            win_rate = np.mean([1 if t['profit_rate'] > self.round_trip_cost else 0 for t in self.episode_trades])
        else:
            win_rate = 0.0
        
        # 평균 거래당 수익 (가중치를 반영한 net profit의 평균)
        if self.episode_trades:
            avg_profit_per_trade = np.mean([
                (t['profit_rate'] - self.round_trip_cost) * 100 * t.get('weight', 1.0)
                for t in self.episode_trades
            ])
        else:
            avg_profit_per_trade = 0.0
            
        # 10.0 ~ 10.0 클리핑 적용 (Sharpe Ratio 안정화)
        sharpe_ratio = float(np.clip(sharpe_ratio, -10.0, 10.0))
        
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
