"""
GRPO 스캘핑 환경

초단위 스캘핑을 위한 Gymnasium 환경을 정의합니다.
시계열 특징 데이터와 5단계 분할 포지션 메타데이터를 사용하여 관측값을 생성하며,
1~5초 급등 패턴 탐지 보상 및 1~5초 미상승 자동 손절 구조를 제공합니다.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import duckdb
from datetime import datetime, timedelta

from lib.rolling_normalization import RollingNormalizer

import inspect

class ContextLogger:
    def __init__(self, default_logger):
        self.default_logger = default_logger

    def _get_logger(self):
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_self = frame.f_back.f_back.f_locals.get('self', None)
                if caller_self is not None:
                    return logging.getLogger(caller_self.__class__.__module__)
        except Exception:
            pass
        return self.default_logger

    def info(self, msg, *args, **kwargs): self._get_logger().info(msg, *args, **kwargs)
    def debug(self, msg, *args, **kwargs): self._get_logger().debug(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs): self._get_logger().warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs): self._get_logger().error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs): self._get_logger().critical(msg, *args, **kwargs)

logger = ContextLogger(logging.getLogger(__name__))


class GRPOScalpingEnv(gym.Env):
    """
    스캘핑을 위한 GRPO 훈련 환경
    
    관측 공간: (seq_len, 43) [특징 28차원 + 5단계 메타데이터 15차원]
    행동 공간: Discrete(3) - 0: 보유, 1: 매수 (+1단계), 2: 매도 (1단계 FIFO 청산)
    
    보상 구조:
    - 5단계 분할 매수/매도 (단계별 독립 진입가 기준 수익률 정산)
    - 1~5초 내 미상승 시 즉시 자동 손절 (-1.0 페널티)
    - 1~5초 내 가격 상승 포착 시 패턴 탐지 보너스 (+0.5)
    - 거래비용: 수수료(0.015%) + 세금(0.18%) (매수/매도 적용)
    
    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명 (기본값: 'datasets')
        seq_len: 시퀀스 길이 (기본값: 3000)
        expected_features: 예상 특징 수 (기본값: 28)
        transaction_cost_rate: 거래 비용 비율 (기본값: 0.00215 = 0.215%)
        max_episode_steps: 에피소드당 최대 스텝 수 (기본값: None, 제한 없음)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        db_path: str,
        table_name: str = 'datasets',
        seq_len: int = 3000,
        expected_features: int = 28,
        transaction_cost_rate: float = 0.00215,
        buy_tax_rate: float = 0.0,
        sell_tax_rate: float = 0.0018,
        no_trade_penalty: float = 0.0,
        max_episode_steps: Optional[int] = None,
        use_raw_data: bool = True,  # ✅ 원본 데이터 사용 여부
        rolling_window_size: int = 1000,  # ✅ Rolling window 크기
        rolling_min_samples: int = 100,  # ✅ 최소 샘플 수
        base_price: float = 100000.0,  # ✅ 기준 가격 (기본값: 10만원)
        max_trades_per_episode: Optional[int] = None, # ✅ 최대 거래 횟수 제한
        step_reward_scale: float = 1.0, # ✅ Dense Step Reward 스케일 조정 비율
        win_bonus: float = 5.0,         # ✅ 거래 수익(수수료 극복) 성공 보너스
        loss_penalty: float = 0.3,      # ✅ 거래 손실 페널티
        buy_signal_bonus: float = 0.5,  # ✅ 매수 신호(거래대금 증가 + 등락률 상승) 보너스
    ):
        super().__init__()
        
        self.base_price = base_price
        self.no_trade_penalty = no_trade_penalty
        self.max_trades_per_episode = max_trades_per_episode
        self.step_reward_scale = step_reward_scale
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.buy_signal_bonus = buy_signal_bonus
        self.accum_trade_value_index = 2
        self.raw_accum_trade_value = None
        
        self.db_path = db_path
        self.table_name = table_name
        self.seq_len = seq_len
        self.expected_features = expected_features
        
        # 거래 비용 설정
        self.transaction_cost_rate = transaction_cost_rate
        self.buy_tax_rate = buy_tax_rate
        self.sell_tax_rate = sell_tax_rate
        
        # 커리큘럼 러닝 스케일링용 타겟 값 저장
        self.target_transaction_cost_rate = transaction_cost_rate
        self.target_buy_tax_rate = buy_tax_rate
        self.target_sell_tax_rate = sell_tax_rate
        
        self.round_trip_cost = (transaction_cost_rate + buy_tax_rate) + (transaction_cost_rate + sell_tax_rate)
        
        # 에피소드 길이 제한
        self.max_episode_steps = max_episode_steps
        
        # ✅ 정규화 설정
        self.use_raw_data = use_raw_data
        self.rolling_window_size = rolling_window_size
        self.rolling_min_samples = rolling_min_samples
        
        # ✅ RollingNormalizer 초기화 (원본 데이터 사용 시)
        if use_raw_data:
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
        
        # 관측 공간: 시퀀스 길이 x (특징 차원 28 + 5단계 메타데이터 15) = (seq_len, 43)
        # 5개 단계 각각의: [is_active, profit_rate, holding_time_norm]
        self.MAX_STAGES = 5
        self.obs_dim = expected_features + (self.MAX_STAGES * 3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.seq_len, self.obs_dim),
            dtype=np.float32
        )
        
        # 행동 공간: 0=보유, 1=매수, 2=매도
        self.action_space = spaces.Discrete(3)
        
        # 데이터베이스 연결
        self.conn = None
        self.feature_columns = None
        self.valid_keys = None
        
        # 임시 연결로 속성 초기화
        self._init_metadata_from_db()
        
    def _init_metadata_from_db(self):
        """임시로 DB 연결을 열고 metadata 등을 세팅"""
        self._connect_db()
        self.feature_columns = self._get_feature_columns()
        try:
            self.return_rate_index = self.feature_columns.index('등락률')
            logger.info(f"Column Mapping Identified: '등락률' at Index {self.return_rate_index}")
        except ValueError as e:
            logger.error(f"Critical Column Missing: {e}")
            raise RuntimeError(f"Required column (등락률) missing from database features")

        try:
            self.accum_trade_value_index = self.feature_columns.index('누적거래대금')
            logger.info(f"Column Mapping Identified: '누적거래대금' at Index {self.accum_trade_value_index}")
        except ValueError:
            self.accum_trade_value_index = 5
            logger.warning("Column '누적거래대금' not found in features, using fallback index 5")
        
        self.valid_keys = []
        self._preload_valid_keys()
        
        # 초기화가 끝났으므로 연결 닫기 (Multiprocessing Pickle 문제 방지)
        self.conn.close()
        self.conn = None

    @property
    def position(self) -> int:
        return 1 if len(getattr(self, 'stages', [])) > 0 else 0

    @property
    def position_steps(self) -> int:
        return len(getattr(self, 'stages', []))

    @property
    def max_split_count(self) -> int:
        return 5

    def __getstate__(self):
        """직렬화 시 DB 연결 제외"""
        state = self.__dict__.copy()
        state['conn'] = None
        return state

    def __setstate__(self, state):
        """역직렬화 시 상태 복원"""
        self.__dict__.update(state)
        self.conn = None

    def _ensure_db_connection(self):
        """스레드/프로세스 내에서 DB 연결 보장"""
        if self.conn is None:
             self._connect_db()
             
        # 에피소드 상태
        self.current_step = 0
        self.stages = []  # List[dict]: {'entry_price': float, 'entry_time': float, 'pattern_rewarded': bool}
        self.current_price = 0.0
        self.current_time = 0.0
        
        # 에피소드 메타데이터
        self.episode_trades = []
        self.episode_rewards = []
        self.loss_holding_violations = 0
        
        # 현재 에피소드 데이터
        self.episode_data = None
        self.episode_metadata = None
        self.episode_length = 0
        
        logger.info(f"GRPOScalpingEnv metadata populated.")
        
        logger.info(f"GRPOScalpingEnv initialized metadata successfully.")

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
                logger.warning(f"Found {len(feature_columns)} features, but model expects {expected_features}.")
                logger.warning(f"EXCLUDED additional features ({len(excluded_features)}): {', '.join(excluded_features)}")
                feature_columns = feature_columns[:expected_features]
            elif len(feature_columns) < expected_features:
                logger.error(f"❌ Insufficient features: found {len(feature_columns)}, expected {expected_features}")
                logger.error(f"Available features: {', '.join(feature_columns)}")
                raise RuntimeError(f"Cannot proceed with {len(feature_columns)} features when {expected_features} are required")
            elif len(feature_columns) < expected_features:
                logger.error(f"Found only {len(feature_columns)} features, but model expects {expected_features}")
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
                    # 메타데이터와 특징 분리 (결측치 처리)
                    # Pandas의 NA/NaN을 0으로 채우거나 필터링합니다. 클린 데이터를 위해 0 채우기 사용.
                    df_features = df[feature_cols].fillna(0.0)
                    metadata = df[['종목코드', '날짜', '시간']].values
                    features = df_features.values.astype(np.float32)
                    
                    # 만약 여전히 무한대 값(Inf)나 NaN이 있다면 기각
                    if not np.isfinite(features).all():
                        logger.warning(f"⚠️ Bad Data (NaN/Inf found): stock={stock_code}, date={date}")
                        attempt += 1
                        continue
                    
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
    
    def _normalize_sequence(self, sequence: np.ndarray, update: bool = True) -> np.ndarray:
        """
        시퀀스를 정규화
        
        Args:
            sequence: (seq_len, n_features) - 원본 데이터
            update: True면 window 업데이트
            
        Returns:
            정규화된 벡터 (seq_len, n_features)
        """
        # ✅ Rolling normalization 적용
        if self.use_raw_data and self.normalizer is not None:
            # RollingNormalizer supports 2D arrays natively.
            # Passing the entire sequence at once is magnitudes faster than a for-loop.
            normalized_seq = self.normalizer.normalize(
                sequence,
                update=update
            )
            return normalized_seq
        return sequence
    
    def _get_current_observation(self) -> np.ndarray:
        """
        현재 관측값 반환
        
        Returns:
            시퀀스 + 포지션 메타데이터 (seq_len, obs_dim)
        """
        # 현재 스텝에서 seq_len만큼의 시퀀스 추출
        start_idx = max(0, self.current_step - self.seq_len + 1)
        end_idx = self.current_step + 1
        
        sequence = self.episode_data[start_idx:end_idx]
        
        # 시퀀스가 seq_len보다 짧으면 앞부분을 0으로 패딩
        if len(sequence) < self.seq_len:
            padding = np.zeros((self.seq_len - len(sequence), sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([padding, sequence])
        
        # 정규화 (최적화 버전: 첫 스텝만 전체 업데이트, 이후 스텝은 1개만 업데이트하여 속도 3000배 향상)
        if self.use_raw_data and self.normalizer is not None:
            if self.normalizer.n_samples == 0:
                normalized_sequence = self._normalize_sequence(sequence, update=True)
            else:
                current_feature = self.episode_data[self.current_step]
                self.normalizer.update(current_feature)
                normalized_sequence = self._normalize_sequence(sequence, update=False)
        else:
            normalized_sequence = sequence
        
        # ✅ 5단계 포지션 정보 메타데이터 추가 (is_active, profit_rate, holding_norm) x 5 = 15
        extra_features_list = []
        for i in range(5):
            if i < len(self.stages):
                st = self.stages[i]
                p_rate = (self.current_price - st['entry_price']) / st['entry_price']
                p_rate = float(np.clip(p_rate, -1.0, 1.0))
                h_sec = self._calculate_seconds_diff(st['entry_time'], self.current_time)
                h_norm = float(min(h_sec / 10.0, 1.0))
                extra_features_list.extend([1.0, p_rate, h_norm])
            else:
                extra_features_list.extend([0.0, 0.0, 0.0])
                
        extra_features = np.array(extra_features_list, dtype=np.float32)
        
        # 각 timestep 마다 동일한 메타데이터 추가 (Broadcasting)
        extra_features_expanded = np.tile(extra_features, (self.seq_len, 1))
        
        # 특징 벡터 (seq_len, 28) + 상태 벡터 (seq_len, 15) = (seq_len, 43)
        observation = np.concatenate([normalized_sequence, extra_features_expanded], axis=-1)
        
        return observation.astype(np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        환경 리셋
        """
        super().reset(seed=seed)
        
        # ✅ DB 연결 지연 초기화 (Multiprocessing Safe)
        self._ensure_db_connection()
        
        # ✅ 에피소드마다 normalizer 리셋
        if self.use_raw_data and self.normalizer is not None:
            self.normalizer.reset()
            logger.debug("Normalizer reset for new episode")
        
        # 에피소드 데이터 샘플링
        self.episode_data, self.episode_metadata = self._sample_episode_start()
        self.episode_length = len(self.episode_data)
        
        # 원본 누적거래대금 추출 (서브클래스에서 미지정 시)
        if self.raw_accum_trade_value is None or len(self.raw_accum_trade_value) != len(self.episode_data):
            idx = getattr(self, 'accum_trade_value_index', 5)
            self.raw_accum_trade_value = self.episode_data[:, idx]
        
        # 🔧 등락률로부터 가격 계산
        self._compute_prices()
        
        # 에피소드 상태 초기화
        self.current_step = self.seq_len - 1  # 최소 seq_len만큼의 히스토리 필요
        self.stages = []
        
        # 현재 가격 및 시간
        self.current_price = self._get_current_price()
        self.current_time = float(self.episode_metadata[self.current_step, 2])
        
        # 에피소드 메타데이터 초기화
        self.episode_trades = []
        self.episode_rewards = []
        self.loss_holding_violations = 0
        
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
        if self.target_transaction_cost_rate > 0:
            ratio = min(1.0, rate / self.target_transaction_cost_rate)
            self.buy_tax_rate = self.target_buy_tax_rate * ratio
            self.sell_tax_rate = self.target_sell_tax_rate * ratio
        self.transaction_cost_rate = rate
        self.round_trip_cost = (self.transaction_cost_rate + self.buy_tax_rate) + (self.transaction_cost_rate + self.sell_tax_rate)
        logger.info(
            f"Transaction cost updated: rate={self.transaction_cost_rate:.6f}, "
            f"buy_tax={self.buy_tax_rate:.6f}, sell_tax={self.sell_tax_rate:.6f} "
            f"(Round trip: {self.round_trip_cost:.6f})"
        )

    def _check_loss_holding_penalty_only(self, holding_time: float) -> Tuple[float, bool]:
        """
        손실 보유 허용 시간 초과 체크 (penalty_only 모드)
        
        임계값을 초과하고 손실 중이면 페널티만 부여.
        강제 청산하지 않고 정책이 학습하도록 유도.
        
        Args:
            holding_time: 보유 시간 (초)
            
        Returns:
            (reward, loss_holding_triggered) 튜플
        """
        reward = 0.0
        loss_holding_triggered = False
        
        # 임계값을 초과하고 손실 중이면 페널티
        if holding_time > self.loss_holding_threshold and self.current_price < self.avg_entry_price:
            loss_holding_triggered = True
            self.loss_holding_violations += 1
            reward = -self.loss_holding_penalty
            
            logger.debug(f"Loss holding penalty: price={self.current_price:.4f}, "
                       f"avg_entry_price={self.avg_entry_price:.4f}, "
                       f"holding_time={holding_time:.2f}s, "
                       f"penalty={self.loss_holding_penalty:.4f}")
        
        return reward, loss_holding_triggered
    
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
        exit_cost = (self.transaction_cost_rate + self.sell_tax_rate) * 100 * weight
        
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
        
        안전장치: 반환값은 항상 max_holding_time 이하로 클램핑됩니다.
        """
        if str(start_time_val).startswith('0') or start_time_val == 0: return 0.0
        
        # 안전 상한선 (기본 300초)
        safe_max = getattr(self, 'max_holding_time', 300.0)
        
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
            
            # 4. 시간 차이 계산
            diff = e_seconds - s_seconds
            if diff < 0:
                # 음수 diff는 파싱 에러이거나 날짜 경계 문제
                # 스캘핑 환경에서 날짜를 넘기는 경우는 없으므로 max_holding_time으로 클램핑
                diff = safe_max
            
            # 5. 안전 클램핑: max_holding_time 초과 방지
            return min(diff, safe_max)
            
        except Exception as e:
            # 파싱 실패 시 안전장치
            logger.warning(f"Time parsing failed: {start_time_val} -> {end_time_val} ({e})")
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
        MAX_STAGES = 5

        # 1. 행동 실행
        if action == 1:  # 매수 (5단계 분할)
            if len(self.stages) < MAX_STAGES:
                if self.max_trades_per_episode is None or len(self.episode_trades) < self.max_trades_per_episode:
                    self.stages.append({
                        'entry_price': self.current_price,
                        'entry_time': self.current_time,
                        'pattern_rewarded': False
                    })
                    buy_weight = 1.0 / MAX_STAGES
                    entry_cost = (self.transaction_cost_rate + self.buy_tax_rate) * 100.0 * buy_weight
                    reward -= entry_cost
                    
                    # ✅ 매수 신호 정렬 보너스 (거래대금 증가 + 등락률 상승 시 긍정적 보상)
                    signal_bonus = self._calculate_buy_signal_reward(buy_weight)
                    reward += signal_bonus
                    
                    logger.debug(f"Buy (Stage {len(self.stages)}/{MAX_STAGES}): price={self.current_price:.1f}, signal_bonus={signal_bonus:.3f}")

        elif action == 2:  # 매도 (1단계씩 FIFO 청산)
            if len(self.stages) > 0:
                stage = self.stages.pop(0)
                holding_time = self._calculate_seconds_diff(stage['entry_time'], self.current_time)
                sell_weight = 1.0 / MAX_STAGES
                profit_rate = (self.current_price - stage['entry_price']) / stage['entry_price']
                
                weighted_profit_reward = profit_rate * 100.0 * sell_weight
                exit_cost = (self.transaction_cost_rate + self.sell_tax_rate) * 100.0 * sell_weight
                trade_reward = weighted_profit_reward - exit_cost
                
                if profit_rate > self.round_trip_cost:
                    bonus = self.win_bonus * sell_weight
                    trade_reward += bonus
                elif profit_rate < 0:
                    penalty = self.loss_penalty * sell_weight
                    trade_reward -= penalty
                    
                reward += trade_reward
                
                trade_info = {
                    'entry_price': stage['entry_price'],
                    'exit_price': self.current_price,
                    'holding_time': holding_time,
                    'profit_rate': profit_rate,
                    'reward': trade_reward,
                    'exit_reason': 'Signal Sell',
                    'weight': sell_weight
                }
                self.episode_trades.append(trade_info)
                logger.debug(f"Sell Stage (FIFO): entry={stage['entry_price']:.1f}, exit={self.current_price:.1f}, profit={profit_rate*100:.2f}%")

        # 2. 매수 후 1~5초 내 미상승 손절 & 상승 패턴 탐지 보너스 체크
        stages_to_remove = []
        for i, stage in enumerate(self.stages):
            holding_sec = self._calculate_seconds_diff(stage['entry_time'], self.current_time)
            profit_rate = (self.current_price - stage['entry_price']) / stage['entry_price']
            sell_weight = 1.0 / MAX_STAGES
            
            if 1.0 <= holding_sec <= 5.0:
                if profit_rate <= 0:
                    # 1~5초 내 가격 미상승 → 즉시 손절
                    weighted_profit_reward = profit_rate * 100.0 * sell_weight
                    exit_cost = (self.transaction_cost_rate + self.sell_tax_rate) * 100.0 * sell_weight
                    sl_reward = weighted_profit_reward - exit_cost - 1.0  # 손절 페널티 -1.0
                    reward += sl_reward
                    
                    trade_info = {
                        'entry_price': stage['entry_price'],
                        'exit_price': self.current_price,
                        'holding_time': holding_sec,
                        'profit_rate': profit_rate,
                        'reward': sl_reward,
                        'exit_reason': '1-5s Non-Rising StopLoss',
                        'weight': sell_weight
                    }
                    self.episode_trades.append(trade_info)
                    stages_to_remove.append(i)
                    self.loss_holding_violations += 1
                else:
                    # 1~5초 내 상승 패턴 탐지 보너스 (+0.5, 1회만 부여)
                    if not stage.get('pattern_rewarded', False):
                        reward += 0.5
                        stage['pattern_rewarded'] = True
            elif holding_sec > 5.0 and profit_rate <= 0:
                # 5초 초과 시에도 미상승/손실 중이면 자동 손절
                weighted_profit_reward = profit_rate * 100.0 * sell_weight
                exit_cost = (self.transaction_cost_rate + self.sell_tax_rate) * 100.0 * sell_weight
                sl_reward = weighted_profit_reward - exit_cost - 0.5
                reward += sl_reward
                
                trade_info = {
                    'entry_price': stage['entry_price'],
                    'exit_price': self.current_price,
                    'holding_time': holding_sec,
                    'profit_rate': profit_rate,
                    'reward': sl_reward,
                    'exit_reason': 'TimeLimit StopLoss',
                    'weight': sell_weight
                }
                self.episode_trades.append(trade_info)
                stages_to_remove.append(i)
                self.loss_holding_violations += 1

        for i in sorted(stages_to_remove, reverse=True):
            self.stages.pop(i)

        # 3. Dense Step Reward (활성 포지션 보유에 따른 변동분 보상)
        if len(self.stages) > 0:
            next_step_idx = self.current_step + 1
            if next_step_idx < len(self.prices):
                next_price = float(self.prices[next_step_idx])
                step_return = (next_price - self.current_price) / self.current_price
                active_weight = len(self.stages) / MAX_STAGES
                step_reward = step_return * self.step_reward_scale * active_weight
                reward += step_reward

        self.episode_rewards.append(reward)
        self.current_step += 1

        # 에피소드 종료 조건 체크
        if self.current_step >= self.episode_length - 1:
            terminated = True
        if self.max_trades_per_episode is not None and len(self.episode_trades) >= self.max_trades_per_episode and len(self.stages) == 0:
            terminated = True
        if self.max_episode_steps is not None and (self.current_step - (self.seq_len - 1)) >= self.max_episode_steps:
            truncated = True

        # 에피소드 종료 시 남아있는 모든 단계 강제 청산
        if terminated or truncated:
            sell_weight = 1.0 / MAX_STAGES
            for stage in self.stages:
                holding_time = self._calculate_seconds_diff(stage['entry_time'], self.current_time)
                profit_rate = (self.current_price - stage['entry_price']) / stage['entry_price']
                weighted_profit_reward = profit_rate * 100.0 * sell_weight
                exit_cost = (self.transaction_cost_rate + self.sell_tax_rate) * 100.0 * sell_weight
                final_trade_reward = weighted_profit_reward - exit_cost
                reward += final_trade_reward
                
                self.episode_trades.append({
                    'entry_price': stage['entry_price'],
                    'exit_price': self.current_price,
                    'holding_time': holding_time,
                    'profit_rate': profit_rate,
                    'reward': final_trade_reward,
                    'weight': sell_weight,
                    'forced_liquidation': True
                })
            self.stages = []

            if len(self.episode_trades) == 0:
                reward -= self.no_trade_penalty

        if not (terminated or truncated):
            self.current_price = self._get_current_price()
            self.current_time = float(self.episode_metadata[self.current_step, 2])
            observation = self._get_current_observation()
        else:
            observation = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)

        info = {
            'position': self.position,
            'position_steps': self.position_steps,
            'current_price': self.current_price,
            'current_time': self.current_time,
            'loss_holding_violations': self.loss_holding_violations
        }
        if terminated or truncated:
            info['episode'] = self._calculate_episode_metadata()

        return observation, reward, terminated, truncated, info
    
    def _calculate_buy_signal_reward(self, buy_weight: float) -> float:
        """
        매수 진입 시점의 시장 신호(등락률 상승 + 거래대금 증가) 평가 및 보너스 산출
        
        조건:
        1. 등락률/가격 상승: 직전 스텝(또는 단기 1~3초) 대비 현재 가격이 상승 중
        2. 거래대금/매수대금 증가: 직전 스텝 대비 누적거래대금 증가(신규 체결 및 매수 유입)
        
        두 조건이 모두 충족될 경우 긍정적인 매수 타이밍으로 판단하여 보너스를 부여합니다.
        """
        if self.current_step <= 0 or self.buy_signal_bonus <= 0:
            return 0.0
            
        prev_step = self.current_step - 1
        
        # 1. 등락률/가격 상승 여부
        is_price_rising = self.current_price > self.prices[prev_step]
        
        # 2. 거래대금/매수대금 증가 여부
        is_trade_val_increasing = False
        if self.raw_accum_trade_value is not None and len(self.raw_accum_trade_value) > self.current_step:
            trade_val_delta = float(self.raw_accum_trade_value[self.current_step] - self.raw_accum_trade_value[prev_step])
            is_trade_val_increasing = (trade_val_delta > 0.0)
        elif self.episode_data is not None and len(self.episode_data) > self.current_step:
            trade_val_delta = float(self.episode_data[self.current_step, self.accum_trade_value_index] - 
                                    self.episode_data[prev_step, self.accum_trade_value_index])
            is_trade_val_increasing = (trade_val_delta > 0.0)
            
        # 3. 긍정적 매수 신호 판정: 등락률 상승 + 거래대금 증가 동시 만족
        if is_price_rising and is_trade_val_increasing:
            bonus = self.buy_signal_bonus * buy_weight
            logger.debug(f"Positive Buy Signal Bonus (+{bonus:.3f}): price rising and trade value increased")
            return bonus
            
        return 0.0

    def _calculate_episode_metadata(self) -> Dict[str, Any]:
        """
        에피소드 종료 시 메타데이터 계산
        
        요구사항 3.7에 따라 다음 메트릭을 계산합니다:
        - 총 수익 (total_return)
        - 거래 횟수 (num_trades)
        - 평균 보유 시간 (avg_holding_time)
        - 샤프 비율 (sharpe_ratio)
        - 손실 보유 위반 횟수 (loss_holding_violations)
        
        Returns:
            에피소드 메타데이터 딕셔너리
        """
        # 총 수익 (거래별 net return의 합)
        if self.episode_trades:
            total_return = sum((t['profit_rate'] - self.round_trip_cost) * 100 * t.get('weight', 1.0) for t in self.episode_trades)
        else:
            total_return = -self.no_trade_penalty
            
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
            'loss_holding_violations': int(self.loss_holding_violations),
            'win_rate': float(win_rate),
            'avg_profit_per_trade': float(avg_profit_per_trade),
            'episode_length': int(self.episode_length),
            'steps_taken': int(self.current_step),
            'trades': self.episode_trades  # 개별 거래 데이터 포함 (평가용)
        }
        
        logger.debug(f"Episode finished: "
                   f"total_return={total_return:.4f}, "
                   f"num_trades={num_trades}, "
                   f"avg_holding_time={avg_holding_time:.2f}s, "
                   f"sharpe_ratio={sharpe_ratio:.4f}, "
                   f"loss_holding_violations={self.loss_holding_violations}, "
                   f"win_rate={win_rate:.2%}")
        
        return metadata
    
    def close(self):
        """환경 종료"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
