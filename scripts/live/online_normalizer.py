"""
학습 호환 온라인 정규화 (Training-Compatible Online Normalization)

실시간 데이터 스트림에서 Rolling window 기반으로 정규화 통계를 계산합니다.
학습 데이터와 동일한 정규화 전략을 사용합니다:
- Log + Z-Score: 누적거래대금, 거래회전율, 체결강도, 매도/매수대기금액
- Z-Score Only: 등락률
- No Normalization: 파생 피처 (이미 정규화됨)
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple
import logging

from lib.normalization import (
    FEATURE_NAMES,
    signed_log1p,
    standard_scale,
    get_normalization_strategy,
    DERIVED_FEATURES
)

logger = logging.getLogger(__name__)


class OnlineNormalizer:
    """
    학습 호환 실시간 데이터 정규화 클래스
    
    학습 데이터와 동일한 정규화 전략을 사용:
    1. Log + Z-Score: 누적거래대금, 거래회전율, 체결강도, 매도/매수대기금액
    2. Z-Score Only: 등락률
    3. No Normalization: 파생 피처 (이미 정규화됨)
    
    최근 N개 샘플의 통계를 Rolling window로 유지합니다.
    초기 워밍업 기간 동안은 Min-Max 스케일링을 사용합니다.
    
    Args:
        num_features: 특징 수 (기본값: 28)
        window_size: Rolling window 크기 (기본값: 200)
        warmup_samples: 워밍업 샘플 수 (기본값: 50)
        clip_range: 정규화 후 클리핑 범위 (기본값: (-5, 5))
    """
    
    def __init__(
        self, 
        num_features: int = 28,
        window_size: int = 200,
        warmup_samples: int = 50,
        clip_range: tuple = (-5.0, 5.0),
        use_training_stats: bool = False,
        training_mean: Optional[np.ndarray] = None,
        training_std: Optional[np.ndarray] = None
    ):
        self.num_features = num_features
        self.window_size = window_size
        self.warmup_samples = warmup_samples
        self.clip_range = clip_range
        
        # 특징 이름
        self.feature_names = FEATURE_NAMES
        
        # 특징별 정규화 전략
        self.strategies = [get_normalization_strategy(name) for name in self.feature_names]
        
        # ✅ 학습 데이터 통계 사용 여부
        self.use_training_stats = use_training_stats
        self.training_mean = training_mean
        self.training_std = training_std
        
        if use_training_stats:
            if training_mean is None or training_std is None:
                raise ValueError("use_training_stats=True requires training_mean and training_std")
            if len(training_mean) != num_features or len(training_std) != num_features:
                raise ValueError(f"Training stats must have {num_features} features")
            logger.info(f"OnlineNormalizer: Using training statistics (mean range: [{training_mean.min():.4f}, {training_mean.max():.4f}])")
        
        # 특징별 rolling buffer (변환 후 값 저장)
        # log_std 전략: 로그 변환 후 값 저장
        # std_only: 원본 값 저장
        # derived: 원본 값 저장 (정규화 안함)
        self.buffers: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(num_features)
        }
        
        # 통계 캐시
        self.mean_cache: Optional[np.ndarray] = None
        self.std_cache: Optional[np.ndarray] = None
        self.cache_valid = False
        
        # 샘플 카운트
        self.sample_count = 0
        
        # Min-Max 범위 (워밍업용)
        self.feature_mins = np.full(num_features, np.inf, dtype=np.float32)
        self.feature_maxs = np.full(num_features, -np.inf, dtype=np.float32)
        
        logger.info(
            f"TrainingCompatibleNormalizer initialized: features={num_features}, "
            f"window={window_size}, warmup={warmup_samples}"
        )
    
    def _transform_for_storage(self, features: np.ndarray) -> np.ndarray:
        """
        버퍼 저장용 변환
        
        log_std 전략: 로그 변환 적용
        나머지: 원본 유지
        """
        transformed = np.zeros_like(features, dtype=np.float32)
        
        for i, strategy in enumerate(self.strategies):
            if strategy == "log_std":
                transformed[i] = signed_log1p(features[i])
            else:
                transformed[i] = features[i]
        
        return transformed
    
    def update(self, features: np.ndarray):
        """
        새로운 샘플로 통계 업데이트
        
        Args:
            features: 원본 특징 벡터 (num_features,)
        """
        if len(features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {len(features)}")
        
        # 변환 (log_std는 로그 변환)
        transformed = self._transform_for_storage(features)
        
        # 각 특징을 버퍼에 추가
        for i, value in enumerate(transformed):
            self.buffers[i].append(float(value))
            
            # Min/Max 업데이트 (워밍업용)
            self.feature_mins[i] = min(self.feature_mins[i], value)
            self.feature_maxs[i] = max(self.feature_maxs[i], value)
        
        # 캐시 무효화
        self.cache_valid = False
        self.sample_count += 1
    
    def _compute_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        현재 버퍼에서 평균과 표준편차 계산
        
        Returns:
            (mean, std) 튜플
        """
        means = np.zeros(self.num_features, dtype=np.float32)
        stds = np.ones(self.num_features, dtype=np.float32)
        
        for i in range(self.num_features):
            buffer = self.buffers[i]
            if len(buffer) > 0:
                buffer_array = np.array(buffer)
                means[i] = np.mean(buffer_array)
                std = np.std(buffer_array, ddof=0)  # 모집단 표준편차
                stds[i] = std if std > 1e-6 else 1.0
        
        return means, stds
    
    def normalize(self, features: np.ndarray, update: bool = True) -> np.ndarray:
        """
        특징 정규화 (학습 방식과 동일)
        
        Args:
            features: 원본 특징 벡터 (num_features,)
            update: True면 이 샘플로 통계 업데이트
            
        Returns:
            정규화된 특징 벡터
        """
        if len(features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {len(features)}")
        
        # ✅ 학습 통계 사용 모드
        if self.use_training_stats:
            # 학습 데이터 통계를 직접 사용 (워밍업 불필요)
            self.mean_cache = self.training_mean
            self.std_cache = self.training_std
            
            # 통계 업데이트는 하되, 정규화에는 학습 통계 사용
            if update:
                self.update(features)
        
        # 워밍업 중이면 Min-Max 스케일링 사용 (학습 통계 미사용 시)
        elif self.sample_count < self.warmup_samples:
            if update:
                self.update(features)
            
            # Min-Max 스케일링: [min, max] -> [-1, 1]
            normalized = np.zeros_like(features, dtype=np.float32)
            for i in range(self.num_features):
                if self.strategies[i] == "derived":
                    # 파생 피처는 그대로
                    normalized[i] = features[i]
                else:
                    min_val = self.feature_mins[i]
                    max_val = self.feature_maxs[i]
                    
                    if max_val > min_val:
                        normalized[i] = 2.0 * (features[i] - min_val) / (max_val - min_val) - 1.0
                    else:
                        normalized[i] = 0.0
            
            return normalized
        
        # 통계 계산 (캐시 사용) - 학습 통계 미사용 시
        else:
            if not self.cache_valid:
                self.mean_cache, self.std_cache = self._compute_statistics()
                self.cache_valid = True
        
        # 특징별 정규화
        normalized = np.zeros_like(features, dtype=np.float32)
        
        for i, strategy in enumerate(self.strategies):
            if strategy == "derived":
                # 파생 피처는 이미 정규화되어 있음
                normalized[i] = features[i]
            elif strategy == "log_std":
                # Log + Z-Score
                log_value = signed_log1p(features[i])
                normalized[i] = standard_scale(
                    log_value,
                    self.mean_cache[i],
                    self.std_cache[i]
                )
            elif strategy == "std_only":
                # Z-Score only
                normalized[i] = standard_scale(
                    features[i],
                    self.mean_cache[i],
                    self.std_cache[i]
                )
            else:
                # 기본값: Z-Score
                normalized[i] = standard_scale(
                    features[i],
                    self.mean_cache[i],
                    self.std_cache[i]
                )
        
        # 클리핑
        normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        
        # ✅ 정규화 결과 검증 (워밍업 완료 후)
        if self.sample_count >= self.warmup_samples:
            max_abs_value = np.abs(normalized).max()
            if max_abs_value > 100:
                logger.warning(
                    f"[NORMALIZER] 정규화 후에도 값이 너무 큼! "
                    f"max_abs={max_abs_value:.2f}, "
                    f"mean_cache(first 5)={self.mean_cache[:5]}, "
                    f"std_cache(first 5)={self.std_cache[:5]}"
                )
        
        # 통계 업데이트
        if update:
            self.update(features)
        
        return normalized
    
    def is_ready(self) -> bool:
        """정규화가 준비되었는지 확인"""
        # 학습 통계 사용 시 즉시 준비됨
        if self.use_training_stats:
            return True
        return self.sample_count >= self.warmup_samples
    
    def get_stats(self) -> dict:
        """현재 통계 반환"""
        if not self.cache_valid:
            self.mean_cache, self.std_cache = self._compute_statistics()
            self.cache_valid = True
        
        return {
            'sample_count': self.sample_count,
            'is_ready': self.is_ready(),
            'mean': self.mean_cache.tolist() if self.mean_cache is not None else None,
            'std': self.std_cache.tolist() if self.std_cache is not None else None,
            'min': self.feature_mins.tolist(),
            'max': self.feature_maxs.tolist()
        }
    
    def reset(self):
        """통계 초기화"""
        for buffer in self.buffers.values():
            buffer.clear()
        
        self.mean_cache = None
        self.std_cache = None
        self.cache_valid = False
        self.sample_count = 0
        self.feature_mins.fill(np.inf)
        self.feature_maxs.fill(-np.inf)
        
        logger.info("OnlineNormalizer reset")


class PerStockNormalizer:
    """
    종목별 온라인 정규화
    
    각 종목마다 별도의 OnlineNormalizer를 유지합니다.
    종목 특성을 더 정확하게 반영할 수 있습니다.
    
    Args:
        num_features: 특징 수
        window_size: Rolling window 크기
        warmup_samples: 워밍업 샘플 수
    """
    
    def __init__(
        self,
        num_features: int,
        window_size: int = 200,
        warmup_samples: int = 50,
        use_training_stats: bool = False,
        training_mean: Optional[np.ndarray] = None,
        training_std: Optional[np.ndarray] = None
    ):
        self.num_features = num_features
        self.window_size = window_size
        self.warmup_samples = warmup_samples
        
        # ✅ 학습 통계 저장
        self.use_training_stats = use_training_stats
        self.training_mean = training_mean
        self.training_std = training_std
        
        # 종목별 normalizer
        self.normalizers: Dict[str, OnlineNormalizer] = {}
        
        if use_training_stats:
            logger.info(f"PerStockNormalizer initialized: features={num_features}, using training statistics")
        else:
            logger.info(f"PerStockNormalizer initialized: features={num_features}")
    
    def normalize(self, stock_code: str, features: np.ndarray, update: bool = True) -> np.ndarray:
        """
        종목별 정규화
        
        Args:
            stock_code: 종목 코드
            features: 특징 벡터
            update: 통계 업데이트 여부
            
        Returns:
            정규화된 특징 벡터
        """
        # 종목별 normalizer 생성
        if stock_code not in self.normalizers:
            self.normalizers[stock_code] = OnlineNormalizer(
                num_features=self.num_features,
                window_size=self.window_size,
                warmup_samples=self.warmup_samples,
                use_training_stats=self.use_training_stats,
                training_mean=self.training_mean,
                training_std=self.training_std
            )
            if self.use_training_stats:
                logger.info(f"Created normalizer for stock: {stock_code} (using training stats)")
            else:
                logger.info(f"Created normalizer for stock: {stock_code}")
        
        return self.normalizers[stock_code].normalize(features, update=update)
    
    def is_ready(self, stock_code: str) -> bool:
        """종목의 정규화가 준비되었는지 확인"""
        if stock_code not in self.normalizers:
            return False
        return self.normalizers[stock_code].is_ready()
    
    def get_stats(self, stock_code: str) -> Optional[dict]:
        """종목별 통계 반환"""
        if stock_code not in self.normalizers:
            return None
        return self.normalizers[stock_code].get_stats()
    
    def reset(self, stock_code: Optional[str] = None):
        """통계 초기화 (특정 종목 또는 전체)"""
        if stock_code:
            if stock_code in self.normalizers:
                self.normalizers[stock_code].reset()
        else:
            for normalizer in self.normalizers.values():
                normalizer.reset()
            logger.info("All normalizers reset")
