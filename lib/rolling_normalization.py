"""
Rolling Window Normalization
실시간 데이터에 적응하는 정규화 시스템
"""
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RollingNormalizer:
    """
    Rolling window 기반 정규화
    최근 N개 데이터의 통계를 사용하여 정규화
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        min_samples: int = 100,
        feature_names: Optional[list] = None,
        seed_mean: Optional[np.ndarray] = None,
        seed_std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            window_size: Rolling window 크기
            min_samples: 정규화에 필요한 최소 샘플 수
            feature_names: 특징 이름 리스트
        """
        self.window_size = window_size
        self.min_samples = min_samples
        self.feature_names = feature_names
        
        # 초기 시드 통계 (학습 통계 warm start)
        self.seed_mean = seed_mean
        self.seed_std = seed_std
        
        # Rolling window 저장소
        self.windows = None  # shape: (window_size, n_features)
        self.current_idx = 0
        self.n_samples = 0
        
        # 통계 캐시
        self.mean_cache = None
        self.std_cache = None
        self.cache_valid = False
        
        logger.info(f"RollingNormalizer initialized: window_size={window_size}, min_samples={min_samples}")
    
    def update(self, features: np.ndarray) -> None:
        """
        새로운 데이터로 window 업데이트
        
        Args:
            features: shape (n_features,) 또는 (batch_size, n_features)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        batch_size, n_features = features.shape
        
        # 첫 업데이트시 window 초기화
        if self.windows is None:
            self.windows = np.zeros((self.window_size, n_features), dtype=np.float32)
        
        # Window에 데이터 추가 (circular buffer)
        for i in range(batch_size):
            self.windows[self.current_idx] = features[i]
            self.current_idx = (self.current_idx + 1) % self.window_size
            self.n_samples = min(self.n_samples + 1, self.window_size)
        
        # 캐시 무효화
        self.cache_valid = False
    
    def _compute_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """통계 계산 (캐시 사용)"""
        if self.cache_valid:
            return self.mean_cache, self.std_cache
        
        if self.n_samples < self.min_samples:
            # 샘플이 부족하면 시드 통계(있으면) 사용, 없으면 기본값
            if self.seed_mean is not None and self.seed_std is not None:
                self.mean_cache = self.seed_mean.astype(np.float32)
                std = self.seed_std.astype(np.float32)
                self.std_cache = np.where(std < 1e-6, 1.0, std)
            else:
                # windows가 아직 초기화되지 않았을 수 있으므로 안전 처리
                n_features = self.windows.shape[1] if self.windows is not None else (
                    len(self.seed_mean) if self.seed_mean is not None else 0
                )
                if n_features <= 0:
                    # 마지막 안전망: 0/1 반환 (호출 측에서 shape 보장 필요)
                    raise ValueError("RollingNormalizer: insufficient samples and no seed statistics; cannot infer feature dimension.")
                self.mean_cache = np.zeros(n_features, dtype=np.float32)
                self.std_cache = np.ones(n_features, dtype=np.float32)
        else:
            # 유효한 데이터만 사용
            valid_data = self.windows[:self.n_samples]
            self.mean_cache = np.mean(valid_data, axis=0)
            self.std_cache = np.std(valid_data, axis=0)
            
            # std가 0인 경우 1로 설정
            self.std_cache = np.where(self.std_cache < 1e-6, 1.0, self.std_cache)
        
        self.cache_valid = True
        return self.mean_cache, self.std_cache
    
    def normalize(self, features: np.ndarray, update: bool = True) -> np.ndarray:
        """
        특징 정규화
        
        Args:
            features: shape (n_features,) 또는 (batch_size, n_features)
            update: True면 window 업데이트
        
        Returns:
            정규화된 특징
        """
        original_shape = features.shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Window 업데이트
        if update:
            self.update(features)
        
        # 통계 계산
        mean, std = self._compute_stats()
        
        # 정규화
        normalized = (features - mean) / std
        
        # 원래 shape으로 복원
        if len(original_shape) == 1:
            normalized = normalized.reshape(-1)
        
        return normalized
    
    def get_stats(self) -> Dict[str, np.ndarray]:
        """현재 통계 반환"""
        mean, std = self._compute_stats()
        return {
            'mean': mean,
            'std': std,
            'n_samples': self.n_samples,
            'window_size': self.window_size
        }
    
    def reset(self) -> None:
        """통계 초기화"""
        self.windows = None
        self.current_idx = 0
        self.n_samples = 0
        self.mean_cache = None
        self.std_cache = None
        self.cache_valid = False
        logger.info("RollingNormalizer reset")


class TimeBasedNormalizer:
    """
    시간대별 정규화
    각 시간대마다 별도의 rolling normalizer 사용
    """
    
    def __init__(
        self,
        time_windows: list = None,
        window_size: int = 500,
        min_samples: int = 50,
        feature_names: Optional[list] = None
    ):
        """
        Args:
            time_windows: 시간대 구간 리스트 [(start, end), ...]
                         예: [(90000, 93000), (93000, 100000), ...]
            window_size: 각 시간대의 rolling window 크기
            min_samples: 정규화에 필요한 최소 샘플 수
            feature_names: 특징 이름 리스트
        """
        if time_windows is None:
            # 기본: 30분 단위
            time_windows = [
                (90000, 93000),   # 09:00-09:30
                (93000, 100000),  # 09:30-10:00
                (100000, 103000), # 10:00-10:30
                (103000, 110000), # 10:30-11:00
            ]
        
        self.time_windows = time_windows
        self.window_size = window_size
        self.min_samples = min_samples
        self.feature_names = feature_names
        
        # 각 시간대별 normalizer
        self.normalizers = {
            f"{start}-{end}": RollingNormalizer(
                window_size=window_size,
                min_samples=min_samples,
                feature_names=feature_names
            )
            for start, end in time_windows
        }
        
        # 기본 normalizer (시간대 매칭 실패시)
        self.default_normalizer = RollingNormalizer(
            window_size=window_size,
            min_samples=min_samples,
            feature_names=feature_names
        )
        
        logger.info(f"TimeBasedNormalizer initialized with {len(time_windows)} time windows")
    
    def _get_time_window(self, time_value: int) -> str:
        """시간값에 해당하는 window key 반환"""
        for start, end in self.time_windows:
            if start <= time_value < end:
                return f"{start}-{end}"
        return "default"
    
    def normalize(
        self,
        features: np.ndarray,
        time_value: int,
        update: bool = True
    ) -> np.ndarray:
        """
        시간대별 정규화
        
        Args:
            features: shape (n_features,) 또는 (batch_size, n_features)
            time_value: 시간값 (예: 90530 = 09:05:30)
            update: True면 window 업데이트
        
        Returns:
            정규화된 특징
        """
        window_key = self._get_time_window(time_value)
        
        if window_key == "default":
            normalizer = self.default_normalizer
        else:
            normalizer = self.normalizers[window_key]
        
        return normalizer.normalize(features, update=update)
    
    def get_stats(self, time_value: Optional[int] = None) -> Dict:
        """
        통계 반환
        
        Args:
            time_value: None이면 모든 시간대 통계 반환
        """
        if time_value is None:
            # 모든 시간대 통계
            return {
                key: normalizer.get_stats()
                for key, normalizer in self.normalizers.items()
            }
        else:
            # 특정 시간대 통계
            window_key = self._get_time_window(time_value)
            if window_key == "default":
                return self.default_normalizer.get_stats()
            else:
                return self.normalizers[window_key].get_stats()
    
    def reset(self) -> None:
        """모든 normalizer 초기화"""
        for normalizer in self.normalizers.values():
            normalizer.reset()
        self.default_normalizer.reset()
        logger.info("TimeBasedNormalizer reset")
