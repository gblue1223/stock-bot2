"""
온라인 정규화 모듈

사전 계산된 통계 없이 실시간으로 정규화를 수행합니다.
Rolling window 방식으로 최근 데이터의 통계를 사용합니다.
"""

import numpy as np
from collections import deque
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class OnlineNormalizer:
    """
    실시간 데이터 정규화 클래스
    
    최근 N개 샘플의 통계(평균, 표준편차)를 계산하여 정규화합니다.
    초기 워밍업 기간 동안은 Min-Max 스케일링을 사용합니다.
    
    장점:
    - 사전 통계 불필요
    - 시장 변동성에 적응
    - 종목 특성 자동 반영
    
    단점:
    - 초기 워밍업 필요 (약 100-200 샘플)
    - 통계가 시간에 따라 변함 (일관성 낮음)
    
    Args:
        num_features: 특징 수
        window_size: Rolling window 크기 (기본값: 200)
        warmup_samples: 워밍업 샘플 수 (기본값: 50)
        clip_range: 정규화 후 클리핑 범위 (기본값: (-5, 5))
    """
    
    def __init__(
        self, 
        num_features: int,
        window_size: int = 200,
        warmup_samples: int = 50,
        clip_range: tuple = (-5.0, 5.0)
    ):
        self.num_features = num_features
        self.window_size = window_size
        self.warmup_samples = warmup_samples
        self.clip_range = clip_range
        
        # 각 특징별 rolling buffer
        self.buffers: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(num_features)
        }
        
        # 통계 캐시
        self.mean_cache: Optional[np.ndarray] = None
        self.std_cache: Optional[np.ndarray] = None
        self.cache_valid = False
        
        # 샘플 카운트
        self.sample_count = 0
        
        # 특징별 관측 범위 (Min-Max 용)
        self.feature_mins = np.full(num_features, np.inf, dtype=np.float32)
        self.feature_maxs = np.full(num_features, -np.inf, dtype=np.float32)
        
        logger.info(
            f"OnlineNormalizer initialized: features={num_features}, "
            f"window={window_size}, warmup={warmup_samples}"
        )
    
    def update(self, features: np.ndarray):
        """
        새로운 샘플로 통계 업데이트
        
        Args:
            features: 특징 벡터 (num_features,)
        """
        if len(features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {len(features)}")
        
        # 각 특징을 버퍼에 추가
        for i, value in enumerate(features):
            self.buffers[i].append(float(value))
            
            # Min/Max 업데이트
            self.feature_mins[i] = min(self.feature_mins[i], value)
            self.feature_maxs[i] = max(self.feature_maxs[i], value)
        
        # 캐시 무효화
        self.cache_valid = False
        self.sample_count += 1
    
    def _compute_statistics(self) -> tuple:
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
                std = np.std(buffer_array)
                stds[i] = std if std > 1e-6 else 1.0  # 표준편차 0 방지
        
        return means, stds
    
    def normalize(self, features: np.ndarray, update: bool = True) -> np.ndarray:
        """
        특징 정규화
        
        Args:
            features: 특징 벡터 (num_features,)
            update: True면 이 샘플로 통계 업데이트
            
        Returns:
            정규화된 특징 벡터
        """
        if len(features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {len(features)}")
        
        # 워밍업 중이면 Min-Max 스케일링 사용
        if self.sample_count < self.warmup_samples:
            if update:
                self.update(features)
            
            # Min-Max 스케일링: [min, max] -> [-1, 1]
            normalized = np.zeros_like(features)
            for i in range(self.num_features):
                min_val = self.feature_mins[i]
                max_val = self.feature_maxs[i]
                
                if max_val > min_val:
                    normalized[i] = 2.0 * (features[i] - min_val) / (max_val - min_val) - 1.0
                else:
                    normalized[i] = 0.0
            
            return normalized
        
        # 통계 계산 (캐시 사용)
        if not self.cache_valid:
            self.mean_cache, self.std_cache = self._compute_statistics()
            self.cache_valid = True
        
        # Z-score 정규화
        normalized = (features - self.mean_cache) / (self.std_cache + 1e-8)
        
        # 클리핑
        normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        
        # 통계 업데이트
        if update:
            self.update(features)
        
        return normalized
    
    def is_ready(self) -> bool:
        """정규화가 준비되었는지 확인"""
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
        warmup_samples: int = 50
    ):
        self.num_features = num_features
        self.window_size = window_size
        self.warmup_samples = warmup_samples
        
        # 종목별 normalizer
        self.normalizers: Dict[str, OnlineNormalizer] = {}
        
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
                warmup_samples=self.warmup_samples
            )
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
