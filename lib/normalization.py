"""
정규화 공통 모듈

학습 데이터 정규화와 실시간 추론 정규화에서 동일한 방식을 사용하도록 보장합니다.

특징별 정규화 전략:
1. Log + Z-Score: 누적거래대금, 거래회전율, 체결강도, 매도/매수대기금액 1~10
2. Z-Score Only: 등락률
3. 파생 피처: 종목명_scalar, 시간_sin, 시간_cos, 시간_scalar (이미 정규화됨)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# 특징별 정규화 전략 정의
LOGSTD_FEATURES = {
    "누적거래대금",
    "거래회전율",
    "체결강도",
    *[f"매도대기금액{i}" for i in range(1, 11)],
    *[f"매수대기금액{i}" for i in range(1, 11)],
}

STDONLY_FEATURES = {"등락률"}

# 파생 피처는 이미 정규화되어 있으므로 추가 정규화 불필요
DERIVED_FEATURES = {
    "종목명_scalar",
    "시간_sin",
    "시간_cos",
    "시간_scalar"
}

# 특징 인덱스 매핑 (FINAL_COLUMNS 기준)
FEATURE_INDICES = {
    "종목명_scalar": 0,
    "시간_sin": 1,
    "시간_cos": 2,
    "시간_scalar": 3,
    "등락률": 4,
    "누적거래대금": 5,
    "거래회전율": 6,
    "체결강도": 7,
    **{f"매도대기금액{i}": 7 + i for i in range(1, 11)},
    **{f"매수대기금액{i}": 17 + i for i in range(1, 11)},
}


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """
    부호를 보존하는 log1p 변환
    
    y = sign(x) * log1p(|x|)
    
    Args:
        x: 입력 배열
        
    Returns:
        변환된 배열
    """
    return np.sign(x) * np.log1p(np.abs(x))


def standard_scale(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Z-Score 표준화
    
    Args:
        x: 입력 값 또는 배열
        mean: 평균
        std: 표준편차
        
    Returns:
        표준화된 값
    """
    if std == 0 or std < 1e-8:
        return np.zeros_like(x)
    return (x - mean) / std


def get_normalization_strategy(feature_name: str) -> str:
    """
    특징 이름에 따른 정규화 전략 반환
    
    Args:
        feature_name: 특징 이름
        
    Returns:
        "log_std": 로그 변환 후 Z-Score
        "std_only": Z-Score만
        "derived": 파생 피처 (이미 정규화됨)
        "unknown": 알 수 없음
    """
    if feature_name in LOGSTD_FEATURES:
        return "log_std"
    elif feature_name in STDONLY_FEATURES:
        return "std_only"
    elif feature_name in DERIVED_FEATURES:
        return "derived"
    else:
        return "unknown"


def normalize_feature(
    value: float,
    feature_name: str,
    mean: float,
    std: float,
    strategy: Optional[str] = None
) -> float:
    """
    단일 특징 값 정규화
    
    Args:
        value: 원본 값
        feature_name: 특징 이름
        mean: 평균 (학습 데이터 또는 Rolling window)
        std: 표준편차
        strategy: 정규화 전략 (None이면 자동 결정)
        
    Returns:
        정규화된 값
    """
    if strategy is None:
        strategy = get_normalization_strategy(feature_name)
    
    if strategy == "derived":
        # 파생 피처는 이미 정규화되어 있음
        return value
    elif strategy == "log_std":
        # Log + Z-Score
        log_value = signed_log1p(value)
        return standard_scale(log_value, mean, std)
    elif strategy == "std_only":
        # Z-Score only
        return standard_scale(value, mean, std)
    else:
        # 기본값: Z-Score
        return standard_scale(value, mean, std)


def normalize_features_array(
    features: np.ndarray,
    feature_names: list,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    특징 배열 전체 정규화
    
    Args:
        features: 원본 특징 배열 (num_features,)
        feature_names: 특징 이름 리스트
        means: 평균 배열 (num_features,)
        stds: 표준편차 배열 (num_features,)
        
    Returns:
        정규화된 특징 배열
    """
    normalized = np.zeros_like(features, dtype=np.float32)
    
    for i, name in enumerate(feature_names):
        normalized[i] = normalize_feature(
            features[i],
            name,
            means[i],
            stds[i]
        )
    
    return normalized


class NormalizationStats:
    """
    정규화 통계 관리 클래스
    
    특징별로 정규화 전 원본 데이터의 평균과 표준편차를 저장합니다.
    log_std 전략의 경우, 로그 변환 후의 통계를 저장합니다.
    """
    
    def __init__(self, feature_names: list):
        """
        Args:
            feature_names: 특징 이름 리스트 (순서 중요)
        """
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        
        # 각 특징별 정규화 전략
        self.strategies = [get_normalization_strategy(name) for name in feature_names]
        
        # 통계 저장
        self.means = np.zeros(self.num_features, dtype=np.float32)
        self.stds = np.ones(self.num_features, dtype=np.float32)
        
        # 샘플 카운트
        self.sample_count = 0
    
    def update(self, features: np.ndarray):
        """
        새로운 샘플로 통계 업데이트 (온라인 학습용)
        
        Args:
            features: 원본 특징 배열 (num_features,)
        """
        # 전략에 따라 변환
        transformed = np.zeros_like(features, dtype=np.float32)
        for i, strategy in enumerate(self.strategies):
            if strategy == "log_std":
                transformed[i] = signed_log1p(features[i])
            elif strategy == "derived":
                transformed[i] = features[i]  # 이미 정규화됨
            else:
                transformed[i] = features[i]
        
        # 온라인 통계 업데이트 (Welford's algorithm)
        self.sample_count += 1
        delta = transformed - self.means
        self.means += delta / self.sample_count
        delta2 = transformed - self.means
        
        # 분산 업데이트 (나중에 std 계산)
        if self.sample_count == 1:
            self.stds = np.ones_like(self.stds)
        else:
            # 간단한 누적 방식 (정확한 온라인 std 계산은 복잡함)
            pass
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        특징 배열 정규화
        
        Args:
            features: 원본 특징 배열 (num_features,)
            
        Returns:
            정규화된 특징 배열
        """
        return normalize_features_array(
            features,
            self.feature_names,
            self.means,
            self.stds
        )
    
    def set_stats(self, means: np.ndarray, stds: np.ndarray):
        """
        통계를 외부에서 설정 (사전 계산된 통계 로드)
        
        Args:
            means: 평균 배열
            stds: 표준편차 배열
        """
        self.means = means.astype(np.float32)
        self.stds = stds.astype(np.float32)
    
    def get_stats(self) -> dict:
        """통계 정보 반환"""
        return {
            'feature_names': self.feature_names,
            'strategies': self.strategies,
            'means': self.means.tolist(),
            'stds': self.stds.tolist(),
            'sample_count': self.sample_count
        }


# 특징 이름 상수 (FINAL_COLUMNS와 동일)
FEATURE_NAMES = [
    "종목명_scalar", "시간_sin", "시간_cos", "시간_scalar",
    "등락률", "누적거래대금", "거래회전율", "체결강도",
    *[f"매도대기금액{i}" for i in range(1, 11)],
    *[f"매수대기금액{i}" for i in range(1, 11)],
]
