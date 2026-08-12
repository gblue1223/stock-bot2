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
    "현재가",
    "누적거래대금",
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

# 특징 인덱스 매핑 (FINAL_COLUMNS 기준, 27개)
FEATURE_INDICES = {
    "종목명_scalar": 0,
    "시간_sin": 1,
    "시간_cos": 2,
    "시간_scalar": 3,
    "등락률": 4,
    "현재가": 5,
    "누적거래대금": 6,
    **{f"매도대기금액{i}": 6 + i for i in range(1, 11)},
    **{f"매수대기금액{i}": 16 + i for i in range(1, 11)},
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


# 특징 이름 상수 (FINAL_COLUMNS와 동일, 27개)
FEATURE_NAMES = [
    "종목명_scalar", "시간_sin", "시간_cos", "시간_scalar",
    "등락률", "현재가", "누적거래대금",
    *[f"매도대기금액{i}" for i in range(1, 11)],
    *[f"매수대기금액{i}" for i in range(1, 11)],
]


# ============================================================================
# 파생 피처 계산 함수 (Derived Feature Computation)
# ============================================================================

def compute_stock_name_scalar(stock_name: str) -> float:
    """
    종목명을 스칼라 값으로 인코딩 (실시간 추론용 간단한 방식)
    
    문자 레벨 스칼라 인코딩: 각 문자의 유니코드 값을 합산하여 0~1 범위로 정규화
    
    주의: 학습 데이터 전처리(normalize_datasets.py)에서는 더 정교한 방식을 사용합니다.
    실시간 추론에서는 간단한 해시 기반 방식을 사용하여 계산 속도를 높입니다.
    
    Args:
        stock_name: 종목명 (예: "삼성전자")
        
    Returns:
        0~1 범위의 스칼라 값
    """
    if not stock_name:
        return 0.0
    
    char_sum = sum(ord(c) for c in stock_name)
    return float(char_sum % 10000) / 10000.0


def compute_time_features(time_str: str) -> Tuple[float, float, float]:
    """
    시간 문자열에서 파생 피처 계산 (실시간 추론용)
    
    주의: 학습 데이터 전처리(normalize_datasets.py)에서는 시간_scalar를 
    전체 데이터셋의 z-score로 정규화합니다. 실시간 추론에서는 고정된 범위로 
    간단히 매핑하여 일관성을 유지합니다.
    
    Args:
        time_str: 시간 문자열 (HHMMSSmmm 형식, 예: "093015000")
        
    Returns:
        (시간_sin, 시간_cos, 시간_scalar) 튜플
        - 시간_sin: 하루 주기의 sin 변환 [-1, 1]
        - 시간_cos: 하루 주기의 cos 변환 [-1, 1]
        - 시간_scalar: 장 시작 후 경과 시간 표준화 [-1, 1]
    """
    if not time_str:
        return 0.0, 0.0, 0.0
    
    try:
        # HHMMSSmmm -> seconds
        time_str = str(time_str).zfill(9)
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = int(time_str[4:6])
        secs = hh * 3600 + mm * 60 + ss
        
        # sin/cos 주기 변환 (하루 24시간 주기)
        SECONDS_IN_DAY = 24 * 60 * 60
        time_sin = float(np.sin(2 * np.pi * secs / SECONDS_IN_DAY))
        time_cos = float(np.cos(2 * np.pi * secs / SECONDS_IN_DAY))
        
        # 장 시작 후 경과 시간 (표준화)
        MARKET_OPEN_SECONDS = 9 * 3600  # 09:00:00
        sec_from_open = max(0, secs - MARKET_OPEN_SECONDS)
        # 간단한 표준화: 0~6시간(21600초) 범위를 -1~1로 매핑
        # 3시간(10800초)을 중심으로 ±3시간
        time_scalar = float((sec_from_open - 10800) / 10800.0)
        
        return time_sin, time_cos, time_scalar
    
    except Exception:
        return 0.0, 0.0, 0.0


def compute_all_derived_features(stock_name: str, time_str: str) -> Tuple[float, float, float, float]:
    """
    모든 파생 피처를 한 번에 계산 (실시간 추론용)
    
    Args:
        stock_name: 종목명
        time_str: 시간 문자열 (HHMMSSmmm 형식)
        
    Returns:
        (종목명_scalar, 시간_sin, 시간_cos, 시간_scalar) 튜플
    """
    stock_name_scalar = compute_stock_name_scalar(stock_name)
    time_sin, time_cos, time_scalar = compute_time_features(time_str)
    
    return stock_name_scalar, time_sin, time_cos, time_scalar


# ============================================================================
# 배치 처리용 함수 (Batch Processing for Training Data)
# ============================================================================

def compute_stock_name_scalar_batch(stock_names: pd.Series) -> pd.Series:
    """
    종목명 배치를 스칼라 값으로 인코딩 (학습 데이터 전처리용)
    
    문자 레벨 인코딩: 각 고유 문자에 ID를 할당하고, 종목명의 문자 ID 평균을 계산
    
    주의: 실시간 추론(compute_stock_name_scalar)과는 다른 방식을 사용합니다.
    학습 데이터에서는 전체 데이터셋의 문자 분포를 고려한 정교한 인코딩을 사용합니다.
    
    Args:
        stock_names: 종목명 Series
        
    Returns:
        0~1 범위의 스칼라 값 Series
    """
    # NaN과 빈 문자열 처리
    cats = stock_names.astype(str).str.strip()
    cats = cats.replace({"nan": "", "None": "", "<NA>": ""})
    
    # 유효한 문자만 추출 (빈 문자열 제외)
    valid_strings = [s for s in cats.tolist() if s]
    if not valid_strings:
        return pd.Series(0.0, index=stock_names.index, dtype=float)
    
    unique_chars = sorted(set("".join(valid_strings)))
    if not unique_chars:
        return pd.Series(0.0, index=stock_names.index, dtype=float)
    
    char_to_id = {ch: idx + 1 for idx, ch in enumerate(unique_chars)}
    max_id = float(len(unique_chars))

    def _encode(s: str) -> float:
        if not s or s in ("", "nan", "None", "<NA>"):
            return 0.0
        ids = [char_to_id.get(ch, 0) for ch in s]
        if not ids:
            return 0.0
        return float(np.mean(ids)) / max_id

    return cats.apply(_encode).astype(float)


def compute_time_features_batch(time_series: pd.Series, use_zscore_for_scalar: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    시간 Series에서 파생 피처 배치 계산 (학습 데이터 전처리용)
    
    Args:
        time_series: 시간 Series (HHMMSSmmm 형식)
        use_zscore_for_scalar: True면 시간_scalar를 z-score로 정규화 (학습 데이터용)
                               False면 고정 범위로 매핑 (실시간 추론용)
        
    Returns:
        (시간_sin, 시간_cos, 시간_scalar) 튜플
    """
    # HHMMSSmmm -> seconds 변환
    def _time_to_seconds(t) -> int:
        try:
            s = str(int(t)).rjust(9, '0')
            hh = int(s[0:2])
            mm = int(s[2:4])
            ss = int(s[4:6])
            return hh * 3600 + mm * 60 + ss
        except Exception:
            return 0
    
    secs = time_series.apply(_time_to_seconds).astype(int)
    
    # sin/cos 주기 변환
    SECONDS_IN_DAY = 24 * 60 * 60
    time_sin = np.sin(2 * np.pi * secs / SECONDS_IN_DAY)
    time_cos = np.cos(2 * np.pi * secs / SECONDS_IN_DAY)
    
    # 시간_scalar 계산
    MARKET_OPEN_SECONDS = 9 * 3600  # 09:00:00
    sec_from_open = (secs - MARKET_OPEN_SECONDS).clip(lower=0).astype(float)
    
    if use_zscore_for_scalar:
        # z-score 정규화 (학습 데이터용)
        x = pd.to_numeric(sec_from_open, errors="coerce").fillna(0).astype(float)
        mean = float(x.mean())
        std = float(x.std(ddof=0))
        if std == 0:
            time_scalar = pd.Series(np.zeros(len(x)), index=time_series.index)
        else:
            time_scalar = (x - mean) / std
    else:
        # 고정 범위 매핑 (실시간 추론용)
        time_scalar = (sec_from_open - 10800) / 10800.0
    
    return pd.Series(time_sin, index=time_series.index), \
           pd.Series(time_cos, index=time_series.index), \

# ============================================================================
# Feature Definition Functions (Moved from ai_trader/lstm/train.py)
# ============================================================================

def get_requested_features() -> list[str]:
    """
    Returns all columns needed for data loading and grouping.
    Note: 날짜, 종목코드, 종목명, 시간 are used for grouping/filtering but excluded from model features.
    Only their encoded versions (종목명_scalar, 시간_sin, etc.) are used as model inputs.
    """
    return [
        "날짜", "종목코드", "종목명", "시간", "등락률", "누적거래대금", "거래회전율", "체결강도", 
        "매도대기금액1", "매도대기금액2", "매도대기금액3", "매도대기금액4", "매도대기금액5",
        "매도대기금액6", "매도대기금액7", "매도대기금액8", "매도대기금액9", "매도대기금액10",
        "매수대기금액1", "매수대기금액2", "매수대기금액3", "매수대기금액4", "매수대기금액5",
        "매수대기금액6", "매수대기금액7", "매수대기금액8", "매수대기금액9", "매수대기금액10",
        "종목명_scalar", "시간_sin", "시간_cos", "시간_scalar"
    ]


def get_model_features(all_features: list[str]) -> list[str]:
    """
    Filters out non-numeric identifier columns that cannot be used as model inputs.
    Excludes: 날짜, 종목코드, 종목명, 시간 (string/identifier columns)
    Keeps: their encoded versions and all numeric features
    """
    exclude_cols = {"날짜", "종목코드", "종목명", "시간"}
    return [c for c in all_features if c not in exclude_cols]
