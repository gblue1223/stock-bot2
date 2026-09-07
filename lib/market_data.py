"""Market data contracts shared by extraction, replay and inference.

Exchange times are HHMMSSmmm, never lexicographically sorted strings. Execution
arrays contain unnormalised KRW prices and share counts; model features have a
separate ordered schema.
"""
from __future__ import annotations

import numpy as np


EXECUTION_COLUMNS = ["현재가"] + [
    f"{side}호가{kind}{level}"
    for side in ("매수", "매도")
    for kind in ("", "수량")
    for level in range(1, 11)
]
PRICE_SCALES = {"krw": 1.0, "million_krw": 1_000_000.0}


def resolve_feature_price_unit(metadata, feature_columns) -> str:
    """Recognize declared units, or the exact historical extract_260811 schema."""
    unit = metadata.get("price_unit")
    legacy = ["현재가", "등락률", "누적거래대금"] + [
        f"{side}대기금액{i}" for side in ("매도", "매수") for i in range(1, 11)
    ] + ["종목명_scalar", "시간_sin", "시간_cos", "시간_scalar"]
    if unit is None and feature_columns == legacy and "schema_version" not in metadata:
        unit = "million_krw"
    if unit is None and "현재가" not in feature_columns:
        unit = "krw"
    if unit not in PRICE_SCALES:
        raise ValueError("Manifest must declare price_unit='krw' or 'million_krw'")
    return unit


def times_to_seconds(values) -> np.ndarray:
    """Convert HHMMSSmmm numbers/strings to seconds since midnight.

    Invalid or absent timestamps become NaN so callers can reject them rather
    than invent a timestamp. Milliseconds are retained.
    """
    arr = np.asarray(values)
    try:
        numbers = arr.astype(np.float64)
    except (TypeError, ValueError):
        return np.asarray([parse_time_seconds(x) for x in arr.flat]).reshape(arr.shape)
    whole = np.floor(numbers)
    hours = np.floor(whole / 10_000_000)
    minutes = np.floor(whole / 100_000) % 100
    seconds = np.floor(whole / 1000) % 100
    millis = whole % 1000
    valid = (np.isfinite(numbers) & (numbers >= 0) & (numbers == whole)
             & (hours < 24) & (minutes < 60) & (seconds < 60))
    return np.where(valid, hours * 3600 + minutes * 60 + seconds + millis / 1000, np.nan)


def parse_time_seconds(value) -> float:
    """Scalar counterpart of :func:`times_to_seconds`. Also accepts HH:MM:SS.mmm."""
    if isinstance(value, str) and ":" in value:
        try:
            hh, mm, ss = value.split(":")
            h, m, s = int(hh), int(mm), float(ss)
            return h * 3600 + m * 60 + s if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60 else float("nan")
        except (TypeError, ValueError):
            return float("nan")
    try:
        return float(times_to_seconds(np.asarray(float(value))))
    except (TypeError, ValueError):
        return float("nan")


def chronological_order(values) -> np.ndarray:
    seconds = times_to_seconds(values)
    if not np.isfinite(seconds).all():
        raise ValueError("Episode contains invalid HHMMSSmmm timestamps")
    return np.argsort(seconds, kind="stable")


def canonical_feature_columns(expected_features: int = 27) -> list[str]:
    from lib.normalization import FEATURE_NAMES
    if expected_features == 27:
        return list(FEATURE_NAMES)
    if expected_features == 28:
        return ["종목명_scalar", "시간_sin", "시간_cos", "시간_scalar",
                "등락률", "누적거래대금", "거래회전율", "체결강도"] + [
                    f"{side}대기금액{i}" for side in ("매도", "매수") for i in range(1, 11)]
    raise ValueError(f"No registered feature schema with {expected_features} columns")


def validate_feature_columns(columns, expected_features: int | None = None) -> list[str]:
    if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
        raise ValueError("Feature schema must be a nonempty list of unique column names")
    if expected_features is not None and len(columns) != expected_features:
        raise ValueError(f"Feature count mismatch: manifest={len(columns)}, expected={expected_features}")
    for required in ("등락률", "누적거래대금"):
        if required not in columns:
            raise ValueError(f"Required feature missing: {required}")
    return list(columns)


def execution_arrays(frame, price_unit: str = "krw") -> dict[str, np.ndarray]:
    """Extract genuine execution prices/depth. Missing depth is never invented."""
    if price_unit not in PRICE_SCALES:
        raise ValueError(f"Unknown price unit: {price_unit}")
    result = {}
    if "호가시간" in frame:
        result["quote_timestamp"] = times_to_seconds(frame["호가시간"])
    elif "quote_timestamp" in frame:
        result["quote_timestamp"] = np.asarray(frame["quote_timestamp"], dtype=np.float64)
    if "현재가" in frame:
        result["last_price"] = np.asarray(frame["현재가"], dtype=np.float64) * PRICE_SCALES[price_unit]
    for side, key in (("매수", "bid"), ("매도", "ask")):
        levels = [i for i in range(1, 11)
                  if f"{side}호가{i}" in frame and f"{side}호가수량{i}" in frame]
        if levels:
            result[key + "_prices"] = np.column_stack([frame[f"{side}호가{i}"] for i in levels]).astype(np.float64) * PRICE_SCALES[price_unit]
            result[key + "_sizes"] = np.column_stack([frame[f"{side}호가수량{i}"] for i in levels]).astype(np.float64)
    return result
