"""Versioned, causal observations shared by training, distillation and inference.

Callers pass raw rows available at the decision timestamp, in schema order.
Prices and stage timestamps are execution data, never normalized features.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from lib.normalization import LOGSTD_FEATURES

SCHEMA_VERSION = 2
MAX_STAGES = 5  # Fixed observation capacity; unused slots stay zero.
STAGE_FIELDS = ("is_active", "profit_rate", "holding_fraction")
NORMALIZATION = "causal_window_log_zscore_v1"


def validate_max_stages(max_stages: int) -> int:
    if isinstance(max_stages, (bool, np.bool_)) or not isinstance(max_stages, (int, np.integer)) or not 1 <= max_stages <= MAX_STAGES:
        raise ValueError("max_stages must be an integer between 1 and 5")
    return int(max_stages)


def action_mask(stage_count: int, max_stages: int = 1) -> np.ndarray:
    """Return valid HOLD/BUY/SELL actions for filled inventory."""
    max_stages = validate_max_stages(max_stages)
    if not isinstance(stage_count, (int, np.integer)) or not 0 <= stage_count <= max_stages:
        raise ValueError("stage_count must be between zero and max_stages")
    return np.array([True, stage_count < max_stages, stage_count > 0], dtype=bool)


class ObservationBuilder:
    def __init__(self, feature_columns: Sequence[str], seq_len: int = 3000,
                 rolling_window_size: int = 1000, rolling_min_samples: int = 100,
                 max_holding_seconds: float = 300.0, feature_price_unit: str = "krw",
                 max_stages: int = 1):
        self.max_stages = validate_max_stages(max_stages)
        self.feature_columns = list(feature_columns)
        if (not self.feature_columns or any(not isinstance(x, str) or not x for x in self.feature_columns)
                or len(set(self.feature_columns)) != len(self.feature_columns)):
            raise ValueError("feature_columns must contain unique, explicit feature names")
        for name, value in (("seq_len", seq_len), ("rolling_window_size", rolling_window_size),
                            ("rolling_min_samples", rolling_min_samples)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if rolling_min_samples > rolling_window_size:
            raise ValueError("rolling_min_samples cannot exceed rolling_window_size")
        if not np.isfinite(max_holding_seconds) or max_holding_seconds <= 0:
            raise ValueError("max_holding_seconds must be positive and finite")
        if feature_price_unit not in ("krw", "million_krw"):
            raise ValueError("feature_price_unit must be 'krw' or 'million_krw'")
        self.feature_price_unit = feature_price_unit
        self.seq_len = int(seq_len)
        self.rolling_window_size = int(rolling_window_size)
        self.rolling_min_samples = int(rolling_min_samples)
        self.max_holding_seconds = float(max_holding_seconds)
        self.obs_dim = len(self.feature_columns) + MAX_STAGES * len(STAGE_FIELDS)
        self.log_indices = [i for i, name in enumerate(self.feature_columns) if name in LOGSTD_FEATURES]

    @property
    def schema(self) -> dict:
        return {"version": SCHEMA_VERSION, "feature_columns": list(self.feature_columns),
                "seq_len": self.seq_len, "rolling_window_size": self.rolling_window_size,
                "rolling_min_samples": self.rolling_min_samples,
                "max_holding_seconds": self.max_holding_seconds, "max_stages": self.max_stages,
                "stage_fields": list(STAGE_FIELDS), "normalization": NORMALIZATION,
                "feature_price_unit": self.feature_price_unit}

    @classmethod
    def from_schema(cls, schema: Mapping) -> "ObservationBuilder":
        if not isinstance(schema, Mapping):
            raise ValueError("Checkpoint has no observation_schema; retrain using the current pipeline")
        required = ("feature_columns", "seq_len", "rolling_window_size",
                    "rolling_min_samples", "max_holding_seconds", "feature_price_unit", "max_stages")
        if any(key not in schema for key in required):
            raise ValueError("Incomplete observation_schema; retrain using the current pipeline")
        builder = cls(**{key: schema[key] for key in required})
        builder.validate_schema(schema)
        return builder

    def validate_schema(self, schema: Mapping) -> None:
        if not isinstance(schema, Mapping) or dict(schema) != self.schema:
            raise ValueError("Incompatible observation_schema (feature order, normalization or stage metadata)")

    def normalize(self, raw_window: np.ndarray) -> np.ndarray:
        """Normalize rows with statistics computed only from this available window.

        As in online inference, all history rows are expressed using statistics at
        the *current* decision. No suffix beyond that decision may be supplied.
        """
        rows = np.asarray(raw_window, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.feature_columns) or not len(rows):
            raise ValueError(f"Expected nonempty raw rows with {len(self.feature_columns)} ordered features")
        if not np.isfinite(rows).all():
            raise ValueError("Observation contains invalid market data")
        transformed = rows.copy()
        if self.log_indices:
            values = transformed[:, self.log_indices]
            transformed[:, self.log_indices] = np.sign(values) * np.log1p(np.abs(values))
        stats_rows = transformed[-self.rolling_window_size:]
        if len(stats_rows) >= self.rolling_min_samples:
            mean = stats_rows.mean(axis=0)
            std = stats_rows.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            transformed = (transformed - mean) / std
        result = transformed.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Normalized observation exceeds finite float32 range")
        return result

    def build(self, raw_window: np.ndarray, stages: Sequence[Mapping] = (),
              current_price: float | None = None,
              current_time_seconds: float | None = None) -> np.ndarray:
        """Build market history plus five FIFO stages, with timestamps in seconds."""
        if len(stages) > self.max_stages:
            raise ValueError("Filled stages exceed configured max_stages")
        # Fix the history span across environment, BC and live callers.
        features = self.normalize(np.asarray(raw_window)[-self.seq_len:])
        state = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        state[-len(features):, :len(self.feature_columns)] = features
        stage_meta = np.zeros((MAX_STAGES, len(STAGE_FIELDS)), dtype=np.float32)
        if stages:
            if current_price is None or not np.isfinite(current_price) or current_price <= 0:
                raise ValueError("A positive raw execution price is required for filled stages")
            if current_time_seconds is None or not np.isfinite(current_time_seconds):
                raise ValueError("A finite current timestamp in seconds is required for filled stages")
        for index, stage in enumerate(stages):
            entry_price = float(stage["entry_price"])
            entry_time = float(stage.get("entry_time_seconds", stage.get("entry_time", np.nan)))
            if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(entry_time):
                raise ValueError("Invalid filled stage price/timestamp")
            if entry_time > current_time_seconds:
                raise ValueError("Stage entry timestamp is after the current observation")
            profit = np.clip((current_price - entry_price) / entry_price, -1.0, 1.0)
            holding = min((current_time_seconds - entry_time) / self.max_holding_seconds, 1.0)
            stage_meta[index] = (1.0, profit, holding)
        state[:, len(self.feature_columns):] = stage_meta.reshape(-1)
        return state
