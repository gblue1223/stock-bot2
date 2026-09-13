"""Raw, chronological NPZ episode replay with a bounded per-process cache."""
from collections import OrderedDict
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ai_trader.grpo.environments.scalping_env_e2e import GRPOScalpingEnv
from lib.market_data import chronological_order, validate_feature_columns, PRICE_SCALES, resolve_feature_price_unit
from lib.market_data import times_to_seconds
from ai_trader.grpo.entry_pattern import validate_entry_pattern, entry_signal

logger = logging.getLogger(__name__)


class GRPOScalpingEnvXLSTM(GRPOScalpingEnv):
    # Entries are immutable raw arrays, never statistics fitted to a future day.
    _episode_cache = OrderedDict()
    _episode_cache_bytes = 0

    def __init__(self, db_path: Optional[str] = None, table_name="datasets",
                 seq_len=3000, expected_features=27, extracted_dir=None,
                 allowed_dates=None, cache_max_bytes=256 * 1024 * 1024, **kwargs):
        self.extracted_dir = Path(extracted_dir).resolve() if extracted_dir else None
        self.manifest_data = None
        self.allowed_dates = None if allowed_dates is None else {str(x) for x in allowed_dates}
        self.cache_max_bytes = max(0, int(cache_max_bytes))
        effective_path = db_path or str(self.extracted_dir or "dummy.db")
        super().__init__(db_path=effective_path, table_name=table_name,
                         seq_len=seq_len, expected_features=expected_features,
                         allowed_dates=allowed_dates, **kwargs)

    def _init_metadata_from_db(self):
        if self.extracted_dir is None:
            return super()._init_metadata_from_db()
        with (self.extracted_dir / "manifest.json").open(encoding="utf-8") as stream:
            self.manifest_data = json.load(stream)
        metadata = self.manifest_data.get("metadata", {})
        if self.entry_pattern_config is not None:
            extracted = validate_entry_pattern(metadata.get('entry_pattern_config'))
            selection_keys = ('min_buy_notional_krw', 'min_price_return', 'min_window_seconds', 'max_window_seconds')
            if extracted is None or any(extracted[key] != self.entry_pattern_config[key] for key in selection_keys):
                raise ValueError('Entry-pattern data missing or selection settings differ; re-extract into a new directory')
        self.feature_columns = validate_feature_columns(metadata.get("feature_columns"), self.expected_features)
        if metadata.get("feature_transform", "raw") != "raw":
            raise ValueError("Only raw episode features are supported; regenerate normalized episodes")
        self.return_rate_index = self.feature_columns.index("등락률")
        if metadata.get("return_rate_index", self.return_rate_index) != self.return_rate_index:
            raise ValueError("Manifest return_rate_index disagrees with feature_columns")
        self.accum_trade_value_index = self.feature_columns.index("누적거래대금")
        self.price_unit = resolve_feature_price_unit(metadata, self.feature_columns)
        if "price_unit" not in metadata and self.price_unit == "million_krw":
            logger.warning("Using explicit extract_260811 legacy price adapter (million KRW); times sorted on load")
        self.price_scale = PRICE_SCALES.get(self.price_unit, 1.0)
        self.valid_keys = [
            (ep["file_path"], ep["stock_code"], ep["date"], ep["length"])
            for ep in self.manifest_data.get("episodes", [])
            if self.allowed_dates is None or str(ep["date"]) in self.allowed_dates
            if self.entry_pattern_config is None or any(
                end >= self.seq_len - 1 and start < ep['length'] - 1
                for start, end in ep.get('entry_signal_ranges', []))
        ]
        if not self.valid_keys:
            raise ValueError("No episodes match the requested dates")

    def _ensure_db_connection(self):
        if self.extracted_dir is None:
            return super()._ensure_db_connection()
        self.current_step = 0
        self.stages = []
        self.current_price = self.current_time = 0.0
        self.episode_trades = []
        self.episode_rewards = []
        self.loss_holding_violations = 0
        self.episode_data = self.episode_metadata = None
        self.episode_length = 0

    @classmethod
    def clear_episode_cache(cls):
        cls._episode_cache.clear()
        cls._episode_cache_bytes = 0

    def _load_cached_episode(self, full_path):
        full_path = full_path.resolve()
        if not full_path.is_relative_to(self.extracted_dir):
            raise ValueError("Episode path escapes extracted_dir")
        stat = full_path.stat()
        key = (str(full_path), stat.st_mtime_ns, stat.st_size, tuple(self.feature_columns), self.price_unit,
               json.dumps(self.entry_pattern_config, sort_keys=True))
        cls = type(self)
        while cls._episode_cache and cls._episode_cache_bytes > self.cache_max_bytes:
            _, (_, _, _, old_size) = cls._episode_cache.popitem(last=False)
            cls._episode_cache_bytes -= old_size
        if key in cls._episode_cache:
            cls._episode_cache.move_to_end(key)
            return cls._episode_cache[key][:3]
        with np.load(full_path, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            metadata = np.asarray(data["metadata"])
            if features.ndim != 2 or features.shape[1] != self.expected_features:
                raise ValueError("Episode feature array disagrees with manifest schema")
            if metadata.ndim != 2 or metadata.shape != (len(features), 3):
                raise ValueError("Episode metadata must have shape (N, 3)")
            if not np.isfinite(features).all():
                raise ValueError("Episode features contain NaN or infinity")
            order = chronological_order(metadata[:, 2])
            features, metadata = features[order], metadata[order]
            execution = {}
            for name in ("last_price", "bid_prices", "ask_prices", "bid_sizes", "ask_sizes", "quote_timestamp"):
                field = "execution_" + name
                if field in data:
                    values = np.asarray(data[field], dtype=np.float64)
                    if len(values) != len(features) or not np.isfinite(values).all():
                        raise ValueError(f"Invalid execution array: {field}")
                    required_ndim = 1 if name in {"last_price", "quote_timestamp"} else 2
                    if values.ndim != required_ndim or np.any(values < 0):
                        raise ValueError(f"Invalid execution shape/values: {field}")
                    execution[name] = values[order]
            for side in ("bid", "ask"):
                if (side + "_prices" in execution) != (side + "_sizes" in execution):
                    raise ValueError(f"Missing price/size pair for {side}")
                if side + "_prices" in execution and execution[side + "_prices"].shape != execution[side + "_sizes"].shape:
                    raise ValueError(f"Execution price/size shape mismatch for {side}")
            if "last_price" not in execution and "현재가" in self.feature_columns:
                execution["last_price"] = features[:, self.feature_columns.index("현재가")].astype(np.float64) * self.price_scale
            if self.entry_pattern_config is not None:
                if 'entry_signal' not in data or 'cumulative_buy_notional_krw' not in data:
                    raise ValueError('Missing entry-pattern source arrays; re-extract data')
                if len(set(metadata[:, 1])) != 1 or len(set(metadata[:, 0])) != 1:
                    raise ValueError('Entry-pattern labels must stay within one stock/date')
                stored = np.asarray(data['entry_signal'])
                cumulative = np.asarray(data['cumulative_buy_notional_krw'], dtype=np.float64)
                if stored.shape != (len(features),) or stored.dtype != np.bool_ or cumulative.shape != stored.shape:
                    raise ValueError('Invalid entry-pattern array shape/type')
                # Signals may include pre-extraction-window history, so recompute
                # only where this file itself provides the entire lookback.
                times = times_to_seconds(metadata[:, 2])
                recomputed = entry_signal(times, execution['last_price'], cumulative[order], self.entry_pattern_config)
                covered = times >= times[0] + self.entry_pattern_config['max_window_seconds']
                if not np.array_equal(stored[order][covered], recomputed[covered]):
                    raise ValueError('Stored entry signals disagree with BUY notional and prices')
                execution['entry_signal'] = stored[order]
        for values in [features, metadata, *execution.values()]:
            values.setflags(write=False)
        size = features.nbytes + metadata.nbytes + sum(x.nbytes for x in execution.values())
        while cls._episode_cache and cls._episode_cache_bytes + size > self.cache_max_bytes:
            _, (_, _, _, old_size) = cls._episode_cache.popitem(last=False)
            cls._episode_cache_bytes -= old_size
        if size <= self.cache_max_bytes:
            cls._episode_cache[key] = (features, metadata, execution, size)
            cls._episode_cache_bytes += size
        return features, metadata, execution

    def _sample_episode_start(self, max_attempts=100):
        if self.extracted_dir is None:
            return super()._sample_episode_start(max_attempts)
        options = getattr(self, "_reset_options", {})
        for attempt in range(max_attempts):
            try:
                idx = int(options["episode_index"]) if "episode_index" in options else int(self.np_random.integers(len(self.valid_keys)))
                file_name, stock, date, _ = self.valid_keys[idx]
                features, metadata, execution = self._load_cached_episode(self.extracted_dir / file_name)
                if self.entry_pattern_config is not None:
                    self._full_entry_signal = execution['entry_signal']
                    self._entry_target_times = times_to_seconds(metadata[:, 2])
                    self._entry_target_prices = execution['last_price']
                start, end = self._episode_slice_bounds(metadata)
                if self.entry_pattern_config is not None:
                    self._episode_entry_signal = self._full_entry_signal[start:end]
                self.episode_execution = {name: values[start:end].copy() for name, values in execution.items()}
                self.raw_accum_trade_value = features[start:end, self.accum_trade_value_index].copy()
                self.episode_key = {"stock_code": str(stock), "date": str(date), "start_index": start}
                return features[start:end].copy(), metadata[start:end].copy()
            except Exception as exc:
                if options:
                    raise
                logger.warning("Episode loading failed (attempt %s): %s", attempt + 1, exc)
        raise RuntimeError(f"Failed to load a valid episode after {max_attempts} attempts")
