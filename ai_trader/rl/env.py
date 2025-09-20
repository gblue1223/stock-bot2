from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_trader.ml.data import load_real_dataframe


@dataclass
class TradingEnvConfig:
    db_path: str = "datasets/datasets.db"
    table: str = "datasets"
    code: Optional[str] = None
    date: Optional[str] = None
    seq_len: int = 60
    target_col: Optional[str] = None  # price-like column for reward
    max_steps: Optional[int] = None    # cap episode length
    scale_obs: bool = False            # optional standardization
    # Reward customization
    reward_mode: str = "delta"        # one of {"delta", "return", "log_return"}
    transaction_cost_bps: float = 0.0  # cost per trade in basis points of price (e.g., 5 = 0.05%)
    holding_cost_bps: float = 0.0      # per-step cost when holding a position (bps of price)
    # Observation formatting
    obs_format: str = "flat"          # "flat" -> (T*F,), "matrix" -> (T, F)


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, cfg: TradingEnvConfig):
        super().__init__()
        self.cfg = cfg
        df, real_cols = load_real_dataframe(cfg.db_path, table=cfg.table, code=cfg.code, date=cfg.date)
        self.real_cols: List[str] = real_cols
        if not self.real_cols:
            raise RuntimeError("No REAL columns available to construct environment")
        self.df = df
        # target column
        self.target_col = cfg.target_col or ("현재가" if "현재가" in real_cols else real_cols[0])
        if self.target_col not in real_cols:
            raise ValueError(f"target_col {self.target_col} not in REAL columns")
        self.prices = df[self.target_col].to_numpy(dtype=np.float32)
        self.features = df[self.real_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

        self.seq_len = cfg.seq_len
        self.ptr = 0  # index points to next tick to append
        self.position = 0  # 0=flat, 1=long
        self.entry_price = 0.0
        self.realized_pnl = 0.0

        n_features = len(self.real_cols)
        if self.cfg.obs_format == "matrix":
            obs_shape = (self.seq_len, n_features)
        else:
            obs_shape = (self.seq_len * n_features,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # hold, long/open, flat/close

        self.max_ptr = len(self.features)
        if self.max_ptr <= self.seq_len + 1:
            raise RuntimeError("Not enough rows in DB to construct an episode. Load more data or reduce seq_len.")
        self.max_steps = cfg.max_steps or (self.max_ptr - self.seq_len - 1)

        # Precompute running stats for optional obs scaling
        self._feat_mean = self.features.mean(axis=0, keepdims=True)
        self._feat_std = self.features.std(axis=0, keepdims=True) + 1e-6

    def _get_window(self) -> np.ndarray:
        w = self.features[self.ptr - self.seq_len : self.ptr]
        if self.cfg.scale_obs:
            w = (w - self._feat_mean) / self._feat_std
        if self.cfg.obs_format == "matrix":
            return w.astype(np.float32)
        return w.reshape(-1).astype(np.float32)

    def _trade_cost(self, price: float) -> float:
        # basis points of price
        return float(self.cfg.transaction_cost_bps) / 10000.0 * float(price)

    def _hold_cost(self, price: float) -> float:
        return float(self.cfg.holding_cost_bps) / 10000.0 * float(price)

    def _reward_from_prices(self, px_t: float, px_next: float, action: int) -> float:
        mode = self.cfg.reward_mode.lower()
        if mode == "delta":
            gain = (px_next - px_t)
        elif mode == "return":
            gain = (px_next / px_t - 1.0) if px_t != 0 else 0.0
        elif mode == "log_return":
            if px_t > 0 and px_next > 0:
                gain = math.log(px_next) - math.log(px_t)
            else:
                gain = 0.0
        else:
            raise ValueError(f"Unsupported reward_mode: {self.cfg.reward_mode}")

        reward = 0.0
        # position-based reward accumulation
        if self.position == 1:
            reward += gain
        # transaction cost on position change
        if action == 1 and self.position == 0:  # open long
            reward -= self._trade_cost(px_t)
        if action == 2 and self.position == 1:  # close long
            reward -= self._trade_cost(px_t)
        # holding cost when in position
        if self.position == 1:
            reward -= self._hold_cost(px_t)
        return float(reward)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # start at a random location with enough room
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        self.ptr = rng.integers(self.seq_len + 1, self.max_ptr - 1)
        self.steps_left = min(self.max_steps, self.max_ptr - self.ptr - 1)
        self.position = 0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        obs = self._get_window()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        # price at current ptr-1 (last in window) and next
        px_t = float(self.prices[self.ptr - 1])
        self.ptr += 1
        px_next = float(self.prices[self.ptr - 1])

        # Position transition logic
        if action == 1:  # long/open or maintain
            if self.position == 0:
                self.position = 1
                self.entry_price = px_t
        elif action == 2:  # flat/close
            if self.position == 1:
                # realize pnl at close
                self.realized_pnl += (px_t - self.entry_price)
            self.position = 0
            self.entry_price = 0.0
        # else: hold -> no change of position

        reward = self._reward_from_prices(px_t, px_next, action)

        self.steps_left -= 1
        terminated = self.ptr >= self.max_ptr - 1
        truncated = self.steps_left <= 0
        obs = self._get_window()
        info = {"pnl": self.realized_pnl, "position": self.position}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
