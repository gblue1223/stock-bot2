from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ml.data import load_real_dataframe


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

        obs_dim = self.seq_len * len(self.real_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
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
        return w.reshape(-1).astype(np.float32)

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

        reward = 0.0
        # Actions: 0=hold, 1=open/maintain long, 2=close/flat
        if action == 1:  # long
            if self.position == 0:  # open new
                self.position = 1
                self.entry_price = px_t
            # reward unrealized delta over step
            reward = (px_next - px_t)
        elif action == 2:  # flat/close
            if self.position == 1:
                pnl = px_t - self.entry_price
                self.realized_pnl += pnl
                reward = pnl  # bonus on close
            self.position = 0
            self.entry_price = 0.0
        else:  # hold
            if self.position == 1:
                reward = (px_next - px_t)

        self.steps_left -= 1
        terminated = self.ptr >= self.max_ptr - 1
        truncated = self.steps_left <= 0
        obs = self._get_window()
        info = {"pnl": self.realized_pnl, "position": self.position}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
