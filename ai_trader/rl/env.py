from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_trader.ml.data import load_real_dataframe
import torch
from ai_trader.ml.models import CNNLSTMAttn, ModelConfig


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
    # Optional analysis encoder
    encoder_ckpt: Optional[str] = None  # path to supervised model checkpoint directory (contains model.pt)
    # Scalping heuristics (optional)
    w_fast: float = 0.0                # weight for fast tape (activity) heuristic
    w_sticky: float = 0.0              # weight for sticky price heuristic
    priority_bonus: float = 0.0        # multiplier applied to combined priority when opening a long
    activity_window: int = 5           # lookback (steps) for fast/sticky computations
    stoploss_wait: int = 3             # steps to wait after entry; if not up, auto-stop


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
        self.hold_steps = 0  # steps since entry for stop-loss logic

        # Optional encoder model (analysis model) to produce state vectors
        self.encoder_model: Optional[CNNLSTMAttn] = None
        self._state_dim: Optional[int] = None
        if self.cfg.encoder_ckpt:
            # Expect ckpt directory with model.pt
            ckpt_path = self._resolve_ckpt_path(self.cfg.encoder_ckpt)
            device = "cpu"
            ckpt = torch.load(ckpt_path, map_location=device)
            enc_cfg = ModelConfig(**{**ckpt["config"]})
            # Ensure sequence length compatibility
            if enc_cfg.seq_len != self.seq_len:
                raise ValueError(
                    f"Encoder seq_len ({enc_cfg.seq_len}) != env seq_len ({self.seq_len}). Re-train or adjust."
                )
            self.encoder_model = CNNLSTMAttn(enc_cfg)
            self.encoder_model.load_state_dict(ckpt["state_dict"])  # type: ignore[arg-type]
            self.encoder_model.eval()
            self._state_dim = self.encoder_model.state_dim

        n_features = len(self.real_cols)
        if self.encoder_model is not None:
            obs_shape = (self._state_dim,)  # state vector dim
        else:
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
        if self.encoder_model is not None:
            # Produce state vector with encoder
            with torch.no_grad():
                x = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)  # [1, T, F]
                z = self.encoder_model.encode(x)  # [1, D]
                return z.squeeze(0).cpu().numpy().astype(np.float32)
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

    @staticmethod
    def _resolve_ckpt_path(ckpt_dir_or_file: str) -> str:
        import os
        if ckpt_dir_or_file.endswith(".pt"):
            return ckpt_dir_or_file
        return os.path.join(ckpt_dir_or_file, "model.pt")

    # --- Heuristic computations ---
    def _fast_tape(self) -> float:
        """Proxy for execution/activity speed: sum of absolute returns over a short window, normalized."""
        k = max(1, int(self.cfg.activity_window))
        if self.ptr < 2:
            return 0.0
        start = max(self.seq_len, self.ptr - k)
        p = self.prices[start - 1 : self.ptr].astype(np.float32)
        if len(p) < 2:
            return 0.0
        rets = np.diff(p)
        denom = np.abs(p[:-1]).mean() + 1e-6
        score = float(np.clip(np.sum(np.abs(rets)) / (denom * k), 0.0, 1.0))
        return score

    def _sticky_price(self) -> float:
        """Higher when recent downside moves are infrequent/mild: 1 - fraction of negative returns."""
        k = max(1, int(self.cfg.activity_window))
        if self.ptr < 2:
            return 0.0
        start = max(self.seq_len, self.ptr - k)
        p = self.prices[start - 1 : self.ptr].astype(np.float32)
        if len(p) < 2:
            return 0.0
        rets = np.diff(p)
        neg_ratio = float((rets < 0).mean())
        return float(np.clip(1.0 - neg_ratio, 0.0, 1.0))

    def _buy_priority(self, price: float) -> float:
        # Combine only fast tape and sticky price (round-figure removed)
        if (self.cfg.w_fast == 0.0 and self.cfg.w_sticky == 0.0):
            return 0.0
        f = self._fast_tape()
        s = self._sticky_price()
        wsum = max(1e-6, float(self.cfg.w_fast + self.cfg.w_sticky))
        return float(np.clip((self.cfg.w_fast * f + self.cfg.w_sticky * s) / wsum, 0.0, 1.0))

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
        self.hold_steps = 0
        obs = self._get_window()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        # price at current ptr-1 (last in window) and next
        px_t = float(self.prices[self.ptr - 1])
        self.ptr += 1
        px_next = float(self.prices[self.ptr - 1])

        opened = False
        closed = False
        stopped = False

        # Position transition logic
        if action == 1:  # long/open or maintain
            if self.position == 0:
                self.position = 1
                self.entry_price = px_t
                self.hold_steps = 0
                opened = True
        elif action == 2:  # flat/close
            if self.position == 1:
                # realize pnl at close
                self.realized_pnl += (px_t - self.entry_price)
                closed = True
            self.position = 0
            self.entry_price = 0.0
            self.hold_steps = 0
        # else: hold -> no change of position

        # base reward from env dynamics
        reward = self._reward_from_prices(px_t, px_next, action)

        # Heuristic bonus when opening a long
        if opened and self.cfg.priority_bonus != 0.0:
            pri = self._buy_priority(px_t)
            reward += float(self.cfg.priority_bonus) * pri

        # Update hold steps and apply 3-second stop-loss if not up
        if self.position == 1:
            self.hold_steps += 1
            if self.hold_steps >= int(self.cfg.stoploss_wait):
                if px_next <= self.entry_price + 1e-12:
                    # force stop at px_next
                    self.realized_pnl += (px_next - self.entry_price)
                    # apply an extra trade cost due to forced close
                    reward -= self._trade_cost(px_next)
                    self.position = 0
                    self.entry_price = 0.0
                    self.hold_steps = 0
                    stopped = True

        self.steps_left -= 1
        terminated = self.ptr >= self.max_ptr - 1
        truncated = self.steps_left <= 0
        obs = self._get_window()
        info = {"pnl": self.realized_pnl, "position": self.position}
        if opened:
            info["opened"] = True
        if closed:
            info["closed"] = True
        if stopped:
            info["stopped"] = True
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
