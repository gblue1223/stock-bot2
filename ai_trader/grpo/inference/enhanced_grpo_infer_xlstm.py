"""Causal xLSTM inference and risk decisions; this module never submits orders."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from lib.observations import ObservationBuilder, action_mask


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Position:
    """One filled FIFO stage. Prices use raw units, times seconds since midnight.

    Update/remove stages only after confirmed fills, never on a signal.
    holding_period is retained for callers without an explicit current clock.
    """
    entry_price: float
    entry_time: float
    current_price: float
    holding_period: float
    cumulative_return: float = 0.0
    max_price: float = 0.0
    trailing_activated: bool = False

    @property
    def profit_rate(self) -> float:
        if not np.isfinite(self.entry_price) or self.entry_price <= 0:
            raise ValueError("Position entry price must be positive and finite")
        return (self.current_price / self.entry_price - 1.0) * 100.0

    @property
    def is_profit(self) -> bool:
        return self.profit_rate > 0


class GRPOInferenceE2EXLSTM:
    def __init__(self, policy_path: Union[str, Path], device: str = "cuda",
                 normalization_stats: Optional[Dict] = None):
        if normalization_stats is not None:
            raise ValueError("External normalization stats are incompatible; use the checkpoint observation_schema")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy = self._load_policy(policy_path)

    def _load_policy(self, policy_path: Union[str, Path]):
        checkpoint = torch.load(policy_path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict) or "policy_state_dict" not in checkpoint:
            raise ValueError("Expected a versioned policy checkpoint, not bare or legacy weights")
        extra_state = checkpoint.get("extra_state") or {}
        self.training_config = dict(extra_state.get("training_config") or {})
        self.observation_builder = ObservationBuilder.from_schema(checkpoint.get("observation_schema"))
        self.observation_schema = self.observation_builder.schema
        state = checkpoint["policy_state_dict"]
        required = ("conv1.weight", "xlstm.cells.0.w_q.weight", "fc1.weight", "policy_head.weight")
        if any(key not in state for key in required):
            raise ValueError("Checkpoint does not contain a complete xLSTM policy")
        obs_dim = int(state["conv1.weight"].shape[1])
        if obs_dim != self.observation_builder.obs_dim or state["policy_head.weight"].shape[0] != 3:
            raise ValueError("Checkpoint dimensions do not match the observation schema and three actions")
        policy = GRPOPolicyE2EXLSTM(
            obs_dim=obs_dim, cnn_channels=int(state["conv1.weight"].shape[0]),
            rnn_hidden_dim=int(state["xlstm.cells.0.w_q.weight"].shape[0]),
            fc_hidden_dim=int(state["fc1.weight"].shape[0]), action_dim=3,
            max_stages=self.observation_builder.max_stages)
        policy.load_state_dict(state, strict=True)
        policy.to(self.device)
        policy.eval()
        return policy

    def set_normalization_stats(self, stats: Dict):
        raise ValueError("Normalization is fixed by the versioned causal observation schema")

    def build_observation(self, raw_window, *, stages=(), current_price=None,
                          current_time_seconds=None, feature_columns=None,
                          feature_price_unit=None) -> np.ndarray:
        if feature_columns is not None and list(feature_columns) != self.observation_builder.feature_columns:
            raise ValueError("Live feature order does not match the checkpoint")
        if feature_price_unit is not None and feature_price_unit != self.observation_builder.feature_price_unit:
            raise ValueError("Live feature price unit does not match the checkpoint")
        if isinstance(raw_window, torch.Tensor):
            raw_window = raw_window.detach().cpu().numpy()
        return self.observation_builder.build(raw_window, stages, current_price, current_time_seconds)

    def predict(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = True, *,
                stages=(), current_price=None, current_time_seconds=None,
                feature_columns=None, feature_price_unit=None) -> Tuple[int, float]:
        """Predict from raw rows; already-normalized full observations are rejected."""
        state = self.build_observation(obs, stages=stages, current_price=current_price,
                                       current_time_seconds=current_time_seconds,
                                       feature_columns=feature_columns, feature_price_unit=feature_price_unit)
        inputs = torch.from_numpy(state).unsqueeze(0).to(self.device)
        valid = torch.as_tensor(action_mask(len(stages), self.observation_builder.max_stages), device=self.device)
        with torch.inference_mode():
            logits, _ = self.policy(inputs)
            logits = logits.masked_fill(~valid.unsqueeze(0), -torch.inf)
            if not torch.isfinite(logits[:, valid]).all():
                raise ValueError("Policy produced nonfinite action logits")
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else distribution.sample()
            index = int(action.item())
            confidence = float(distribution.probs[0, index].item())
        return index, confidence


class EnhancedGRPOInferenceXLSTM:
    """Apply the training risk limits by default; extra exits require opt-in."""

    def __init__(self, base_inference: GRPOInferenceE2EXLSTM,
                 min_buy_confidence: float = 0.0, min_sell_confidence: float = 0.0,
                 stop_loss_rate: Optional[float] = None, take_profit_rate: Optional[float] = None,
                 trailing_stop_activation_rate: Optional[float] = None,
                 trailing_stop_callback_rate: float = 1.0,
                 stagnation_exit_seconds: float = 0.0, stagnation_threshold: float = 0.5,
                 enable_auto_exit: bool = True, max_holding_seconds: Optional[float] = None):
        self.base_inference = base_inference
        self.min_buy_confidence = min_buy_confidence
        self.min_sell_confidence = min_sell_confidence
        training_config = getattr(base_inference, "training_config", {})
        self.stop_loss_rate = (-float(training_config.get("stop_loss_pct", 2.0))
                               if stop_loss_rate is None else stop_loss_rate)
        self.take_profit_rate = take_profit_rate
        self.trailing_stop_activation_rate = trailing_stop_activation_rate
        self.trailing_stop_callback_rate = trailing_stop_callback_rate
        self.stagnation_exit_seconds = stagnation_exit_seconds
        self.stagnation_threshold = stagnation_threshold
        self.enable_auto_exit = enable_auto_exit
        self.max_holding_seconds = (base_inference.observation_builder.max_holding_seconds
                                    if max_holding_seconds is None else max_holding_seconds)
        if not np.isfinite(self.max_holding_seconds) or self.max_holding_seconds <= 0:
            raise ValueError("max_holding_seconds must be positive and finite")
        if not 0 <= min_buy_confidence <= 1 or not 0 <= min_sell_confidence <= 1:
            raise ValueError("Confidence thresholds must be in [0, 1]")
        if not np.isfinite(self.stop_loss_rate) or self.stop_loss_rate >= 0:
            raise ValueError("stop_loss_rate must be finite and negative")
        for name, threshold in (("take_profit_rate", take_profit_rate),
                                ("trailing_stop_activation_rate", trailing_stop_activation_rate)):
            if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
                raise ValueError(f"{name} must be None or finite and positive")
        if (not np.isfinite(trailing_stop_callback_rate) or trailing_stop_callback_rate <= 0
                or not np.isfinite(stagnation_exit_seconds) or stagnation_exit_seconds < 0
                or not np.isfinite(stagnation_threshold)):
            raise ValueError("Invalid trailing or stagnation threshold")
        self.stats = dict.fromkeys(("total_predictions", "buy_signals", "buy_filtered",
                                   "sell_signals", "auto_exits", "stop_losses", "take_profits",
                                   "max_holding_exits", "trailing_exits", "stagnation_exits"), 0)

    def predict(self, sequence: np.ndarray,
                current_position: Optional[Union[Position, Sequence[Position]]] = None,
                current_price: float = 0.0, deterministic: bool = True, *,
                current_time_seconds: Optional[float] = None, feature_columns=None,
                feature_price_unit=None) -> Tuple[int, float, Dict]:
        self.stats["total_predictions"] += 1
        positions = ([] if current_position is None else
                     [current_position] if isinstance(current_position, Position) else list(current_position))
        if len(positions) > self.base_inference.observation_builder.max_stages:
            raise ValueError("Filled stages exceed checkpoint max_stages")
        if positions:
            if not np.isfinite(current_price) or current_price <= 0:
                raise ValueError("Risk decisions require the latest positive raw execution price")
            if current_time_seconds is None:
                current_time_seconds = max(p.entry_time + p.holding_period for p in positions)
            if not np.isfinite(current_time_seconds):
                raise ValueError("Risk decisions require a finite timestamp in seconds")
        stages = []
        for position in positions:
            if not np.isfinite(position.entry_price) or position.entry_price <= 0:
                raise ValueError("Invalid filled entry price")
            if not np.isfinite(position.entry_time) or position.entry_time > current_time_seconds:
                raise ValueError("Invalid filled entry timestamp")
            position.current_price = current_price
            position.holding_period = current_time_seconds - position.entry_time
            position.cumulative_return = current_price / position.entry_price - 1.0
            position.max_price = max(position.entry_price, position.max_price, current_price)
            peak_profit = (position.max_price / position.entry_price - 1.0) * 100.0
            position.trailing_activated |= (self.trailing_stop_activation_rate is not None
                                            and peak_profit >= self.trailing_stop_activation_rate)
            stages.append({"entry_price": position.entry_price, "entry_time_seconds": position.entry_time})
        if self.enable_auto_exit:
            for position in positions:
                pct = position.profit_rate
                reason, counter = None, None
                if position.holding_period >= self.max_holding_seconds:
                    reason, counter = "Maximum Holding Time", "max_holding_exits"
                elif pct <= self.stop_loss_rate:
                    reason, counter = "Stop Loss Hit", "stop_losses"
                elif self.take_profit_rate is not None and pct >= self.take_profit_rate:
                    reason, counter = "Take Profit Hit", "take_profits"
                elif (self.trailing_stop_activation_rate is not None and position.trailing_activated and
                      (position.max_price / position.entry_price - 1.0) * 100.0 - pct >= self.trailing_stop_callback_rate):
                    reason, counter = "Trailing Stop Hit", "trailing_exits"
                elif (self.stagnation_exit_seconds > 0 and
                      position.holding_period >= self.stagnation_exit_seconds and pct <= self.stagnation_threshold):
                    reason, counter = "Stagnation Exit", "stagnation_exits"
                if reason:
                    self.stats["auto_exits"] += 1
                    self.stats[counter] += 1
                    # Signal only. Execution must liquidate all filled stages
                    # and keep retrying until confirmed fills establish flatness.
                    return Action.SELL.value, 1.0, {"reason": reason, "profit_rate": pct,
                                                   "holding_seconds": position.holding_period,
                                                   "risk_exit_all": True, "stage_count": len(stages)}
        action, confidence = self.base_inference.predict(
            sequence, deterministic, stages=stages, current_price=current_price,
            current_time_seconds=current_time_seconds, feature_columns=feature_columns,
            feature_price_unit=feature_price_unit)
        info = {"raw_action": action, "confidence": confidence}
        if not action_mask(len(stages), self.base_inference.observation_builder.max_stages)[action]:
            info.update(filtered=True, filter_reason="Invalid action for filled inventory")
            return Action.HOLD.value, confidence, info
        if action == Action.BUY.value:
            self.stats["buy_signals"] += 1
            if confidence < self.min_buy_confidence:
                self.stats["buy_filtered"] += 1
                action = Action.HOLD.value
                info.update(filtered=True, filter_reason="Buy confidence below threshold")
        elif action == Action.SELL.value:
            self.stats["sell_signals"] += 1
            if confidence < self.min_sell_confidence:
                action = Action.HOLD.value
                info.update(filtered=True, filter_reason="Sell confidence below threshold")
        return action, confidence, info
