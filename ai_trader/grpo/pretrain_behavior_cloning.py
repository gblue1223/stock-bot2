#!/usr/bin/env python3
"""Distill a compatible GRU teacher using the same causal observations as trading."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import logging
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ai_trader.grpo.policies.scalping_policy_e2e import GRPOPolicyE2E
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from ai_trader.grpo.evaluation import chronological_date_split, normalize_date
from lib.market_data import chronological_order, times_to_seconds, resolve_feature_price_unit
from lib.observations import ObservationBuilder, validate_max_stages

logger = logging.getLogger(__name__)


class OfflineEpisodeDataset(Dataset):
    """Causal windows with deterministic hypothetical filled-stage examples.

    Synthetic stage prices/times come from the already visible prefix. They are
    examples for distillation, not evidence of historical executable fills.
    """
    def __init__(self, extracted_dir: str, seq_len: int, step_size: int = 50,
                 observation_schema=None, simulate_positions: bool = True,
                 max_cached_episodes: int = 2, allowed_dates=None):
        if step_size < 1 or max_cached_episodes < 1:
            raise ValueError("step_size and max_cached_episodes must be positive")
        self.extracted_dir = Path(extracted_dir).resolve()
        self.seq_len = seq_len
        self.step_size = step_size
        self.simulate_positions = simulate_positions
        self.max_cached_episodes = max_cached_episodes
        with (self.extracted_dir / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        metadata = manifest.get("metadata", {})
        columns = metadata.get("feature_columns")
        if metadata.get("feature_transform", "raw") != "raw":
            raise ValueError("Distillation requires raw feature episodes")
        price_unit = resolve_feature_price_unit(metadata, columns)
        self.observation_builder = (ObservationBuilder(columns, seq_len=seq_len, feature_price_unit=price_unit)
                                    if observation_schema is None else ObservationBuilder.from_schema(observation_schema))
        if self.observation_builder.execution_observations:
            raise ValueError("Schema v4 distillation requires recorded account states and execution states; "
                             "raw market episodes and synthetic stages cannot supply historical net "
                             "liquidation costs, pending orders or episode timing")
        if self.observation_builder.account_observations:
            raise ValueError("Schema v3 distillation requires recorded account states; raw market episodes "
                             "and synthetic stage examples do not provide cash, exposure or pending orders")
        if list(columns or []) != self.observation_builder.feature_columns or seq_len != self.observation_builder.seq_len:
            raise ValueError("Dataset feature order/sequence length does not match the teacher observation schema")
        if price_unit != self.observation_builder.feature_price_unit:
            raise ValueError("Dataset feature price unit does not match the teacher observation schema")
        self.observation_schema = self.observation_builder.schema
        episodes = manifest.get("episodes", [])
        if allowed_dates is None:
            allowed_dates = chronological_date_split(ep["date"] for ep in episodes)["train"]
        self.allowed_dates = {normalize_date(day) for day in allowed_dates}
        self.episodes = [ep for ep in episodes if normalize_date(ep["date"]) in self.allowed_dates]
        self.samples = [(episode_index, start)
                        for episode_index, episode in enumerate(self.episodes)
                        for start in range(0, int(episode["length"]) - seq_len + 1, step_size)]
        if not self.samples:
            raise ValueError("No complete windows in the selected distillation data")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def _load_episode(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        path = (self.extracted_dir / self.episodes[index]["file_path"]).resolve()
        if not path.is_relative_to(self.extracted_dir):
            raise ValueError("Episode path escapes the extracted data directory")
        with np.load(path, allow_pickle=False) as archive:
            raw = archive["features"].astype(np.float32)
            metadata = archive["metadata"]
            if raw.ndim != 2 or raw.shape[1] != len(self.observation_builder.feature_columns):
                raise ValueError("Episode features disagree with the declared schema")
            if metadata.ndim != 2 or metadata.shape != (len(raw), 3) or not np.isfinite(raw).all():
                raise ValueError("Invalid episode feature/metadata rows")
            if len(raw) != int(self.episodes[index]["length"]):
                raise ValueError("Episode length disagrees with manifest")
            if {normalize_date(day) for day in metadata[:, 1]} != {normalize_date(self.episodes[index]["date"])}:
                raise ValueError("Episode row dates disagree with the manifest date")
            order = chronological_order(metadata[:, 2])
            raw = raw[order]
            seconds = times_to_seconds(metadata[order, 2])
            if "execution_last_price" in archive:
                prices = archive["execution_last_price"].astype(np.float64)[order]
            elif "현재가" in self.observation_builder.feature_columns:
                # The feature unit cancels in stage-return ratios.
                prices = raw[:, self.observation_builder.feature_columns.index("현재가")].astype(np.float64)
            else:
                prices = None
            if prices is not None and (prices.shape != (len(raw),) or not np.isfinite(prices).all() or (prices <= 0).any()):
                raise ValueError("Invalid raw execution prices for synthetic stage examples")
        result = raw, seconds, prices
        self._cache[index] = result
        if len(self._cache) > self.max_cached_episodes:
            self._cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        episode_index, start = self.samples[index]
        raw, seconds, prices = self._load_episode(episode_index)
        end = start + self.seq_len
        stages = []
        current_price = None if prices is None else float(prices[end - 1])
        if self.simulate_positions and prices is not None:
            count = index % (self.observation_builder.max_stages + 1)
            # Never create positions earlier than the schema's maximum holding horizon.
            eligible = np.flatnonzero(seconds[start:end] >= seconds[end - 1] - self.observation_builder.max_holding_seconds)
            if count and len(eligible):
                offsets = eligible[np.linspace(0, len(eligible) - 1, count, dtype=int)]
                stages = [{"entry_price": float(prices[start + offset]),
                           "entry_time_seconds": float(seconds[start + offset])} for offset in offsets]
        state = self.observation_builder.build(raw[start:end], stages, current_price, float(seconds[end - 1]))
        return torch.from_numpy(state)


def distillation_loss(teacher_logits, teacher_values, student_logits, student_values,
                      states, feature_count: int, temperature: float, max_stages: int = 1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    max_stages = validate_max_stages(max_stages)
    inventory = (states[:, -1, -15::3] > 0.5).sum(dim=-1)
    valid = torch.stack((torch.ones_like(inventory, dtype=torch.bool),
                         inventory < max_stages, inventory > 0), dim=-1)
    # Finite sentinel avoids KL's 0 * infinity when an action is invalid.
    sentinel = torch.finfo(student_logits.dtype).min
    teacher_scaled = (teacher_logits / temperature).masked_fill(~valid, sentinel)
    student_scaled = (student_logits / temperature).masked_fill(~valid, sentinel)
    policy_loss = F.kl_div(F.log_softmax(student_scaled, dim=-1),
                          F.softmax(teacher_scaled, dim=-1), reduction="batchmean") * temperature ** 2
    value_loss = F.mse_loss(student_values, teacher_values)
    return policy_loss + 0.5 * value_loss


def pretrain(teacher_path: str, extracted_dir: str, output_path: str,
             epochs: int = 5, batch_size: int = 64, lr: float = 1e-4,
             temperature: float = 2.0, device: str = "cuda",
             seq_len: int | None = None, step_size: int = 200,
             train_end_date=None, validation_end_date=None, embargo_dates: int = 0):
    if epochs < 1 or batch_size < 1 or lr <= 0 or temperature <= 0:
        raise ValueError("epochs, batch_size, learning rate and temperature must be positive")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(teacher_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "policy_state_dict" not in checkpoint:
        raise ValueError("A versioned teacher checkpoint is required")
    builder = ObservationBuilder.from_schema(checkpoint.get("observation_schema"))
    if seq_len is not None and seq_len != builder.seq_len:
        raise ValueError("Distillation sequence length must match the teacher observation schema")
    with (Path(extracted_dir) / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    date_splits = chronological_date_split((ep["date"] for ep in manifest["episodes"]),
                                           train_end_date, validation_end_date,
                                           embargo_dates=embargo_dates)
    teacher_splits = checkpoint.get("extra_state", {}).get("date_splits", {})
    if not teacher_splits.get("train"):
        raise ValueError("Teacher training-date provenance is missing; holdout safety cannot be verified")
    teacher_dates = {normalize_date(day) for day in teacher_splits["train"]}
    teacher_validation_dates = {normalize_date(day) for day in teacher_splits.get("validation", [])}
    heldout_dates = set(date_splits["validation"]) | set(date_splits["test"])
    if teacher_dates & heldout_dates or max(teacher_dates) > max(date_splits["train"]):
        raise ValueError("Teacher training dates contaminate the distillation holdout period")
    if teacher_validation_dates and max(teacher_validation_dates) >= min(date_splits["test"]):
        raise ValueError("Teacher validation dates contaminate the distillation test period")
    dataset = OfflineEpisodeDataset(extracted_dir, seq_len=builder.seq_len, step_size=step_size,
                                    observation_schema=builder.schema, allowed_dates=date_splits["train"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    state = checkpoint["policy_state_dict"]
    required = ("conv1.weight", "gru.weight_hh_l0", "fc1.weight", "policy_head.weight")
    if any(key not in state for key in required):
        raise ValueError("Teacher is not a complete GRU policy checkpoint")
    model_args = {
        "obs_dim": int(state["conv1.weight"].shape[1]),
        "cnn_channels": int(state["conv1.weight"].shape[0]),
        "rnn_hidden_dim": int(state["gru.weight_hh_l0"].shape[1]),
        "fc_hidden_dim": int(state["fc1.weight"].shape[0]),
        "action_dim": int(state["policy_head.weight"].shape[0]),
    }
    if model_args["obs_dim"] != builder.obs_dim or model_args["action_dim"] != 3:
        raise ValueError("Teacher dimensions disagree with the current observation schema")
    teacher = GRPOPolicyE2E(**model_args).to(device)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    student = GRPOPolicyE2EXLSTM(**model_args, max_stages=builder.max_stages).to(device)
    for name in ("conv1", "bn1", "conv2", "bn2"):
        getattr(student, name).load_state_dict(getattr(teacher, name).state_dict(), strict=True)
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    for epoch in range(epochs):
        loss_sum = 0.0
        for states in loader:
            states = states.to(device)
            with torch.no_grad():
                teacher_logits, teacher_values = teacher(states)
            student_logits, student_values = student(states)
            loss = distillation_loss(teacher_logits, teacher_values, student_logits, student_values,
                                     states, len(builder.feature_columns), temperature, builder.max_stages)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite distillation loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())
        logger.info("Distillation epoch %s/%s loss=%.6f", epoch + 1, epochs, loss_sum / len(loader))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = dict(date_splits, train=sorted(set(date_splits["train"]) | teacher_dates),
                      validation=sorted(set(date_splits["validation"]) | teacher_validation_dates))
    torch.save({"policy_state_dict": student.state_dict(), "observation_schema": builder.schema,
                "config": {**model_args, "policy_type": student.__class__.__name__,
                           "seq_len": builder.seq_len},
                "extra_state": {"date_splits": provenance,
                                "source_teacher_train_dates": sorted(teacher_dates),
                                "source_teacher_validation_dates": sorted(teacher_validation_dates)}}, destination)
    return student


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal GRU to xLSTM distillation")
    parser.add_argument("--teacher_policy", required=True)
    parser.add_argument("--extracted_dir", default="data/extracted_episodes")
    parser.add_argument("--output_path", default="models/pretrain/xlstm_distilled.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--train_end_date", default=None)
    parser.add_argument("--validation_end_date", default=None)
    parser.add_argument("--embargo_dates", type=int, default=0)
    args = parser.parse_args()
    pretrain(args.teacher_policy, args.extracted_dir, args.output_path, args.epochs, args.batch_size,
             args.lr, args.temperature, args.device, args.seq_len, args.step_size,
             args.train_end_date, args.validation_end_date, args.embargo_dates)
