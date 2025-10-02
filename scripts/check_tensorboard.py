#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check TensorBoard event logs to verify if training is progressing normally.

Usage examples:
  # one-off summary
  python scripts/check_tensorboard.py --event-file "D:/Workspace/Project/stock-bot/stock-bot2/models/supervised/tensorboard_logs/events.out.tfevents.1759406251.Raphael-PC.46232.0"
  python scripts/check_tensorboard.py --logdir "D:/Workspace/Project/stock-bot/stock-bot2/models/supervised/tensorboard_logs"

  # watch mode: poll every 15s and exit when idle for 120s
  python scripts/check_tensorboard.py --logdir ".../tensorboard_logs" --watch --interval 15 --idle-seconds 120

It prints:
- available scalar tags
- last/best values for key tags (train/val loss, accuracy)
- trend over recent steps
- simple anomaly checks (NaN/inf, stalled metrics, non-decreasing val loss)
"""

from __future__ import annotations
import argparse
import glob
import math
import os
from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

from tensorboard.backend.event_processing import event_accumulator as ea

# 각 태그 설명 및 성공 기준
# - Loss/Train_Epoch: 에폭 단위 학습 손실. 낮을수록 좋음. 시간이 지날수록 하락 추세가 이상적.
# - Loss/Validation_Epoch: 에폭 단위 검증 손실. 낮을수록 좋음. 학습 손실과 함께 하락하며 과적합 시 상승 반전 가능.
#   성공: 최근 구간(예: 5 스텝)에서 하락(개선)하거나, 베이스라인(분류: ~1.0986(3클래스 CE), 회귀: 초기값) 대비 유의하게 낮아짐.
# - Loss/Train_Batch: 배치 단위 학습 손실. 노이즈가 큼. 전체적으로 완만한 하락 추세가 정상. 빈번한 NaN/Inf는 실패.
# - Metrics/Val_Accuracy: 검증 정확도(분류). 높을수록 좋음. 3-클래스는 랜덤 기준선 ≈ 0.333.
#   성공: 기준선 대비 지속적 상승, 에폭 경과에 따라 점진적으로 향상.
# - Metrics/Val_Pearson_r: 회귀 상관계수 r. [-1,1]. 1에 가까울수록 좋음.
#   성공: 0 이상에서 점진적 상승(>0.2, >0.5 등 도메인에 따라 목표 설정).
# - Metrics/Val_R2: 결정계수 R^2. (-∞, 1]. 1에 가까울수록 좋음.
#   성공: 0 이상 달성 후 점진적 상승(>0.1, >0.3 등 도메인에 따라 목표 설정).
# - Learning_Rate/Epoch: 에폭 단위 러닝레이트. 스케줄러 작동 시 감소 가능.
#   성공: 손실 정체 시 적절히 감소하며 NaN/Inf 없음.
# - Learning_Rate: 배치 단위 러닝레이트(있을 경우). Epoch과 동일 해석.
KEY_TAGS = [
    "Loss/Train_Epoch",
    "Loss/Validation_Epoch",
    "Loss/Train_Batch",
    "Metrics/Val_Accuracy",
    "Metrics/Val_Pearson_r",
    "Metrics/Val_R2",
    "Learning_Rate/Epoch",
    "Learning_Rate",
]

@dataclass
class Series:
    steps: List[int]
    values: List[float]


def find_event_file(logdir: str | None, event_file: str | None) -> str:
    if event_file and os.path.isfile(event_file):
        return event_file
    if logdir and os.path.isdir(logdir):
        # pick newest event file
        candidates = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
        if not candidates:
            raise FileNotFoundError(f"No event files found in logdir: {logdir}")
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    raise FileNotFoundError("Provide --event-file or a valid --logdir containing TensorBoard event files")


def load_scalars(event_path: str) -> Dict[str, Series]:
    acc = ea.EventAccumulator(event_path, size_guidance={
        ea.SCALARS: 0,  # load all
    })
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    out: Dict[str, Series] = {}
    for t in tags:
        scalars = acc.Scalars(t)
        steps = [s.step for s in scalars]
        vals = [float(s.value) for s in scalars]
        out[t] = Series(steps=steps, values=vals)
    return out


def newest_event_file_in(logdir: str) -> str | None:
    candidates = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def get_event_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def max_step(series: Dict[str, Series]) -> int:
    m = -1
    for s in series.values():
        if s.steps:
            m = max(m, s.steps[-1])
    return m


def last_k_trend(vals: List[float], k: int = 5) -> Tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    take = vals[-k:] if len(vals) >= k else vals[:]
    return (take[0], take[-1])


def summarize(series: Dict[str, Series]) -> None:
    print("=== Available Scalar Tags ===")
    for t in sorted(series.keys()):
        print(f"- {t} (n={len(series[t].values)})")
    print("=============================")

    def print_summary(tag: str, better: str) -> None:
        if tag not in series:
            return
        s = series[tag]
        if not s.values:
            return
        last = s.values[-1]
        best = (min(s.values) if better == "lower" else max(s.values))
        best_step = s.steps[s.values.index(best)]
        v0, v1 = last_k_trend(s.values, 5)
        trend = ("down" if (v1 < v0) else ("up" if (v1 > v0) else "flat")) if (not math.isnan(v0) and not math.isnan(v1)) else "n/a"
        # 성공 해석 가이드 출력 보조
        guide = ""
        if tag == "Loss/Validation_Epoch":
            guide = "(낮을수록 좋음, 최근 하락 추세면 양호)"
        elif tag == "Loss/Train_Epoch":
            guide = "(낮을수록 좋음, 과도한 하락+검증악화는 과적합)"
        elif tag == "Loss/Train_Batch":
            guide = "(노이즈 큼, 전반적 하락 추세가 정상)"
        elif tag == "Metrics/Val_Accuracy":
            guide = "(높을수록 좋음, 3-클래스 기준선≈0.333 초과 및 상승 추세면 양호)"
        elif tag == "Metrics/Val_Pearson_r":
            guide = "(높을수록 좋음, 0 이상 유지 및 점진적 상승)"
        elif tag == "Metrics/Val_R2":
            guide = "(높을수록 좋음, 0 이상 유지 및 점진적 상승)"
        elif tag == "Learning_Rate/Epoch" or tag == "Learning_Rate":
            guide = "(스케줄러에 의해 점진적 감소 가능, NaN/Inf 금지)"
        print(f"{tag}: last={last:.6f} best={best:.6f} @step={best_step} trend(5)={trend} {guide}")

    print("\n=== Key Metrics ===")
    print_summary("Loss/Train_Epoch", better="lower")
    print_summary("Loss/Validation_Epoch", better="lower")
    print_summary("Metrics/Val_Accuracy", better="higher")
    print_summary("Metrics/Val_Pearson_r", better="higher")
    print_summary("Metrics/Val_R2", better="higher")
    print_summary("Learning_Rate/Epoch", better="lower")

    print("\n=== Anomaly Checks ===")
    # NaN/Inf check and monotonicity for val loss
    def has_nan_or_inf(vals: List[float]) -> bool:
        return any((math.isnan(v) or math.isinf(v)) for v in vals)

    if "Loss/Validation_Epoch" in series and series["Loss/Validation_Epoch"].values:
        vals = series["Loss/Validation_Epoch"].values
        print(f"ValLoss length={len(vals)} nan/inf={has_nan_or_inf(vals)}")
        if len(vals) >= 3:
            # non-increasing over last k implies potential plateau/regression
            v0, v1 = last_k_trend(vals, 5)
            if not math.isnan(v0) and not math.isnan(v1):
                if v1 >= v0:
                    print("[WARN] Validation loss did not improve over recent steps (non-decreasing trend)")
    else:
        print("[INFO] No Validation loss scalars found")

    # Accuracy increasing check (direction3)
    if "Metrics/Val_Accuracy" in series and series["Metrics/Val_Accuracy"].values:
        acc_vals = series["Metrics/Val_Accuracy"].values
        v0, v1 = last_k_trend(acc_vals, 5)
        if not math.isnan(v0) and not math.isnan(v1):
            if v1 <= v0:
                print("[WARN] Validation accuracy did not increase over recent steps")

    # Batch loss spikes check
    if "Loss/Train_Batch" in series and len(series["Loss/Train_Batch"].values) >= 10:
        bvals = series["Loss/Train_Batch"].values[-50:]
        if has_nan_or_inf(bvals):
            print("[WARN] NaN/Inf detected in recent batch losses")
        if max(bvals) > (min(bvals) * 5.0 + 1e-6):
            print("[WARN] Large spikes observed in recent batch losses")


def main():
    parser = argparse.ArgumentParser(description="Check TensorBoard event logs for training progress")
    parser.add_argument("--event-file", default=None, help="Path to a specific events.out.tfevents.* file")
    parser.add_argument("--logdir", default=None, help="Directory containing TensorBoard event files")
    parser.add_argument("--watch", action="store_true", help="Periodically check the log(s) until idle and then exit")
    parser.add_argument("--interval", type=int, default=15, help="Polling interval seconds in watch mode")
    parser.add_argument("--idle-seconds", type=int, default=120, help="Consider training finished if no update for this many seconds")
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration status in watch mode")
    args = parser.parse_args()

    if not args.watch:
        event_path = find_event_file(args.logdir, args.event_file)
        print(f"Using event file: {event_path}")
        series = load_scalars(event_path)
        if not series:
            print("No scalars found in the event file.")
            return
        summarize(series)
        return

    # Watch mode
    if not args.logdir and not args.event_file:
        raise FileNotFoundError("--watch requires either --logdir or --event-file")

    last_event_path = None
    last_mtime = 0.0
    last_step = -1
    last_change_ts = time.time()

    try:
        while True:
            # Resolve event file each iteration if a logdir is given (handles new files)
            if args.logdir:
                event_path = newest_event_file_in(args.logdir)
                if not event_path:
                    if args.verbose:
                        print("[watch] no event files yet...")
                    time.sleep(max(1, args.interval))
                    continue
            else:
                event_path = args.event_file

            # Detect file switch
            if event_path != last_event_path:
                print(f"[watch] using event file: {event_path}")
                last_event_path = event_path
                last_mtime = 0.0
                last_step = -1
                last_change_ts = time.time()

            current_mtime = get_event_mtime(event_path)
            series = {}
            try:
                series = load_scalars(event_path)
            except Exception as e:
                if args.verbose:
                    print(f"[watch] reload failed: {e}")

            current_step = max_step(series) if series else -1
            if args.verbose:
                print(f"[watch] mtime={current_mtime:.0f} step={current_step}")

            # On update: summarize once and reset idle timer
            if current_mtime != last_mtime or current_step != last_step:
                if series:
                    summarize(series)
                last_mtime = current_mtime
                last_step = current_step
                last_change_ts = time.time()
            else:
                idle = time.time() - last_change_ts
                if idle >= args.idle_seconds:
                    print(f"[watch] idle for {int(idle)}s (>= {args.idle_seconds}s). Assuming training is finished. Exiting.")
                    break

            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        print("[watch] interrupted by user")


if __name__ == "__main__":
    main()
