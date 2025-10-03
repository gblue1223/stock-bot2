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
import sys

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
# - Loss/Train_Chunk: 청크 단위 학습 손실 평균. 낮을수록 좋음. 에폭 종료 전 중간 추세 확인용.
# - Loss/Validation_Chunk: 청크 단위 검증 손실 평균. 낮을수록 좋음. 최근 하락 추세면 양호.
# - Metrics/Val_Accuracy_Chunk: 청크 단위 정확도(누적 로직 기반). 높을수록 좋음. 3-클래스 기준선 ≈0.333 초과 및 상승 추세면 양호.
# - Metrics/Val_Accuracy_Chunk_logits: 청크 단위 정확도(모델 로짓 argmax 직접 산출). 위와 유사하되, 산출 경로가 다르므로 둘이 유사해야 정상.
# - Metrics/Val_True_Class{0,1,2}_Count_Chunk: 청크 내 타깃 클래스별 개수. 분포가 특정 클래스에 과도하게 치우치지 않는지 확인.
# - Metrics/Val_Pred_Class{0,1,2}_Count_Chunk: 청크 내 예측 클래스별 개수. 다수 클래스 편향 여부 확인.
#   성공: 타깃/예측 분포가 유사하고, 소수 클래스에 대한 예측도 점차 개선.


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
    print("=== 사용 가능한 스칼라 태그 목록 ===")
    for t in sorted(series.keys()):
        print(f"- {t} (개수={len(series[t].values)})")
    print("==============================")

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
        print(f"{tag}: 마지막={last:.6f} 최고={best:.6f} @단계={best_step} 추세(5)={trend} {guide}")

    print("\n=== 핵심 지표 ===")
    print_summary("Loss/Train_Epoch", better="lower")
    print_summary("Loss/Validation_Epoch", better="lower")
    print_summary("Metrics/Val_Accuracy", better="higher")
    print_summary("Metrics/Val_Pearson_r", better="higher")
    print_summary("Metrics/Val_R2", better="higher")
    print_summary("Learning_Rate/Epoch", better="lower")

    print("\n=== 이상 징후 점검 ===")
    # NaN/Inf check and monotonicity for val loss
    def has_nan_or_inf(vals: List[float]) -> bool:
        return any((math.isnan(v) or math.isinf(v)) for v in vals)

    if "Loss/Validation_Epoch" in series and series["Loss/Validation_Epoch"].values:
        vals = series["Loss/Validation_Epoch"].values
        print(f"검증 손실 길이={len(vals)} NaN/Inf 포함={has_nan_or_inf(vals)}")
        if len(vals) >= 3:
            # non-increasing over last k implies potential plateau/regression
            v0, v1 = last_k_trend(vals, 5)
            if not math.isnan(v0) and not math.isnan(v1):
                if v1 >= v0:
                    print("[경고] 최근 구간에서 검증 손실이 개선되지 않았습니다(비감소 추세)")
    else:
        print("[정보] 검증 손실 스칼라가 없습니다 (에폭/청크 태그 모두 미존재)")

    # Accuracy increasing check (direction3)
    if "Metrics/Val_Accuracy" in series and series["Metrics/Val_Accuracy"].values:
        acc_vals = series["Metrics/Val_Accuracy"].values
        v0, v1 = last_k_trend(acc_vals, 5)
        if not math.isnan(v0) and not math.isnan(v1):
            if v1 <= v0:
                print("[경고] 최근 구간에서 검증 정확도가 상승하지 않았습니다")

    # Batch loss spikes check
    if "Loss/Train_Batch" in series and len(series["Loss/Train_Batch"].values) >= 10:
        bvals = series["Loss/Train_Batch"].values[-50:]
        if has_nan_or_inf(bvals):
            print("[경고] 최근 배치 손실에서 NaN/Inf가 감지되었습니다")
        if max(bvals) > (min(bvals) * 5.0 + 1e-6):
            print("[경고] 최근 배치 손실에서 큰 스파이크가 관측되었습니다")


# ---------------- STATS + EXPORTS -----------------
def compute_stats(values: List[float]) -> Dict[str, float]:
    import statistics as stat
    if not values:
        return {"n": 0}
    n = len(values)
    mean = stat.fmean(values)
    std = stat.pstdev(values) if n > 1 else 0.0
    vmin, vmax = min(values), max(values)
    last = values[-1]
    first = values[0]
    pct = ((last - first) / (abs(first) + 1e-9)) * 100.0 if n > 1 else 0.0
    # simple slope via last_k trend
    v0, v1 = last_k_trend(values, min(10, n))
    slope = (v1 - v0) / max(1, min(10, n) - 1) if not (math.isnan(v0) or math.isnan(v1)) else 0.0
    return {"n": n, "mean": mean, "std": std, "min": vmin, "max": vmax, "last": last, "pct": pct, "slope": slope}


def print_stats(series: Dict[str, Series], tags: List[str], last: int) -> None:
    print("\n=== 통계(최근) ===")
    for tag in tags:
        if tag not in series or not series[tag].values:
            continue
        vals = series[tag].values[-last:] if last > 0 else series[tag].values
        st = compute_stats(vals)
        print(f"{tag}: 개수={st['n']} 마지막={st['last']:.6g} 평균={st['mean']:.6g} 표준편차={st['std']:.6g} 최소={st['min']:.6g} 최대={st['max']:.6g} 변화율={st['pct']:.3f}% 기울기~={st['slope']:.3g}\n")


def print_class_distribution(series: Dict[str, Series], last: int) -> None:
    # Aggregate last-N counts if available and show proportions for true/pred
    true_tags = [
        'Metrics/Val_True_Class0_Count_Chunk',
        'Metrics/Val_True_Class1_Count_Chunk',
        'Metrics/Val_True_Class2_Count_Chunk',
    ]
    pred_tags = [
        'Metrics/Val_Pred_Class0_Count_Chunk',
        'Metrics/Val_Pred_Class1_Count_Chunk',
        'Metrics/Val_Pred_Class2_Count_Chunk',
    ]
    def agg(tags: List[str]) -> List[float]:
        out = []
        for t in tags:
            if t in series and series[t].values:
                vals = series[t].values[-last:] if last > 0 else series[t].values
                out.append(float(sum(vals)))
            else:
                out.append(0.0)
        s = sum(out)
        return [v / s if s > 0 else 0.0 for v in out]
    true_prop = agg(true_tags)
    pred_prop = agg(pred_tags)
    if sum(true_prop) > 0 or sum(pred_prop) > 0:
        print("=== 클래스 분포(최근) ===")
        print(f"실제: C0={true_prop[0]:.3f} C1={true_prop[1]:.3f} C2={true_prop[2]:.3f}")
        print(f"예측: C0={pred_prop[0]:.3f} C1={pred_prop[1]:.3f} C2={pred_prop[2]:.3f}")


# ---------------- Automated Diagnosis -----------------
def _get_vals(series: Dict[str, Series], tag: str, last: int) -> List[float]:
    if tag not in series or not series[tag].values:
        return []
    vals = series[tag].values
    return vals[-last:] if last > 0 else vals


def analyze_training(series: Dict[str, Series], last: int = 200) -> Tuple[List[str], List[str]]:
    """Analyze training progress and return (warnings, successes)"""
    warnings: List[str] = []
    successes: List[str] = []

    # 1) Validation loss presence
    val_epoch = _get_vals(series, "Loss/Validation_Epoch", last)
    val_chunk = _get_vals(series, "Loss/Validation_Chunk", last)
    if not val_epoch and not val_chunk:
        warnings.append("검증 손실 스칼라를 찾을 수 없습니다 (에폭/청크 모두). 검증이 올바르게 수행되지 않을 수 있습니다.")
    else:
        successes.append("✅ 검증 손실 데이터가 정상적으로 기록되고 있습니다")

    # Choose a validation loss signal to analyze trend/level
    vloss = val_epoch if val_epoch else val_chunk

    # 2) NaN/Inf checks on losses
    def has_bad(vals: List[float]) -> bool:
        return any((math.isnan(v) or math.isinf(v)) for v in vals)
    train_batch = _get_vals(series, "Loss/Train_Batch", last)
    train_chunk = _get_vals(series, "Loss/Train_Chunk", last)
    train_epoch = _get_vals(series, "Loss/Train_Epoch", last)
    if has_bad(train_batch) or has_bad(train_chunk) or has_bad(train_epoch) or has_bad(vloss):
        warnings.append("최근 손실값에서 NaN/Inf가 감지되었습니다. 데이터와 학습률을 확인하세요.")
    else:
        successes.append("✅ 손실값이 안정적으로 계산되고 있습니다 (NaN/Inf 없음)")

    # 3) Chance-level plateau for 3-class CE (~ln(3)≈1.0986)
    if vloss:
        last_v = vloss[-1]
        v0, v1 = last_k_trend(vloss, min(10, len(vloss)))
        near_chance = abs(last_v - math.log(3)) < 0.02  # within ~0.02 of ln(3)
        non_decreasing = (not math.isnan(v0) and not math.isnan(v1) and v1 >= v0)
        if near_chance and non_decreasing:
            warnings.append(f"검증 손실이 우연 수준에 정체되어 있습니다 (~ln(3)≈1.0986). 현재={last_v:.4f}. 예측 붕괴 또는 학습 부족 가능성.")
        elif last_v < math.log(3) - 0.1:  # significantly better than chance
            successes.append(f"🎯 검증 손실이 우연 수준보다 유의하게 낮습니다 (현재={last_v:.4f} < 1.099)")

    # 4) Class collapse detection using chunk diagnostics (streaming path)
    true0 = sum(_get_vals(series, 'Metrics/Val_True_Class0_Count_Chunk', last))
    true1 = sum(_get_vals(series, 'Metrics/Val_True_Class1_Count_Chunk', last))
    true2 = sum(_get_vals(series, 'Metrics/Val_True_Class2_Count_Chunk', last))
    pred0 = sum(_get_vals(series, 'Metrics/Val_Pred_Class0_Count_Chunk', last))
    pred1 = sum(_get_vals(series, 'Metrics/Val_Pred_Class1_Count_Chunk', last))
    pred2 = sum(_get_vals(series, 'Metrics/Val_Pred_Class2_Count_Chunk', last))
    pred_sum = pred0 + pred1 + pred2
    true_sum = true0 + true1 + true2
    
    if pred_sum > 0:
        p0, p1, p2 = pred0 / pred_sum, pred1 / pred_sum, pred2 / pred_sum
        collapsed = False
        if p1 > 0.98 and p0 < 0.01 and p2 < 0.01:
            warnings.append("예측 붕괴 감지: 거의 모든 검증이 클래스 1로 예측됩니다.")
            collapsed = True
        if p0 > 0.98 and p1 < 0.01 and p2 < 0.01:
            warnings.append("예측 붕괴 감지: 거의 모든 검증이 클래스 0으로 예측됩니다.")
            collapsed = True
        if p2 > 0.98 and p0 < 0.01 and p1 < 0.01:
            warnings.append("예측 붕괴 감지: 거의 모든 검증이 클래스 2로 예측됩니다.")
            collapsed = True
        
        if not collapsed and min(p0, p1, p2) > 0.05:  # all classes have some predictions
            successes.append(f"🎲 모든 클래스에 대한 예측이 균형적입니다 (C0:{p0:.2f}, C1:{p1:.2f}, C2:{p2:.2f})")

    # 5) Accuracy at majority baseline
    acc_chunk = _get_vals(series, 'Metrics/Val_Accuracy_Chunk', last)
    acc_epoch = _get_vals(series, 'Metrics/Val_Accuracy', last)
    vacc = acc_epoch[-1] if acc_epoch else (acc_chunk[-1] if acc_chunk else None)
    if vacc is not None and true_sum > 0:
        majority = max(true0, true1, true2) / true_sum if true_sum > 0 else None
        if majority is not None:
            if abs(vacc - majority) < 0.02 and pred_sum > 0:
                warnings.append(f"검증 정확도가 다수 클래스 기준선과 유사합니다 (정확도={vacc:.3f}, 기준선={majority:.3f}). 편향된 예측 가능성.")
            elif vacc > majority + 0.05:  # significantly better than majority baseline
                successes.append(f"📈 검증 정확도가 다수 클래스 기준선을 상회합니다 (정확도={vacc:.3f} > 기준선={majority:.3f})")
            
            # Check if accuracy is above random baseline for 3-class
            if vacc > 0.4:  # well above 1/3 for 3-class
                successes.append(f"🎯 검증 정확도가 우연 수준을 크게 상회합니다 (정확도={vacc:.3f} >> 0.333)")

    # 6) Stalled learning: no improvement trend on validation loss and accuracy not increasing
    loss_improving = False
    acc_improving = False
    
    if vloss and len(vloss) >= 5:
        v0, v1 = last_k_trend(vloss, 5)
        if not math.isnan(v0) and not math.isnan(v1):
            if v1 >= v0:
                warnings.append("최근 구간에서 검증 손실이 개선되지 않았습니다 (비감소 추세).")
            else:
                loss_improving = True
                successes.append("📉 검증 손실이 최근 구간에서 개선되고 있습니다")
    
    acc_for_trend = acc_epoch if acc_epoch else acc_chunk
    if acc_for_trend and len(acc_for_trend) >= 5:
        a0, a1 = last_k_trend(acc_for_trend, 5)
        if not math.isnan(a0) and not math.isnan(a1):
            if a1 <= a0:
                warnings.append("최근 구간에서 검증 정확도가 상승하지 않았습니다.")
            else:
                acc_improving = True
                successes.append("📈 검증 정확도가 최근 구간에서 상승하고 있습니다")
    
    # Overall learning progress
    if loss_improving and acc_improving:
        successes.append("🚀 학습이 순조롭게 진행되고 있습니다 (손실 감소 + 정확도 상승)")

    return warnings, successes


def save_png_charts(series: Dict[str, Series], tags: List[str], last: int, outdir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[정보] matplotlib가 설치되어 있지 않습니다. 설치: pip install matplotlib")
        return
    os.makedirs(outdir, exist_ok=True)
    for tag in tags:
        if tag not in series or not series[tag].values:
            continue
        vals = series[tag].values[-last:] if last > 0 else series[tag].values
        steps = list(range(len(vals)))
        plt.figure(figsize=(10, 4))
        plt.plot(steps, vals, label=tag)
        plt.title(tag)
        plt.xlabel('step')
        plt.ylabel('value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fname = os.path.join(outdir, f"{tag.replace('/', '_')}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        # print(f"[PNG] saved: {fname}")


def save_html_report(series: Dict[str, Series], tags: List[str], last: int, outfile: str) -> None:
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot as plot_html
    except Exception:
        print("[정보] plotly가 설치되어 있지 않습니다. 설치: pip install plotly")
        return
    figs = []
    for tag in tags:
        if tag not in series or not series[tag].values:
            continue
        vals = series[tag].values[-last:] if last > 0 else series[tag].values
        steps = list(range(len(vals)))
        figs.append(go.Scatter(x=steps, y=vals, name=tag, mode='lines'))
    if not figs:
        print("[정보] HTML 리포트로 렌더링할 데이터가 없습니다")
        return
    layout = dict(title='Training Metrics', xaxis_title='step', yaxis_title='value')
    fig = go.Figure(data=figs, layout=layout)
    plot_html(fig, filename=outfile, auto_open=False, include_plotlyjs='cdn')
    # print(f"[HTML] saved: {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Check TensorBoard event logs for training progress")
    parser.add_argument("--event-file", default=None, help="Path to a specific events.out.tfevents.* file")
    parser.add_argument("--logdir", default=None, help="Directory containing TensorBoard event files")
    parser.add_argument("--watch", action="store_true", help="Periodically check the log(s) until idle and then exit")
    parser.add_argument("--interval", type=int, default=15, help="Polling interval seconds in watch mode")
    parser.add_argument("--idle-seconds", type=int, default=120, help="Consider training finished if no update for this many seconds")
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration status in watch mode")
    # Output options
    parser.add_argument("--plot-tags", nargs='*', default=[
        "Loss/Train_Batch",
        "Loss/Train_Chunk",
        "Loss/Validation_Chunk",
        "Metrics/Val_Accuracy_Chunk",
        "Metrics/Val_Accuracy_Chunk_logits",
        "Metrics/Val_True_Class0_Count_Chunk",
        "Metrics/Val_True_Class1_Count_Chunk",
        "Metrics/Val_True_Class2_Count_Chunk",
        "Metrics/Val_Pred_Class0_Count_Chunk",
        "Metrics/Val_Pred_Class1_Count_Chunk",
        "Metrics/Val_Pred_Class2_Count_Chunk",
    ], help="Tags to include in outputs (stats/png/html)")
    parser.add_argument("--plot-last", type=int, default=200, help="Number of recent points to use (per tag)")
    parser.add_argument("--save-png", default=None, help="Directory to save PNG charts (requires matplotlib)")
    parser.add_argument("--out-html", default=None, help="Path to save interactive HTML report (requires plotly)")
    parser.add_argument("--stats-only", action="store_true", help="Only print numeric stats (no summarize)")
    args = parser.parse_args()

    if not args.watch:
        event_path = find_event_file(args.logdir, args.event_file)
        print(f"사용할 이벤트 파일: {event_path}")
        series = load_scalars(event_path)
        if not series:
            print("이벤트 파일에서 스칼라를 찾지 못했습니다.")
            return
        if not args.stats_only:
            summarize(series)
        print_stats(series, tags=args.plot_tags, last=args.plot_last)
        print_class_distribution(series, last=args.plot_last)
        # 자동 진단
        warnings, successes = analyze_training(series, last=args.plot_last)
        
        if successes:
            print("\n=== 🎉 학습 성공 지표 ===")
            for i, msg in enumerate(successes, 1):
                print(f"[{i}] {msg}")
        
        if warnings:
            print("\n=== ⚠️  주의사항 ===")
            for i, msg in enumerate(warnings, 1):
                print(f"[{i}] {msg}")
            print("⚠️  위 사항들을 검토해보세요. (계속 진행됩니다)")
        
        if not warnings and not successes:
            print("\n=== ℹ️  진단 정보 부족 ===")
            print("충분한 데이터가 없어 자동 진단을 수행할 수 없습니다.")
        if args.save_png:
            save_png_charts(series, tags=args.plot_tags, last=args.plot_last, outdir=args.save_png)
        if args.out_html:
            save_html_report(series, tags=args.plot_tags, last=args.plot_last, outfile=args.out_html)
        return

    # Watch mode
    if not args.logdir and not args.event_file:
        raise FileNotFoundError("--watch requires either --logdir or --event-file")

    last_event_path = None
    last_mtime = 0.0
    last_step = -1
    last_change_ts = time.time()
    update_index = 0

    try:
        while True:
            # Resolve event file each iteration if a logdir is given (handles new files)
            if args.logdir:
                event_path = newest_event_file_in(args.logdir)
                if not event_path:
                    if args.verbose:
                        print("[watch] 이벤트 파일이 아직 없습니다...")
                    time.sleep(max(1, args.interval))
                    continue
            else:
                event_path = args.event_file

            # Detect file switch
            if event_path != last_event_path:
                print(f"[watch] 사용할 이벤트 파일: {event_path}")
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
                    print(f"[watch] 재로딩 실패: {e}")

            current_step = max_step(series) if series else -1
            if args.verbose:
                print(f"[watch] 수정시각={current_mtime:.0f} 스텝={current_step}")

            # On update: summarize once and reset idle timer
            if current_mtime != last_mtime or current_step != last_step:
                # header to separate updates
                update_index += 1
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                delta_step = (current_step - last_step) if last_step >= 0 and current_step >= 0 else 0
                print("\n" + "=" * 20 + f" 업데이트 #{update_index} " + "=" * 20)
                print(f"[시간] {ts}")
                print(f"[이벤트] {event_path}")
                print(f"[스텝] 이전={last_step} -> 현재={current_step} (Δ={delta_step})")
                print("-" * 56)
                if series:
                    if not args.stats_only:
                        summarize(series)
                    print_stats(series, tags=args.plot_tags, last=args.plot_last)
                    print_class_distribution(series, last=args.plot_last)
                    # 워치 모드 자동 진단
                    warnings, successes = analyze_training(series, last=args.plot_last)
                    
                    if successes:
                        print("\n=== 🎉 학습 성공 지표 ===")
                        for i, msg in enumerate(successes, 1):
                            print(f"[{i}] {msg}")
                    
                    if warnings:
                        print("\n=== ⚠️  주의사항 ===")
                        for i, msg in enumerate(warnings, 1):
                            print(f"[{i}] {msg}")
                        print("⚠️  위 사항들을 검토해보세요. (모니터링 계속)")
                    
                    if not warnings and not successes:
                        print("\n=== ℹ️  진단 정보 부족 ===")
                        print("충분한 데이터가 없어 자동 진단을 수행할 수 없습니다.")
                    if args.save_png:
                        save_png_charts(series, tags=args.plot_tags, last=args.plot_last, outdir=args.save_png)
                    if args.out_html:
                        save_html_report(series, tags=args.plot_tags, last=args.plot_last, outfile=args.out_html)
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
