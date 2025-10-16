#!/usr/bin/env python3
"""
빠른 GRPO 검증/백테스트 스크립트

- 훈련이 끝난 후 정책을 홀드아웃 데이터로 빠르게 검증합니다.
- 내부 `ai_trader.grpo.backtest.run_backtest` 유틸리티를 사용합니다.
- 정책 체크포인트 경로를 지정하지 않으면, 기본 체크포인트 디렉토리에서 최신 파일을 자동 선택합니다.

사용 예시:
    python -m scripts.grpo.quick_validation \
        --embedding-model C:\\path\\to\\autoencoder\\model.pt \
        --db C:\\path\\to\\datasets_norm_all.duckdb \
        --policy-dir models/grpo_scalping/checkpoints \
        --test-start 2024-12-01 \
        --test-end 2024-12-31 \
        --out results/quick_backtest.json \
        --device cuda
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List
import torch
import sys

# 프로젝트 루트를 Python 경로에 추가하여 스크립트 직접 실행을 지원
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ai_trader.grpo.backtest import run_backtest
from ai_trader.grpo.evaluation import GRPOEvaluationMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Quick validation (backtest) for trained GRPO policy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 필수
    # quick_train_grpo.py와 동일한 기본 경로를 제공하여 무인 실행 가능
    parser.add_argument('--embedding-model', type=str,
                        default=r"C:\\Users\\user\\Workspace\\datasets@20251005\\autoencoder\\20251012_013148\\model.pt",
                        help='임베딩(오토인코더) 모델 체크포인트 경로 (.pt)')
    parser.add_argument('--db', type=str,
                        default=r"C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb",
                        help='DuckDB 데이터베이스 경로')

    # 정책 경로 (택1): 개별 파일 또는 디렉토리 내 최신 파일 자동 선택
    parser.add_argument('--policy-path', type=str, default=None,
                        help='정책(Policy) 체크포인트 경로 (.pt)')
    parser.add_argument('--policy-dir', type=str, default='models/grpo_scalping/checkpoints',
                        help='정책 체크포인트 디렉토리 (최신 파일 자동 선택)')

    # 백테스트 파라미터
    parser.add_argument('--table', type=str, default='datasets', help='테이블명')
    parser.add_argument('--seq-len', type=int, default=60, help='시퀀스 길이')
    parser.add_argument('--test-start', type=str, default='2025-09-01', help='테스트 시작 날짜 YYYY-MM-DD')
    parser.add_argument('--test-end', type=str, default='2025-09-30', help='테스트 종료 날짜 YYYY-MM-DD')
    parser.add_argument('--stocks', type=str, nargs='*', default=None, help='테스트할 종목코드 리스트')

    # 비용/슬리피지/룰
    parser.add_argument('--transaction-cost-rate', type=float, default=0.00215, help='거래 비용 비율 (0.215%)')
    parser.add_argument('--slippage-rate', type=float, default=0.0001, help='슬리피지 비율 (0.01%)')
    parser.add_argument('--quick-exit-threshold', type=float, default=1.5, help='빠른 손절 임계값 (초)')
    parser.add_argument('--quick-exit-penalty', type=float, default=0.01, help='빠른 손절 페널티')

    # 실행/출력
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None, help='디바이스 (미지정 시 자동 감지)')
    parser.add_argument('--out', type=str, default=None, help='결과 저장 경로 (.json)')
    parser.add_argument('--verbose', action='store_true', help='상세 출력')
    parser.add_argument('--print-eval', action='store_true', help='evaluator로 상세 메트릭 표 출력')

    # 알림 시스템 옵션 (선택)
    parser.add_argument('--disable-alerts', action='store_true', help='성능 알림 비활성화')

    return parser.parse_args()


def _pick_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """디렉토리에서 가장 최근 수정된 .pt 체크포인트를 반환"""
    p = Path(checkpoint_dir)
    if not p.exists() or not p.is_dir():
        logger.warning(f"Checkpoint directory not found or not a directory: {checkpoint_dir}")
        return None

    candidates: List[Path] = sorted(p.glob('*.pt'), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        logger.warning(f"No .pt files found in checkpoint directory: {checkpoint_dir}")
        return None

    latest = str(candidates[0])
    logger.info(f"Selected latest checkpoint: {latest}")
    return latest


def main() -> int:
    args = parse_args()

    # 경로 체크
    if not os.path.exists(args.embedding_model):
        logger.error(f"Embedding model not found: {args.embedding_model}")
        return 1
    if not os.path.exists(args.db):
        logger.error(f"Database not found: {args.db}")
        return 1

    # 디바이스 자동 감지 (미지정 시)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 정책 경로 결정
    policy_path = args.policy_path
    if policy_path is None:
        policy_path = _pick_latest_checkpoint(args.policy_dir)
        if policy_path is None:
            logger.error("Cannot determine policy checkpoint. Provide --policy-path or ensure --policy-dir has .pt files.")
            return 1

    logger.info("Starting quick validation (backtest)...")
    logger.info(f"Embedding: {args.embedding_model}")
    logger.info(f"Policy:    {policy_path}")
    logger.info(f"DB:        {args.db}")
    logger.info(f"Device:    {device}")
    if args.test_start or args.test_end:
        logger.info(f"Test period: {args.test_start or 'BEGIN'} ~ {args.test_end or 'END'}")

    try:
        results = run_backtest(
            embedding_model_path=args.embedding_model,
            policy_path=policy_path,
            db_path=args.db,
            table_name=args.table,
            test_start_date=args.test_start,
            test_end_date=args.test_end,
            stock_codes=args.stocks,
            seq_len=args.seq_len,
            transaction_cost_rate=args.transaction_cost_rate,
            slippage_rate=args.slippage_rate,
            quick_exit_threshold=args.quick_exit_threshold,
            quick_exit_penalty=args.quick_exit_penalty,
            device=device,
            output_path=args.out,
            verbose=args.verbose,
            enable_alerts=not args.disable_alerts,
        )

        # 결과 저장 (run_backtest 내부에서 저장을 처리하지 않은 경우에 대비)
        if args.out:
            from ai_trader.grpo.backtest import BacktestSimulator  # for save utility if needed
            # 간단 저장: JSON 직렬화 가능한 형태로 저장
            import json
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {out_path}")

        # 간단 요약 출력
        metrics = results.get('metrics', {})
        summary = results.get('summary', {})
        logger.info("Validation Summary:")
        logger.info(f"  Total Episodes: {summary.get('num_episodes', 0)}")
        logger.info(f"  Total Trades:   {summary.get('total_trades', 0)}")
        logger.info(f"  Win Rate:       {metrics.get('win_rate', 0.0):.2%}")
        logger.info(f"  Sharpe:         {metrics.get('sharpe_ratio', 0.0):.4f}")
        logger.info(f"  Max Drawdown:   {metrics.get('max_drawdown', 0.0):.2%}")
        logger.info(f"  Avg Profit/tr:  {metrics.get('avg_profit_per_trade', 0.0):.4f}")

        # evaluator를 사용해 에피소드로부터 메트릭 재계산/출력 (선택)
        if args.print_eval:
            episodes = results.get('episodes', [])
            evaluator = GRPOEvaluationMetrics(episodes)
            eval_metrics = evaluator.compute_metrics()
            # 상세 표 출력
            evaluator.print_metrics(eval_metrics)

        return 0
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
