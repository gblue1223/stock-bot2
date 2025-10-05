"""
GRPO 백테스팅 예제

이 스크립트는 홀드아웃 테스트 데이터에서 GRPO 에이전트의 백테스팅을 실행하는 예제입니다.
현실적인 거래 비용 및 슬리피지를 적용하여 성능을 평가합니다.
"""

import argparse
import logging
from pathlib import Path

from ai_trader.grpo.backtest import run_backtest

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='GRPO 백테스팅 예제')
    
    # 모델 경로
    parser.add_argument(
        '--embedding-model',
        type=str,
        required=True,
        help='임베딩 모델 체크포인트 경로'
    )
    parser.add_argument(
        '--policy',
        type=str,
        required=True,
        help='정책 체크포인트 경로'
    )
    
    # 데이터베이스 설정
    parser.add_argument(
        '--db',
        type=str,
        default=r'C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb',
        help='DuckDB 데이터베이스 경로'
    )
    parser.add_argument(
        '--table',
        type=str,
        default='datasets',
        help='테이블명'
    )
    
    # 테스트 기간 설정
    parser.add_argument(
        '--test-start-date',
        type=str,
        default=None,
        help='테스트 시작 날짜 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--test-end-date',
        type=str,
        default=None,
        help='테스트 종료 날짜 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--stock-codes',
        type=str,
        nargs='+',
        default=None,
        help='테스트할 종목 코드 리스트'
    )
    
    # 모델 설정
    parser.add_argument(
        '--seq-len',
        type=int,
        default=60,
        help='시퀀스 길이'
    )
    
    # 거래 비용 설정
    parser.add_argument(
        '--transaction-cost',
        type=float,
        default=0.00215,
        help='거래 비용 비율 (기본값: 0.00215 = 0.215%%)'
    )
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.0001,
        help='슬리피지 비율 (기본값: 0.0001 = 0.01%%)'
    )
    
    # 빠른 손절 룰 설정
    parser.add_argument(
        '--quick-exit-threshold',
        type=float,
        default=1.5,
        help='빠른 손절 시간 임계값 (초)'
    )
    parser.add_argument(
        '--quick-exit-penalty',
        type=float,
        default=0.01,
        help='빠른 손절 룰 위반 페널티'
    )
    
    # 디바이스 설정
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='디바이스 (cuda 또는 cpu)'
    )
    
    # 출력 설정
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='결과 저장 경로 (.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='진행 상황 출력'
    )
    
    args = parser.parse_args()
    
    # 백테스팅 실행
    logger.info("Starting backtest...")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Policy: {args.policy}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Test period: {args.test_start_date} ~ {args.test_end_date}")
    logger.info(f"Transaction cost: {args.transaction_cost*100:.3f}%")
    logger.info(f"Slippage: {args.slippage*100:.3f}%")
    
    results = run_backtest(
        embedding_model_path=args.embedding_model,
        policy_path=args.policy,
        db_path=args.db,
        table_name=args.table,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        stock_codes=args.stock_codes,
        seq_len=args.seq_len,
        transaction_cost_rate=args.transaction_cost,
        slippage_rate=args.slippage,
        quick_exit_threshold=args.quick_exit_threshold,
        quick_exit_penalty=args.quick_exit_penalty,
        device=args.device,
        output_path=args.output,
        verbose=args.verbose
    )
    
    logger.info("Backtest completed successfully")
    
    # 결과 요약 출력
    summary = results['summary']
    metrics = results['metrics']
    
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    print(f"Episodes:                {summary['num_episodes']}")
    print(f"Stocks:                  {summary['num_stocks']}")
    print(f"Days:                    {summary['num_days']}")
    print(f"Total Trades:            {summary['total_trades']}")
    print(f"Win Rate:                {metrics['win_rate']:.2%}")
    print(f"Avg Profit per Trade:    {metrics['avg_profit_per_trade']:.4f}")
    print(f"Max Drawdown:            {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio:            {metrics['sharpe_ratio']:.4f}")
    print(f"Avg Holding Time:        {metrics['avg_holding_time']:.2f}s")
    print("=" * 70 + "\n")
    
    # 성능 기준 검증 (요구사항 7.2, 7.3)
    print("Performance Criteria Check:")
    print("-" * 70)
    
    criteria_met = True
    
    # 승률 > 50%
    if metrics['win_rate'] > 0.50:
        print(f"✓ Win Rate > 50%: {metrics['win_rate']:.2%}")
    else:
        print(f"✗ Win Rate > 50%: {metrics['win_rate']:.2%}")
        criteria_met = False
    
    # 샤프 비율 > 1.0
    if metrics['sharpe_ratio'] > 1.0:
        print(f"✓ Sharpe Ratio > 1.0: {metrics['sharpe_ratio']:.4f}")
    else:
        print(f"✗ Sharpe Ratio > 1.0: {metrics['sharpe_ratio']:.4f}")
        criteria_met = False
    
    # 최대 낙폭 < 10%
    if metrics['max_drawdown'] < 0.10:
        print(f"✓ Max Drawdown < 10%: {metrics['max_drawdown']:.2%}")
    else:
        print(f"✗ Max Drawdown < 10%: {metrics['max_drawdown']:.2%}")
        criteria_met = False
    
    # 평균 보유 시간 < 30초
    if metrics['avg_holding_time'] < 30:
        print(f"✓ Avg Holding Time < 30s: {metrics['avg_holding_time']:.2f}s")
    else:
        print(f"✗ Avg Holding Time < 30s: {metrics['avg_holding_time']:.2f}s")
        criteria_met = False
    
    print("-" * 70)
    
    if criteria_met:
        print("✓ All performance criteria met!")
    else:
        print("✗ Some performance criteria not met")
    
    print()


if __name__ == '__main__':
    main()
