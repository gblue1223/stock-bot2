#!/usr/bin/env python3
"""
실시간 프로그램 매매 10분할 자동매매 실행 스크립트

사용 예시:
    1. 드라이런 (모의 실행):
       python -m algo_trader.program.main --stocks "005930,000660" --dry-run
       
    2. NXT (Nextrade 대체거래소) 10분할 트레이딩:
       python -m algo_trader.program.main --stocks "005930,000660" --exchange nxt --budget 2000000

    3. SOR (최유리 스마트 라우팅) 10분할 트레이딩:
       python -m algo_trader.program.main --stocks "005930,000660" --exchange sor --budget 2000000
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가 및 .env 우선 로드
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "koapys" / "kiwoom_rest_api" / "src"))

env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from algo_trader.program.config import ProgramTradingConfig
from algo_trader.program.trader import RealtimeProgramTrader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AlgoTraderProgramMain")


def main():
    parser = argparse.ArgumentParser(description="실시간 프로그램 매매 10분할 트레이딩 실행기")
    parser.add_argument(
        "--stocks", "-s",
        type=str,
        default="005930,000660",
        help="대상 종목코드 목록 (쉼표로 구분, 예: 005930,000660)"
    )
    parser.add_argument(
        "--exchange", "-e",
        type=str,
        default="krx",
        help="거래소 구분 (krx: 한국거래소, nxt: Nextrade 대체거래소, sor/al: 최유리 스마트 라우팅)"
    )
    parser.add_argument(
        "--budget", "-b",
        type=float,
        default=1_000_000.0,
        help="종목당 총 투자 예산 (원, 기본값: 1,000,000원)"
    )
    parser.add_argument(
        "--splits", "-n",
        type=int,
        default=10,
        help="분할 매매 횟수 (기본값: 10분할)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=300,
        help="분할 매수 실행 최소 간격 (초 단위, 기본값: 300초)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=2.0,
        help="손절 비율 (%%, 기본값: 2.0)"
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=3.0,
        help="익절 비율 (%%, 기본값: 3.0)"
    )
    parser.add_argument(
        "--close-time",
        type=str,
        default="15:15:00",
        help="장 마감 전 자동 청산 시각 (HH:MM:SS, 기본값: 15:15:00)"
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=100000.0,
        help="프로그램 매도 시그널 이탈 감도 기준 (백만원 단위, 기본값: 100,000백만원 = -1,000억원)"
    )
    parser.add_argument(
        "--account", "-a",
        type=str,
        help="계좌번호 (선택사항, 입력하지 않으면 첫번째 계좌 자동 선택)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="드라이런 모드 (실제 주문을 내지 않고 로그로만 확인)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="최대 사이클 실행 횟수 (테스트용, 기본값: 무한)"
    )

    args = parser.parse_args()

    stock_list = [code.strip() for code in args.stocks.split(",") if code.strip()]
    if not stock_list:
        logger.error("대상 종목 코드가 지정되지 않았습니다.")
        sys.exit(1)

    # 거래소 코드 맵핑 (krx -> KRX, nxt -> NXT, sor -> SOR)
    ex_upper = args.exchange.upper()
    if ex_upper in ["NXT", "NX"]:
        stex_code = "NXT"
    elif ex_upper in ["SOR", "AL", "SOL", "INTEGRATED"]:
        stex_code = "SOR"
    else:
        stex_code = "KRX"

    config = ProgramTradingConfig(
        target_stocks=stock_list,
        split_count=args.splits,
        total_budget_per_stock=args.budget,
        interval_seconds=args.interval,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        market_close_time=args.close_time,
        account_no=args.account,
        dry_run=args.dry_run,
        stock_exchange_type=stex_code,
        sell_net_buy_trend_threshold=args.sell_threshold
    )

    trader = RealtimeProgramTrader(config=config)
    trader.start_loop(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
