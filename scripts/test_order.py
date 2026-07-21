#!/usr/bin/env python3
"""
주식 1주 매수 후 즉시 매도 테스트 스크립트 (`scripts/test_order.py`)

사용법:
    1. 일반 KRX 거래:
       .venv64/Scripts/python scripts/test_order.py --code 005930 --qty 1

    2. SOR (Smart Order Routing / 최유리 스마트 라우팅) 거래:
       .venv64/Scripts/python scripts/test_order.py --code 005930 --qty 1 --exchange sor
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "koapys" / "kiwoom_rest_api" / "src"))

load_dotenv(project_root / ".env")

from koapys import KoapyRestSimple, OrderType, OrderBookType

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestOrder")


def run_order_test(stock_code: str = "005930", quantity: int = 1, exchange: str = "krx"):
    exchange_upper = exchange.upper()
    
    # 키움 API 거래소 구분 코드 (1: KRX, 2: NXT, 3: SOR/통합)
    if exchange_upper in ["SOR", "AL", "SOL", "INTEGRATED"]:
        stex_type = "3"
        exchange_name = "SOR / SOL (최유리 통합 스마트 주문, stex_tp=3)"
    elif exchange_upper in ["NXT", "NX"]:
        stex_type = "2"
        exchange_name = "NXT (Nextrade 대체거래소, stex_tp=2)"
    else:
        stex_type = "1"
        exchange_name = "KRX (한국거래소, stex_tp=1)"

    # 종목코드에 접미사가 붙어있다면 원본 6자리 추출
    clean_code = stock_code.split("_")[0]

    logger.info(f"키움 REST API 클라이언트 초기화 중... (거래 방식: {exchange_name})")
    client = KoapyRestSimple(simulation=False)
    
    try:
        client.ensure_connected()
        logger.info("✅ 키움 REST API 연결 성공!")
    except Exception as e:
        logger.error(f"❌ API 연결 실패: {e}")
        logger.error("키움증권 개발자센터에서 접속 IP/단말기 등록 및 API Key 상태를 확인해주세요.")
        return

    accounts = client.get_account_list()
    account_no = accounts[0] if accounts else None
    logger.info(f"사용 계좌번호: {account_no}")

    # 1. 1주 매수 주문 송신
    logger.info(f"🚀 [{clean_code}] {quantity}주 시장가 매수 주문 송신 중... ({exchange_name})")
    try:
        buy_res = client.send_order(
            rqname=f"TEST_BUY_{exchange_upper}",
            account_no=account_no,
            order_type=OrderType.BUY,
            code=clean_code,
            quantity=quantity,
            price=0,
            hoga=OrderBookType.MARKET
        )
        logger.info(f"✅ 매수 주문 응답: {buy_res}")
    except Exception as e:
        logger.error(f"❌ 매수 주문 실패: {e}")
        return

    # 체결 대기 (3초)
    logger.info("3초간 체결 대기 중...")
    time.sleep(3)

    # 2. 1주 매도 주문 송신
    logger.info(f"🔥 [{clean_code}] {quantity}주 시장가 매도 주문 송신 중... ({exchange_name})")
    try:
        sell_res = client.send_order(
            rqname=f"TEST_SELL_{exchange_upper}",
            account_no=account_no,
            order_type=OrderType.SELL,
            code=clean_code,
            quantity=quantity,
            price=0,
            hoga=OrderBookType.MARKET
        )
        logger.info(f"✅ 매도 주문 응답: {sell_res}")
        logger.info(f"🎉 [{clean_code}] 1주 매수 후 매도 테스트 프로세스가 완료되었습니다! ({exchange_name})")
    except Exception as e:
        logger.error(f"❌ 매도 주문 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1주 매수 후 매도 테스트 스크립트 (KRX / SOR 지원)")
    parser.add_argument("--code", type=str, default="005930", help="종목코드 (기본: 005930)")
    parser.add_argument("--qty", type=int, default=1, help="주문 수량 (기본: 1주)")
    parser.add_argument("--exchange", "-e", type=str, default="krx", help="거래 방식 (krx, sor/sol, nxt)")
    args = parser.parse_args()

    run_order_test(stock_code=args.code, quantity=args.qty, exchange=args.exchange)
