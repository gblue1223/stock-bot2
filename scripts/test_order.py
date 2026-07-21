#!/usr/bin/env python3
"""
삼성전자 1주 매수 후 즉시 매도 테스트 스크립트 (`scripts/test_order.py`)

사용법:
    .venv64/Scripts/python scripts/test_order.py --code 005930 --qty 1
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


def run_order_test(stock_code: str = "005930", quantity: int = 1):
    logger.info("키움 REST API 클라이언트 초기화 중...")
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
    logger.info(f"🚀 [{stock_code}] {quantity}주 시장가 매수 주문 송신 중...")
    try:
        buy_res = client.send_order(
            rqname="TEST_BUY_1",
            account_no=account_no,
            order_type=OrderType.BUY,
            code=stock_code,
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
    logger.info(f"🔥 [{stock_code}] {quantity}주 시장가 매도 주문 송신 중...")
    try:
        sell_res = client.send_order(
            rqname="TEST_SELL_1",
            account_no=account_no,
            order_type=OrderType.SELL,
            code=stock_code,
            quantity=quantity,
            price=0,
            hoga=OrderBookType.MARKET
        )
        logger.info(f"✅ 매도 주문 응답: {sell_res}")
        logger.info("🎉 1주 매수 후 매도 테스트 프로세스가 완료되었습니다!")
    except Exception as e:
        logger.error(f"❌ 매도 주문 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1주 매수 후 매도 테스트 스크립트")
    parser.add_argument("--code", type=str, default="005930", help="종목코드 (기본: 005930)")
    parser.add_argument("--qty", type=int, default=1, help="주문 수량 (기본: 1주)")
    args = parser.parse_args()

    run_order_test(stock_code=args.code, quantity=args.qty)
