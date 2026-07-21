#!/usr/bin/env python3
"""
주식 1주 매수 후 즉시 매도 테스트 스크립트 (`scripts/test_order.py`)

사용법:
    1. 일반 KRX 거래 (시장가):
       .venv64/Scripts/python scripts/test_order.py --code 005930 --qty 1

    2. NXT (Nextrade 대체거래소 애프터마켓 15:30~20:00 - 지정가 필수):
       .venv64/Scripts/python scripts/test_order.py --code 005930 --qty 1 --exchange nxt --price 263000

    3. SOR (Smart Order Routing 최유리 주문):
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
from kiwoom_rest_api.koreanstock.stockinfo import StockInfo

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestOrder")


def run_order_test(stock_code: str = "005930", quantity: int = 1, exchange: str = "krx", price: int = 0):
    exchange_upper = exchange.upper()
    
    # 키움 API 거래소 구분 코드 (KRX, NXT, SOR)
    if exchange_upper in ["NXT", "NX"]:
        stex_code = "NXT"
        exchange_name = "NXT (Nextrade 대체거래소 애프터마켓, dmst_stex_tp='NXT')"
        is_nxt = True
    elif exchange_upper in ["SOR", "AL", "SOL", "INTEGRATED"]:
        stex_code = "SOR"
        exchange_name = "SOR / SOL (최유리 통합 스마트 주문, dmst_stex_tp='SOR')"
        is_nxt = False
    else:
        stex_code = "KRX"
        exchange_name = "KRX (한국거래소, dmst_stex_tp='KRX')"
        is_nxt = False

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

    # 호가 및 가격 설정
    target_price = price
    if price > 0:
        hoga_type = OrderBookType.LIMIT
        order_desc = f"지정가 {target_price:,.0f}원"
    else:
        if is_nxt:
            try:
                stock_info_api = StockInfo(base_url=client._base_url, token_manager=client._token_manager)
                info_res = stock_info_api.basic_stock_information_request_ka10001(clean_code)
                cur_prc = abs(int(info_res.get("stk_prc") or info_res.get("cur_prc") or 55000))
                target_price = cur_prc
                hoga_type = OrderBookType.LIMIT
                logger.info(f"💡 NXT 애프터마켓 지정을 위해 현재가({cur_prc:,.0f}원)로 지정가 자동 설정")
                order_desc = f"지정가 {target_price:,.0f}원"
            except Exception:
                hoga_type = OrderBookType.LIMIT
                target_price = 55000
                order_desc = f"지정가 {target_price:,.0f}원"
        else:
            hoga_type = OrderBookType.MARKET
            target_price = 0
            order_desc = "시장가"

    # 1. 1주 매수 주문 송신 (dmst_stex_tp 파라미터 전달)
    logger.info(f"🚀 [{clean_code}] {quantity}주 {order_desc} 매수 주문 송신 중... ({exchange_name})")
    try:
        buy_res = client.send_order(
            rqname=f"TEST_BUY_{exchange_upper}",
            account_no=account_no,
            order_type=OrderType.BUY,
            code=clean_code,
            quantity=quantity,
            price=target_price,
            hoga=hoga_type,
            dmst_stex_tp=stex_code
        )
        logger.info(f"✅ 매수 주문 응답: {buy_res}")
    except Exception as e:
        logger.error(f"❌ 매수 주문 실패: {e}")
        return

    # 체결 대기 (3초)
    logger.info("3초간 체결 대기 중...")
    time.sleep(3)

    # 2. 1주 매도 주문 송신 (dmst_stex_tp 파라미터 전달)
    logger.info(f"🔥 [{clean_code}] {quantity}주 {order_desc} 매도 주문 송신 중... ({exchange_name})")
    try:
        sell_res = client.send_order(
            rqname=f"TEST_SELL_{exchange_upper}",
            account_no=account_no,
            order_type=OrderType.SELL,
            code=clean_code,
            quantity=quantity,
            price=target_price,
            hoga=hoga_type,
            dmst_stex_tp=stex_code
        )
        logger.info(f"✅ 매도 주문 응답: {sell_res}")
        logger.info(f"🎉 [{clean_code}] 1주 매수 후 매도 테스트 프로세스가 완료되었습니다! ({exchange_name})")
    except Exception as e:
        logger.error(f"❌ 매도 주문 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1주 매수 후 매도 테스트 스크립트 (KRX / NXT / SOR 지원)")
    parser.add_argument("--code", type=str, default="005930", help="종목코드 (기본: 005930)")
    parser.add_argument("--qty", type=int, default=1, help="주문 수량 (기본: 1주)")
    parser.add_argument("--exchange", "-e", type=str, default="krx", help="거래 방식 (krx, nxt, sor/sol)")
    parser.add_argument("--price", "-p", type=int, default=0, help="지정가 가격 (0일 경우 시장가, NXT의 경우 현재가 자동 지정)")
    args = parser.parse_args()

    run_order_test(stock_code=args.code, quantity=args.qty, exchange=args.exchange, price=args.price)
