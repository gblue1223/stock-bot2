#!/usr/bin/env python3
"""
Koapys API 연결 테스트 스크립트

실시간 데이터 수집 및 특징 추출 테스트
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from koapys.client import KoapyRestSimple
from koapys.types import OrderType, OrderBookType

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_connection():
    """Koapys 연결 테스트"""
    logger.info("=" * 80)
    logger.info("[TEST] Koapys Connection")
    logger.info("=" * 80)
    
    try:
        # 클라이언트 초기화
        koapys = KoapyRestSimple(
            base_url=os.getenv('KOAPYS_BASE_URL', 'http://localhost:5000'),
            api_key=os.getenv('KOAPYS_API_KEY'),
            simulation=True
        )
        
        # 연결 확인
        koapys.ensure_connected()
        logger.info("[OK] Connection successful")
        
        # 계좌 목록
        accounts = koapys.get_account_list()
        logger.info(f"[OK] Accounts: {accounts}")
        
        koapys.close()
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Connection failed: {e}")
        return False


def test_market_data(stock_code: str = '005930'):
    """시장 데이터 조회 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info(f"[TEST] Market Data ({stock_code})")
    logger.info("=" * 80)
    
    try:
        koapys = KoapyRestSimple(
            base_url=os.getenv('KOAPYS_BASE_URL', 'http://localhost:5000'),
            simulation=True
        )
        
        # 종목 기본 정보
        logger.info("\n[1] Stock Basic Info:")
        basic_info = koapys.get_stock_basic_info(stock_code)
        logger.info(f"  Data: {basic_info}")
        
        # 현재 가격
        logger.info("\n[2] Current Price:")
        current_price = koapys.get_current_price(stock_code)
        logger.info(f"  Price: {current_price:,}원")
        
        # 상한가
        logger.info("\n[3] Upper Limit Price:")
        upper_limit = koapys.get_upper_limit_price(stock_code)
        logger.info(f"  Upper Limit: {upper_limit:,}원")
        
        koapys.close()
        return True, basic_info
        
    except Exception as e:
        logger.error(f"[ERROR] Market data failed: {e}")
        return False, None


def test_feature_extraction(market_data: dict):
    """특징 추출 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("[TEST] Feature Extraction")
    logger.info("=" * 80)
    
    try:
        features = []
        
        # 1. 등락률
        current_price = float(market_data.get('현재가', 0))
        base_price = float(market_data.get('기준가', current_price))
        if base_price > 0:
            change_rate = (current_price - base_price) / base_price * 100
        else:
            change_rate = 0.0
        features.append(change_rate)
        logger.info(f"[1] 등락률: {change_rate:.2f}%")
        
        # 2. 누적거래대금
        volume_amount = float(market_data.get('누적거래대금', 0))
        features.append(volume_amount / 1e9)
        logger.info(f"[2] 누적거래대금: {volume_amount/1e9:.2f}억원")
        
        # 3. 거래회전율
        volume = float(market_data.get('거래량', 0))
        listed_shares = float(market_data.get('상장주식수', 1))
        turnover_rate = (volume / listed_shares * 100) if listed_shares > 0 else 0.0
        features.append(turnover_rate)
        logger.info(f"[3] 거래회전율: {turnover_rate:.4f}%")
        
        # 4. 체결강도
        buy_volume = float(market_data.get('매수체결량', 0))
        sell_volume = float(market_data.get('매도체결량', 1))
        strength = (buy_volume / sell_volume * 100) if sell_volume > 0 else 100.0
        features.append(strength)
        logger.info(f"[4] 체결강도: {strength:.2f}")
        
        # 5-14. 매도대기금액
        logger.info(f"\n[5-14] 매도대기금액:")
        for i in range(1, 11):
            sell_price = float(market_data.get(f'매도호가{i}', 0))
            sell_qty = float(market_data.get(f'매도호가수량{i}', 0))
            sell_amount = sell_price * sell_qty / 1e6
            features.append(sell_amount)
            if i <= 3:  # 처음 3개만 출력
                logger.info(f"  호가{i}: {sell_amount:.2f}백만원")
        
        # 15-24. 매수대기금액
        logger.info(f"\n[15-24] 매수대기금액:")
        for i in range(1, 11):
            buy_price = float(market_data.get(f'매수호가{i}', 0))
            buy_qty = float(market_data.get(f'매수호가수량{i}', 0))
            buy_amount = buy_price * buy_qty / 1e6
            features.append(buy_amount)
            if i <= 3:  # 처음 3개만 출력
                logger.info(f"  호가{i}: {buy_amount:.2f}백만원")
        
        # numpy array로 변환
        features_array = np.array(features, dtype=np.float32)
        
        logger.info(f"\n[RESULT] Feature Vector:")
        logger.info(f"  Shape: {features_array.shape}")
        logger.info(f"  Dtype: {features_array.dtype}")
        logger.info(f"  Range: [{features_array.min():.2f}, {features_array.max():.2f}]")
        logger.info(f"  Mean: {features_array.mean():.2f}")
        logger.info(f"  Std: {features_array.std():.2f}")
        
        # NaN 체크
        if np.isnan(features_array).any():
            logger.warning(f"[WARNING] NaN values detected!")
            nan_indices = np.where(np.isnan(features_array))[0]
            logger.warning(f"  NaN indices: {nan_indices}")
        else:
            logger.info(f"[OK] No NaN values")
        
        return True, features_array
        
    except Exception as e:
        logger.error(f"[ERROR] Feature extraction failed: {e}", exc_info=True)
        return False, None


def test_order_simulation(stock_code: str = '005930'):
    """주문 시뮬레이션 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info(f"[TEST] Order Simulation ({stock_code})")
    logger.info("=" * 80)
    
    try:
        koapys = KoapyRestSimple(
            base_url=os.getenv('KOAPYS_BASE_URL', 'http://localhost:5000'),
            simulation=True
        )
        
        # 계좌 정보
        accounts = koapys.get_account_list()
        account_no = accounts[0] if accounts else '0000000000'
        
        # 현재 가격
        current_price = koapys.get_current_price(stock_code)
        
        # 매수 주문 (시뮬레이션)
        logger.info(f"\n[1] Buy Order:")
        logger.info(f"  Stock: {stock_code}")
        logger.info(f"  Price: {current_price:,}원")
        logger.info(f"  Quantity: 10주")
        
        buy_result = koapys.send_order(
            rqname=f"TEST_BUY_{datetime.now().strftime('%H%M%S')}",
            account_no=account_no,
            order_type=OrderType.BUY,
            code=stock_code,
            quantity=10,
            price=current_price,
            hoga=OrderBookType.MARKET_IOC
        )
        logger.info(f"  Result: {buy_result}")
        
        # 매도 주문 (시뮬레이션)
        logger.info(f"\n[2] Sell Order:")
        sell_result = koapys.send_order(
            rqname=f"TEST_SELL_{datetime.now().strftime('%H%M%S')}",
            account_no=account_no,
            order_type=OrderType.SELL,
            code=stock_code,
            quantity=10,
            price=current_price,
            hoga=OrderBookType.MARKET_IOC
        )
        logger.info(f"  Result: {sell_result}")
        
        koapys.close()
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Order simulation failed: {e}")
        return False


def main():
    """메인 함수"""
    logger.info("=" * 80)
    logger.info("[START] Koapys API Test Suite")
    logger.info("=" * 80)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Base URL: {os.getenv('KOAPYS_BASE_URL', 'http://localhost:5000')}")
    logger.info(f"Simulation: True")
    logger.info("=" * 80)
    
    results = {}
    
    # 1. 연결 테스트
    results['connection'] = test_connection()
    
    # 2. 시장 데이터 테스트
    results['market_data'], market_data = test_market_data('005930')
    
    # 3. 특징 추출 테스트
    if results['market_data'] and market_data:
        results['feature_extraction'], features = test_feature_extraction(market_data)
    else:
        results['feature_extraction'] = False
        logger.warning("[SKIP] Feature extraction test (no market data)")
    
    # 4. 주문 시뮬레이션 테스트
    results['order_simulation'] = test_order_simulation('005930')
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("[SUMMARY] Test Results")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        logger.info(f"{status} {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n[SUCCESS] All tests passed!")
        logger.info("=" * 80)
        logger.info("Next Steps:")
        logger.info("1. Run live trading in simulation mode")
        logger.info("2. Monitor for 1 week")
        logger.info("3. Start real trading with small amount")
        logger.info("=" * 80)
    else:
        logger.error("\n[FAILURE] Some tests failed!")
        logger.error("Please fix the issues before proceeding.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
