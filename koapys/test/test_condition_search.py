"""
조건검색 기능 테스트

키움 REST API의 ka10172 (조건식 목록), ka10173 (조건검색) 테스트
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from koapys.client import KoapyRestSimple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_condition_search():
    """조건검색 기능 테스트"""
    
    logger.info("=== 조건검색 테스트 시작 ===")
    
    # 클라이언트 초기화
    client = KoapyRestSimple(simulation=False)
    
    try:
        # 1. 연결 확인
        logger.info("연결 확인 중...")
        client.ensure_connected()
        logger.info("✓ 연결 성공")
        
        # 2. 조건식 목록 로드
        logger.info("\n조건식 목록 로드 중...")
        if client.get_condition_load():
            logger.info("✓ 조건식 목록 로드 성공")
        else:
            logger.warning("⚠ 조건식 목록 로드 실패")
            return
        
        # 3. 조건식 목록 조회
        logger.info("\n조건식 목록 조회 중...")
        conditions = client.get_condition_name_list()
        
        if not conditions:
            logger.warning("⚠ 등록된 조건식이 없습니다.")
            logger.info("HTS에서 조건검색식을 먼저 등록해주세요.")
            return
        
        logger.info(f"✓ {len(conditions)}개의 조건식 발견:")
        for idx, (cond_idx, cond_name) in enumerate(conditions, 1):
            logger.info(f"  {idx}. [{cond_idx}] {cond_name}")
        
        # 4. 첫 번째 조건식으로 검색 실행
        if conditions:
            condition_index, condition_name = conditions[0]
            logger.info(f"\n조건검색 실행: '{condition_name}' (인덱스: {condition_index})")
            
            stocks = client.get_stocks_by_condition(condition_name)
            
            if stocks:
                logger.info(f"✓ {len(stocks)}개 종목 발견:")
                for i, stock_code in enumerate(stocks[:10], 1):  # 최대 10개만 표시
                    logger.info(f"  {i}. {stock_code}")
                
                if len(stocks) > 10:
                    logger.info(f"  ... 외 {len(stocks) - 10}개")
            else:
                logger.info("조건에 맞는 종목이 없습니다.")
        
        logger.info("\n=== 조건검색 테스트 완료 ===")
        
    except Exception as e:
        logger.error(f"❌ 테스트 중 오류 발생: {e}", exc_info=True)
        return False
    
    return True

def test_condition_search_with_listener():
    """비동기 리스너를 사용한 조건검색 테스트"""
    
    logger.info("\n=== 비동기 조건검색 테스트 시작 ===")
    
    client = KoapyRestSimple(simulation=False)
    
    try:
        client.ensure_connected()
        
        # 조건식 목록 가져오기
        conditions = client.get_condition_name_list()
        
        if not conditions:
            logger.warning("등록된 조건식이 없습니다.")
            return
        
        condition_name = conditions[0][1]
        logger.info(f"비동기 조건검색 실행: '{condition_name}'")
        
        # 리스너 정의
        def on_stocks_received(stocks):
            logger.info(f"✓ 비동기 결과 수신: {len(stocks)}개 종목")
            for i, stock_code in enumerate(stocks[:5], 1):
                logger.info(f"  {i}. {stock_code}")
        
        # 비동기 실행
        client.get_stocks_by_condition(condition_name, listener=on_stocks_received)
        
        # 비동기 처리 대기
        import time
        time.sleep(2)
        
        logger.info("=== 비동기 조건검색 테스트 완료 ===")
        
    except Exception as e:
        logger.error(f"❌ 비동기 테스트 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    # 기본 조건검색 테스트
    test_condition_search()
    
    # 비동기 조건검색 테스트
    # test_condition_search_with_listener()
