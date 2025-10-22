#!/usr/bin/env python3
"""
주식 실시간 데이터 수신 예제

이 예제는 주식체결(0B) 및 주식호가(0C) 실시간 데이터를 수신하는 방법을 보여줍니다.
"""

import asyncio
import os
import sys
import logging
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from koapys import KoapyRestSimple

# KoapyRestSimple을 통해 액세스 토큰 가져오기
try:
    koapy_client = KoapyRestSimple()
    ACCESS_TOKEN = koapy_client.access_token
    print(f"✓ 액세스 토큰 로드 성공 (길이: {len(ACCESS_TOKEN)})")
except Exception as e:
    print(f"⚠️  액세스 토큰 로드 실패: {e}")
    ACCESS_TOKEN = os.environ.get("KIWOOM_ACCESS_TOKEN", "your_access_token_here")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def example_simple_realtime():
    """간단한 실시간 데이터 수신 예제"""
    print("=== 간단한 실시간 데이터 수신 예제 ===")
    
    from kiwoom_rest_api.koreanstock.realtime_stock import RealtimeStockClient, StockRealtimeData
    
    # 클라이언트 생성
    client = RealtimeStockClient(access_token=ACCESS_TOKEN)
    
    # 콜백 함수 설정
    def on_trade_data(stock_code: str, data: StockRealtimeData):
        """주식체결 데이터 수신 시 호출"""
        print(f"[체결] {data.종목명}({stock_code}): 현재가={data.현재가:,} 등락률={data.등락률}% "
              f"거래량={data.거래량:,} 체결강도={data.체결강도}")
    
    def on_quote_data(stock_code: str, data: StockRealtimeData):
        """주식호가 데이터 수신 시 호출"""
        print(f"[호가] {data.종목명}({stock_code}): "
              f"매도1={data.매도호가1:,}({data.매도호가수량1:,}) "
              f"매수1={data.매수호가1:,}({data.매수호가수량1:,})")
    
    client.on_trade_data = on_trade_data
    client.on_quote_data = on_quote_data
    
    # 클라이언트 시작
    await client.start()
    
    # 종목 등록 (삼성전자, SK하이닉스)
    await client.register_stocks(
        stock_codes=['005930', '000660'],
        include_trade=True,
        include_quote=True
    )
    
    # 30초간 실행
    try:
        await asyncio.sleep(30)
    finally:
        await client.stop()


async def example_with_manager():
    """Manager를 사용한 고급 예제 (히스토리 저장)"""
    print("=== Manager를 사용한 고급 예제 ===")
    
    from kiwoom_rest_api.koreanstock.realtime_stock import RealtimeStockManager
    
    # 매니저 생성
    manager = RealtimeStockManager(access_token=ACCESS_TOKEN)
    
    # 시작
    await manager.start()
    
    # 종목 추가
    await manager.add_stocks(
        stock_codes=['005930', '000660', '035720'],  # 삼성전자, SK하이닉스, 카카오
        include_trade=True,
        include_quote=True
    )
    
    # 20초간 실행
    try:
        await asyncio.sleep(20)
    finally:
        # 히스토리 출력
        for stock_code in manager.registered_stocks:
            history = manager.get_history(stock_code, limit=5)
            print(f"\n[{stock_code}] 최근 5개 데이터:")
            for i, data in enumerate(history, 1):
                print(f"  {i}. 시간={data.get('시간', 'N/A')} "
                      f"등락률={data.get('등락률', 0):.2f}% "
                      f"누적거래대금={data.get('누적거래대금', 0):,.0f}")
        
        await manager.stop()


async def example_final_columns_format():
    """FINAL_COLUMNS 형식으로 데이터 추출 예제"""
    print("=== FINAL_COLUMNS 형식 데이터 추출 예제 ===")
    
    from kiwoom_rest_api.koreanstock.realtime_stock import (
        RealtimeStockClient, 
        StockRealtimeData,
        FINAL_COLUMNS
    )
    
    # 클라이언트 생성
    client = RealtimeStockClient(access_token=ACCESS_TOKEN)
    
    # 데이터 업데이트 콜백
    def on_data_update(stock_code: str, data: StockRealtimeData):
        """데이터 업데이트 시 FINAL_COLUMNS 형식으로 출력"""
        # FINAL_COLUMNS 형식으로 변환
        data_dict = data.to_dict()
        
        # 주요 필드만 출력
        print(f"\n[{stock_code}] FINAL_COLUMNS 데이터:")
        print(f"  종목명: {data_dict['종목명']}")
        print(f"  시간: {data_dict['시간']}")
        print(f"  등락률: {data_dict['등락률']:.2f}%")
        print(f"  누적거래대금: {data_dict['누적거래대금']:,.0f}")
        print(f"  거래회전율: {data_dict['거래회전율']:.2f}%")
        print(f"  체결강도: {data_dict['체결강도']:.2f}%")
        
        # 매도/매수 대기금액 상위 3개
        print(f"  매도대기금액1-3: {data_dict['매도대기금액1']:.2f}, "
              f"{data_dict['매도대기금액2']:.2f}, {data_dict['매도대기금액3']:.2f} (백만원)")
        print(f"  매수대기금액1-3: {data_dict['매수대기금액1']:.2f}, "
              f"{data_dict['매수대기금액2']:.2f}, {data_dict['매수대기금액3']:.2f} (백만원)")
        
        # 전체 FINAL_COLUMNS 컬럼 확인
        print(f"\n  사용 가능한 컬럼 ({len(FINAL_COLUMNS)}개):")
        print(f"  {', '.join(FINAL_COLUMNS[:10])}...")
    
    client.on_data_update = on_data_update
    
    # 시작
    await client.start()
    
    # 종목 등록
    await client.register_stocks(
        stock_codes=['005930'],  # 삼성전자
        include_trade=True,
        include_quote=True
    )
    
    # 15초간 실행 (데이터 업데이트 시마다 출력됨)
    try:
        await asyncio.sleep(15)
    finally:
        await client.stop()


async def example_custom_processing():
    """커스텀 데이터 처리 예제"""
    print("=== 커스텀 데이터 처리 예제 ===")
    
    from kiwoom_rest_api.koreanstock.realtime_stock import RealtimeStockClient, StockRealtimeData
    
    # 데이터 저장을 위한 딕셔너리
    accumulated_data = {}
    
    # 클라이언트 생성
    client = RealtimeStockClient(access_token=ACCESS_TOKEN)
    
    # 데이터 업데이트 콜백
    def on_data_update(stock_code: str, data: StockRealtimeData):
        """데이터를 누적하고 통계 계산"""
        if stock_code not in accumulated_data:
            accumulated_data[stock_code] = {
                'count': 0,
                'max_price': 0,
                'min_price': float('inf'),
                'total_volume': 0
            }
        
        acc = accumulated_data[stock_code]
        acc['count'] += 1
        
        if data.현재가 > 0:
            acc['max_price'] = max(acc['max_price'], data.현재가)
            acc['min_price'] = min(acc['min_price'], data.현재가)
        
        if data.누적거래량 > 0:
            acc['total_volume'] = data.누적거래량
        
        # 10개 업데이트마다 통계 출력
        if acc['count'] % 10 == 0:
            print(f"\n[{data.종목명}({stock_code})] 통계 (업데이트 {acc['count']}회):")
            print(f"  최고가: {acc['max_price']:,}")
            print(f"  최저가: {acc['min_price']:,}")
            print(f"  누적거래량: {acc['total_volume']:,}")
            if acc['max_price'] > 0 and acc['min_price'] < float('inf'):
                volatility = (acc['max_price'] - acc['min_price']) / acc['min_price'] * 100
                print(f"  변동폭: {volatility:.2f}%")
    
    client.on_data_update = on_data_update
    
    # 시작
    await client.start()
    
    # 종목 등록
    await client.register_stocks(
        stock_codes=['005930', '000660'],
        include_trade=True,
        include_quote=True
    )
    
    # 30초간 실행
    try:
        await asyncio.sleep(30)
    finally:
        # 최종 통계 출력
        print("\n=== 최종 통계 ===")
        for stock_code, stats in accumulated_data.items():
            print(f"{stock_code}: {stats}")
        
        await client.stop()


async def main():
    """메인 함수"""
    print("주식 실시간 데이터 수신 예제")
    print("=" * 50)
    
    if not ACCESS_TOKEN or ACCESS_TOKEN == "your_access_token_here":
        print("⚠️  경고: ACCESS_TOKEN이 설정되지 않았습니다.")
        print("KoapyRestSimple을 통해 토큰을 가져오거나 환경변수를 설정하세요.")
        return
    
    # 예제들 실행
    examples = [
        ("간단한 실시간 데이터 수신", example_simple_realtime),
        ("Manager를 사용한 고급 예제", example_with_manager),
        ("FINAL_COLUMNS 형식 데이터 추출", example_final_columns_format),
        ("커스텀 데이터 처리", example_custom_processing),
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*50}")
        print(f"{name} 예제 실행 중...")
        print(f"{'='*50}")
        try:
            await example_func()
        except Exception as e:
            print(f"예제 실행 중 오류 발생: {e}")
        
        print(f"\n{name} 예제 완료")
        await asyncio.sleep(2)  # 예제 간 간격
    
    print("\n" + "="*50)
    print("모든 예제 완료!")


if __name__ == "__main__":
    # 비동기 예제들 실행
    asyncio.run(main())
