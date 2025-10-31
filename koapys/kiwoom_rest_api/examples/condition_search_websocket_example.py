#!/usr/bin/env python3
"""
조건검색 WebSocket 클라이언트 사용 예제

이 예제는 키움증권의 조건검색 실시간 WebSocket API를 사용하는 방법을 보여줍니다.
ConditionSearchClient 클래스를 사용하여 간편하게 조건검색을 수행합니다.
"""

import asyncio
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from koapys import KoapyRestSimple, ConditionSearchClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 상세 디버깅이 필요한 경우 아래 주석 해제
# logging.getLogger('kiwoom_rest_api.websocket').setLevel(logging.DEBUG)
# logging.getLogger('websockets').setLevel(logging.DEBUG)

async def example_condition_search_realtime():
    """조건검색 실시간 WebSocket 예제 (단일 조건식)"""
    print("=== 조건검색 실시간 WebSocket 예제 ===")
    
    # KoapyRestSimple 클라이언트 생성
    koapy_client = KoapyRestSimple()
    koapy_client.ensure_connected()
    
    # WebSocketClient 시작 (KoapyRestSimple에서 관리)
    await koapy_client.start_websocket()
    
    # ConditionSearchClient 생성
    client = ConditionSearchClient(client=koapy_client.websocket)
    
    selected_condition = None
    
    # 콜백 함수 설정
    def on_condition_list(conditions):
        """조건식 목록 수신 시 호출"""
        nonlocal selected_condition
        print(f"\n✓ 조건식 목록 수신: {len(conditions)}개")
        for cond in conditions:
            print(f"  [{cond.index}] {cond.name}")
        
        if conditions:
            selected_condition = conditions[0]
            print(f"\n선택된 조건식: [{selected_condition.index}] {selected_condition.name}")
        else:
            print("\n⚠️ 등록된 조건식이 없습니다. HTS에서 조건식을 먼저 등록해주세요.")
    
    def on_condition_registered(cond_idx, cond_name):
        """조건검색 등록 완료 시 호출"""
        print(f"✓ 조건검색 등록 성공: [{cond_idx}] {cond_name}")
        print("   실시간 종목 데이터 수신 대기 중...")
    
    def on_stock_in(code, name, cond_idx):
        """종목 편입 시 호출"""
        print(f"  [편입] {code} - {name}")
    
    def on_stock_out(code, name, cond_idx):
        """종목 이탈 시 호출"""
        print(f"  [이탈] {code} - {name}")
    
    def on_error(error):
        """오류 발생 시 호출"""
        print(f"✗ 오류 발생: {error}")
    
    # 콜백 등록
    client.on_condition_list = on_condition_list
    client.on_condition_registered = on_condition_registered
    client.on_stock_in = on_stock_in
    client.on_stock_out = on_stock_out
    client.on_error = on_error
    
    try:
        # 1. 조건식 목록 로드
        print("\n1. 조건식 목록 조회 중...")
        conditions = await client.load_conditions()
        
        if not conditions:
            print("조건식이 없습니다. 종료합니다.")
            return
        
        # 2. 첫 번째 조건식 등록
        if selected_condition:
            print(f"\n2. 조건검색 실시간 등록 중...")
            await client.register_condition(
                condition_index=selected_condition.index,
                condition_name=selected_condition.name
            )
        
        # 3. 60초간 실시간 데이터 수신
        print("\n3. 실시간 데이터 수신 중 (60초)...")
        print("   조건에 맞는 종목이 발생하면 실시간으로 표시됩니다.\n")
        await asyncio.sleep(60)
        
    finally:
        # 4. 조건검색 해지 및 종료
        if selected_condition:
            print("\n4. 조건검색 해지 중...")
            await client.unregister_condition(
                condition_index=selected_condition.index
            )
            await asyncio.sleep(1)
        
        client.cleanup()  # 콜백 정리
        await koapy_client.stop_websocket()
        koapy_client.close()
        
        # 최종 결과 출력
        stocks = client.get_received_stocks()
        total_stocks = sum(len(s) for s in stocks.values())
        print(f"\n=== 최종 수신 종목 수: {total_stocks}개 ===")
        for cond_idx, stock_list in stocks.items():
            if stock_list:
                print(f"\n조건식 [{cond_idx}]: {len(stock_list)}개 종목")
                for stock in stock_list[:20]:  # 최대 20개 표시
                    print(f"  - {stock}")

async def example_multiple_condition_search():
    """여러 조건식 동시 실행 예제"""
    print("=== 여러 조건검색 동시 실행 예제 ===")
    
    # KoapyRestSimple 클라이언트 생성
    koapy_client = KoapyRestSimple()
    koapy_client.ensure_connected()
    
    # WebSocketClient 시작 (KoapyRestSimple에서 관리)
    await koapy_client.start_websocket()
    
    # ConditionSearchClient 생성
    client = ConditionSearchClient(client=koapy_client.websocket)
    
    selected_conditions = []
    
    # 콜백 함수 설정
    def on_condition_list(conditions):
        """조건식 목록 수신 시 호출"""
        print(f"\n✓ 조건식 목록 수신: {len(conditions)}개")
        for cond in conditions:
            print(f"  [{cond.index}] {cond.name}")
        
        if len(conditions) < 2:
            print("\n⚠️ 2개 이상의 조건식이 필요합니다.")
        else:
            # 최대 2개 조건식 선택
            selected_conditions.clear()
            selected_conditions.extend(conditions[:2])
            print(f"\n선택된 조건식: {len(selected_conditions)}개")
            for cond in selected_conditions:
                print(f"  [{cond.index}] {cond.name}")
    
    def on_condition_registered(cond_idx, cond_name):
        """조건검색 등록 완료 시 호출"""
        print(f"✓ 조건검색 등록 성공: [{cond_idx}] {cond_name}")
    
    def on_stock_in(code, name, cond_idx):
        """종목 편입 시 호출"""
        print(f"  [{cond_idx}] 편입: {code} - {name}")
    
    def on_stock_out(code, name, cond_idx):
        """종목 이탈 시 호출"""
        print(f"  [{cond_idx}] 이탈: {code} - {name}")
    
    def on_error(error):
        """오류 발생 시 호출"""
        print(f"✗ 오류 발생: {error}")
    
    # 콜백 등록
    client.on_condition_list = on_condition_list
    client.on_condition_registered = on_condition_registered
    client.on_stock_in = on_stock_in
    client.on_stock_out = on_stock_out
    client.on_error = on_error
    
    try:
        # 1. 조건식 목록 로드
        print("\n1. 조건식 목록 조회 중...")
        conditions = await client.load_conditions()
        
        if len(conditions) < 2:
            print("조건식이 2개 미만입니다. 종료합니다.")
            return
        
        # 2. 여러 조건식 등록
        print(f"\n2. 조건검색 실시간 등록 중... ({len(selected_conditions)}개)")
        await client.register_multiple_conditions(selected_conditions, delay=0.5)
        
        # 3. 60초간 실시간 데이터 수신
        print("\n3. 실시간 데이터 수신 중 (60초)...")
        print("   조건에 맞는 종목이 발생하면 실시간으로 표시됩니다.\n")
        await asyncio.sleep(60)
        
    finally:
        # 4. 모든 조건검색 해지 및 종료
        if selected_conditions:
            print("\n4. 조건검색 해지 중...")
            await client.unregister_multiple_conditions(selected_conditions)
            await asyncio.sleep(1)
        
        client.cleanup()  # 콜백 정리
        await koapy_client.stop_websocket()
        koapy_client.close()
        
        # 최종 결과 출력
        stocks = client.get_received_stocks()
        print("\n=== 최종 결과 ===")
        for cond_idx, stock_list in stocks.items():
            cond_name = next((c.name for c in selected_conditions if c.index == cond_idx), cond_idx)
            print(f"[{cond_idx}] {cond_name}: {len(stock_list)}개 종목")
            if stock_list:
                for stock in stock_list[:10]:  # 최대 10개 표시
                    print(f"  - {stock}")

async def example_realtime_condition_detailed():
    """실시간 조건검색 데이터 (type='02') 상세 예제
    
    이 예제는 REAL 타입 중 type='02' (조건검색 실시간 데이터)의 처리 과정을 
    자세히 보여줍니다. 종목 편입/이탈 시 상세 정보를 출력합니다.
    """
    print("=== 실시간 조건검색 데이터 (REAL type='02') 상세 예제 ===")
    
    # KoapyRestSimple 클라이언트 생성
    koapy_client = KoapyRestSimple()
    koapy_client.ensure_connected()
    
    # WebSocketClient 시작
    await koapy_client.start_websocket()
    
    # ConditionSearchClient 생성
    client = ConditionSearchClient(client=koapy_client.websocket)
    
    selected_condition = None
    stock_in_count = 0
    stock_out_count = 0
    
    # 실시간 REAL 데이터 처리를 위한 상세 콜백
    async def on_realtime_data(realtime_data):
        """WebSocket의 모든 REAL 데이터를 가로채서 type='02' 확인"""
        if realtime_data.trnm == 'REAL':
            for item in realtime_data.data:
                if isinstance(item, dict) and item.get('type') == '02':
                    # type='02'는 조건검색 실시간 데이터
                    values = item.get('values', {})
                    print(f"\n[REAL type='02' 수신]")
                    print(f"  종목코드: {values.get('9001', 'N/A')}")
                    print(f"  동작: {'편입' if values.get('843') == 'I' else '이탈' if values.get('843') == 'D' else '알수없음'} ({values.get('843')})")
                    print(f"  체결시간: {values.get('20', 'N/A')}")
                    print(f"  매도/수구분: {values.get('907', 'N/A')}")
    
    # 콜백 함수 설정
    def on_condition_list(conditions):
        """조건식 목록 수신 시 호출"""
        nonlocal selected_condition
        print(f"\n✓ 조건식 목록 수신: {len(conditions)}개")
        for cond in conditions:
            print(f"  [{cond.index}] {cond.name}")
        
        if conditions:
            selected_condition = conditions[0]
            print(f"\n선택된 조건식: [{selected_condition.index}] {selected_condition.name}")
        else:
            print("\n⚠️ 등록된 조건식이 없습니다. HTS에서 조건식을 먼저 등록해주세요.")
    
    def on_condition_registered(cond_idx, cond_name):
        """조건검색 등록 완료 시 호출"""
        print(f"\n✓ 조건검색 등록 성공: [{cond_idx}] {cond_name}")
        print("=" * 60)
        print("실시간 조건검색 데이터 (REAL type='02') 수신 대기 중...")
        print("조건에 맞는 종목이 발생하면 편입/이탈 정보가 표시됩니다.")
        print("=" * 60)
    
    def on_stock_in(code, name, cond_idx):
        """종목 편입 시 호출 (REAL type='02', action='I')"""
        nonlocal stock_in_count
        stock_in_count += 1
        print(f"\n🔵 [편입 #{stock_in_count}] 조건식[{cond_idx}]")
        print(f"   종목코드: {code}")
        if name:
            print(f"   종목명: {name}")
        print(f"   현재 편입 종목 수: {len(client.received_stocks.get(cond_idx, []))}개")
    
    def on_stock_out(code, name, cond_idx):
        """종목 이탈 시 호출 (REAL type='02', action='D')"""
        nonlocal stock_out_count
        stock_out_count += 1
        print(f"\n🔴 [이탈 #{stock_out_count}] 조건식[{cond_idx}]")
        print(f"   종목코드: {code}")
        if name:
            print(f"   종목명: {name}")
        print(f"   현재 편입 종목 수: {len(client.received_stocks.get(cond_idx, []))}개")
    
    def on_error(error):
        """오류 발생 시 호출"""
        print(f"\n✗ 오류 발생: {error}")
    
    # 콜백 등록
    client.on_condition_list = on_condition_list
    client.on_condition_registered = on_condition_registered
    client.on_stock_in = on_stock_in
    client.on_stock_out = on_stock_out
    client.on_error = on_error
    
    # WebSocket REAL 데이터 리스너 추가 (type='02' 모니터링용)
    koapy_client.websocket.on('data', on_realtime_data)
    
    try:
        # 1. 조건식 목록 로드
        print("\n1. 조건식 목록 조회 중...")
        conditions = await client.load_conditions()
        
        if not conditions:
            print("조건식이 없습니다. 종료합니다.")
            return
        
        # 2. 첫 번째 조건식 등록
        if selected_condition:
            print(f"\n2. 조건검색 실시간 등록 중...")
            await client.register_condition(
                condition_index=selected_condition.index,
                condition_name=selected_condition.name
            )
        
        # 3. 120초간 실시간 데이터 수신 (type='02' 데이터 대기)
        print("\n3. 실시간 REAL type='02' 데이터 수신 중 (120초)...")
        print("   ※ REAL 메시지의 type='02'는 조건검색 실시간 데이터입니다.")
        print("   ※ values 필드 내용:")
        print("      - '9001': 종목코드")
        print("      - '843': 편입('I') / 이탈('D')")
        print("      - '20': 체결시간")
        print("      - '907': 매도/수 구분")
        print()
        
        await asyncio.sleep(120)
        
    finally:
        # 4. 조건검색 해지 및 종료
        if selected_condition:
            print("\n4. 조건검색 해지 중...")
            await client.unregister_condition(
                condition_index=selected_condition.index
            )
            await asyncio.sleep(1)
        
        # 리스너 제거
        koapy_client.websocket.remove_listener('data', on_realtime_data)
        client.cleanup()
        await koapy_client.stop_websocket()
        koapy_client.close()
        
        # 최종 통계 출력
        print("\n" + "=" * 60)
        print("=== 실시간 조건검색 (REAL type='02') 처리 통계 ===")
        print("=" * 60)
        print(f"총 편입 이벤트: {stock_in_count}회")
        print(f"총 이탈 이벤트: {stock_out_count}회")
        
        stocks = client.get_received_stocks()
        total_stocks = sum(len(s) for s in stocks.values())
        print(f"\n최종 편입 종목 수: {total_stocks}개")
        
        for cond_idx, stock_list in stocks.items():
            if stock_list:
                print(f"\n조건식 [{cond_idx}]: {len(stock_list)}개 종목")
                for i, stock in enumerate(stock_list[:30], 1):  # 최대 30개 표시
                    print(f"  {i:2d}. {stock}")

async def main():
    """메인 함수"""
    print("키움증권 조건검색 WebSocket 예제")
    print("=" * 50)
    
    
    # 예제 선택
    print("\n1. 단일 조건검색 실시간")
    print("2. 여러 조건검색 동시 실행")
    print("3. 실시간 조건검색 데이터 (REAL type='02') 상세 예제")
    
    # 기본적으로 첫 번째 예제 실행
    # await example_condition_search_realtime()
    
    # 여러 조건검색 예제 실행하려면 아래 주석 해제
    # await example_multiple_condition_search()
    
    # 실시간 type='02' 상세 예제 실행하려면 아래 주석 해제
    await example_realtime_condition_detailed()

if __name__ == "__main__":
    asyncio.run(main())
