#!/usr/bin/env python3
"""
조건검색 테스트 스크립트 - WebSocket으로 조건 충족 종목 확인
"""
import sys
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from koapys import KoapyRestSimple
from koapys.kiwoom_rest_api.src.kiwoom_rest_api.koreanstock.condition_search import ConditionSearchClient

async def test_condition_search():
    """조건검색 WebSocket 테스트"""
    print("=" * 80)
    print("조건검색 테스트 (WebSocket)")
    print("=" * 80)
    
    # Koapys 클라이언트 초기화
    koapys = KoapyRestSimple(simulation=False)
    koapys.ensure_connected()
    print(f"✓ Connected to Koapys API")
    
    # WebSocket 시작
    print("\n1. WebSocket 연결 중...")
    await koapys.start_websocket()
    print("✓ WebSocket connected")
    
    # ConditionSearchClient 생성
    condition_client = ConditionSearchClient(client=koapys.websocket)
    
    # 조건식 목록 로드
    print("\n2. 조건식 목록 조회 중...")
    conditions = await condition_client.load_conditions()
    
    if not conditions:
        print("❌ 조건식이 없습니다!")
        await koapys.stop_websocket()
        return
    
    print(f"✓ 조건식 {len(conditions)}개 발견:")
    for i, cond in enumerate(conditions):
        print(f"  [{i}] {cond.index}: {cond.name}")
    
    # 첫 번째 조건식으로 검색
    selected = conditions[0]
    cond_idx = selected.index
    cond_name = selected.name
    
    print(f"\n3. 조건검색 등록 및 종목 수신: [{cond_idx}] {cond_name}")
    
    # 수신된 종목 저장
    received_stocks = []
    target_code = '356680'
    found = False
    
    def on_stock_in(code, name, cond_idx):
        nonlocal found
        code_clean = code[1:] if code.startswith('A') else code
        received_stocks.append({'code': code_clean, 'name': name})
        
        if code_clean == target_code:
            print(f"  🎯 {code_clean} ({name}) ← 타겟 종목 발견!")
            found = True
        else:
            if len(received_stocks) <= 20:
                print(f"  [{len(received_stocks)}] {code_clean} ({name})")
    
    condition_client.on_stock_in = on_stock_in
    
    # 조건검색 등록
    await condition_client.register_condition(
        condition_index=cond_idx,
        condition_name=cond_name
    )
    
    # 초기 종목 수신 대기 (2초)
    await asyncio.sleep(2)
    
    print(f"\n4. 결과 확인")
    print(f"✓ 총 {len(received_stocks)}개 종목 수신")
    
    if len(received_stocks) > 20:
        print(f"   (처음 20개만 출력, 나머지 {len(received_stocks) - 20}개 생략)")
    
    if found:
        print(f"\n✅ {target_code}이 조건에 포함되어 있습니다!")
    else:
        print(f"\n❌ {target_code}이 조건에 포함되지 않습니다.")
        print(f"   조건식 '{cond_name}'의 설정을 HTS에서 확인하세요.")
        print(f"   또는 실시간 편입을 기다려야 할 수 있습니다 (현재는 초기 종목만 확인)")
    
    # 정리
    await condition_client.unregister_condition(cond_idx)
    condition_client.cleanup()
    await koapys.stop_websocket()
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    asyncio.run(test_condition_search())
