#!/usr/bin/env python3
"""
조건검색 WebSocket 클라이언트 사용 예제

이 예제는 키움증권의 조건검색 실시간 WebSocket API를 사용하는 방법을 보여줍니다.
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# kiwoom_rest_api 모듈 경로 추가
kiwoom_api_src = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(kiwoom_api_src))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 상세 디버깅이 필요한 경우 아래 주석 해제
# logging.getLogger('kiwoom_rest_api.websocket').setLevel(logging.DEBUG)
# logging.getLogger('websockets').setLevel(logging.DEBUG)

async def example_condition_search_realtime():
    """조건검색 실시간 WebSocket 예제"""
    print("=== 조건검색 실시간 WebSocket 예제 ===")
    
    from kiwoom_rest_api import WebSocketClient, RealTimeData
    from kiwoom_rest_api.auth.token import TokenManager
    
    # 토큰 매니저 생성
    token_manager = TokenManager()
    client = WebSocketClient(access_token=token_manager.get_token())
    
    # 조건식 목록 및 선택된 조건식 저장
    condition_list = []
    selected_condition = None
    
    # 수신된 종목 저장
    received_stocks = []
    
    # 콜백 함수들 설정
    async def on_data_received(realtime_data: RealTimeData):
        """실시간 데이터 수신 시 호출"""
        nonlocal selected_condition
        
        # 1. 조건식 목록 응답
        if realtime_data.trnm == 'CNSRLST':
            print(f"\n✓ 조건식 목록 수신")
            if realtime_data.return_code == 0:
                condition_list.clear()
                # 데이터 형식: [['0', 'koa-시가베팅'], ['1', 'koa-관종-3%이상'], ...]
                for item in realtime_data.data:
                    if isinstance(item, list) and len(item) >= 2:
                        cond_idx = item[0]
                        cond_nm = item[1]
                        condition_list.append({'cond_idx': cond_idx, 'cond_nm': cond_nm})
                        print(f"  [{cond_idx}] {cond_nm}")
                
                if condition_list:
                    # 첫 번째 조건식 선택
                    selected_condition = condition_list[0]
                    print(f"\n선택된 조건식: [{selected_condition['cond_idx']}] {selected_condition['cond_nm']}")
                    
                    # 선택한 조건식으로 실시간 등록
                    print(f"\n2. 조건검색 실시간 등록 중...")
                    await client.register_condition_search_ka10173(
                        condition_index=selected_condition['cond_idx'],
                        condition_name=selected_condition['cond_nm']
                    )
                else:
                    print("\n⚠️ 등록된 조건식이 없습니다. HTS에서 조건식을 먼저 등록해주세요.")
            else:
                print(f"✗ 조건식 목록 조회 실패: {realtime_data.return_msg}")
        
        # 2. 조건검색 등록 응답
        elif realtime_data.trnm == 'CNSRREQ':
            if realtime_data.return_code == 0:
                print("✓ 조건검색 등록 성공 - 실시간 종목 데이터 수신 대기 중...")
            else:
                print(f"✗ 조건검색 등록 실패: {realtime_data.return_msg}")
        
        # 3. 실시간 조건검색 데이터
        elif realtime_data.trnm == 'COND_REAL':
            # 조건검색 실시간 데이터
            print(f"\n실시간 조건검색 데이터 수신: {len(realtime_data.data)}개 항목")
            for item in realtime_data.data:
                stock_code = item.get('stk_cd', '')
                stock_name = item.get('stk_nm', '')
                action = item.get('action', '')  # 'in' or 'out'
                
                if action == 'in':
                    print(f"  [편입] {stock_code} - {stock_name}")
                    if stock_code not in received_stocks:
                        received_stocks.append(stock_code)
                elif action == 'out':
                    print(f"  [이탈] {stock_code} - {stock_name}")
                    if stock_code in received_stocks:
                        received_stocks.remove(stock_code)
                else:
                    print(f"  [종목] {stock_code} - {stock_name}")
                    if stock_code not in received_stocks:
                        received_stocks.append(stock_code)
    
    async def on_connected():
        """연결 성공 시 호출"""
        print("✓ WebSocket 서버에 연결되었습니다")
    
    async def on_logged_in():
        """로그인 성공 시 호출"""
        print("✓ WebSocket 서버 로그인 성공")
        
        # 로그인 후 조건식 목록 조회
        print("\n1. 조건식 목록 조회 중...")
        await client.condition_list_request_ka10171()
    
    async def on_error(error: Exception):
        """오류 발생 시 호출"""
        print(f"✗ 오류 발생: {error}")
    
    # 콜백 함수들 등록
    client.on_data = on_data_received
    client.on_connect = on_connected
    client.on_login = on_logged_in
    client.on_error = on_error
    
    # 클라이언트 시작
    await client.start()
    
    # 60초간 실행 (조건 충족 종목 실시간 수신)
    print("\n3. 실시간 데이터 수신 중 (60초)...")
    print("   조건에 맞는 종목이 발생하면 실시간으로 표시됩니다.")
    try:
        await asyncio.sleep(60)
    finally:
        # 조건검색 해지
        if selected_condition:
            print("\n4. 조건검색 해지 중...")
            await client.unregister_condition_search_ka10174(
                condition_index=selected_condition['cond_idx']
            )
            await asyncio.sleep(1)  # 해지 메시지 전송 대기
        
        await client.stop()
        
        print(f"\n최종 수신 종목 수: {len(received_stocks)}개")
        if received_stocks:
            print("수신된 종목:")
            for stock in received_stocks[:20]:  # 최대 20개 표시
                print(f"  - {stock}")

async def example_multiple_condition_search():
    """여러 조건식 동시 실행 예제"""
    print("=== 여러 조건검색 동시 실행 예제 ===")
    
    from kiwoom_rest_api import WebSocketClient, RealTimeData
    from kiwoom_rest_api.auth.token import TokenManager
    
    # 토큰 매니저 생성
    token_manager = TokenManager()
    client = WebSocketClient(access_token=token_manager.get_token())
    
    # 조건식 목록 및 선택된 조건식들 저장
    condition_list = []
    selected_conditions = []
    
    # 조건식별 종목 저장
    condition_stocks = {}
    
    async def on_data_received(realtime_data: RealTimeData):
        """실시간 데이터 수신"""
        # 조건식 목록 응답
        if realtime_data.trnm == 'CNSRLST':
            print(f"\n✓ 조건식 목록 수신")
            if realtime_data.return_code == 0:
                condition_list.clear()
                # 데이터 형식: [['0', 'koa-시가베팅'], ['1', 'koa-관종-3%이상'], ...]
                for item in realtime_data.data:
                    if isinstance(item, list) and len(item) >= 2:
                        cond_idx = item[0]
                        cond_nm = item[1]
                        condition_list.append({'cond_idx': cond_idx, 'cond_nm': cond_nm})
                        print(f"  [{cond_idx}] {cond_nm}")
                
                if len(condition_list) < 2:
                    print("\n⚠️ 2개 이상의 조건식이 필요합니다.")
                    return
                
                # 최대 2개 조건식 사용
                selected_conditions.clear()
                selected_conditions.extend(condition_list[:2])
                
                # 조건식별 종목 저장 초기화
                for cond in selected_conditions:
                    condition_stocks[cond['cond_idx']] = []
                
                # 모든 조건식 등록
                print("\n✓ 조건식 등록 중...")
                for cond in selected_conditions:
                    print(f"  등록: [{cond['cond_idx']}] {cond['cond_nm']}")
                    await client.register_condition_search_ka10173(
                        condition_index=cond['cond_idx'],
                        condition_name=cond['cond_nm'],
                        group_no=cond['cond_idx']  # 각 조건식을 별도 그룹으로
                    )
                    await asyncio.sleep(0.5)  # 등록 간격
        
        # 실시간 조건검색 데이터
        elif realtime_data.trnm == 'COND_REAL':
            for item in realtime_data.data:
                cond_idx = item.get('cond_idx', '')
                stock_code = item.get('stk_cd', '')
                stock_name = item.get('stk_nm', '')
                
                if cond_idx in condition_stocks:
                    print(f"[{cond_idx}] {stock_code} - {stock_name}")
                    if stock_code not in condition_stocks[cond_idx]:
                        condition_stocks[cond_idx].append(stock_code)
    
    async def on_logged_in():
        """로그인 후 조건식 목록 조회"""
        print("✓ 로그인 성공, 조건식 목록 조회 중...")
        await client.condition_list_request_ka10171()
    
    client.on_data = on_data_received
    client.on_login = on_logged_in
    
    await client.start()
    
    try:
        await asyncio.sleep(60)
    finally:
        # 모든 조건식 해지
        print("\n조건식 해지 중...")
        for cond in selected_conditions:
            await client.unregister_condition_search_ka10174(
                condition_index=cond['cond_idx'],
                group_no=cond['cond_idx']
            )
        
        await client.stop()
        
        # 결과 출력
        print("\n=== 최종 결과 ===")
        for cond_idx, stocks in condition_stocks.items():
            cond_name = next((c['cond_nm'] for c in selected_conditions if c['cond_idx'] == cond_idx), cond_idx)
            print(f"[{cond_idx}] {cond_name}: {len(stocks)}개 종목")

async def main():
    """메인 함수"""
    print("키움증권 조건검색 WebSocket 예제")
    print("=" * 50)
    
    
    # 예제 선택
    print("\n1. 단일 조건검색 실시간")
    print("2. 여러 조건검색 동시 실행")
    
    # 기본적으로 첫 번째 예제 실행
    await example_condition_search_realtime()
    
    # 여러 조건검색 예제 실행하려면 아래 주석 해제
    # await example_multiple_condition_search()

if __name__ == "__main__":
    asyncio.run(main())
