from dotenv import load_dotenv
import logging
# from pytest_httpx import HTTPXMock # 실제 실행 시 불필요
# import pytest # 실제 실행 시 불필요
import asyncio
import os
import time # 시간 측정을 위해 추가

from kiwoom_rest_api.koreanstock.stockinfo import StockInfo
from kiwoom_rest_api.auth.token import TokenManager
from kiwoom_rest_api.core.base import APIError # APIError 임포트

# --- 실제 비동기 API 호출 함수 ---
# 함수 이름을 좀 더 명확하게 하고, 어떤 호출인지 구분하기 위해 인덱스를 받도록 수정
async def fetch_basic_stock_info(index: int, stock_code: str, stock_info_instance: StockInfo):
    """지정된 종목 코드의 기본 정보를 비동기적으로 가져옵니다."""
    print(f"[Task {index}] 시작: {stock_code} 정보 요청")
    start_time = time.monotonic()
    try:
        result = await stock_info_instance.basic_stock_information_request_ka10001(stock_code)
        end_time = time.monotonic()
        print(f"[Task {index}] 완료: {stock_code} 정보 가져옴 (성공). 소요 시간: {end_time - start_time:.2f}초")
        # 결과의 일부만 출력하거나 필요한 정보만 반환할 수 있습니다.
        # print(f"[Task {index}] 결과: {result.get('output', '결과 없음')}")
        return index, "성공" if str(result.get("return_code")) == "0" else "실패", result # 인덱스, 상태, 결과 코드 반환
    except APIError as e:
        end_time = time.monotonic()
        print(f"[Task {index}] 완료: {stock_code} 정보 가져옴 (API 오류: HTTP {e.status_code}). 소요 시간: {end_time - start_time:.2f}초")
        print(f"[Task {index}] 오류 메시지: {e.message}")
        return index, f"API 오류 {e.status_code}", e.message # 인덱스, 상태, 오류 메시지 반환
    except Exception as e:
        end_time = time.monotonic()
        print(f"[Task {index}] 완료: {stock_code} 정보 가져옴 (기타 오류: {type(e).__name__}). 소요 시간: {end_time - start_time:.2f}초")
        print(f"[Task {index}] 오류 메시지: {e}")
        return index, f"오류 {type(e).__name__}", str(e) # 인덱스, 상태, 오류 메시지 반환

# --- 메인 비동기 실행 함수 ---
async def main():
    """여러 API 호출을 동시에 실행합니다."""
    # --- 환경 변수 로드 및 키 확인 ---
    load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env") # .env 파일 경로 확인
    api_key = os.getenv('KIWOOM_API_KEY')
    api_secret = os.getenv('KIWOOM_API_SECRET')

    if not api_key or not api_secret:
        print("오류: KIWOOM_API_KEY 또는 KIWOOM_API_SECRET 환경 변수가 설정되지 않았습니다.")
        return

    # --- 공유 객체 생성 ---
    # TokenManager는 한 번만 생성하여 공유 (주의: 동시 토큰 갱신 시 문제 없는지 확인 필요)
    token_manager = TokenManager()
    # StockInfo 인스턴스도 한 번만 생성
    stock_info_async = StockInfo(
        base_url="https://api.kiwoom.com", # !!! 중요: 올바른 Open API URL 사용 !!!
        token_manager=token_manager,
        use_async=True
    )

    # --- 동시 실행할 작업 목록 ---
    # 여러 종목 코드 또는 동일 종목 코드 여러 번 호출
    stock_codes_to_fetch = ["005930", "000660", "035720", "005930", "000660","005930", "000660", "035720", "005930", "000660"] # 예시: 삼성전자, SK하이닉스, 카카오
    num_concurrent_tasks = len(stock_codes_to_fetch)
    tasks = []
    for i, code in enumerate(stock_codes_to_fetch):
        # 각 작업에 인덱스, 종목 코드, 공유 StockInfo 인스턴스 전달
        task = asyncio.create_task(fetch_basic_stock_info(i + 1, code, stock_info_async))
        tasks.append(task)

    print(f"\n--- {num_concurrent_tasks}개의 비동기 작업 동시 실행 시작 ---")
    overall_start_time = time.monotonic()

    # asyncio.gather를 사용하여 모든 작업을 동시에 실행하고 결과(또는 예외)를 기다림
    # return_exceptions=True: 작업 중 예외가 발생해도 다른 작업은 계속 진행하고 예외 객체를 반환
    results = await asyncio.gather(*tasks, return_exceptions=True)

    overall_end_time = time.monotonic()
    print(f"--- 모든 작업 완료 ---")
    print(f"총 소요 시간: {overall_end_time - overall_start_time:.2f}초")

    # --- 결과 처리 ---
    print("\n--- 각 작업 결과 요약 ---")
    successful_calls = 0
    failed_calls = 0
    for i, result in enumerate(results):
        task_index = i + 1 # fetch_basic_stock_info 에서 반환된 인덱스와 동일
        if isinstance(result, Exception):
            # gather에서 예외를 반환한 경우 (일반적으로 작업 내부에서 처리되지 않은 예외)
            print(f"[Task {task_index}] 실패 (Gather에서 예외 발생): {result}")
            failed_calls += 1
        else:
            # fetch_basic_stock_info 에서 반환된 튜플 (index, status, detail)
            idx, status, detail = result
            print(f"[Task {idx}] 상태: {status}")
            if status != "성공":
                print(f"  - 상세: {detail}") # 필요 시 상세 오류 출력
                failed_calls += 1
            else:
                successful_calls += 1
                print(f"  - 결과 코드: {result}") # 성공 시 결과 코드 출력

    print(f"\n요약: 성공 {successful_calls}건, 실패 {failed_calls}건")

# --- 스크립트 실행 부분 ---
if __name__ == "__main__":
    print("\n\n비동기 동시 실행 테스트 시작...\n\n")
    # 메인 비동기 함수 실행
    asyncio.run(main())
    print("\n\n테스트 실행 완료.")