from dotenv import load_dotenv
import json
# import logging


# # 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# 환경 변수 로드
load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env")

from kiwoom_rest_api.koreanstock.etf import ETF
from kiwoom_rest_api.auth.token import TokenManager

# 토큰 매니저 초기화
token_manager = TokenManager()

# StockInfo 인스턴스 생성 (base_url 수정)
etf = ETF(base_url="https://api.kiwoom.com", token_manager=token_manager)

def print_result(result_name, result, print_result):
    if isinstance(result, dict):
        if str(result.get("return_code")) == "0":
            if print_result:
                print(f"{result_name} 응답:\n", json.dumps(result, indent=4, ensure_ascii=False))
            else:
                print(f"{result_name} 응답: 성공")
        else:
            print(f"{result_name} 응답: 실패\n", json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print(f"{result_name} is not a dictionary.")


try:
    print("\n\n test 실행")
    
    print_result("ka40001_result", etf.etf_return_rate_request_ka40001(
        stock_code="069500",
        etf_object_index_code="207",
        period="3"
    ), print_result=False)
    
    print_result("ka40002_result", etf.etf_stock_info_request_ka40002(
        stock_code="069500"
    ), print_result=False)
    
    print_result("ka40003_result", etf.etf_daily_trend_request_ka40003(
        stock_code="069500"
    ), print_result=False)
    
    print_result("ka40004_result", etf.etf_overall_market_price_request_ka40004(
        tax_type="0",
        nav_pre="0",
        management_company="0000",
        tax_yn="0",
        trace_index="0",
        exchange_type="1"
    ), print_result=False)
    
    print_result("ka40006_result", etf.etf_time_segment_trend_request_ka40006(
        stock_code="069500"
    ), print_result=False)
    
    print_result("ka40007_result", etf.etf_time_segment_execution_request_ka40007(
        stock_code="069500"
    ), print_result=False)
    
    print_result("ka40008_result", etf.etf_datewise_execution_request_ka40008(
        stock_code="069500"
    ), print_result=False)
    
    print_result("ka40009_result", etf.etf_timewise_execution_request_ka40009(
        stock_code="069500"
    ), print_result=False)
    
    print_result("ka40010_result", etf.etf_timewise_trend_request_ka40010(
        stock_code="069500"
    ), print_result=False)
    

except Exception as e:
    print("에러 발생:", str(e))



