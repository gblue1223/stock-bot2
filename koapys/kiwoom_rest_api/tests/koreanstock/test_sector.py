from dotenv import load_dotenv
import json
# import logging


# # 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# 환경 변수 로드
load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env")

from kiwoom_rest_api.koreanstock.sector import Sector
from kiwoom_rest_api.auth.token import TokenManager

# 토큰 매니저 초기화
token_manager = TokenManager()

# StockInfo 인스턴스 생성 (base_url 수정)
sector = Sector(base_url="https://api.kiwoom.com", token_manager=token_manager)

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
    
    print_result("ka10010_result", sector.industry_program_trading_request_ka10010(
        stock_code="005930"
    ), print_result=False)
    
    print_result("ka10051_result", sector.industrywise_investor_net_buy_request_ka10051(
        mrkt_tp="0",  # 코스피
        amt_qty_tp="0",  # 금액
        stex_tp="3",  # 통합
        base_dt="20241107"  # 기준일자
    ), print_result=False)
    
    print_result("ka20001_result", sector.industry_current_price_request_ka20001(
        mrkt_tp="0",  # 코스피
        inds_cd="001"  # 종합(KOSPI)
    ), print_result=False)
    
    print_result("ka20002_result", sector.industrywise_stock_price_request_ka20002(
        mrkt_tp="0",  # 코스피
        inds_cd="001",  # 종합(KOSPI)
        stex_tp="1"  # KRX
    ), print_result=False)
    
    print_result("ka20003_result", sector.all_industries_index_request_ka20003(
        inds_cd="001"  # 종합(KOSPI)
    ), print_result=False)
    
    print_result("ka20009_result", sector.industry_daily_current_price_request_ka20009(
        mrkt_tp="0",  # 코스피
        inds_cd="001"  # 종합(KOSPI)
    ), print_result=False)

except Exception as e:
    print("에러 발생:", str(e))



