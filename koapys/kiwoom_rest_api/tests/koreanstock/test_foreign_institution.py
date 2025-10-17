from dotenv import load_dotenv
import json
# import logging


# # 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# 환경 변수 로드
load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env")

from kiwoom_rest_api.koreanstock.foreign_institution import ForeignInstitution
from kiwoom_rest_api.auth.token import TokenManager

# 토큰 매니저 초기화
token_manager = TokenManager()

# StockInfo 인스턴스 생성 (base_url 수정)
foreign_institution = ForeignInstitution(base_url="https://api.kiwoom.com", token_manager=token_manager)

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
    
    print_result("ka10008_result", foreign_institution.foreign_investor_stockwise_trading_trend_request_ka10008(
        stock_code="005930"
    ), print_result=False)
    
    print_result("ka10009_result", foreign_institution.institutional_stock_request_ka10009(
        stock_code="005930"
    ), print_result=False)
    
    print_result("ka10131_result", foreign_institution.institution_foreign_consecutive_trading_status_request_ka10131(
        dt="1",
        mrkt_tp="001",
        netslmt_tp="2",
        stk_inds_tp="0",
        amt_qty_tp="0",
        stex_tp="1"
    ), print_result=False)
   

except Exception as e:
    print("에러 발생:", str(e))



