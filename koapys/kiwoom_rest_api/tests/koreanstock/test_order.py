from dotenv import load_dotenv
import json
# import logging


# # 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# 환경 변수 로드
load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env")

from kiwoom_rest_api.koreanstock.order import Order
from kiwoom_rest_api.auth.token import TokenManager

# 토큰 매니저 초기화
token_manager = TokenManager()

# StockInfo 인스턴스 생성 (base_url 수정)
order = Order(base_url="https://api.kiwoom.com", token_manager=token_manager)

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
    
    print_result("kt10000_result", order.stock_buy_order_request_kt10000(
        dmst_stex_tp="KRX",
        stk_cd="005930",
        ord_qty="1000000",
        trde_tp="3"
    ), print_result=False)
    
    print_result("kt10001_result", order.stock_sell_order_request_kt10001(
        dmst_stex_tp="KRX",
        stk_cd="005930",
        ord_qty="1",
        trde_tp="3"
    ), print_result=False)
    
    print_result("kt10002_result", order.stock_modify_order_request_kt10002(
        dmst_stex_tp="KRX",
        orig_ord_no="0000139",
        stk_cd="005930",
        mdfy_qty="1",
        mdfy_uv="199700"
    ), print_result=False)
    
    print_result("kt10003_result", order.stock_cancel_order_request_kt10003(
        dmst_stex_tp="KRX",
        orig_ord_no="0000140",
        stk_cd="005930",
        cncl_qty="1"
    ), print_result=False)
    
    
except Exception as e:
    print("에러 발생:", str(e))



