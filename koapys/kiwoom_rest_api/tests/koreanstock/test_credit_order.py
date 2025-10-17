from dotenv import load_dotenv
import json
# import logging


# # 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# 환경 변수 로드
load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env")

from kiwoom_rest_api.koreanstock.credit_order import CreditOrder
from kiwoom_rest_api.auth.token import TokenManager

# 토큰 매니저 초기화
token_manager = TokenManager()

# StockInfo 인스턴스 생성 (base_url 수정)
credit_order = CreditOrder(base_url="https://api.kiwoom.com", token_manager=token_manager)

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
    
    import time
    
    print_result("margin_buy_order_request_kt10006_result", credit_order.margin_buy_order_request_kt10006(
        dmst_stex_tp="KRX",
        stk_cd="005930",
        ord_qty="1",
        ord_uv="1",
        trde_tp="0"
    ), print_result=False)
    
    print_result("margin_sell_order_request_kt10007_result", credit_order.margin_sell_order_request_kt10007(
        dmst_stex_tp="KRX",
        stk_cd="005930",
        ord_qty="1",
        ord_uv="1",
        trde_tp="0",
        crd_deal_tp="33",
        crd_loan_dt=time.strftime("%Y%m%d", time.localtime())
    ), print_result=False)
    
    print_result("margin_modify_order_request_kt10008_result", credit_order.margin_modify_order_request_kt10008(
        dmst_stex_tp="KRX",
        orig_ord_no="0000455",
        stk_cd="005930",
        mdfy_qty="1",
        mdfy_uv="1"
    ), print_result=False)
    
    print_result("margin_cancel_order_request_kt10009_result", credit_order.margin_cancel_order_request_kt10009(
        dmst_stex_tp="KRX",
        orig_ord_no="0001615",
        stk_cd="005930",
        cncl_qty="1"
    ), print_result=False)
    
    
    
except Exception as e:
    print("에러 발생:", str(e))



