from dotenv import load_dotenv
import json
# import logging


# # 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# 환경 변수 로드
load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env")

from kiwoom_rest_api.koreanstock.theme import Theme
from kiwoom_rest_api.auth.token import TokenManager

# 토큰 매니저 초기화
token_manager = TokenManager()

# StockInfo 인스턴스 생성 (base_url 수정)
theme = Theme(base_url="https://api.kiwoom.com", token_manager=token_manager)

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
    
    print_result("ka90001_result", theme.theme_group_list_request_ka90001(
        qry_tp="0",
        date_tp="10",
        flu_pl_amt_tp="1",
        stex_tp="1"
    ), print_result=False)
    
    print_result("ka90002_result", theme.theme_component_stocks_request_ka90002(
        thema_grp_cd="100",
        stex_tp="1",
        date_tp="2"
    ), print_result=False)
    
    
except Exception as e:
    print("에러 발생:", str(e))



