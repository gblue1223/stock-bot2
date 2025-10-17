from dotenv import load_dotenv
import json
# import logging


# # 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# 환경 변수 로드
load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env")

from kiwoom_rest_api.koreanstock.account import Account
from kiwoom_rest_api.auth.token import TokenManager

# 토큰 매니저 초기화
token_manager = TokenManager()

# StockInfo 인스턴스 생성 (base_url 수정)
account = Account(base_url="https://api.kiwoom.com", token_manager=token_manager)

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
    
    print_result("ka10072_result", account.realized_profit_by_date_stock_request_ka10072(
        stock_code="005930",
        start_date="20241128"
    ), print_result=False)
    
    print_result("ka10073_result", account.realized_profit_by_period_stock_request_ka10073(
        stock_code="005930",
        start_date="20241128",
        end_date="20241128"
    ), print_result=False)

    print_result("ka10074_result", account.daily_realized_profit_request_ka10074(
        start_date="20240301",
        end_date="20240331"
    ), print_result=False)
    
    print_result("ka10075_result", account.unfilled_orders_request_ka10075(
        all_stk_tp="1",
        trde_tp="0",
        stex_tp="0",
        stock_code="005930"
    ), print_result=False)
    
    print_result("ka10076_result", account.filled_orders_request_ka10076(
        qry_tp="1",
        sell_tp="0",
        stex_tp="0",
        stock_code="005930"
    ), print_result=False)
    
    print_result("ka10077_result", account.today_realized_profit_detail_request_ka10077(
        stock_code="005930"
    ), print_result=False)
    
    print_result("ka10085_result", account.account_return_rate_request_ka10085(
        stex_tp="0"  # 통합 거래소
    ), print_result=False)
    
    print_result("ka10088_result", account.unfilled_split_order_detail_request_ka10088(
        order_no="8"
    ), print_result=False)
    
    print_result("ka10170_result", account.today_trading_journal_request_ka10170(
        ottks_tp="1",
        ch_crd_tp="0",
        base_dt="20241128"
    ), print_result=False)
    
    print_result("kt00001_result", account.deposit_detail_status_request_kt00001(
        qry_tp="1"
    ), print_result=False)
    
    print_result("kt00002_result", account.daily_estimated_deposit_asset_status_request_kt00002(
        start_dt="20241128",
        end_dt="20241128"
    ), print_result=False)
    
    print_result("kt00003_result", account.estimated_asset_inquiry_request_kt00003(
        qry_tp="0"  # 전체 조회
    ), print_result=False)
    
    print_result("kt00004_result", account.account_evaluation_status_request_kt00004(
        qry_tp="0",  # 전체 조회
        dmst_stex_tp="KRX"  # 한국거래소
    ), print_result=False)
    
    print_result("kt00005_result", account.filled_position_request_kt00005(
        dmst_stex_tp="KRX"  # 한국거래소
    ), print_result=False)
    
    print_result("kt00007_result", account.account_order_execution_detail_request_kt00007(
        qry_tp="1",  # 주문순
        stk_bond_tp="0",  # 전체
        sell_tp="0",  # 전체
        dmst_stex_tp="%",  # 전체
        stock_code=""  # 전체
    ), print_result=False)
    
    print_result("kt00008_result", account.next_day_settlement_schedule_request_kt00008(
        strt_dcd_seq=""  # 전체
    ), print_result=False)
    
    print_result("kt00009_result", account.account_order_execution_status_request_kt00009(
        stk_bond_tp="0",  # 전체
        mrkt_tp="0",  # 전체
        sell_tp="0",  # 전체
        qry_tp="0",  # 전체
        dmst_stex_tp="KRX"  # 한국거래소
    ), print_result=False)
    
    print_result("kt00010_result", account.withdrawable_order_amount_request_kt00010(
        stock_code="005930",  # 삼성전자
        trde_tp="2",  # 매수
        uv="267000"  # 매수가격
    ), print_result=False)
    
    print_result("kt00011_result", account.orderable_quantity_by_margin_ratio_request_kt00011(
        stock_code="005930"  # 삼성전자
    ), print_result=False)
    
    print_result("kt00012_result", account.orderable_quantity_by_credit_guarantee_ratio_request_kt00012(
        stock_code="005930"  # 삼성전자
    ), print_result=False)
    
    print_result("kt00013_result", account.margin_detail_inquiry_request_kt00013(), print_result=False)
    
    print_result("kt00015_result", account.comprehensive_transaction_history_request_kt00015(
        start_date="20241121",
        end_date="20241125",
        transaction_type="0"  # 전체
    ), print_result=False)
    
    print_result("kt00016_result", account.daily_account_return_detail_status_request_kt00016(
        from_date="20241111",
        to_date="20241125"
    ), print_result=False)
    
    print_result("kt00017_result", account.today_account_status_by_account_request_kt00017(), print_result=False)
    
    print_result("kt00018_result", account.account_evaluation_balance_detail_request_kt00018(
        query_type="1",  # 합산
        domestic_exchange_type="KRX"  # 한국거래소
    ), print_result=False)
    
except Exception as e:
    print("에러 발생:", str(e))



