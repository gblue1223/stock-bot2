from dotenv import load_dotenv
import json
# import logging


# # 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# 환경 변수 로드
load_dotenv("C:/projects/pypi/kiwoom-rest-api/.env")

from kiwoom_rest_api.koreanstock.market_condition import MarketCondition
from kiwoom_rest_api.auth.token import TokenManager

# 토큰 매니저 초기화
token_manager = TokenManager()

# StockInfo 인스턴스 생성 (base_url 수정)
market_condition = MarketCondition(base_url="https://api.kiwoom.com", token_manager=token_manager)

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
    
    print_result("ka10004_result", market_condition.stock_quote_request_ka10004(
        stock_code="005930"
    ), print_result=False)

    print_result("ka10005_result", market_condition.stock_daily_weekly_monthly_time_request_ka10005(
        stock_code="005930"
    ), print_result=False)

    print_result("ka10006_result", market_condition.stock_minute_time_request_ka10006(
        stock_code="005930"
    ), print_result=False)

    print_result("ka10007_result", market_condition.market_price_table_info_request_ka10007(
        stock_code="005930"
    ), print_result=False)

    print_result("ka10011_result", market_condition.rights_issue_overall_price_request_ka10011(
        rights_type="00"  # 전체
    ), print_result=False)

    print_result("ka10044_result", market_condition.daily_institutional_trading_items_request_ka10044(
        start_date="20241106",
        end_date="20241107",
        trade_type="1",         # 순매도
        market_type="001",      # 코스피
        stock_exchange_type="3" # 통합
    ), print_result=False)

    print_result("ka10045_result", market_condition.stockwise_institutional_trading_trend_request_ka10045(
        stock_code="005930",
        start_date="20241007",
        end_date="20241107",
        org_institution_price_type="1",   # 기관 매수단가
        foreign_institution_price_type="1" # 외인 매수단가
    ), print_result=False)

    print_result("ka10046_result", market_condition.execution_strength_by_hour_request_ka10046(
        stock_code="005930"
    ), print_result=False)

    print_result("ka10047_result", market_condition.execution_strength_by_day_request_ka10047(
        stock_code="005930"
    ), print_result=False)

    print_result("ka10063_result", market_condition.intraday_investor_trading_request_ka10063(
        market_type="000",
        amount_quantity_type="1",
        investor_type="6",
        foreign_all="0",
        simultaneous_net_buy_type="0",
        stock_exchange_type="3"
    ), print_result=False)

    print_result("ka10066_result", market_condition.post_market_investor_trading_request_ka10066(
        market_type="000",
        amount_quantity_type="1",
        trade_type="0",
        stock_exchange_type="3"
    ), print_result=False)

    print_result("ka10078_result", market_condition.brokerwise_stock_trading_trend_request_ka10078(
        member_company_code="001",
        stock_code="005930",
        start_date="20241106",
        end_date="20241107"
    ), print_result=False)

    print_result("ka10086_result", market_condition.daily_stock_price_request_ka10086(
        stock_code="005930",
        query_date="20241125",
        indicator_type="0"
    ), print_result=False)

    print_result("ka10087_result", market_condition.after_hours_single_price_request_ka10087(
        stock_code="005930"
    ), print_result=False)

    print_result("ka90005_result", market_condition.program_trading_trend_by_time_request_ka90005(
        date="20241101",
        amount_quantity_type="1",
        market_type="P00101",
        minute_tick_type="1",
        stock_exchange_type="1"
    ), print_result=False)

    print_result("ka90006_result", market_condition.program_trading_arbitrage_balance_trend_request_ka90006(
        date="20241125",
        stock_exchange_type="1"
    ), print_result=False)

    print_result("ka90007_result", market_condition.cumulative_program_trading_trend_request_ka90007(
        date="20240525",
        amount_quantity_type="1",
        market_type="0",
        stock_exchange_type="3"
    ), print_result=False)

    print_result("ka90008_result", market_condition.stockwise_program_trading_by_hour_request_ka90008(
        amount_quantity_type="1",
        stock_code="005930",
        date="20241125"
    ), print_result=False)

    print_result("ka90010_result", market_condition.program_trading_trend_by_date_request_ka90010(
        date="20241125",
        amount_quantity_type="1",
        market_type="P00101",
        minute_tick_type="0",
        stock_exchange_type="1"
    ), print_result=False)

    print_result("ka90013_result", market_condition.stockwise_program_trading_by_day_request_ka90013(
        stock_code="005930"
    ), print_result=False)

except Exception as e:
    print("에러 발생:", str(e))



