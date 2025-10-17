from typing import Dict, Optional, Any

from kiwoom_rest_api.core.sync_client import make_request

def get_investor_trend(
    stock_code: str,
    period: str = "D",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    투자자별 매매동향 (KA-STOCK-013)
    
    Args:
        stock_code: 종목코드 (6자리)
        period: 기간분류코드 (D:일봉, W:주봉, M:월봉)
        start_date: 조회 시작 날짜 (YYYYMMDD)
        end_date: 조회 끝 날짜 (YYYYMMDD)
        access_token: OAuth 액세스 토큰
    
    Returns:
        투자자별 매매동향 데이터
    """
    endpoint = "/stock/investor"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
        "FID_PERIOD_DIV_CODE": period,
    }
    
    if start_date:
        params["FID_INPUT_DATE_1"] = start_date
    
    if end_date:
        params["FID_INPUT_DATE_2"] = end_date
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )

def get_market_investor_trend(
    market_code: str = "0",
    period: str = "D",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    전체 시장별 투자자 매매동향 (KA-STOCK-014)
    
    Args:
        market_code: 시장분류코드 (0:전체, 1:코스피, 2:코스닥)
        period: 기간분류코드 (D:일봉, W:주봉, M:월봉)
        start_date: 조회 시작 날짜 (YYYYMMDD)
        end_date: 조회 끝 날짜 (YYYYMMDD)
        access_token: OAuth 액세스 토큰
    
    Returns:
        시장별 투자자 매매동향 데이터
    """
    endpoint = "/stock/investor/market"
    params = {
        "FID_COND_MRKT_DIV_CODE": market_code,
        "FID_PERIOD_DIV_CODE": period,
    }
    
    if start_date:
        params["FID_INPUT_DATE_1"] = start_date
    
    if end_date:
        params["FID_INPUT_DATE_2"] = end_date
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )

def get_program_trading(
    market_code: str = "0",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    프로그램 매매현황 (KA-STOCK-015)
    
    Args:
        market_code: 시장분류코드 (0:전체, 1:코스피, 2:코스닥)
        start_date: 조회 시작 날짜 (YYYYMMDD)
        end_date: 조회 끝 날짜 (YYYYMMDD)
        access_token: OAuth 액세스 토큰
    
    Returns:
        프로그램 매매현황 데이터
    """
    endpoint = "/stock/program"
    params = {
        "FID_COND_MRKT_DIV_CODE": market_code,
    }
    
    if start_date:
        params["FID_INPUT_DATE_1"] = start_date
    
    if end_date:
        params["FID_INPUT_DATE_2"] = end_date
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )
