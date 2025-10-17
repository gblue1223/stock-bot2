from typing import Dict, Optional, Any

from kiwoom_rest_api.core.sync_client import make_request

def get_per_analysis(
    market_code: str = "0",
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    PER/PBR/배당수익률 (KA-STOCK-010)
    
    Args:
        market_code: 시장분류코드 (0:전체, 1:코스피, 2:코스닥)
        access_token: OAuth 액세스 토큰
    
    Returns:
        PER/PBR/배당수익률 데이터
    """
    endpoint = "/stock/per"
    params = {
        "FID_COND_MRKT_DIV_CODE": market_code,
    }
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )

def get_rapid_price_change(
    market_code: str = "0",
    sort_code: str = "1",
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    급등락 종목 (KA-STOCK-009)
    
    Args:
        market_code: 시장분류코드 (0:전체, 1:코스피, 2:코스닥)
        sort_code: 정렬구분 (1:급등, 2:급락)
        access_token: OAuth 액세스 토큰
    
    Returns:
        급등락 종목 데이터
    """
    endpoint = "/stock/rapid"
    params = {
        "FID_COND_MRKT_DIV_CODE": market_code,
        "FID_INPUT_ISCD": sort_code,
    }
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )

def get_price_ranges(
    stock_code: str,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    가격 매물대 조회 (KA-STOCK-012)
    
    Args:
        stock_code: 종목코드 (6자리)
        access_token: OAuth 액세스 토큰
    
    Returns:
        가격 매물대 데이터
    """
    endpoint = "/stock/price-ranges"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )

def get_stock_trend(
    stock_code: str,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    주가 이격도 추이 (KA-STOCK-011)
    
    Args:
        stock_code: 종목코드 (6자리)
        access_token: OAuth 액세스 토큰
    
    Returns:
        주가 이격도 데이터
    """
    endpoint = "/stock/trend"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )
