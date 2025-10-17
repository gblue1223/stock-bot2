from typing import Dict, Optional, Any

from kiwoom_rest_api.core.sync_client import make_request

def get_trading_volume(
    stock_code: str,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    거래량 급증 종목 (KA-STOCK-008)
    
    Args:
        stock_code: 종목코드 (6자리)
        access_token: OAuth 액세스 토큰
    
    Returns:
        거래량 급증 데이터
    """
    endpoint = "/stock/volume"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )

def get_execution_price(
    stock_code: str,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    체결가 추이 (KA-STOCK-005)
    
    Args:
        stock_code: 종목코드 (6자리)
        access_token: OAuth 액세스 토큰
    
    Returns:
        체결가 추이 데이터
    """
    endpoint = "/stock/execution"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )

def get_orderbook(
    stock_code: str,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    호가 정보 조회 (KA-STOCK-006)
    
    Args:
        stock_code: 종목코드 (6자리)
        access_token: OAuth 액세스 토큰
    
    Returns:
        호가 정보 데이터
    """
    endpoint = "/stock/orderbook"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )

def get_trading_brokers(
    stock_code: str,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    거래원 정보 조회 (KA-STOCK-007)
    
    Args:
        stock_code: 종목코드 (6자리)
        access_token: OAuth 액세스 토큰
    
    Returns:
        거래원 정보 데이터
    """
    endpoint = "/stock/broker"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
    }
    
    return make_request(
        endpoint=endpoint,
        params=params,
        access_token=access_token,
    )
