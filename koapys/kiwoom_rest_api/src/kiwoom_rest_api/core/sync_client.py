from typing import Any, Dict, Optional

import httpx

from kiwoom_rest_api.core.base import prepare_request_params, process_response

def make_request(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    access_token: Optional[str] = None,
    timeout: Optional[float] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    
    """Make a synchronous HTTP request to the Kiwoom API"""
    request_params = prepare_request_params(
        endpoint=endpoint,
        method=method,
        params=params,
        data=data,
        headers=headers,
        access_token=access_token,
        timeout=timeout,
    )
    
    # 추가: kwargs에서 json 데이터 처리
    if 'json' in kwargs and method in ["POST", "PUT", "PATCH"]:
        request_params["json"] = kwargs['json']
    
    with httpx.Client() as client:
        response = client.request(
            method=request_params["method"],
            url=request_params["url"],
            params=request_params.get("params"),
            json=request_params.get("json"),
            headers=request_params["headers"],
            timeout=request_params["timeout"],
        )
                
        return process_response(response)
