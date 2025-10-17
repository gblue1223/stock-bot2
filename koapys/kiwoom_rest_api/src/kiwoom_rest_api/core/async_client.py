from typing import Any, Dict, Optional

import httpx

from kiwoom_rest_api.core.base import prepare_request_params, process_response_async

async def make_request_async(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    access_token: Optional[str] = None,
    timeout: Optional[float] = None,
    **kwargs  # Add **kwargs
) -> Dict[str, Any]:
    """Make an asynchronous HTTP request to the Kiwoom API"""
    request_params = prepare_request_params(
        endpoint=endpoint,
        method=method,
        params=params,
        data=data,
        headers=headers,
        access_token=access_token,
        timeout=timeout,
    )

    # Handle 'json' data from kwargs
    json_data = kwargs.get('json')
    if json_data and method in ["POST", "PUT", "PATCH"]:
        # Prioritize explicitly passed 'json' data
        request_params["json"] = json_data
        # If 'data' was also prepared, 'json' takes precedence here.
        # Remove 'data' if 'json' is being used to avoid conflicts in httpx
        request_params.pop("data", None)

    async with httpx.AsyncClient() as client:
        # This should return an httpx.Response object
        response: httpx.Response = await client.request(
            method=request_params["method"],
            url=request_params["url"],
            params=request_params.get("params"),
            json=request_params.get("json"),
            data=request_params.get("data"),
            headers=request_params["headers"],
            timeout=request_params["timeout"],
        )

    return await process_response_async(response)
