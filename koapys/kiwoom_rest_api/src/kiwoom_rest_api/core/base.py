from typing import Any, Dict, Optional, Union
import json
from urllib.parse import urljoin
import httpx
import inspect # Import inspect

from kiwoom_rest_api.config import get_base_url, get_headers, DEFAULT_TIMEOUT

class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, status_code: int, message: str, error_data: dict = None):
        self.status_code = status_code
        self.message = message
        self.error_data = error_data or {}
        super().__init__(f"API Error (HTTP {status_code}): {message}")

    def __str__(self):
        return f"API Error (HTTP {self.status_code}): {self.message}"

def make_url(endpoint: str) -> str:
    """Create a full URL from an endpoint"""
    if endpoint.startswith(('http://', 'https://')):
        return endpoint
    
    # Ensure endpoint starts with a forward slash
    if not endpoint.startswith('/'):
        endpoint = f"/{endpoint}"
        
    print("\n\n##  full url  ##\n\n", urljoin(get_base_url(), endpoint))
    
    return urljoin(get_base_url(), endpoint)

def process_response(response: Any) -> Dict[str, Any]:
    """Process API response and handle errors"""
    if not hasattr(response, 'status_code'):
        raise ValueError(f"Invalid response object: {response}")
    
    if 200 <= response.status_code < 300:
        if not response.text:
            return {}
        
        try:
            response_json = response.json()
            
            access_control_expose_headers = response.headers.get("access-control-expose-headers")
            if access_control_expose_headers:
                access_control_expose_headers = access_control_expose_headers.split(",")
                
                for header in access_control_expose_headers:
                    response_json[header] = response.headers.get(header)
                
                return response_json
                
            return response_json
        
        except json.JSONDecodeError:
            return {"content": response.text}
    
    # Handle error responses
    error_message = "Unknown error"
    error_data = None
    
    try:
        error_data = response.json()
        error_message = error_data.get("message", "Unknown error")
    except (json.JSONDecodeError, AttributeError):
        if response.text:
            error_message = response.text
    
    raise APIError(response.status_code, error_message, error_data)

def prepare_request_params(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    access_token: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Prepare request parameters for HTTP request"""
    # 헤더 정규화
    normalized_headers = {}
    if headers:
        for key, value in headers.items():
            # 모든 헤더 키를 소문자로 변환하여 중복 방지
            normalized_headers[key.lower()] = value
    
    # 기본 헤더 설정
    default_headers = {
        "content-type": "application/json;charset=UTF-8",
    }
    
    # API 키 추가
    from kiwoom_rest_api.config import get_api_key, get_api_secret
    default_headers["appkey"] = get_api_key()
    default_headers["appsecret"] = get_api_secret()
    
    # 헤더 병합 (사용자 정의 헤더가 기본 헤더보다 우선)
    merged_headers = {**default_headers, **normalized_headers}
    
    # 액세스 토큰 추가
    if access_token:
        merged_headers["authorization"] = f"Bearer {access_token}"
    
    # URL 구성
    from kiwoom_rest_api.config import get_base_url
    url = endpoint if endpoint.startswith(("http://", "https://")) else f"{get_base_url()}{endpoint}"
    
    # 요청 파라미터 구성
    request_params = {
        "url": url,
        "method": method,
        "headers": merged_headers,
        "timeout": timeout or DEFAULT_TIMEOUT,
    }
    
    # 쿼리 파라미터 추가
    if params:
        request_params["params"] = params
    
    # POST/PUT/PATCH 요청용 데이터 추가
    if method in ["POST", "PUT", "PATCH"] and data:
        if merged_headers.get("content-type", "").startswith("application/json"):
            request_params["json"] = data
        else:
            request_params["data"] = data
    
    return request_params

async def process_response_async(response: httpx.Response) -> Dict[str, Any]:
    if not isinstance(response, httpx.Response):
        print("ERROR: process_response_async did not receive an httpx.Response object!")
        raise TypeError(f"Expected httpx.Response, but got {type(response)}")

    # --- 추가 디버깅 ---
    json_method = getattr(response, 'json', None)
    is_json_coro = inspect.iscoroutinefunction(json_method)
    print(f"DEBUG: inspect.iscoroutinefunction(response.json) = {is_json_coro}")
    # --- 추가 디버깅 끝 ---

    try:
        # 성공(200) 응답 처리
        if response.status_code == 200:
            try:
                # 여기서 여전히 TypeError 발생 가능성 있음
                json_data = await response.json()
                
                access_control_expose_headers = response.headers.get("access-control-expose-headers")
                if access_control_expose_headers:
                    access_control_expose_headers = access_control_expose_headers.split(",")
                    
                    for header in access_control_expose_headers:
                        json_data[header] = response.headers.get(header)
                
                
                if isinstance(json_data, dict) and str(json_data.get("return_code")) != "0":
                     error_message = json_data.get("return_msg", "Unknown API error message")
                     raise APIError(response.status_code, error_message, json_data)
                return json_data
            except json.JSONDecodeError:
                 # 여기서도 TypeError 발생 가능성 있음
                 raw_text_content = await response.text()
                 error_message = f"Failed to decode JSON response. Content: {raw_text_content[:200]}"
                 raise APIError(response.status_code, error_message, {"raw_content": raw_text_content})
            except TypeError as te: # await 실패 시
                 print(f"ERROR: TypeError on SUCCESS path await: {te}")
                 # await 없이 직접 접근 시도 (진단용)
                 try:
                    json_data = response.json() # await 없이 호출
                     
                       
                    access_control_expose_headers = response.headers.get("access-control-expose-headers")
                    if access_control_expose_headers:
                        access_control_expose_headers = access_control_expose_headers.split(",")
                        
                        for header in access_control_expose_headers:
                            json_data[header] = response.headers.get(header)
                            
                    if isinstance(json_data, dict) and str(json_data.get("return_code")) != "0":
                        error_message = json_data.get("return_msg", "Unknown API error message")
                        raise APIError(response.status_code, error_message, json_data)
                    return json_data # 성공하면 반환
                 except Exception as direct_err:
                     print(f"ERROR: Direct access failed after TypeError: {direct_err}")
                     raw_text_content = getattr(response, 'text', 'N/A') # text 속성 접근 시도
                     raise APIError(response.status_code, f"TypeError processing SUCCESS response: {te}. Raw content: {raw_text_content[:200]}", {"raw_content": raw_text_content})


        # HTTP 에러(400 등) 처리
        else:
            error_message = f"HTTP Error {response.status_code}"
            error_data = {"status_code": response.status_code}
            raw_text_content = "Could not retrieve error content"

            # --- 진단: await 없이 text 속성 직접 접근 시도 ---
            try:
                if hasattr(response, 'text') and isinstance(response.text, str):
                    print("DEBUG: Accessing response.text directly as attribute.")
                    raw_text_content = response.text
                    error_data["raw_content"] = raw_text_content
                    error_message += f". Content: {raw_text_content[:500]}" # 내용 조금 더 보기

                    # 텍스트 내용으로 JSON 파싱 시도
                    try:
                        error_json = json.loads(raw_text_content)
                        error_msg1 = error_json.get("msg1", "No msg1 found in error JSON")
                        error_message = f"HTTP Error {response.status_code}: {error_msg1}" # 에러 메시지 개선
                        error_data.update(error_json)
                    except json.JSONDecodeError:
                        print("DEBUG: Error response body is not JSON.")
                else:
                     print("DEBUG: response.text is not a direct string attribute.")
                     # 여기서 await response.text()를 시도하면 TypeError 발생 가능성 높음
            except Exception as e_diag:
                print(f"ERROR: Exception during diagnostic access of response text: {e_diag}")
            # --- 진단 끝 ---

            # 최종 에러 발생
            raise APIError(response.status_code, error_message, error_data)

    except httpx.RequestError as e:
        # 네트워크 관련 에러
        raise APIError(500, f"Request failed: {str(e)}", {"exception": str(e)})
