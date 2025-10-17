from typing import Optional
from kiwoom_rest_api.core.sync_client import make_request
from kiwoom_rest_api.core.async_client import make_request_async

class KiwoomBaseAPI:
    def __init__(
        self,
        base_url: str = None,
        token_manager=None,
        use_async: bool = False,
        resource_url: str = ""
    ):
        self.base_url = base_url
        self.token_manager = token_manager
        self.use_async = use_async
        self.resource_url = resource_url
        self._request_func = make_request_async if use_async else make_request

    def _get_access_token(self) -> Optional[str]:
        if self.token_manager:
            return self.token_manager.get_token()
        return None

    async def _get_access_token_async(self) -> Optional[str]:
        if self.token_manager and hasattr(self.token_manager, 'get_token_async'):
            return await self.token_manager.get_token_async()
        return self._get_access_token()

    def _make_request(self, method: str, url: str, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["content-type"] = "application/json;charset=UTF-8"
        if self.token_manager:
            access_token = self._get_access_token()
            headers["Authorization"] = f"Bearer {access_token}"
        return make_request(endpoint=url, method=method, headers=headers, **kwargs)

    async def _make_request_async(self, method: str, url: str, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["content-type"] = "application/json;charset=UTF-8"
        if self.token_manager:
            access_token = await self._get_access_token_async()
            headers["Authorization"] = f"Bearer {access_token}"
        return await make_request_async(endpoint=url, method=method, headers=headers, **kwargs)

    def _execute_request(self, method: str, resource_url: str = None, **kwargs):
        # resource_url이 제공되면 임시로 사용, 아니면 기본값 사용
        url_resource = resource_url if resource_url is not None else self.resource_url
        url = f"{self.base_url}{url_resource}" if self.base_url else f"/{url_resource}"
        if self.use_async:
            return self._make_request_async(method, url, **kwargs)
        return self._make_request(method, url, **kwargs)
