from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import time

from kiwoom_rest_api.config import get_api_key, get_api_secret, TOKEN_URL
from kiwoom_rest_api.core.sync_client import make_request

class TokenManager:
    """Manages OAuth tokens for Kiwoom API"""
    
    def __init__(self):
        self._access_token = None
        self._token_expiry = None
        self._refresh_token = None
        self._refresh_expiry = None
    
    @property
    def access_token(self) -> Optional[str]:
        """Get the current access token, refreshing if necessary"""
        if self._is_access_token_valid():
            return self._access_token
        
        # Try to refresh the token
        if self._can_refresh_token():
            self._refresh_access_token()
            return self._access_token
        
        # Get a new token
        self._request_new_token()
        return self._access_token
    
    def get_token(self) -> str:
        """Get the current access token (alias for access_token property)"""
        return self.access_token
    
    def _is_access_token_valid(self) -> bool:
        """Check if the current access token is valid"""
        if not self._access_token or not self._token_expiry:
            return False
        
        # Add a small buffer (30 seconds) to avoid edge cases
        return datetime.now() < self._token_expiry - timedelta(seconds=30)
    
    def _can_refresh_token(self) -> bool:
        """Check if we can refresh the current token"""
        if not self._refresh_token or not self._refresh_expiry:
            return False
        
        return datetime.now() < self._refresh_expiry - timedelta(seconds=30)
    
    def _request_new_token(self) -> None:
        """Request a new access token"""
        response = make_request(
            endpoint=TOKEN_URL,
            method="POST",
            data={
                "grant_type": "client_credentials",
                "appkey": get_api_key(),
                "secretkey": get_api_secret(),
            },
        )
        
        self._update_token_info(response)
    
    def _refresh_access_token(self) -> None:
        """Refresh the access token using the refresh token"""
        response = make_request(
            endpoint=TOKEN_URL,
            method="POST",
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "appkey": get_api_key(),
                "appsecret": get_api_secret(),
            },
        )
        
        self._update_token_info(response)
    
    def _update_token_info(self, token_response: Dict[str, Any]) -> None:
        """Update token information from the API response"""
        self._access_token = token_response.get("token")
        
        # Calculate expiry time
        if "expires_in" in token_response:
            self._token_expiry = datetime.now() + timedelta(seconds=token_response["expires_in"])
            
        if "expires_dt" in token_response:
            self._token_expiry = datetime.strptime(token_response["expires_dt"], "%Y%m%d%H%M%S")
        
        # Update refresh token if provided
        refresh_token = token_response.get("refresh_token")
        if refresh_token:
            self._refresh_token = refresh_token
            
            if "refresh_token_expires_in" in token_response:
                self._refresh_expiry = datetime.now() + timedelta(seconds=token_response["refresh_token_expires_in"])

# Convenience functions
def get_access_token() -> str:
    """Get a valid access token"""
    manager = TokenManager()
    return manager.get_token()



if __name__ == "__main__":
    print(get_access_token())

