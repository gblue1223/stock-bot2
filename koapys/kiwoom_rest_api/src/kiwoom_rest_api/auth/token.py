from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging

from kiwoom_rest_api.config import get_api_key, get_api_secret, TOKEN_URL
from kiwoom_rest_api.core.sync_client import make_request

logger = logging.getLogger(__name__)

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
        try:
            api_key = get_api_key()
            api_secret = get_api_secret()
            
            # API 키 검증 (보안을 위해 일부만 표시)
            if not api_key or not api_secret:
                logger.error(f"Missing API credentials: key={'SET' if api_key else 'MISSING'}, secret={'SET' if api_secret else 'MISSING'}")
                self._access_token = None
                return
            
            logger.debug(f"Requesting token with API key: {api_key[:10]}... (length: {len(api_key)})")
            
            response = make_request(
                endpoint=TOKEN_URL,
                method="POST",
                data={
                    "grant_type": "client_credentials",
                    "appkey": api_key,
                    "secretkey": api_secret,
                },
            )
            logger.debug(f"Token request response type: {type(response)}, content: {response}")
            self._update_token_info(response)
        except Exception as e:
            logger.error(f"Failed to request new token: {e}", exc_info=True)
            raise
    
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
        logger.debug(f"Token response keys: {list(token_response.keys())}")
        
        # Check for error response
        return_code = token_response.get("return_code")
        return_msg = token_response.get("return_msg", "")
        
        if return_code != 0:
            logger.error(f"Token request failed: code={return_code}, msg={return_msg}")
            logger.error(f"Full response: {token_response}")
            self._access_token = None
            return
        
        # Try multiple possible keys for access token
        self._access_token = (
            token_response.get("token") or 
            token_response.get("access_token") or
            token_response.get("accessToken")
        )
        
        if self._access_token:
            logger.info(f"Access token updated successfully (length: {len(self._access_token)})")
        else:
            logger.warning(f"Failed to extract access token from response. Available keys: {list(token_response.keys())}")
            logger.warning(f"return_code={return_code}, return_msg={return_msg}")
        
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

