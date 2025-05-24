import hmac
import os
from fastapi import HTTPException, status, Security
from fastapi.security import APIKeyHeader, APIKeyQuery
from dotenv import load_dotenv

load_dotenv()

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

class AuthManager:
    def __init__(self):
        self.valid_keys = os.getenv("API_KEYS", "").split(",")
        self.hmac_secret = os.getenv("HMAC_SECRET", "").encode()

    async def validate_api_key(
        self, 
        header_key: str = Security(api_key_header),
        query_key: str = Security(api_key_query)
    ) -> str:
        """多位置API Key验证"""
        if not (header_key or query_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API Key"
            )
        
        key = header_key or query_key
        if key not in self.valid_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API Key"
            )
        return key

    def verify_hmac(self, data: bytes, signature: str) -> bool:
        """HMAC签名验证"""
        digest = hmac.new(
            self.hmac_secret, 
            data, 
            'sha256'
        ).hexdigest()
        return hmac.compare_digest(digest, signature)

auth_manager = AuthManager()