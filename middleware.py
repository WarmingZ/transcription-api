"""
Middleware для авторизації та логування
"""

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import logging
from typing import Optional
from auth import get_api_key_manager

logger = logging.getLogger(__name__)

class APIKeyAuth(HTTPBearer):
    """Авторизація через API ключ"""
    
    def __init__(self, auto_error: bool = True):
        super(APIKeyAuth, self).__init__(auto_error=auto_error)
        self.api_key_manager = get_api_key_manager()
    
    async def __call__(self, request: Request) -> Optional[str]:
        credentials: HTTPAuthorizationCredentials = await super(APIKeyAuth, self).__call__(request)
        
        if credentials:
            if not self.verify_api_key(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Недійсний API ключ",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return credentials.credentials
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API ключ не надано",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def verify_api_key(self, api_key: str, update_stats: bool = True) -> bool:
        """Перевіряє API ключ"""
        return self.api_key_manager.validate_api_key(api_key, update_stats)

# Глобальний об'єкт авторизації
api_key_auth = APIKeyAuth()

class AdminAPIKeyAuth(APIKeyAuth):
    """Авторизація для адмін ендпоінтів (без оновлення статистики)"""
    
    def verify_api_key(self, api_key: str) -> bool:
        """Перевіряє API ключ без оновлення статистики"""
        return self.api_key_manager.validate_api_key(api_key, update_stats=False)

# Глобальний об'єкт авторизації для адміністраторів
admin_api_key_auth = AdminAPIKeyAuth()

class LoggingMiddleware:
    """Middleware для логування запитів"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Логуємо запит
            logger.info(f"📥 {request.method} {request.url.path} - {request.client.host}")
            
            # Логуємо запити до захищених ендпоінтів
            if request.url.path.startswith("/transcribe") or request.url.path.startswith("/api-keys"):
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    api_key = auth_header.split(" ")[1]
                    key_info = api_key_auth.api_key_manager.get_key_info(api_key)
                    if key_info:
                        logger.info(f"🔐 Запит до {request.url.path} від {key_info['name']}")
                    else:
                        logger.warning(f"⚠️ Недійсний API ключ для {request.url.path}")
                else:
                    logger.warning(f"⚠️ Відсутній заголовок авторизації для {request.url.path}")
        
        await self.app(scope, receive, send)

