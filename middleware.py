"""
Middleware для авторизації API токенів
"""
from fastapi import HTTPException, Header, Request
from typing import Optional
import logging

logger = logging.getLogger(__name__)

async def verify_api_key(request: Request, authorization: Optional[str] = Header(None)):
    """Перевіряє API ключ з заголовка Authorization"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Відсутній токен авторизації")
    
    # Перевіряємо формат "Bearer TOKEN"
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Невірний формат токена. Використовуйте: Bearer YOUR_TOKEN")
    
    api_key = authorization[7:]  # Видаляємо "Bearer "
    
    # Імпортуємо менеджер тут щоб уникнути циклічних імпортів
    from api_auth import api_key_manager
    
    if not api_key_manager.verify_api_key(api_key):
        raise HTTPException(status_code=401, detail="Недійсний або неактивний API ключ")
    
    return api_key

async def verify_master_token(request: Request, authorization: Optional[str] = Header(None)):
    """Перевіряє master токен"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Відсутній master токен")
    
    # Перевіряємо формат "Bearer TOKEN"
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Невірний формат токена. Використовуйте: Bearer YOUR_TOKEN")
    
    master_token = authorization[7:]  # Видаляємо "Bearer "
    
    # Імпортуємо менеджер тут щоб уникнути циклічних імпортів
    from api_auth import api_key_manager
    
    if not api_key_manager.verify_master_token(master_token):
        raise HTTPException(status_code=401, detail="Недійсний master токен")
    
    return master_token

def get_master_token_from_query(request: Request) -> Optional[str]:
    """Отримує master токен з query параметра"""
    return request.query_params.get("master_token")

def verify_master_token_from_query(request: Request) -> str:
    """Перевіряє master токен з query параметра"""
    master_token = get_master_token_from_query(request)
    if not master_token:
        raise HTTPException(status_code=401, detail="Відсутній master токен в параметрах")
    
    from api_auth import api_key_manager
    
    if not api_key_manager.verify_master_token(master_token):
        raise HTTPException(status_code=401, detail="Недійсний master токен")
    
    return master_token
