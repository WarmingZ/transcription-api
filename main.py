from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import tempfile
import os
import httpx
from typing import Optional, List, Dict, Any
import logging
import secrets
from datetime import datetime, timedelta
from models import LocalTranscriptionService
from auth import get_api_key_manager
from middleware import LoggingMiddleware, admin_api_key_auth

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ukrainian Audio Transcription API (Local Models)",
    description="API для транскрипції українського аудіо/відео з визначенням дикторів (локальні моделі)",
    version="1.0.0"
)

# Додаємо CORS middleware для правильного кодування
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Додаємо middleware для логування
app.add_middleware(LoggingMiddleware)

# Підключення статичних файлів
app.mount("/static", StaticFiles(directory="static"), name="static")

# Глобальна змінна для сервісу транскрипції
transcription_service = None

# CSRF токени для адмін панелі
csrf_tokens = {}
csrf_token_expiry = {}

# Rate limiting для адмін операцій
admin_rate_limit = {}

def generate_csrf_token(api_key: str) -> str:
    """Генерує CSRF токен для API ключа"""
    token = secrets.token_urlsafe(32)
    csrf_tokens[api_key] = token
    csrf_token_expiry[api_key] = datetime.now() + timedelta(hours=1)
    return token

def validate_csrf_token(api_key: str, token: str) -> bool:
    """Перевіряє CSRF токен"""
    if api_key not in csrf_tokens:
        return False
    
    if api_key in csrf_token_expiry and datetime.now() > csrf_token_expiry[api_key]:
        # Токен прострочений
        del csrf_tokens[api_key]
        del csrf_token_expiry[api_key]
        return False
    
    return csrf_tokens[api_key] == token

def check_rate_limit(api_key: str, operation: str, limit: int = 10, window_minutes: int = 5) -> bool:
    """Перевіряє rate limit для операції"""
    now = datetime.now()
    key = f"{api_key}:{operation}"
    
    if key not in admin_rate_limit:
        admin_rate_limit[key] = []
    
    # Очищуємо старі запити
    admin_rate_limit[key] = [
        req_time for req_time in admin_rate_limit[key]
        if now - req_time < timedelta(minutes=window_minutes)
    ]
    
    # Перевіряємо ліміт
    if len(admin_rate_limit[key]) >= limit:
        return False
    
    # Додаємо поточний запит
    admin_rate_limit[key].append(now)
    return True

# Функції авторизації
def verify_api_key(authorization: str = Header(None)) -> str:
    """Перевіряє API ключ з заголовка Authorization"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="API ключ не надано. Додайте заголовок: Authorization: Bearer YOUR_API_KEY"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Неправильний формат авторизації. Використовуйте: Bearer YOUR_API_KEY"
        )
    
    api_key = authorization.split(" ")[1]
    api_key_manager = get_api_key_manager()
    
    if not api_key_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Недійсний або неактивний API ключ"
        )
    
    return api_key

def get_current_user(api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Отримує інформацію про поточного користувача"""
    api_key_manager = get_api_key_manager()
    key_info = api_key_manager.get_key_info(api_key)
    return {
        "api_key": api_key,
        "name": key_info["name"],
        "permissions": key_info["permissions"],
        "usage_count": key_info["usage_count"],
        "is_admin": key_info.get("is_admin", False)
    }

def verify_admin_api_key(authorization: str = Header(None)) -> str:
    """Перевіряє API ключ для адмін ендпоінтів (без оновлення статистики)"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="API ключ не надано. Додайте заголовок: Authorization: Bearer YOUR_API_KEY"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Неправильний формат авторизації. Використовуйте: Bearer YOUR_API_KEY"
        )
    
    api_key = authorization.split(" ")[1]
    api_key_manager = get_api_key_manager()
    
    # Використовуємо адмін middleware (без оновлення статистики)
    if not admin_api_key_auth.verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Недійсний або неактивний API ключ"
        )
    
    return api_key

def verify_admin_user(api_key: str = Depends(verify_admin_api_key)) -> Dict[str, Any]:
    """Перевіряє чи є користувач адміністратором"""
    api_key_manager = get_api_key_manager()
    
    if not api_key_manager.is_admin(api_key):
        raise HTTPException(
            status_code=403,
            detail="Доступ заборонено. Потрібні адміністраторські права."
        )
    
    key_info = api_key_manager.get_key_info(api_key)
    return {
        "api_key": api_key,
        "name": key_info["name"],
        "permissions": key_info["permissions"],
        "usage_count": key_info["usage_count"],
        "is_admin": True
    }

class TranscriptionRequest(BaseModel):
    url: Optional[HttpUrl] = None
    language: str = "uk"  # Українська мова за замовчуванням
    model_size: str = "small"  # Розмір моделі Whisper
    enhance_audio: bool = True  # Попередня обробка аудіо
    use_diarization: bool = False  # Використовувати діаризацію

class TranscriptionResponse(BaseModel):
    text: str
    segments: List[Dict[str, Any]]
    speakers: Optional[List[Dict[str, Any]]] = None
    duration: float
    language: str
    diarization_type: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

@app.on_event("startup")
async def load_models():
    """Завантаження локальних моделей при запуску сервера"""
    global transcription_service
    
    try:
        logger.info("Ініціалізація локального сервісу транскрипції...")
        transcription_service = LocalTranscriptionService()
        
        if transcription_service.load_models():
            logger.info("Локальні моделі завантажені успішно")
        else:
            logger.error("Не вдалося завантажити моделі")
            raise RuntimeError("Моделі не завантажені")
        
    except Exception as e:
        logger.error(f"Помилка при завантаженні моделей: {e}")
        raise RuntimeError(f"Не вдалося ініціалізувати сервіс: {e}")

async def download_file_from_url(url: str) -> str:
    """Завантаження файлу з URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(str(url))
            response.raise_for_status()
            
            # Створюємо тимчасовий файл
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Помилка завантаження файлу: {str(e)}")

# Функції транскрипції тепер використовують локальний сервіс

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_file(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: str = Form("uk"),
    model_size: str = Form("small"),
    use_diarization: bool = Form(False),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Транскрипція аудіо/відео файлу з визначенням дикторів
    
    Параметри:
    - file: Завантажений файл (аудіо або відео)
    - url: URL посилання на файл
    - language: Мова транскрипції (за замовчуванням 'uk' для української)
    - model_size: Розмір моделі Whisper (tiny, base, small, medium, large, auto)
    - use_diarization: Використовувати діаризацію Оператор/Клієнт (True/False)
    """
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Необхідно надати або файл, або URL")
    
    if file and url:
        raise HTTPException(status_code=400, detail="Надайте або файл, або URL, але не обидва")
    
    # Валідація розміру моделі
    if model_size not in ["tiny", "base", "small", "medium", "large", "auto"]:
        raise HTTPException(status_code=400, detail="Розмір моделі повинен бути: tiny, base, small, medium, large або auto")
    
    temp_file_path = None
    
    try:
        # Обробка файлу або URL
        if file:
            # Збереження завантаженого файлу
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}")
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            temp_file_path = temp_file.name
            
        elif url:
            # Завантаження файлу з URL
            temp_file_path = await download_file_from_url(url)
        
        # Транскрипція з або без діаризації
        if use_diarization:
            logger.info(f"Початок транскрипції з діаризацією файлу: {temp_file_path}")
            processed_result = transcription_service.transcribe_with_diarization(temp_file_path, language)
        else:
            logger.info(f"Початок простої транскрипції файлу: {temp_file_path}")
            processed_result = transcription_service.transcribe_simple(temp_file_path, language)
        
        logger.info("Транскрипція завершена успішно")
        return TranscriptionResponse(**processed_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неочікувана помилка: {e}")
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка сервера: {str(e)}")
    
    finally:
        # Очищення тимчасових файлів
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Тимчасовий файл видалено: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Не вдалося видалити тимчасовий файл: {e}")

@app.post("/transcribe-with-diarization", response_model=TranscriptionResponse)
async def transcribe_with_diarization(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: str = Form("uk"),
    model_size: str = Form("small"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Транскрипція аудіо/відео файлу з діаризацією Оператор/Клієнт
    
    Параметри:
    - file: Завантажений файл (аудіо або відео)
    - url: URL посилання на файл
    - language: Мова транскрипції (за замовчуванням 'uk' для української)
    - model_size: Розмір моделі Whisper (tiny, base, small, medium, large, auto)
    """
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Необхідно надати або файл, або URL")
    
    if file and url:
        raise HTTPException(status_code=400, detail="Надайте або файл, або URL, але не обидва")
    
    # Валідація розміру моделі
    if model_size not in ["tiny", "base", "small", "medium", "large", "auto"]:
        raise HTTPException(status_code=400, detail="Розмір моделі повинен бути: tiny, base, small, medium, large або auto")
    
    temp_file_path = None
    
    try:
        # Обробка файлу або URL
        if file:
            # Збереження завантаженого файлу
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}")
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            temp_file_path = temp_file.name
            
        elif url:
            # Завантаження файлу з URL
            temp_file_path = await download_file_from_url(url)
        
        # Транскрипція з діаризацією
        logger.info(f"Початок транскрипції з діаризацією файлу: {temp_file_path}")
        processed_result = transcription_service.transcribe_with_diarization(temp_file_path, language)
        
        logger.info("Транскрипція з діаризацією завершена успішно")
        return TranscriptionResponse(**processed_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неочікувана помилка: {e}")
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка сервера: {str(e)}")
    
    finally:
        # Очищення тимчасових файлів
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Тимчасовий файл видалено: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Не вдалося видалити тимчасовий файл: {e}")

@app.get("/health")
async def health_check():
    """Перевірка стану сервісу"""
    return {
        "status": "healthy",
        "models_loaded": transcription_service is not None and transcription_service.models_loaded,
        "whisper_loaded": transcription_service is not None and transcription_service.whisper_model.model is not None
    }

# Ендпоінти для управління API ключами (ТІЛЬКИ ДЛЯ АДМІНІСТРАТОРІВ)
@app.get("/admin/api-keys")
async def list_api_keys(admin_user: Dict[str, Any] = Depends(verify_admin_user)):
    """Показує всі API ключі (тільки для адміністраторів)"""
    api_key = admin_user["api_key"]
    
    # Перевіряємо rate limit
    if not check_rate_limit(api_key, "list_keys", limit=20, window_minutes=5):
        raise HTTPException(
            status_code=429,
            detail="Занадто багато запитів. Спробуйте пізніше."
        )
    
    api_key_manager = get_api_key_manager()
    csrf_token = generate_csrf_token(api_key)
    
    return {
        "keys": api_key_manager.list_api_keys_with_keys(),
        "admin_user": admin_user["name"],
        "csrf_token": csrf_token
    }

@app.post("/admin/api-keys/create")
async def create_api_key(
    name: str = Form(...),
    description: str = Form(""),
    is_admin: bool = Form(False),
    csrf_token: str = Form(...),
    admin_user: Dict[str, Any] = Depends(verify_admin_user)
):
    """Створює новий API ключ (тільки для адміністраторів)"""
    api_key = admin_user["api_key"]
    
    # Перевіряємо CSRF токен
    if not validate_csrf_token(api_key, csrf_token):
        raise HTTPException(
            status_code=403,
            detail="Недійсний CSRF токен"
        )
    
    # Перевіряємо rate limit
    if not check_rate_limit(api_key, "create_key", limit=5, window_minutes=10):
        raise HTTPException(
            status_code=429,
            detail="Занадто багато спроб створення ключів. Спробуйте пізніше."
        )
    
    # Валідація даних
    if len(name.strip()) < 2:
        raise HTTPException(
            status_code=400,
            detail="Назва ключа має бути мінімум 2 символи"
        )
    
    if len(name.strip()) > 100:
        raise HTTPException(
            status_code=400,
            detail="Назва ключа не може бути довшою за 100 символів"
        )
    
    api_key_manager = get_api_key_manager()
    new_key = api_key_manager.generate_api_key(name.strip(), description.strip(), is_admin)
    
    # Логуємо створення ключа
    logger.info(f"🔑 Адміністратор {admin_user['name']} створив новий API ключ: {name}")
    
    return {
        "message": "API ключ створено успішно",
        "api_key": new_key,
        "name": name,
        "description": description,
        "is_admin": is_admin,
        "created_by": admin_user["name"]
    }

@app.post("/admin/api-keys/revoke")
async def revoke_api_key(
    api_key: str = Form(...),
    csrf_token: str = Form(...),
    admin_user: Dict[str, Any] = Depends(verify_admin_user)
):
    """Відкликає API ключ (тільки для адміністраторів)"""
    admin_api_key = admin_user["api_key"]
    
    # Перевіряємо CSRF токен
    if not validate_csrf_token(admin_api_key, csrf_token):
        raise HTTPException(
            status_code=403,
            detail="Недійсний CSRF токен"
        )
    
    # Перевіряємо rate limit
    if not check_rate_limit(admin_api_key, "revoke_key", limit=10, window_minutes=5):
        raise HTTPException(
            status_code=429,
            detail="Занадто багато спроб відкликання ключів. Спробуйте пізніше."
        )
    
    api_key_manager = get_api_key_manager()
    
    # Не дозволяємо відкликати власний ключ
    if api_key == admin_api_key:
        raise HTTPException(
            status_code=400,
            detail="Не можна відкликати власний адміністраторський ключ"
        )
    
    if api_key_manager.revoke_api_key(api_key):
        # Логуємо відкликання ключа
        logger.info(f"🚫 Адміністратор {admin_user['name']} відкликав API ключ: {api_key[:10]}...")
        return {
            "message": "API ключ відкликано успішно",
            "api_key": api_key,
            "revoked_by": admin_user["name"]
        }
    else:
        raise HTTPException(status_code=404, detail="API ключ не знайдено")

@app.post("/admin/api-keys/activate")
async def activate_api_key(
    api_key: str = Form(...),
    csrf_token: str = Form(...),
    admin_user: Dict[str, Any] = Depends(verify_admin_user)
):
    """Активує API ключ (тільки для адміністраторів)"""
    admin_api_key = admin_user["api_key"]
    
    # Перевіряємо CSRF токен
    if not validate_csrf_token(admin_api_key, csrf_token):
        raise HTTPException(
            status_code=403,
            detail="Недійсний CSRF токен"
        )
    
    # Перевіряємо rate limit
    if not check_rate_limit(admin_api_key, "activate_key", limit=10, window_minutes=5):
        raise HTTPException(
            status_code=429,
            detail="Занадто багато спроб активації ключів. Спробуйте пізніше."
        )
    
    api_key_manager = get_api_key_manager()
    
    if api_key_manager.activate_api_key(api_key):
        # Логуємо активацію ключа
        logger.info(f"✅ Адміністратор {admin_user['name']} активував API ключ: {api_key[:10]}...")
        return {
            "message": "API ключ активовано успішно",
            "api_key": api_key,
            "activated_by": admin_user["name"]
        }
    else:
        raise HTTPException(status_code=404, detail="API ключ не знайдено")

# Публічний ендпоінт для перевірки статусу ключа
@app.get("/api-key/status")
async def check_api_key_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Перевіряє статус поточного API ключа"""
    return {
        "valid": True,
        "name": current_user["name"],
        "is_admin": current_user["is_admin"],
        "usage_count": current_user["usage_count"],
        "permissions": current_user["permissions"]
    }

@app.get("/")
async def root():
    """Перенаправлення на веб-інтерфейс"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

@app.get("/admin")
async def admin_panel():
    """Перенаправлення на адмін панель"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/admin.html")

@app.get("/api")
async def api_info():
    """Інформація про API"""
    return {
        "message": "Ukrainian Audio Transcription API (Local Models)",
        "version": "1.0.0",
        "description": "API для транскрипції українського аудіо/відео з визначенням дикторів (локальні моделі)",
        "authentication": {
            "type": "API Key",
            "header": "Authorization: Bearer YOUR_API_KEY",
            "required_for": ["/transcribe", "/transcribe-with-diarization", "/api-keys/*"]
        },
        "endpoints": {
            "transcribe": "/transcribe (POST, requires auth)",
            "transcribe_with_diarization": "/transcribe-with-diarization (POST, requires auth)",
            "health": "/health (GET, public)",
            "docs": "/docs (GET, public)",
            "web_interface": "/static/index.html (GET, public)",
            "admin_panel": "/admin (GET, public) - Веб-адмін панель",
            "api_key_status": "/api-key/status (GET, requires auth)",
            "admin_list_keys": "/admin/api-keys (GET, requires admin auth)",
            "admin_create_key": "/admin/api-keys/create (POST, requires admin auth)",
            "admin_revoke_key": "/admin/api-keys/revoke (POST, requires admin auth)",
            "admin_activate_key": "/admin/api-keys/activate (POST, requires admin auth)"
        },
        "features": [
            "Локальна транскрипція (faster-whisper)",
            "Проста діаризація Оператор/Клієнт (WebRTC VAD)",
            "Підтримка файлів та URL",
            "Українська мова",
            "API Key авторизація",
            "Управління API ключами",
            "Логування використання"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
