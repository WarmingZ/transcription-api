from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import tempfile
import os
import httpx
from typing import Optional, List, Dict, Any
import logging
from models import LocalTranscriptionService

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


# Глобальна змінна для сервісу транскрипції
transcription_service = None



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
    use_diarization: bool = Form(False)
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
        logger.info(f"📝 Параметри запиту: model_size={model_size}, language={language}, use_diarization={use_diarization}")
        if use_diarization:
            logger.info(f"Початок транскрипції з діаризацією файлу: {temp_file_path}")
            processed_result = transcription_service.transcribe_with_diarization(temp_file_path, language, model_size)
        else:
            logger.info(f"Початок простої транскрипції файлу: {temp_file_path}")
            processed_result = transcription_service.transcribe_simple(temp_file_path, language, model_size)
        
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
    model_size: str = Form("small")
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
        logger.info(f"📝 Параметри запиту: model_size={model_size}, language={language}")
        logger.info(f"Початок транскрипції з діаризацією файлу: {temp_file_path}")
        processed_result = transcription_service.transcribe_with_diarization(temp_file_path, language, model_size)
        
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


@app.get("/api")
async def api_info():
    """Інформація про API"""
    return {
        "message": "Ukrainian Audio Transcription API (Local Models)",
        "version": "1.0.0",
        "description": "API для транскрипції українського аудіо/відео з визначенням дикторів (локальні моделі)",
        "endpoints": {
            "transcribe": "/transcribe (POST, public)",
            "transcribe_with_diarization": "/transcribe-with-diarization (POST, public)",
            "health": "/health (GET, public)",
            "docs": "/docs (GET, public)",
            "api_info": "/api (GET, public)"
        },
        "features": [
            "Локальна транскрипція (faster-whisper)",
            "Проста діаризація Оператор/Клієнт (WebRTC VAD)",
            "Підтримка файлів та URL",
            "Українська мова",
            "Оптимізація для CPU та GPU"
        ],
        "supported_formats": [
            "Аудіо: WAV, MP3, M4A, FLAC, OGG",
            "Відео: MP4, AVI, MOV, MKV"
        ],
        "model_sizes": ["tiny", "base", "small", "medium", "large", "auto"],
        "languages": ["uk", "en", "ru", "pl", "de", "fr", "es", "it"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
