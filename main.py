from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Header, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import tempfile
import os
import httpx
import time
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
import logging
from models import LocalTranscriptionService
from middleware import verify_api_key, verify_master_token, verify_master_token_from_query
from api_auth import api_key_manager

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


# Глобальні змінні для сервісу транскрипції та черги
transcription_service = None
task_queue = None
tasks = {}  # task_id -> TaskStatus
executor = None
worker_tasks = []  # Зберігаємо посилання на воркер-таски



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

class GenerateKeyRequest(BaseModel):
    client_name: str

class GenerateKeyResponse(BaseModel):
    api_key: str
    client_name: str
    created_at: str

class DeleteKeyRequest(BaseModel):
    api_key: str

class UpdateKeyNotesRequest(BaseModel):
    api_key: str
    notes: str

class ToggleKeyStatusRequest(BaseModel):
    api_key: str

class APIKeyInfo(BaseModel):
    key: str
    client_name: str
    created_at: str
    active: bool
    usage_count: int
    last_used: Optional[str]
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_processing_time: float
    average_processing_time: float
    notes: str

class TaskStatus(BaseModel):
    task_id: str
    status: str  # queued, processing, completed, failed
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: int  # 0-100
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    file_name: str
    language: str
    model_size: str
    use_diarization: bool

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

@app.on_event("startup")
async def load_models():
    """Завантаження локальних моделей при запуску сервера"""
    global transcription_service, task_queue, executor, worker_tasks
    
    try:
        logger.info("Ініціалізація локального сервісу транскрипції...")
        transcription_service = LocalTranscriptionService()
        
        if transcription_service.load_models():
            logger.info("Локальні моделі завантажені успішно")
        else:
            logger.error("Не вдалося завантажити моделі")
            raise RuntimeError("Моделі не завантажені")
        
        # Виводимо інформацію про master токен
        api_key_manager.print_startup_info()
        
        # Ініціалізуємо чергу та executor
        task_queue = asyncio.Queue()
        executor = ThreadPoolExecutor(max_workers=2)
        
        # Запускаємо воркери для обробки черги
        logger.info("Запуск воркерів для обробки черги транскрипції...")
        for i in range(2):  # Запускаємо 2 воркери
            worker_task = asyncio.create_task(worker())
            worker_tasks.append(worker_task)
            logger.info(f"Воркер {i+1} запущено")
        
    except Exception as e:
        logger.error(f"Помилка при завантаженні моделей: {e}")
        raise RuntimeError(f"Не вдалося ініціалізувати сервіс: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Очищення ресурсів при завершенні сервера"""
    global executor, worker_tasks
    
    logger.info("Завершення роботи сервера...")
    
    # Скасовуємо всі воркер-таски
    for worker_task in worker_tasks:
        worker_task.cancel()
    
    # Чекаємо завершення воркерів
    if worker_tasks:
        await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    # Закриваємо executor
    if executor:
        executor.shutdown(wait=True)
    
    logger.info("Сервер завершив роботу")

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

def save_task_status(task_id: str, task_status: TaskStatus):
    """Збереження статусу задачі в JSON файл з автоматичним очищенням старих задач"""
    try:
        tasks_file = "data/tasks.json"
        os.makedirs("data", exist_ok=True)
        
        # Завантажуємо існуючі задачі
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r', encoding='utf-8') as f:
                all_tasks = json.load(f)
        else:
            all_tasks = {}
        
        # Оновлюємо статус задачі
        all_tasks[task_id] = task_status.dict()
        
        # Очищуємо старі задачі (старші 7 днів)
        cleaned_tasks = clean_old_tasks(all_tasks)
        
        # Зберігаємо назад
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_tasks, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Помилка збереження статусу задачі {task_id}: {e}")

def clean_old_tasks(all_tasks: dict, max_age_days: int = 7) -> dict:
    """Очищення старих задач з файлу"""
    from datetime import datetime, timedelta
    
    try:
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_tasks = {}
        removed_count = 0
        
        for task_id, task_data in all_tasks.items():
            try:
                # Парсимо дату створення задачі
                created_at_str = task_data.get('created_at', '')
                if created_at_str:
                    created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                    
                    # Якщо задача старша за cutoff_date, видаляємо її
                    if created_at < cutoff_date:
                        removed_count += 1
                        continue
                
                # Зберігаємо задачу
                cleaned_tasks[task_id] = task_data
                
            except Exception as e:
                logger.warning(f"Помилка обробки задачі {task_id}: {e}")
                # Якщо не можемо розпарсити дату, зберігаємо задачу
                cleaned_tasks[task_id] = task_data
        
        if removed_count > 0:
            logger.info(f"Автоматично очищено {removed_count} старих задач (старші {max_age_days} днів)")
        
        return cleaned_tasks
        
    except Exception as e:
        logger.error(f"Помилка очищення старих задач: {e}")
        return all_tasks  # Повертаємо оригінальний словник при помилці

def load_task_status(task_id: str) -> Optional[TaskStatus]:
    """Завантаження статусу задачі з JSON файлу"""
    try:
        tasks_file = "data/tasks.json"
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r', encoding='utf-8') as f:
                all_tasks = json.load(f)
                if task_id in all_tasks:
                    return TaskStatus(**all_tasks[task_id])
    except Exception as e:
        logger.error(f"Помилка завантаження статусу задачі {task_id}: {e}")
    return None

def process_transcription_task_sync(task_id: str, file_path: str, language: str, model_size: str, use_diarization: bool, api_key: str):
    """Синхронна обробка задачі транскрипції"""
    try:
        # Оновлюємо статус на "processing"
        task_status = tasks[task_id]
        task_status.status = "processing"
        task_status.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        task_status.progress = 10
        save_task_status(task_id, task_status)
        
        logger.info(f"Початок обробки задачі {task_id}")
        
        # Виконуємо транскрипцію синхронно
        if use_diarization:
            result = transcription_service.transcribe_with_diarization(file_path, language, model_size)
        else:
            result = transcription_service.transcribe_simple(file_path, language, model_size)
        
        # Оновлюємо статус на "completed"
        task_status.status = "completed"
        task_status.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        task_status.progress = 100
        task_status.result = result
        save_task_status(task_id, task_status)
        
        # Логуємо успішне використання API
        processing_time = time.time() - time.mktime(time.strptime(task_status.started_at, "%Y-%m-%d %H:%M:%S"))
        api_key_manager.log_api_usage(api_key, success=True, processing_time=processing_time)
        
        logger.info(f"Задача {task_id} завершена успішно")
        
    except Exception as e:
        # Оновлюємо статус на "failed"
        task_status = tasks[task_id]
        task_status.status = "failed"
        task_status.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        task_status.error = str(e)
        save_task_status(task_id, task_status)
        
        # Логуємо невдале використання API
        processing_time = time.time() - time.mktime(time.strptime(task_status.started_at, "%Y-%m-%d %H:%M:%S"))
        api_key_manager.log_api_usage(api_key, success=False, processing_time=processing_time)
        
        logger.error(f"Помилка обробки задачі {task_id}: {e}")
    
    finally:
        # Видаляємо тимчасовий файл
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Тимчасовий файл видалено: {file_path}")
            except Exception as e:
                logger.warning(f"Не вдалося видалити тимчасовий файл: {e}")

async def worker():
    """Воркер для обробки задач з черги"""
    worker_id = id(asyncio.current_task())
    logger.info(f"Воркер {worker_id} запущено")
    
    while True:
        try:
            # Очікуємо задачу з черги
            task_data = await task_queue.get()
            logger.info(f"Воркер {worker_id} отримав задачу {task_data['task_id']}")
            
            # Обробляємо задачу в окремому потоці
            await asyncio.get_event_loop().run_in_executor(
                executor,
                process_transcription_task_sync,
                task_data['task_id'],
                task_data['file_path'],
                task_data['language'],
                task_data['model_size'],
                task_data['use_diarization'],
                task_data['api_key']
            )
            
            # Позначаємо задачу як виконану
            task_queue.task_done()
            logger.info(f"Воркер {worker_id} завершив задачу {task_data['task_id']}")
            
        except asyncio.CancelledError:
            logger.info(f"Воркер {worker_id} отримав сигнал завершення")
            break
        except Exception as e:
            logger.error(f"Помилка воркера {worker_id}: {e}")
            await asyncio.sleep(1)  # Пауза перед наступною спробою

# Функції транскрипції тепер використовують локальний сервіс

@app.post("/transcribe", response_model=TaskResponse)
async def transcribe_audio_file(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: str = Form("uk"),
    model_size: str = Form("small"),
    use_diarization: bool = Form(False),
    api_key: str = Depends(verify_api_key)
):
    """
    Транскрипція аудіо/відео файлу з визначенням дикторів (асинхронна через чергу)
    
    Параметри:
    - file: Завантажений файл (аудіо або відео)
    - url: URL посилання на файл
    - language: Мова транскрипції (за замовчуванням 'uk' для української)
    - model_size: Розмір моделі Whisper (tiny, base, small, medium, large, auto)
    - use_diarization: Використовувати діаризацію Оператор/Клієнт (True/False)
    
    Повертає task_id для відстеження статусу через /task/{task_id}
    """
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Необхідно надати або файл, або URL")
    
    if file and url:
        raise HTTPException(status_code=400, detail="Надайте або файл, або URL, але не обидва")
    
    # Валідація розміру моделі
    if model_size not in ["tiny", "base", "small", "medium", "large", "auto"]:
        raise HTTPException(status_code=400, detail="Розмір моделі повинен бути: tiny, base, small, medium, large або auto")
    
    # Генеруємо унікальний ID задачі
    task_id = str(uuid.uuid4())
    
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
            file_name = file.filename
            
        elif url:
            # Завантаження файлу з URL
            temp_file_path = await download_file_from_url(url)
            file_name = url.split('/')[-1] if '/' in url else "downloaded_file"
        
        # Створюємо статус задачі
        task_status = TaskStatus(
            task_id=task_id,
            status="queued",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            started_at=None,
            completed_at=None,
            progress=0,
            result=None,
            error=None,
            file_name=file_name,
            language=language,
            model_size=model_size,
            use_diarization=use_diarization
        )
        
        # Зберігаємо статус задачі
        tasks[task_id] = task_status
        save_task_status(task_id, task_status)
        
        # Додаємо задачу в чергу
        await task_queue.put({
            'task_id': task_id,
            'file_path': temp_file_path,
            'language': language,
            'model_size': model_size,
            'use_diarization': use_diarization,
            'api_key': api_key
        })
        
        logger.info(f"Задача {task_id} додана в чергу для файлу {file_name}")
        
        return TaskResponse(
            task_id=task_id,
            status="queued",
            message=f"Файл {file_name} додано в чергу обробки. Використовуйте /task/{task_id} для відстеження статусу."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неочікувана помилка: {e}")
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка сервера: {str(e)}")

@app.post("/transcribe-with-diarization", response_model=TranscriptionResponse)
async def transcribe_with_diarization(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: str = Form("uk"),
    model_size: str = Form("small"),
    api_key: str = Depends(verify_api_key)
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
        start_time = time.time()
        
        try:
            processed_result = transcription_service.transcribe_with_diarization(temp_file_path, language, model_size)
            
            # Логуємо успішне використання
            processing_time = time.time() - start_time
            api_key_manager.log_api_usage(api_key, success=True, processing_time=processing_time)
            
            logger.info("Транскрипція з діаризацією завершена успішно")
            return TranscriptionResponse(**processed_result)
            
        except Exception as e:
            # Логуємо невдале використання
            processing_time = time.time() - start_time
            api_key_manager.log_api_usage(api_key, success=False, processing_time=processing_time)
            raise e
        
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
    queue_size = task_queue.qsize() if task_queue else 0
    active_tasks = len([t for t in tasks.values() if t.status == "processing"])
    max_workers = executor._max_workers if executor else 0
    
    return {
        "status": "healthy",
        "models_loaded": transcription_service is not None and transcription_service.models_loaded,
        "whisper_loaded": transcription_service is not None and transcription_service.whisper_model.model is not None,
        "queue_size": queue_size,
        "active_tasks": active_tasks,
        "max_workers": max_workers,
        "worker_tasks": len(worker_tasks)
    }

@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Отримання статусу задачі транскрипції"""
    # Спочатку перевіряємо в пам'яті
    if task_id in tasks:
        return tasks[task_id]
    
    # Якщо не знайдено в пам'яті, завантажуємо з файлу
    task_status = load_task_status(task_id)
    if task_status:
        return task_status
    
    raise HTTPException(status_code=404, detail="Задача не знайдена")

@app.get("/tasks")
async def list_tasks(limit: int = 50, status: Optional[str] = None):
    """Отримання списку задач з фільтрацією"""
    try:
        tasks_file = "data/tasks.json"
        if not os.path.exists(tasks_file):
            return {"tasks": [], "total": 0}
        
        with open(tasks_file, 'r', encoding='utf-8') as f:
            all_tasks = json.load(f)
        
        # Фільтруємо за статусом якщо вказано
        if status:
            filtered_tasks = {k: v for k, v in all_tasks.items() if v.get('status') == status}
        else:
            filtered_tasks = all_tasks
        
        # Сортуємо за часом створення (найновіші спочатку)
        sorted_tasks = sorted(
            filtered_tasks.items(), 
            key=lambda x: x[1].get('created_at', ''), 
            reverse=True
        )
        
        # Обмежуємо кількість результатів
        limited_tasks = sorted_tasks[:limit]
        
        return {
            "tasks": [TaskStatus(**task_data) for _, task_data in limited_tasks],
            "total": len(filtered_tasks),
            "limit": limit,
            "status_filter": status
        }
        
    except Exception as e:
        logger.error(f"Помилка отримання списку задач: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка отримання списку задач: {str(e)}")

@app.delete("/task/{task_id}")
async def cancel_task(task_id: str, api_key: str = Depends(verify_api_key)):
    """Скасування задачі (тільки якщо вона ще в черзі)"""
    if task_id not in tasks:
        task_status = load_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Задача не знайдена")
        tasks[task_id] = task_status
    
    task_status = tasks[task_id]
    
    if task_status.status == "completed":
        raise HTTPException(status_code=400, detail="Задача вже завершена")
    
    if task_status.status == "processing":
        raise HTTPException(status_code=400, detail="Задача вже обробляється, не можна скасувати")
    
    if task_status.status == "failed":
        raise HTTPException(status_code=400, detail="Задача вже провалена")
    
    # Видаляємо задачу з черги (якщо вона там є)
    # Примітка: це спрощена реалізація, в реальному проекті потрібно більш складну логіку
    task_status.status = "cancelled"
    task_status.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
    save_task_status(task_id, task_status)
    
    return {"message": f"Задача {task_id} скасована"}


# Адмін endpoints
@app.post("/admin/generate-key", response_model=GenerateKeyResponse)
async def generate_api_key(
    request: GenerateKeyRequest,
    master_token: str = Depends(verify_master_token)
):
    """Генерує новий API ключ (потребує master токен)"""
    try:
        api_key = api_key_manager.generate_api_key(request.client_name)
        key_info = api_key_manager.get_api_key_info(api_key)
        
        return GenerateKeyResponse(
            api_key=api_key,
            client_name=key_info["client_name"],
            created_at=key_info["created_at"]
        )
    except Exception as e:
        logger.error(f"Помилка генерації API ключа: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка генерації ключа: {str(e)}")

@app.post("/admin/delete-key")
async def delete_api_key(
    request: DeleteKeyRequest,
    master_token: str = Depends(verify_master_token)
):
    """Видаляє API ключ (потребує master токен)"""
    try:
        success = api_key_manager.delete_api_key(request.api_key)
        if success:
            return {"message": "API ключ успішно видалено"}
        else:
            raise HTTPException(status_code=404, detail="API ключ не знайдено")
    except Exception as e:
        logger.error(f"Помилка видалення API ключа: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка видалення ключа: {str(e)}")

@app.get("/admin/list-keys")
async def list_api_keys(master_token: str = Depends(verify_master_token)):
    """Отримує список всіх API ключів (потребує master токен)"""
    try:
        keys = api_key_manager.list_api_keys()
        stats = api_key_manager.get_stats()
        
        return {
            "keys": keys,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Помилка отримання списку ключів: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка отримання списку: {str(e)}")

@app.post("/admin/update-key-notes")
async def update_key_notes(
    request: UpdateKeyNotesRequest,
    master_token: str = Depends(verify_master_token)
):
    """Оновлює нотатки для API ключа (потребує master токен)"""
    try:
        success = api_key_manager.update_api_key_notes(request.api_key, request.notes)
        if success:
            return {"message": "Нотатки успішно оновлено"}
        else:
            raise HTTPException(status_code=404, detail="API ключ не знайдено")
    except Exception as e:
        logger.error(f"Помилка оновлення нотаток: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка оновлення нотаток: {str(e)}")

@app.post("/admin/toggle-key-status")
async def toggle_key_status(
    request: ToggleKeyStatusRequest,
    master_token: str = Depends(verify_master_token)
):
    """Перемикає статус API ключа (потребує master токен)"""
    try:
        success = api_key_manager.toggle_api_key_status(request.api_key)
        if success:
            key_info = api_key_manager.get_api_key_info(request.api_key)
            status = "активний" if key_info.get("active", True) else "неактивний"
            return {"message": f"API ключ тепер {status}"}
        else:
            raise HTTPException(status_code=404, detail="API ключ не знайдено")
    except Exception as e:
        logger.error(f"Помилка зміни статусу ключа: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка зміни статусу: {str(e)}")

@app.get("/admin/key-details/{api_key}")
async def get_key_details(
    api_key: str,
    master_token: str = Depends(verify_master_token)
):
    """Отримує детальну інформацію про API ключ (потребує master токен)"""
    try:
        key_info = api_key_manager.get_api_key_info(api_key)
        if key_info:
            return {
                "key": api_key,
                "client_name": key_info["client_name"],
                "created_at": key_info["created_at"],
                "active": key_info.get("active", True),
                "usage_count": key_info.get("usage_count", 0),
                "last_used": key_info.get("last_used"),
                "total_requests": key_info.get("total_requests", 0),
                "successful_requests": key_info.get("successful_requests", 0),
                "failed_requests": key_info.get("failed_requests", 0),
                "total_processing_time": round(key_info.get("total_processing_time", 0), 2),
                "average_processing_time": round(key_info.get("total_processing_time", 0) / max(key_info.get("total_requests", 1), 1), 2),
                "notes": key_info.get("notes", "")
            }
        else:
            raise HTTPException(status_code=404, detail="API ключ не знайдено")
    except Exception as e:
        logger.error(f"Помилка отримання деталей ключа: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка отримання деталей: {str(e)}")

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Адмін панель для управління API ключами"""
    # Перевіряємо master токен з query параметра
    master_token = request.query_params.get("master_token")
    if not master_token or not api_key_manager.verify_master_token(master_token):
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Admin Panel - Access Denied</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 50px; text-align: center; }
                .error { color: #d32f2f; background: #ffebee; padding: 20px; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="error">
                <h1>🔒 Access Denied</h1>
                <p>Недійсний або відсутній master токен</p>
                <p>Використовуйте: <code>/admin?master_token=YOUR_MASTER_TOKEN</code></p>
            </div>
        </body>
        </html>
        """, status_code=401)
    
    # Отримуємо список ключів
    try:
        keys = api_key_manager.list_api_keys()
        stats = api_key_manager.get_stats()
    except Exception as e:
        keys = []
        stats = {"total_keys": 0, "active_keys": 0, "inactive_keys": 0}
    
    # Генеруємо HTML
    keys_html = ""
    for key in keys:
        status_class = "active" if key["active"] else "inactive"
        keys_html += f"""
        <tr class="{status_class}">
            <td><code>{key["key"][:20]}...</code></td>
            <td>{key["client_name"]}</td>
            <td>{key["created_at"][:19]}</td>
            <td>
                <button onclick="deleteKey('{key["key"]}')" class="delete-btn">Видалити</button>
            </td>
        </tr>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Admin Panel</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 10px; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; flex: 1; }}
            .stat-number {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
            .form-section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; font-weight: bold; }}
            .active {{ background: #e8f5e8; }}
            .inactive {{ background: #ffe8e8; }}
            input[type="text"] {{ width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
            button {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }}
            .generate-btn {{ background: #4caf50; color: white; }}
            .delete-btn {{ background: #f44336; color: white; }}
            .generate-btn:hover {{ background: #45a049; }}
            .delete-btn:hover {{ background: #da190b; }}
            .new-key {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0; display: none; }}
            .new-key code {{ background: #f0f0f0; padding: 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔑 API Admin Panel</h1>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{stats["total_keys"]}</div>
                    <div>Всього ключів</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats["active_keys"]}</div>
                    <div>Активних</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats["inactive_keys"]}</div>
                    <div>Неактивних</div>
                </div>
            </div>
            
            <div class="form-section">
                <h3>➕ Створити новий API ключ</h3>
                <input type="text" id="clientName" placeholder="Назва клієнта" />
                <button class="generate-btn" onclick="generateKey()">Генерувати ключ</button>
                <div id="newKey" class="new-key"></div>
            </div>
            
            <div class="form-section">
                <h3>📋 Список API ключів</h3>
                <table>
                    <thead>
                        <tr>
                            <th>API Ключ</th>
                            <th>Клієнт</th>
                            <th>Створено</th>
                            <th>Дії</th>
                        </tr>
                    </thead>
                    <tbody>
                        {keys_html}
                    </tbody>
                </table>
            </div>
        </div>
        
        <script>
            async function generateKey() {{
                const clientName = document.getElementById('clientName').value;
                if (!clientName) {{
                    alert('Введіть назву клієнта');
                    return;
                }}
                
                try {{
                    const response = await fetch('/admin/generate-key', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer {master_token}'
                        }},
                        body: JSON.stringify({{ client_name: clientName }})
                    }});
                    
                    if (response.ok) {{
                        const data = await response.json();
                        const newKeyDiv = document.getElementById('newKey');
                        newKeyDiv.innerHTML = `
                            <h4>✅ Новий API ключ створено!</h4>
                            <p><strong>Клієнт:</strong> ${{data.client_name}}</p>
                            <p><strong>API ключ:</strong> <code>${{data.api_key}}</code></p>
                            <p><strong>Створено:</strong> ${{data.created_at}}</p>
                            <p style="color: #d32f2f;"><strong>⚠️ Збережіть цей ключ! Він більше не буде показаний.</strong></p>
                        `;
                        newKeyDiv.style.display = 'block';
                        document.getElementById('clientName').value = '';
                        setTimeout(() => location.reload(), 2000);
                    }} else {{
                        alert('Помилка створення ключа');
                    }}
                }} catch (error) {{
                    alert('Помилка: ' + error.message);
                }}
            }}
            
            async function deleteKey(apiKey) {{
                if (!confirm('Ви впевнені, що хочете видалити цей API ключ?')) {{
                    return;
                }}
                
                try {{
                    const response = await fetch('/admin/delete-key', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer {master_token}'
                        }},
                        body: JSON.stringify({{ api_key: apiKey }})
                    }});
                    
                    if (response.ok) {{
                        alert('API ключ видалено');
                        location.reload();
                    }} else {{
                        alert('Помилка видалення ключа');
                    }}
                }} catch (error) {{
                    alert('Помилка: ' + error.message);
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(html_content)

@app.get("/admin-panel")
async def admin_panel_static():
    """Об'єднана адмін панель з розширеними функціями"""
    return FileResponse("static/admin.html")

@app.get("/transcription")
async def transcription_page():
    """Веб-сторінка для транскрипції аудіо/відео"""
    return FileResponse("static/transcription.html")

@app.get("/api")
async def api_info():
    """Інформація про API"""
    return {
        "message": "Ukrainian Audio Transcription API (Local Models)",
        "version": "1.0.0",
        "description": "API для транскрипції українського аудіо/відео з визначенням дикторів (локальні моделі)",
        "endpoints": {
            "transcribe": "/transcribe (POST, requires API key, returns task_id)",
            "transcribe_with_diarization": "/transcribe-with-diarization (POST, requires API key)",
            "task_status": "/task/{task_id} (GET, public, check task status)",
            "list_tasks": "/tasks (GET, public, list all tasks with filtering)",
            "cancel_task": "/task/{task_id} (DELETE, requires API key, cancel queued task)",
            "health": "/health (GET, public, includes queue status)",
            "docs": "/docs (GET, public)",
            "api_info": "/api (GET, public)",
            "admin": "/admin (GET, requires master token)",
            "admin_panel": "/admin-panel (GET, unified admin panel with advanced features)",
            "transcription": "/transcription (GET, web interface for audio/video transcription)",
            "admin_generate_key": "/admin/generate-key (POST, requires master token)",
            "admin_delete_key": "/admin/delete-key (POST, requires master token)",
            "admin_list_keys": "/admin/list-keys (GET, requires master token)",
            "admin_update_notes": "/admin/update-key-notes (POST, requires master token)",
            "admin_toggle_status": "/admin/toggle-key-status (POST, requires master token)",
            "admin_key_details": "/admin/key-details/{api_key} (GET, requires master token)"
        },
        "features": [
            "Локальна транскрипція (faster-whisper)",
            "Quantized моделі для CPU",
            "Проста діаризація Оператор/Клієнт (WebRTC VAD)",
            "Підтримка файлів та URL",
            "Українська мова",
            "Оптимізація для CPU та GPU",
            "Система API токенів"
        ],
        "supported_formats": [
            "Аудіо: WAV, MP3, M4A, FLAC, OGG",
            "Відео: MP4, AVI, MOV, MKV"
        ],
        "model_sizes": ["tiny", "base", "small", "medium", "large", "auto"],
        "languages": ["uk", "en", "ru", "pl", "de", "fr", "es", "it"],
        "note": "Для використання API потрібен API ключ. Отримайте його у адміністратора."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
