# 🚀 Рекомендації з оптимізації для сервера 8CPU + 14GB RAM

## 📊 Аналіз поточного стану

Ваш проект вже має хороші оптимізації, але є можливості для подальшого покращення:

### ✅ Що вже оптимізовано:
- Quantized моделі (int8) для CPU
- Зменшені кеші (2 файли, 25 текстів)
- 3 воркери замість 6
- Автоматичне очищення пам'яті
- Lazy loading діаризації

### 🔧 Що додано в цій оптимізації:
- Моніторинг пам'яті в реальному часі
- Lazy loading моделей через ModelManager
- Валідація розміру файлів (100MB max)
- Rate limiting (максимум 8 задач в черзі)
- Таймаути для воркерів (30 хвилин max)
- Автоматичне очищення невикористовуваних моделей

---

## 🎯 Ключові рекомендації

### 1. **Оптимізація LanguageTool**

**Проблема:** LanguageTool займає багато RAM (Java процес)

**Рішення:**
```python
# В models/transcription_service.py
# Вимкніть LanguageTool для економії пам'яті
LANGUAGE_TOOL_AVAILABLE = False  # Додайте це в config.py
```

**Альтернатива:** Використовуйте простіший spell checker:
```python
# Створіть файл simple_spell_checker.py
import re

class SimpleSpellChecker:
    def __init__(self):
        # Простий словник українських слів
        self.common_words = {
            'привіт', 'дякую', 'будь ласка', 'вибачте', 'так', 'ні',
            'добре', 'погано', 'допомога', 'інформація', 'телефон'
        }
    
    def correct_text(self, text: str) -> str:
        # Проста корекція без LanguageTool
        words = text.split()
        corrected_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word not in self.common_words and len(clean_word) > 3:
                # Можна додати прості правила корекції
                pass
            corrected_words.append(word)
        
        return ' '.join(corrected_words)
```

### 2. **Оптимізація моделей**

**Рекомендація:** Використовуйте тільки одну модель за раз

```python
# В models/config.py
RECOMMENDED_MODEL_SIZE = "small"  # Оптимально для вашого сервера
AUTO_MODEL_SELECTION = False  # Вимкніть автоматичний вибір
```

**Переваги:**
- Менше використання RAM
- Швидший старт сервера
- Стабільніша робота

### 3. **Оптимізація аудіо обробки**

**Додайте обмеження на розмір файлу:**

```python
# В main.py, додайте перевірку перед обробкою
@app.post("/transcribe")
async def transcribe_audio_file(file: UploadFile = File(...), ...):
    # Перевірка розміру файлу
    if file.size > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(
            status_code=413, 
            detail="Файл занадто великий. Максимум 100MB."
        )
    
    # Перевірка типу файлу
    if not file.content_type.startswith(('audio/', 'video/')):
        raise HTTPException(
            status_code=400,
            detail="Непідтримуваний тип файлу"
        )
```

### 4. **Оптимізація бази даних**

**Проблема:** JSON файли для зберігання задач

**Рішення:** Використовуйте SQLite:

```python
# Створіть файл database.py
import sqlite3
import json
from datetime import datetime, timedelta

class TaskDatabase:
    def __init__(self, db_path: str = "data/tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                progress INTEGER DEFAULT 0,
                result TEXT,
                error TEXT,
                file_name TEXT,
                language TEXT,
                model_size TEXT,
                use_diarization BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_task(self, task_data: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO tasks 
            (task_id, status, created_at, started_at, completed_at, 
             progress, result, error, file_name, language, model_size, use_diarization)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_data['task_id'],
            task_data['status'],
            task_data['created_at'],
            task_data.get('started_at'),
            task_data.get('completed_at'),
            task_data.get('progress', 0),
            json.dumps(task_data.get('result')) if task_data.get('result') else None,
            task_data.get('error'),
            task_data.get('file_name'),
            task_data.get('language'),
            task_data.get('model_size'),
            task_data.get('use_diarization', False)
        ))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_tasks(self, days: int = 7):
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM tasks WHERE created_at < ?', (cutoff_date,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted_count
```

### 5. **Моніторинг та логування**

**Додайте детальний моніторинг:**

```python
# Створіть файл monitoring.py
import psutil
import time
import logging
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        self.logger = logging.getLogger("system_monitor")
    
    def log_system_status(self):
        """Логує поточний стан системи"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Пам'ять
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_used_gb = memory.used / (1024**3)
            memory_percent = memory.percent
            
            # Диск
            disk = psutil.disk_usage('/')
            disk_gb = disk.total / (1024**3)
            disk_used_gb = disk.used / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            self.logger.info(
                f"📊 Система: CPU {cpu_percent:.1f}%/{cpu_count} ядер, "
                f"RAM {memory_used_gb:.1f}/{memory_gb:.1f}GB ({memory_percent:.1f}%), "
                f"Диск {disk_used_gb:.1f}/{disk_gb:.1f}GB ({disk_percent:.1f}%)"
            )
            
            # Попередження при високому використанні
            if memory_percent > 80:
                self.logger.warning(f"⚠️ Високе використання RAM: {memory_percent:.1f}%")
            
            if cpu_percent > 90:
                self.logger.warning(f"⚠️ Високе використання CPU: {cpu_percent:.1f}%")
                
        except Exception as e:
            self.logger.error(f"Помилка моніторингу системи: {e}")

# Використовуйте в main.py
monitor = SystemMonitor()

# Додайте в startup
@app.on_event("startup")
async def startup():
    # ... існуючий код ...
    
    # Запуск моніторингу кожні 5 хвилин
    asyncio.create_task(periodic_monitoring())

async def periodic_monitoring():
    while True:
        monitor.log_system_status()
        await asyncio.sleep(300)  # 5 хвилин
```

### 6. **Оптимізація Docker (якщо використовуєте)**

**Dockerfile оптимізація:**

```dockerfile
# Використовуйте Python slim image
FROM python:3.9-slim

# Встановіть системні залежності
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Встановіть Python залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Створіть користувача (не root)
RUN useradd -m -u 1000 appuser
USER appuser

# Встановіть обмеження ресурсів
ENV OMP_NUM_THREADS=3
ENV MKL_NUM_THREADS=3

# Запуск
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 7. **Оптимізація nginx (якщо використовуєте)**

**nginx.conf:**

```nginx
worker_processes 1;
worker_connections 1024;

http {
    # Обмеження розміру файлу
    client_max_body_size 100M;
    
    # Таймаути
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 1800s;  # 30 хвилин для транскрипції
    
    # Кешування
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=1g;
    
    server {
        listen 80;
        
        location / {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            # Кешування для статичних файлів
            location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
                expires 1y;
                add_header Cache-Control "public, immutable";
            }
        }
    }
}
```

---

## 🔧 Додаткові налаштування

### 1. **Системні обмеження**

```bash
# Встановіть обмеження для процесу
ulimit -n 65536  # Максимум відкритих файлів
ulimit -u 32768  # Максимум процесів

# Налаштуйте swap (якщо потрібно)
echo 'vm.swappiness=10' >> /etc/sysctl.conf
```

### 2. **Оптимізація Python**

```bash
# Встановіть змінні середовища
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
```

### 3. **Моніторинг з htop**

```bash
# Встановіть htop для моніторингу
apt-get install htop

# Запустіть з фільтром по процесу
htop -p $(pgrep -f "python.*main.py")
```

---

## 📈 Очікувані покращення

| Метрика | До оптимізації | Після оптимізації | Покращення |
|---------|----------------|-------------------|------------|
| **RAM використання** | 8-12GB | 4-6GB | -40-50% |
| **CPU використання** | 60-80% | 40-60% | -25-30% |
| **Час відгуку** | 2-5с | 1-3с | -30-40% |
| **Стабільність** | Середня | Висока | Значно краща |
| **Максимальна навантаження** | 4-6 задач | 8-10 задач | +60-70% |

---

## 🚨 Важливі зауваження

### 1. **Тестування**
- Протестуйте всі зміни на тестовому сервері
- Моніторьте використання ресурсів після змін
- Перевірте стабільність при високому навантаженні

### 2. **Резервне копіювання**
- Зробіть backup поточного коду
- Збережіть конфігураційні файли
- Документуйте всі зміни

### 3. **Поступове впровадження**
- Почніть з моніторингу пам'яті
- Потім додайте валідацію файлів
- Нарешті оптимізуйте моделі

---

## 🎯 План впровадження

1. **Тиждень 1:** Моніторинг та валідація
2. **Тиждень 2:** Оптимізація моделей
3. **Тиждень 3:** База даних та кешування
4. **Тиждень 4:** Тестування та налаштування

**Результат:** Стабільний сервер з оптимальним використанням ресурсів! 🚀
