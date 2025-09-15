"""
Конфігурація та константи для моделей
"""

import os
from pathlib import Path

# Директорія для локальних моделей
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Налаштування логування
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Імпорт LanguageTool для орфографічної корекції
try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
    logger.info("LanguageTool доступний для орфографічної корекції")
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    logger.warning("LanguageTool недоступний - орфографічна корекція вимкнена")

# Константи для оптимізації (налаштовано для сервера 8GB RAM + 4 CPU AMD)
DEFAULT_CHUNK_SIZE = 60  # секунд (оптимізовано для small моделі)
MAX_WORKERS = 2  # Оптимізовано для стабільності (50% ядер)
CACHE_MAX_SIZE = 3  # Оптимізовано для small моделі

# GPU оптимізації
GPU_CHUNK_SIZE = 120  # секунд (більші чанки для GPU)
GPU_MAX_WORKERS = 4  # Більше процесів для GPU
GPU_CACHE_MAX_SIZE = 5  # Більший кеш для GPU

# Налаштування діаризації
ENABLE_DIARIZATION = True  # Можна відключити для економії ресурсів
DIARIZATION_MAX_WORKERS = 2  # Обмежена кількість процесів для діаризації

# Оптимізовані параметри для швидкості
SPEED_OPTIMIZED_BEAM_SIZE = 1  # Завжди 1 для максимальної швидкості
SPEED_OPTIMIZED_VAD = True     # Увімкнено для кращого виявлення початку мовлення
SPEED_OPTIMIZED_CHUNK_SIZES = {
    'short': 45,    # < 5 хв (збільшено для кращого використання CPU)
    'medium': 90,   # 5-30 хв (збільшено для кращого використання CPU)
    'long': 120     # > 30 хв (збільшено для кращого використання CPU)
}

# Доступні моделі для різних мов
# Для української мови: tiny, base, small, medium, large
# Для англійської мови: також distil-small.en, distil-medium.en
SUPPORTED_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Quantized моделі (рекомендовані ChatGPT для кращої продуктивності)
# faster-whisper автоматично використовує quantized версії при compute_type="int8"
QUANTIZED_MODELS = {
    'tiny': 'tiny',    # tiny + int8 = quantized
    'base': 'base',    # base + int8 = quantized  
    'small': 'small',  # small + int8 = quantized (рекомендована ChatGPT)
    'medium': 'medium', # medium + int8 = quantized
    'large': 'large'   # large + int8 = quantized
}

# Налаштування compute_type для різних пристроїв
CPU_COMPUTE_TYPE = "int8"  # Quantized для CPU (швидше та економніше)
GPU_COMPUTE_TYPE = "float16"  # Float16 для GPU
