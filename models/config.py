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

# Константи для оптимізації
DEFAULT_CHUNK_SIZE = 60  # секунд (збільшено для уникнення проблем з короткими файлами)
MAX_WORKERS = 16
CACHE_MAX_SIZE = 3

# Оптимізовані параметри для швидкості
SPEED_OPTIMIZED_BEAM_SIZE = 1  # Завжди 1 для максимальної швидкості
SPEED_OPTIMIZED_VAD = True     # Увімкнено для кращого виявлення початку мовлення
SPEED_OPTIMIZED_CHUNK_SIZES = {
    'short': 60,    # < 5 хв (збільшено з 20 до 60)
    'medium': 120,  # 5-30 хв (збільшено з 30 до 120)
    'long': 180     # > 30 хв (збільшено з 45 до 180)
}

# Доступні моделі для різних мов
# Для української мови: tiny, base, small, medium, large
# Для англійської мови: також distil-small.en, distil-medium.en
SUPPORTED_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
