#!/usr/bin/env python3
"""
Скрипт для попереднього завантаження моделей
"""
import os
import sys
import logging
from pathlib import Path

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_whisper_model(model_size="small"):
    """Завантаження faster-whisper моделі в локальну директорію проекту"""
    try:
        from faster_whisper import WhisperModel
        
        # Створюємо директорію для моделей в проекті
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Завантаження faster-whisper {model_size} моделі в {models_dir}...")
        model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8",
            download_root=str(models_dir)  # Завантажуємо в локальну директорію
        )
        logger.info(f"✅ faster-whisper {model_size} модель завантажена успішно в {models_dir}")
        return True
    except Exception as e:
        logger.error(f"❌ Помилка завантаження faster-whisper {model_size}: {e}")
        return False

def download_all_whisper_models():
    """Завантаження всіх підтримуваних Whisper моделей"""
    models = ["small", "medium"]
    results = {}
    
    for model_size in models:
        logger.info(f"Завантаження {model_size} моделі...")
        results[model_size] = download_whisper_model(model_size)
    
    return results

def download_speechbrain_model():
    """Завантаження SpeechBrain моделі"""
    try:
        from speechbrain.pretrained import EncoderClassifier
        logger.info("Завантаження SpeechBrain моделі...")
        
        # Створюємо директорію для моделей
        models_dir = Path("pretrained_models")
        models_dir.mkdir(exist_ok=True)
        
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(models_dir / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cpu"}  # Використовуємо CPU для завантаження
        )
        logger.info("✅ SpeechBrain модель завантажена успішно")
        return True
    except Exception as e:
        logger.error(f"❌ Помилка завантаження SpeechBrain: {e}")
        return False

def check_dependencies():
    """Перевірка залежностей"""
    logger.info("Перевірка залежностей...")
    
    required_packages = [
        "torch",
        "torchaudio", 
        "faster_whisper",
        "speechbrain",
        "librosa",
        "scikit-learn",
        "numpy",
        "scipy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error(f"Відсутні пакети: {', '.join(missing_packages)}")
        logger.error("Встановіть їх командою: pip install -r requirements.txt")
        return False
    
    logger.info("✅ Всі залежності встановлені")
    return True

def main():
    """Головна функція"""
    logger.info("🚀 Запуск завантаження моделей")
    logger.info("=" * 50)
    
    # Перевірка залежностей
    if not check_dependencies():
        sys.exit(1)
    
    # Завантаження моделей
    logger.info("Завантаження faster-whisper моделей (small та medium)...")
    whisper_results = download_all_whisper_models()
    speechbrain_ok = download_speechbrain_model()
    
    logger.info("=" * 50)
    
    # Перевіряємо результати
    whisper_ok = any(whisper_results.values())  # Хоча б одна модель завантажилась
    
    if whisper_ok and speechbrain_ok:
        logger.info("🎉 Всі моделі завантажені успішно!")
        logger.info("Доступні faster-whisper моделі:")
        for model, success in whisper_results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {model}")
        logger.info("🚀 faster-whisper працює в 4-10 разів швидше ніж OpenAI Whisper!")
        logger.info(f"📁 Моделі збережені в локальній директорії: {Path('models').absolute()}")
        logger.info("Тепер ви можете запустити API: python main.py")
    else:
        logger.error("⚠️ Деякі моделі не завантажились")
        if not whisper_ok:
            logger.error("- Жодна faster-whisper модель не завантажена")
        if not speechbrain_ok:
            logger.error("- SpeechBrain модель не завантажена")
        sys.exit(1)

if __name__ == "__main__":
    main()
