#!/usr/bin/env python3
"""
Скрипт для перевірки завантажених моделей
"""
import os
from pathlib import Path
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_models():
    """Перевіряє завантажені моделі в проекті"""
    models_dir = Path("models")
    
    print("🔍 Перевірка моделей faster-whisper...")
    print("=" * 50)
    
    if not models_dir.exists():
        print("❌ Директорія models/ не знайдена")
        print("💡 Запустіть: python setup_models.py")
        return
    
    print(f"📁 Директорія моделей: {models_dir.absolute()}")
    print()
    
    # Перевіряємо faster-whisper моделі
    faster_whisper_models = ["small", "medium"]
    
    for model_size in faster_whisper_models:
        model_path = models_dir / f"faster-whisper-{model_size}"
        
        if model_path.exists():
            # Підраховуємо розмір
            total_size = 0
            file_count = 0
            
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            size_mb = total_size / (1024 * 1024)
            print(f"✅ {model_size}: {size_mb:.1f} MB ({file_count} файлів)")
        else:
            print(f"❌ {model_size}: не завантажена")
    
    print()
    
    # Перевіряємо SpeechBrain моделі
    speechbrain_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
    
    if speechbrain_dir.exists():
        total_size = 0
        file_count = 0
        
        for file_path in speechbrain_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        size_mb = total_size / (1024 * 1024)
        print(f"✅ SpeechBrain: {size_mb:.1f} MB ({file_count} файлів)")
    else:
        print("❌ SpeechBrain: не завантажена")
    
    print()
    print("💡 Для завантаження моделей запустіть: python setup_models.py")

if __name__ == "__main__":
    check_models()
