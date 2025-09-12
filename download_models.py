#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для завантаження моделей після клонування репозиторію
"""

import os
import logging
from models import LocalTranscriptionService

def download_models():
    """Завантажуємо моделі автоматично"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Початок завантаження моделей...")
    
    try:
        # Створюємо сервіс транскрипції
        service = LocalTranscriptionService()
        
        # Завантажуємо моделі
        logger.info("📥 Завантаження faster-whisper моделей...")
        service.load_models()
        
        logger.info("✅ Всі моделі завантажені успішно!")
        logger.info("🎯 Моделі збережені в:")
        logger.info("   - models/faster-whisper-medium/")
        logger.info("   - models/faster-whisper-small/")
        
    except Exception as e:
        logger.error(f"❌ Помилка завантаження моделей: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("📦 Завантаження моделей для Ukrainian Transcription API")
    print("=" * 60)
    
    if download_models():
        print("\n🎉 Готово! Моделі завантажені.")
        print("🚀 Тепер можна запускати: python main.py")
    else:
        print("\n❌ Помилка завантаження моделей!")
        print("💡 Спробуйте запустити скрипт ще раз")
