#!/usr/bin/env python3
"""
Скрипт для запуску тестів системи транскрипції
"""

import sys
import os
import unittest
from pathlib import Path

# Додаємо поточну директорію до шляху Python
sys.path.insert(0, str(Path(__file__).parent))

def run_tests():
    """Запускає всі тести"""
    print("🧪 ЗАПУСК ТЕСТІВ СИСТЕМИ ТРАНСКРИПЦІЇ")
    print("=" * 50)
    
    # Завантажуємо тести
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('test_transcription')
    
    # Запускаємо тести
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Виводимо результати
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТИ ТЕСТУВАННЯ:")
    print(f"✅ Успішних тестів: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Невдалих тестів: {len(result.failures)}")
    print(f"💥 Помилок: {len(result.errors)}")
    print(f"📈 Загалом тестів: {result.testsRun}")
    
    if result.failures:
        print("\n❌ НЕВДАЛІ ТЕСТИ:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ПОМИЛКИ:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Повертаємо код виходу
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
