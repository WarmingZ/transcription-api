#!/usr/bin/env python3
"""
Скрипт для управління діаризацією та оптимізації ресурсів сервера
"""

import os
import sys
from pathlib import Path

def check_diarization_status():
    """Перевіряє поточний стан діаризації"""
    print("🔍 ПЕРЕВІРКА СТАНУ ДІАРИЗАЦІЇ")
    print("=" * 50)
    
    # Читаємо конфігурацію
    config_path = Path("models/config.py")
    if not config_path.exists():
        print("❌ Файл конфігурації не знайдено")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Перевіряємо налаштування
    if "ENABLE_DIARIZATION = True" in content:
        print("✅ Діаризація УВІМКНЕНА")
    elif "ENABLE_DIARIZATION = False" in content:
        print("❌ Діаризація ВИМКНЕНА")
    else:
        print("⚠️ Налаштування діаризації не знайдено")
    
    # Перевіряємо кількість worker процесів
    if "DIARIZATION_MAX_WORKERS = 2" in content:
        print("🔧 Максимум 2 worker процеси для діаризації")
    elif "DIARIZATION_MAX_WORKERS = 1" in content:
        print("🔧 Максимум 1 worker процес для діаризації")
    else:
        print("⚠️ Налаштування worker процесів не знайдено")
    
    print()

def disable_diarization():
    """Відключає діаризацію для економії ресурсів"""
    print("🔧 ВІДКЛЮЧЕННЯ ДІАРИЗАЦІЇ")
    print("=" * 50)
    
    config_path = Path("models/config.py")
    if not config_path.exists():
        print("❌ Файл конфігурації не знайдено")
        return
    
    # Читаємо поточний вміст
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Замінюємо налаштування
    if "ENABLE_DIARIZATION = True" in content:
        content = content.replace("ENABLE_DIARIZATION = True", "ENABLE_DIARIZATION = False")
        
        # Зберігаємо зміни
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Діаризація відключена")
        print("💡 Перезапустіть сервер для застосування змін")
    else:
        print("⚠️ Діаризація вже відключена або налаштування не знайдено")
    
    print()

def enable_diarization():
    """Включає діаризацію"""
    print("🔧 ВКЛЮЧЕННЯ ДІАРИЗАЦІЇ")
    print("=" * 50)
    
    config_path = Path("models/config.py")
    if not config_path.exists():
        print("❌ Файл конфігурації не знайдено")
        return
    
    # Читаємо поточний вміст
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Замінюємо налаштування
    if "ENABLE_DIARIZATION = False" in content:
        content = content.replace("ENABLE_DIARIZATION = False", "ENABLE_DIARIZATION = True")
        
        # Зберігаємо зміни
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Діаризація включена")
        print("💡 Перезапустіть сервер для застосування змін")
    else:
        print("⚠️ Діаризація вже включена або налаштування не знайдено")
    
    print()

def optimize_diarization():
    """Оптимізує налаштування діаризації для сервера 8GB RAM + 4 CPU AMD"""
    print("🚀 ОПТИМІЗАЦІЯ ДІАРИЗАЦІЇ")
    print("=" * 50)
    
    config_path = Path("models/config.py")
    if not config_path.exists():
        print("❌ Файл конфігурації не знайдено")
        return
    
    # Читаємо поточний вміст
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Оптимізуємо налаштування
    optimizations = [
        ("DIARIZATION_MAX_WORKERS = 2", "DIARIZATION_MAX_WORKERS = 1"),  # Менше процесів
    ]
    
    changes_made = False
    for old, new in optimizations:
        if old in content:
            content = content.replace(old, new)
            changes_made = True
            print(f"✅ {old} → {new}")
    
    if changes_made:
        # Зберігаємо зміни
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n💡 Оптимізація завершена")
        print("💡 Перезапустіть сервер для застосування змін")
    else:
        print("⚠️ Немає змін для оптимізації")
    
    print()

def show_diarization_info():
    """Показує інформацію про діаризацію"""
    print("📋 ІНФОРМАЦІЯ ПРО ДІАРИЗАЦІЮ")
    print("=" * 50)
    
    print("🎯 Що таке діаризація:")
    print("   • Розділення аудіо на сегменти за дикторами")
    print("   • Призначення ролей 'Оператор' та 'Клієнт'")
    print("   • Використовує WebRTC VAD для виявлення мовлення")
    
    print("\n⚡ Навантаження на сервер:")
    print("   • Додаткове використання RAM для VAD")
    print("   • Паралельна обробка сегментів")
    print("   • Додаткові процеси для аналізу аудіо")
    
    print("\n🔧 Оптимізації:")
    print("   • Lazy loading - ініціалізується тільки при потребі")
    print("   • Обмежена кількість worker процесів")
    print("   • Зменшена агресивність VAD")
    print("   • Можливість повного відключення")
    
    print("\n📊 Рекомендації для сервера 8GB RAM + 4 CPU AMD:")
    print("   • Діаризація включена: 1-2 worker процеси")
    print("   • Діаризація відключена: економія ~500MB RAM")
    print("   • Використовуйте тільки при необхідності")
    
    print()

def show_usage_stats():
    """Показує статистику використання діаризації"""
    print("📊 СТАТИСТИКА ВИКОРИСТАННЯ")
    print("=" * 50)
    
    # Перевіряємо чи є логи
    log_files = list(Path(".").glob("*.log"))
    if log_files:
        print(f"📁 Знайдено {len(log_files)} лог файлів")
        for log_file in log_files[:3]:  # Показуємо тільки перші 3
            print(f"   • {log_file.name}")
    else:
        print("📁 Лог файли не знайдено")
    
    print("\n💡 Для моніторингу використання:")
    print("   • Перевірте логи сервера")
    print("   • Використовуйте htop або top для моніторингу CPU/RAM")
    print("   • Запустіть сервер з --log-level debug")
    
    print()

def main():
    """Головна функція"""
    print("🔧 УПРАВЛІННЯ ДІАРИЗАЦІЄЮ")
    print("=" * 60)
    print("Сервер: 8GB RAM + 4 CPU AMD")
    print("=" * 60)
    
    while True:
        print("\n📋 ВИБЕРІТЬ ДІЮ:")
        print("1. Перевірити стан діаризації")
        print("2. Відключити діаризацію (економія ресурсів)")
        print("3. Включити діаризацію")
        print("4. Оптимізувати налаштування")
        print("5. Показати інформацію про діаризацію")
        print("6. Показати статистику використання")
        print("7. Вихід")
        
        choice = input("\nВведіть номер (1-7): ").strip()
        
        if choice == "1":
            check_diarization_status()
        elif choice == "2":
            disable_diarization()
        elif choice == "3":
            enable_diarization()
        elif choice == "4":
            optimize_diarization()
        elif choice == "5":
            show_diarization_info()
        elif choice == "6":
            show_usage_stats()
        elif choice == "7":
            print("👋 До побачення!")
            break
        else:
            print("❌ Невірний вибір. Спробуйте ще раз.")

if __name__ == "__main__":
    main()
