#!/bin/bash

# Скрипт для ініціалізації Git репозиторію та завантаження на GitHub

echo "🚀 Налаштування Git репозиторію для Transcription API"
echo "=================================================="

# Перевірка наявності Git
if ! command -v git &> /dev/null; then
    echo "❌ Git не встановлений. Встановіть Git спочатку:"
    echo "   macOS: xcode-select --install"
    echo "   Ubuntu/Debian: sudo apt install git"
    echo "   Windows: https://git-scm.com/download/win"
    exit 1
fi

# Ініціалізація Git репозиторію
echo "📁 Ініціалізація Git репозиторію..."
git init

# Додавання файлів
echo "📝 Додавання файлів до репозиторію..."
git add .

# Перший коміт
echo "💾 Створення першого коміту..."
git commit -m "Initial commit: Ukrainian Audio Transcription API

- Додано FastAPI додаток для транскрипції аудіо
- Підтримка української мови з faster-whisper
- Діаризація дикторів (Оператор/Клієнт)
- Оптимізації для CPU/GPU
- Веб-інтерфейс для тестування
- Автоматичне завантаження моделей"

echo ""
echo "✅ Git репозиторій ініціалізовано!"
echo ""
echo "📋 Наступні кроки:"
echo "1. Створіть репозиторій на GitHub: https://github.com/new"
echo "2. Назвіть репозиторій: transcription-api"
echo "3. Зробіть репозиторій публічним (рекомендовано)"
echo "4. НЕ додавайте README, .gitignore або ліцензію (вони вже є)"
echo "5. Виконайте команди нижче:"
echo ""
echo "git remote add origin https://github.com/WarmingZ/transcription-api.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "🎯 Після цього ваш код буде доступний на GitHub!"
echo ""
echo "📖 Для детальної інструкції дивіться SETUP.md"
