#!/bin/bash

# Скрипт для налаштування сервера після деплою

echo "🚀 Налаштування сервера після деплою..."

# Створюємо директорію для даних
mkdir -p data

# Перевіряємо чи існують файли конфігурації
if [ ! -f "data/master_token.txt" ]; then
    echo "📝 Створюємо master токен..."
    # Master токен буде створений автоматично при першому запуску
fi

if [ ! -f "data/api_keys.json" ]; then
    echo "📄 Створюємо файл API ключів..."
    echo "{}" > data/api_keys.json
fi

# Встановлюємо права доступу
chmod 600 data/master_token.txt 2>/dev/null || true
chmod 600 data/api_keys.json 2>/dev/null || true

echo "✅ Налаштування завершено!"
echo ""
echo "📋 Наступні кроки:"
echo "1. Запустіть сервер: python main.py"
echo "2. Подивіться логи - там буде master токен"
echo "3. Відкрийте адмін панель: http://your-server:8000/admin-panel"
echo "4. Введіть master токен для доступу"
echo ""
echo "🔒 Файли data/ не будуть завантажені на GitHub (додано в .gitignore)"
