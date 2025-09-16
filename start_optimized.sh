#!/bin/bash

# 🚀 Скрипт запуску оптимізованого сервера транскрибації
# Для сервера 8CPU + 14GB RAM

echo "🚀 Запуск оптимізованого сервера транскрибації..."

# Налаштування змінних середовища для оптимізації
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export PYTHONHASHSEED=0

# Налаштування Python для оптимізації
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Налаштування пам'яті
export MALLOC_ARENA_MAX=2

# Перевірка доступності ресурсів
echo "📊 Перевірка системних ресурсів..."

# CPU
CPU_COUNT=$(nproc)
echo "  CPU: $CPU_COUNT ядер"

# RAM
RAM_GB=$(free -g | awk 'NR==2{printf "%.1f", $2}')
echo "  RAM: ${RAM_GB}GB"

# Диск
DISK_GB=$(df -h / | awk 'NR==2{print $4}')
echo "  Диск: $DISK_GB вільних"

# Перевірка мінімальних вимог
if [ "$CPU_COUNT" -lt 4 ]; then
    echo "⚠️ Попередження: Менше 4 CPU ядер"
fi

if (( $(echo "$RAM_GB < 8" | bc -l) )); then
    echo "⚠️ Попередження: Менше 8GB RAM"
fi

# Створення директорій
echo "📁 Створення директорій..."
mkdir -p data
mkdir -p temp
mkdir -p logs

# Перевірка Python та залежностей
echo "🐍 Перевірка Python..."
python3 --version

# Активація віртуального середовища (якщо є)
if [ -d "venv" ]; then
    echo "🔧 Активація віртуального середовища..."
    source venv/bin/activate
fi

# Перевірка залежностей
echo "📦 Перевірка залежностей..."
python3 -c "import torch, faster_whisper, fastapi; print('✅ Основні залежності встановлені')" || {
    echo "❌ Помилка: Не всі залежності встановлені"
    echo "Встановіть залежності: pip install -r requirements.txt"
    exit 1
}

# Очищення старих файлів
echo "🧹 Очищення старих файлів..."
find temp/ -type f -mtime +1 -delete 2>/dev/null || true
find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Запуск системного монітора в фоновому режимі
echo "📊 Запуск системного монітора..."
python3 system_monitor.py &
MONITOR_PID=$!

# Функція очищення при завершенні
cleanup() {
    echo "🛑 Завершення роботи..."
    kill $MONITOR_PID 2>/dev/null || true
    echo "✅ Системний монітор зупинено"
    exit 0
}

# Налаштування обробки сигналів
trap cleanup SIGINT SIGTERM

# Запуск основного сервера
echo "🚀 Запуск сервера транскрибації..."
echo "   Host: 0.0.0.0"
echo "   Port: 8000"
echo "   Workers: 1 (оптимізовано для CPU)"
echo "   Threads: 3 (оптимізовано для 8CPU)"
echo ""
echo "📝 Логи зберігаються в logs/server.log"
echo "📊 Моніторинг системи в logs/system_monitor.log"
echo ""
echo "🔗 Доступ до API: http://localhost:8000"
echo "📖 Документація: http://localhost:8000/docs"
echo ""

# Запуск сервера з оптимізованими параметрами
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-log \
    --log-level info \
    --timeout-keep-alive 30 \
    --limit-concurrency 10 \
    --limit-max-requests 1000 \
    --backlog 100 \
    --log-config logging.conf 2>/dev/null || \
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --access-log \
    --log-level info \
    --timeout-keep-alive 30
