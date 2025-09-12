# 🎤 Ukrainian Audio Transcription API

**Локальний API для транскрипції українського аудіо/відео з визначенням дикторів**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Особливості

- 🎯 **Локальна обробка** - всі дані залишаються на вашому сервері
- 🇺🇦 **Українська мова** - оптимізовано для української мови
- 🎤 **Діаризація** - просте визначення дикторів (Оператор/Клієнт)
- ⚡ **Висока швидкість** - використання faster-whisper
- 🔧 **Гнучкість** - підтримка різних форматів аудіо/відео
- 🌐 **REST API** - простий інтерфейс для інтеграції
- 📱 **Веб-інтерфейс** - зручний інтерфейс для тестування

## 🚀 Швидкий старт

### 1️⃣ Клонування репозиторію
```bash
git clone https://github.com/WarmingZ/transcription-api.git
cd transcription-api
```

### 2️⃣ Встановлення залежностей
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg libsndfile1 libsndfile1-dev

# macOS
brew install ffmpeg libsndfile
```

### 3️⃣ Створення віртуального середовища
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# або
venv\Scripts\activate     # Windows
```

### 4️⃣ Встановлення Python залежностей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5️⃣ Завантаження моделей
```bash
python download_models.py
```

### 6️⃣ Запуск API
```bash
python main.py
```

**Відкрийте браузер: `http://localhost:8000`** 🌐

## 📖 Детальна інструкція

Для повної інструкції з налаштування та вирішення проблем дивіться [SETUP.md](SETUP.md)

## 🎯 API Endpoints

| Endpoint | Метод | Опис |
|----------|-------|------|
| `/transcribe` | POST | Проста транскрипція |
| `/transcribe-with-diarization` | POST | Транскрипція з діаризацією |
| `/health` | GET | Перевірка стану сервісу |
| `/docs` | GET | Документація API |
| `/` | GET | Веб-інтерфейс |

## 💡 Приклад використання

```bash
# Проста транскрипція
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "language=uk"

# Транскрипція з діаризацією
curl -X POST "http://localhost:8000/transcribe-with-diarization" \
  -F "file=@audio.mp3" \
  -F "language=uk"
```

## ⚙️ Налаштування

### Змінні середовища
```bash
# API ключ для аутентифікації (опціонально)
export API_KEY="your-secret-key"

# Розмір моделі (tiny, base, small, medium, large)
export MODEL_SIZE="small"
```

### Моделі Whisper
- **tiny** - найшвидша, найменша якість (~0.1x реального часу)
- **base** - швидка, базова якість (~0.2x реального часу)
- **small** - збалансована (~0.3x реального часу) ⭐ **рекомендовано**
- **medium** - висока якість (~0.5x реального часу)
- **large** - найкраща якість (~1.0x реального часу)

## 🏗️ Архітектура

```
transcription-api/
├── main.py                 # FastAPI додаток
├── models/                 # Модулі для роботи з моделями
│   ├── __init__.py
│   ├── config.py          # Конфігурація
│   ├── whisper_model.py   # faster-whisper
│   ├── diarization.py     # Діаризація дикторів
│   └── transcription_service.py  # Основний сервіс
├── download_models.py     # Завантаження моделей
├── requirements.txt       # Python залежності
├── static/                # Веб-інтерфейс
└── README.md             # Документація
```

## 🔧 Системні вимоги

| Компонент | Мінімум | Рекомендовано |
|-----------|---------|---------------|
| **CPU** | 2 ядра | 4+ ядер |
| **RAM** | 4GB | 8GB+ (16GB для large) |
| **Python** | 3.9+ | 3.11+ |
| **GPU** | - | CUDA-сумісна (опціонально) |

## 🐛 Вирішення проблем

### Помилка "No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### Помилка "ffmpeg not found"
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Недостатньо пам'яті
- Використовуйте меншу модель (`tiny` або `base`)
- Зменшіть розмір аудіо файлів
- Закрийте інші програми

## 🔒 Безпека

### Для продакшену:
1. Встановіть API ключ: `export API_KEY="your-secret-key"`
2. Налаштуйте файрвол
3. Використовуйте HTTPS
4. Обмежте доступ до API

## 📊 Продуктивність

### Оптимізації:
- ✅ **Multiprocessing** - паралельна обробка аудіо сегментів
- ✅ **Оптимізовані параметри** - beam_size=1, vad_filter=False
- ✅ **Швидке ресемплінгу** - використання soxr
- ✅ **Адаптивні чанки** - динамічний розмір сегментів
- ✅ **GPU підтримка** - автоматичне визначення пристрою

## 📝 Логи

Логи зберігаються в:
- Консоль (stdout)
- Файли в директорії `logs/` (якщо створена)

## 🤝 Підтримка

При виникненні проблем:
1. Перевірте логи в консолі
2. Переконайтеся, що всі залежності встановлені
3. Перевірте доступність пам'яті
4. Створіть [issue](https://github.com/WarmingZ/transcription-api/issues) в GitHub

## 📄 Ліцензія

Цей проект розповсюджується під ліцензією MIT. Дивіться [LICENSE](LICENSE) для деталей.

## 🙏 Подяки

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - швидка транскрипція
- [speechbrain](https://speechbrain.github.io/) - діаризація дикторів
- [FastAPI](https://fastapi.tiangolo.com/) - веб-фреймворк