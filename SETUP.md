# 🚀 Швидкий запуск Transcription API

## 📋 Системні вимоги

- **Python 3.9+**
- **8GB+ RAM** (рекомендовано 16GB)
- **FFmpeg** для обробки аудіо
- **libsndfile** для роботи з аудіо файлами

## ⚡ Швидкий старт

### 1️⃣ Клонування репозиторію
```bash
git clone https://github.com/WarmingZ/transcription-api.git
cd transcription-api
```

### 2️⃣ Встановлення системних залежностей

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg libsndfile1 libsndfile1-dev
```

#### macOS:
```bash
brew install ffmpeg libsndfile
```

#### Windows:
```bash
# Встановіть FFmpeg та libsndfile через conda або вручну
conda install ffmpeg libsndfile
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

## 🌐 Використання

### Веб-інтерфейс
Відкрийте браузер: `http://localhost:8000`

### API endpoints
- **POST** `/transcribe` - проста транскрипція
- **POST** `/transcribe-with-diarization` - транскрипція з діаризацією
- **GET** `/health` - перевірка стану сервісу
- **GET** `/docs` - документація API

### Приклад використання
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

## 🔧 Налаштування

### Змінні середовища
```bash
# API ключ для аутентифікації (опціонально)
export API_KEY="your-secret-key"

# Розмір моделі (tiny, base, small, medium, large)
export MODEL_SIZE="small"
```

### Конфігурація моделей
Моделі автоматично завантажуються при першому запуску:
- **tiny** - найшвидша, найменша якість
- **base** - швидка, базова якість  
- **small** - збалансована (рекомендовано)
- **medium** - висока якість
- **large** - найкраща якість, найповільніша

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

### Помилка "libsndfile not found"
```bash
# Ubuntu/Debian
sudo apt install libsndfile1 libsndfile1-dev

# macOS
brew install libsndfile
```

### Недостатньо пам'яті
- Використовуйте меншу модель (`tiny` або `base`)
- Зменшіть розмір аудіо файлів
- Закрийте інші програми

## 📊 Продуктивність

### Рекомендовані налаштування:
- **CPU**: 4+ ядер
- **RAM**: 8GB+ (16GB для моделі large)
- **GPU**: CUDA-сумісна карта (опціонально)

### Час транскрипції:
- **tiny**: ~0.1x реального часу
- **base**: ~0.2x реального часу
- **small**: ~0.3x реального часу
- **medium**: ~0.5x реального часу
- **large**: ~1.0x реального часу

## 🔒 Безпека

### Для продакшену:
1. Встановіть API ключ: `export API_KEY="your-secret-key"`
2. Налаштуйте файрвол
3. Використовуйте HTTPS
4. Обмежте доступ до API

## 📝 Логи

Логи зберігаються в:
- Консоль (stdout)
- Файли в директорії `logs/` (якщо створена)

## 🆘 Підтримка

При виникненні проблем:
1. Перевірте логи в консолі
2. Переконайтеся, що всі залежності встановлені
3. Перевірте доступність пам'яті
4. Створіть issue в GitHub репозиторії
