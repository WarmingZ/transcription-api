# 🔑 Admin Panel Guide

## Доступ до адмін панелі

У вас є **два варіанти** адмін панелі:

### 1. Динамічна адмін панель (вбудована в API)
```
http://localhost:8000/admin?master_token=YOUR_MASTER_TOKEN
```

### 2. Статична адмін панель (окрема HTML сторінка)
```
http://localhost:8000/admin-panel
```

## 🔐 Master токен

Ваш master токен:
```
weXmUvHm5p0eUUnfvNX0a7xPiOAEVN2uPfb6BoPpV_Y
```

## Функції адмін панелі

### ✅ Створення API ключів
- Введіть назву клієнта
- Натисніть "Генерувати ключ"
- **ВАЖЛИВО**: Збережіть ключ! Він більше не буде показаний

### 📋 Перегляд списку ключів
- Показує всі активні та неактивні ключі
- Відображає назву клієнта та дату створення
- Автоматично оновлюється кожні 30 секунд

### 🗑️ Видалення ключів
- Натисніть "Видалити" біля потрібного ключа
- Підтвердіть видалення
- Ключ буде видалено назавжди

### 📊 Статистика
- Всього ключів
- Активних ключів
- Неактивних ключів

## Використання API ключів

### Приклад запиту з API ключем:
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@audio.wav" \
  -F "language=uk" \
  -F "model_size=small"
```

### Приклад запиту з URL:
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "url=https://example.com/audio.mp3" \
  -F "language=uk"
```

## Безпека

- ✅ Master токен захищає всі адмін функції
- ✅ API ключі генеруються через `secrets.token_urlsafe(32)`
- ✅ Файли токенів зберігаються поза веб-каталогом
- ✅ Всі зміни зберігаються в реальному часі
- ✅ Статична сторінка не потребує серверного рендерингу

## Структура файлів

```
data/
├── api_keys.json      # Всі API ключі
└── master_token.txt   # Master токен

static/
└── admin.html         # Статична адмін панель
```

## API Endpoints

- `GET /admin-panel` - Статична адмін панель
- `GET /admin` - Динамічна адмін панель (потребує master токен)
- `POST /admin/generate-key` - Створення API ключа
- `POST /admin/delete-key` - Видалення API ключа
- `GET /admin/list-keys` - Список всіх ключів
- `POST /transcribe` - Транскрипція (потребує API ключ)
- `POST /transcribe-with-diarization` - Транскрипція з діаризацією (потребує API ключ)
