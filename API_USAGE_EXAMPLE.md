# 🔑 API Usage Example

## Отримання Master токена

Master токен зберігається у файлі `data/master_token.txt`:

```bash
cat data/master_token.txt
# weXmUvHm5p0eUUnfvNX0a7xPiOAEVN2uPfb6BoPpV_Y
```

## Доступ до адмін панелі

Відкрийте браузер:
```
http://localhost:8000/admin?master_token=weXmUvHm5p0eUUnfvNX0a7xPiOAEVN2uPfb6BoPpV_Y
```

## Створення API ключа через curl

```bash
curl -X POST "http://localhost:8000/admin/generate-key" \
  -H "Authorization: Bearer weXmUvHm5p0eUUnfvNX0a7xPiOAEVN2uPfb6BoPpV_Y" \
  -H "Content-Type: application/json" \
  -d '{"client_name": "My Client"}'
```

## Використання API ключа для транскрипції

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Authorization: Bearer v51df5BfL1TjJYGdEt5erwysTEyV60GgHnM5WDFhr0o" \
  -F "file=@audio.wav" \
  -F "language=uk" \
  -F "model_size=small"
```

## Видалення API ключа

```bash
curl -X POST "http://localhost:8000/admin/delete-key" \
  -H "Authorization: Bearer weXmUvHm5p0eUUnfvNX0a7xPiOAEVN2uPfb6BoPpV_Y" \
  -H "Content-Type: application/json" \
  -d '{"api_key": "v51df5BfL1TjJYGdEt5erwysTEyV60GgHnM5WDFhr0o"}'
```

## Список всіх ключів

```bash
curl -X GET "http://localhost:8000/admin/list-keys" \
  -H "Authorization: Bearer weXmUvHm5p0eUUnfvNX0a7xPiOAEVN2uPfb6BoPpV_Y"
```

## Структура файлів

- `data/api_keys.json` - зберігає всі API ключі
- `data/master_token.txt` - master токен для адміністрування
- Файли знаходяться поза веб-каталогом для безпеки
