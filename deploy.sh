#!/bin/bash
set -e

cd /root/transcription-api
git pull origin main
source venv/bin/activate
screen -S transcription-api -X quit || true
screen -dmS transcription-api bash -c "python main.py > logs/app.log 2>&1"
echo "✅ Деплой завершено!"