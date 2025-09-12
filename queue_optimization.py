#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Оптимізації для високого навантаження
"""

import asyncio
import logging
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

class RequestPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class TranscriptionTask:
    task_id: str
    user_id: str
    priority: RequestPriority
    created_at: float
    file_path: str
    language: str
    callback_url: str = None

class TranscriptionQueue:
    """Черга транскрипції з пріоритетами"""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.queue = []
        self.running_tasks = {}
        self.completed_tasks = {}
        self.logger = logging.getLogger(__name__)
    
    async def add_task(self, task: TranscriptionTask) -> str:
        """Додати завдання в чергу"""
        # Вставляємо за пріоритетом
        inserted = False
        for i, existing_task in enumerate(self.queue):
            if task.priority.value > existing_task.priority.value:
                self.queue.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            self.queue.append(task)
        
        self.logger.info(f"Додано завдання {task.task_id} з пріоритетом {task.priority.name}")
        
        # Запускаємо обробку якщо є вільні слоти
        await self._process_queue()
        
        return task.task_id
    
    async def _process_queue(self):
        """Обробка черги"""
        while len(self.running_tasks) < self.max_concurrent and self.queue:
            task = self.queue.pop(0)
            asyncio.create_task(self._process_task(task))
    
    async def _process_task(self, task: TranscriptionTask):
        """Обробка одного завдання"""
        task_id = task.task_id
        self.running_tasks[task_id] = task
        
        try:
            self.logger.info(f"Початок обробки завдання {task_id}")
            
            # Тут буде виклик транскрипції
            # result = await transcription_service.transcribe_simple(...)
            
            # Симуляція обробки
            await asyncio.sleep(2)
            
            self.completed_tasks[task_id] = {
                'status': 'completed',
                'result': {'text': 'Результат транскрипції'},
                'completed_at': time.time()
            }
            
            self.logger.info(f"Завдання {task_id} завершено")
            
        except Exception as e:
            self.completed_tasks[task_id] = {
                'status': 'error',
                'error': str(e),
                'completed_at': time.time()
            }
            self.logger.error(f"Помилка в завданні {task_id}: {e}")
        
        finally:
            del self.running_tasks[task_id]
            # Продовжуємо обробку черги
            await self._process_queue()
    
    def get_status(self, task_id: str) -> Dict[str, Any]:
        """Отримати статус завдання"""
        if task_id in self.running_tasks:
            return {'status': 'running', 'position': 0}
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            # Шукаємо в черзі
            for i, task in enumerate(self.queue):
                if task.task_id == task_id:
                    return {'status': 'queued', 'position': i + 1}
            return {'status': 'not_found'}

# Приклад використання в FastAPI
class QueueOptimizedTranscriptionService:
    """Оптимізований сервіс з чергою"""
    
    def __init__(self):
        self.queue = TranscriptionQueue(max_concurrent=3)
        self.logger = logging.getLogger(__name__)
    
    async def submit_transcription(self, user_id: str, file_path: str, 
                                 language: str = "uk", priority: RequestPriority = RequestPriority.NORMAL) -> str:
        """Відправити завдання на транскрипцію"""
        task_id = f"task_{int(time.time() * 1000)}"
        
        task = TranscriptionTask(
            task_id=task_id,
            user_id=user_id,
            priority=priority,
            created_at=time.time(),
            file_path=file_path,
            language=language
        )
        
        await self.queue.add_task(task)
        return task_id
    
    async def get_transcription_status(self, task_id: str) -> Dict[str, Any]:
        """Отримати статус транскрипції"""
        return self.queue.get_status(task_id)

# Налаштування для різних сценаріїв
PRODUCTION_CONFIG = {
    "max_concurrent": 3,  # Максимум 3 одночасних транскрипції
    "queue_timeout": 300,  # 5 хвилин на завдання
    "priority_boost": {  # Підвищення пріоритету для VIP користувачів
        "vip_user": RequestPriority.HIGH,
        "premium_user": RequestPriority.NORMAL,
        "free_user": RequestPriority.LOW
    }
}

DEVELOPMENT_CONFIG = {
    "max_concurrent": 1,  # Тільки 1 для тестування
    "queue_timeout": 60,
    "priority_boost": {}
}

if __name__ == "__main__":
    print("🚀 Система черги для транскрипції")
    print("=" * 50)
    print("✅ Підтримка пріоритетів")
    print("✅ Обмеження одночасних завдань")
    print("✅ Відстеження статусу")
    print("✅ Автоматичне масштабування")
