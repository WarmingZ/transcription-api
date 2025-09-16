"""
Моніторинг пам'яті для оптимізації сервера транскрибації
"""

import psutil
import logging
import time
from typing import Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Моніторинг та управління пам'яттю"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
        self.start_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Отримує поточне використання пам'яті"""
        try:
            # Системна пам'ять
            system_memory = psutil.virtual_memory()
            
            # Пам'ять процесу
            process_memory = self.process.memory_info()
            
            return {
                'system_total_gb': system_memory.total / (1024**3),
                'system_available_gb': system_memory.available / (1024**3),
                'system_used_percent': system_memory.percent,
                'process_rss_mb': process_memory.rss / (1024**2),
                'process_vms_mb': process_memory.vms / (1024**2),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Помилка отримання інформації про пам'ять: {e}")
            return {}
    
    def log_memory_status(self, context: str = ""):
        """Логує поточний стан пам'яті"""
        memory_info = self.get_memory_usage()
        if memory_info:
            logger.info(f"🧠 Пам'ять {context}: "
                       f"Система {memory_info['system_used_percent']:.1f}% "
                       f"({memory_info['system_available_gb']:.1f}GB вільних), "
                       f"Процес {memory_info['process_rss_mb']:.1f}MB")
    
    def check_memory_pressure(self) -> bool:
        """Перевіряє чи є тиск на пам'ять"""
        memory_info = self.get_memory_usage()
        if memory_info:
            return memory_info['system_used_percent'] > self.max_memory_percent
        return False
    
    def force_garbage_collection(self):
        """Примусове очищення пам'яті"""
        import gc
        collected = gc.collect()
        logger.info(f"🗑️ Garbage collection: очищено {collected} об'єктів")
    
    @contextmanager
    def memory_context(self, context_name: str):
        """Контекстний менеджер для моніторингу пам'яті"""
        self.log_memory_status(f"до {context_name}")
        start_memory = self.get_memory_usage()
        
        try:
            yield self
        finally:
            end_memory = self.get_memory_usage()
            if start_memory and end_memory:
                memory_diff = end_memory['process_rss_mb'] - start_memory['process_rss_mb']
                logger.info(f"🧠 Пам'ять після {context_name}: "
                           f"{memory_diff:+.1f}MB зміна")
            
            # Примусове очищення якщо потрібно
            if self.check_memory_pressure():
                self.force_garbage_collection()

# Глобальний монітор пам'яті
memory_monitor = MemoryMonitor()
