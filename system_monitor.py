#!/usr/bin/env python3
"""
Системний монітор для сервера транскрибації
Автоматичне очищення пам'яті та моніторинг ресурсів
"""

import psutil
import time
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Системний монітор з автоматичним очищенням"""
    
    def __init__(self):
        self.memory_threshold = 80.0  # Поріг використання RAM
        self.cpu_threshold = 90.0     # Поріг використання CPU
        self.disk_threshold = 90.0    # Поріг використання диску
        self.monitoring_interval = 60  # Інтервал моніторингу (секунди)
        self.cleanup_interval = 300    # Інтервал очищення (секунди)
        self.running = True
        
        # Статистика
        self.stats = {
            'memory_warnings': 0,
            'cpu_warnings': 0,
            'disk_warnings': 0,
            'cleanups_performed': 0,
            'start_time': datetime.now()
        }
        
        # Налаштування сигналів для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Обробка сигналів завершення"""
        logger.info(f"Отримано сигнал {signum}, завершуємо роботу...")
        self.running = False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Отримує поточну інформацію про систему"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Пам'ять
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_percent = memory.percent
            
            # Диск
            disk = psutil.disk_usage('/')
            disk_gb = disk.total / (1024**3)
            disk_used_gb = disk.used / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            # Процеси
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if proc.info['memory_percent'] > 1.0:  # Процеси що використовують >1% RAM
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Сортуємо по використанню пам'яті
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else None
                },
                'memory': {
                    'total_gb': memory_gb,
                    'used_gb': memory_used_gb,
                    'available_gb': memory_available_gb,
                    'percent': memory_percent
                },
                'disk': {
                    'total_gb': disk_gb,
                    'used_gb': disk_used_gb,
                    'free_gb': disk_free_gb,
                    'percent': disk_percent
                },
                'top_processes': processes[:10]  # Топ 10 процесів
            }
            
        except Exception as e:
            logger.error(f"Помилка отримання інформації про систему: {e}")
            return {}
    
    def log_system_status(self):
        """Логує поточний стан системи"""
        info = self.get_system_info()
        if not info:
            return
        
        cpu = info['cpu']
        memory = info['memory']
        disk = info['disk']
        
        # Основна інформація
        logger.info(
            f"📊 Система: CPU {cpu['percent']:.1f}%/{cpu['count']} ядер, "
            f"RAM {memory['used_gb']:.1f}/{memory['total_gb']:.1f}GB ({memory['percent']:.1f}%), "
            f"Диск {disk['used_gb']:.1f}/{disk['total_gb']:.1f}GB ({disk['percent']:.1f}%)"
        )
        
        # Топ процеси
        if info['top_processes']:
            logger.info("🔝 Топ процеси по RAM:")
            for proc in info['top_processes'][:5]:
                logger.info(f"  {proc['name']} (PID {proc['pid']}): {proc['memory_percent']:.1f}% RAM")
        
        # Попередження
        warnings = []
        
        if memory['percent'] > self.memory_threshold:
            warnings.append(f"⚠️ Високе використання RAM: {memory['percent']:.1f}%")
            self.stats['memory_warnings'] += 1
        
        if cpu['percent'] > self.cpu_threshold:
            warnings.append(f"⚠️ Високе використання CPU: {cpu['percent']:.1f}%")
            self.stats['cpu_warnings'] += 1
        
        if disk['percent'] > self.disk_threshold:
            warnings.append(f"⚠️ Високе використання диску: {disk['percent']:.1f}%")
            self.stats['disk_warnings'] += 1
        
        if warnings:
            for warning in warnings:
                logger.warning(warning)
    
    def cleanup_system(self):
        """Виконує очищення системи"""
        try:
            logger.info("🧹 Початок очищення системи...")
            
            # Очищення тимчасових файлів
            temp_dirs = ['/tmp', '/var/tmp', 'temp']
            cleaned_files = 0
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    try:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    # Видаляємо файли старші 1 години
                                    if os.path.getmtime(file_path) < time.time() - 3600:
                                        os.unlink(file_path)
                                        cleaned_files += 1
                                except (OSError, PermissionError):
                                    pass
                    except (OSError, PermissionError):
                        pass
            
            # Очищення Python кешу
            import gc
            collected = gc.collect()
            
            # Очищення логів (якщо вони занадто великі)
            log_files = ['system_monitor.log', 'app.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    try:
                        file_size = os.path.getsize(log_file)
                        if file_size > 100 * 1024 * 1024:  # 100MB
                            # Обрізаємо лог файл
                            with open(log_file, 'w') as f:
                                f.write(f"# Лог файл обрізано {datetime.now()}\n")
                            logger.info(f"📝 Обрізано лог файл {log_file}")
                    except (OSError, PermissionError):
                        pass
            
            logger.info(f"✅ Очищення завершено: {cleaned_files} файлів, {collected} об'єктів GC")
            self.stats['cleanups_performed'] += 1
            
        except Exception as e:
            logger.error(f"Помилка очищення системи: {e}")
    
    def check_and_cleanup(self):
        """Перевіряє стан системи та виконує очищення при необхідності"""
        info = self.get_system_info()
        if not info:
            return
        
        memory = info['memory']
        cpu = info['cpu']
        
        # Автоматичне очищення при високому використанні ресурсів
        should_cleanup = False
        
        if memory['percent'] > self.memory_threshold:
            logger.warning(f"🚨 Критичне використання RAM: {memory['percent']:.1f}%")
            should_cleanup = True
        
        if cpu['percent'] > self.cpu_threshold:
            logger.warning(f"🚨 Критичне використання CPU: {cpu['percent']:.1f}%")
            should_cleanup = True
        
        if should_cleanup:
            self.cleanup_system()
    
    def print_stats(self):
        """Виводить статистику роботи"""
        uptime = datetime.now() - self.stats['start_time']
        
        logger.info("📈 Статистика моніторингу:")
        logger.info(f"  Час роботи: {uptime}")
        logger.info(f"  Попереджень RAM: {self.stats['memory_warnings']}")
        logger.info(f"  Попереджень CPU: {self.stats['cpu_warnings']}")
        logger.info(f"  Попереджень диску: {self.stats['disk_warnings']}")
        logger.info(f"  Очищень виконано: {self.stats['cleanups_performed']}")
    
    def run(self):
        """Основний цикл моніторингу"""
        logger.info("🚀 Запуск системного монітора...")
        logger.info(f"Пороги: RAM {self.memory_threshold}%, CPU {self.cpu_threshold}%, Диск {self.disk_threshold}%")
        
        last_cleanup = time.time()
        
        try:
            while self.running:
                # Логування стану системи
                self.log_system_status()
                
                # Перевірка та очищення
                self.check_and_cleanup()
                
                # Періодичне очищення
                if time.time() - last_cleanup > self.cleanup_interval:
                    self.cleanup_system()
                    last_cleanup = time.time()
                
                # Очікування
                time.sleep(self.monitoring_interval)
                
        except KeyboardInterrupt:
            logger.info("Отримано сигнал завершення")
        except Exception as e:
            logger.error(f"Помилка в основному циклі: {e}")
        finally:
            self.print_stats()
            logger.info("👋 Системний монітор завершив роботу")

def main():
    """Головна функція"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("""
Системний монітор для сервера транскрибації

Використання:
  python system_monitor.py              # Запуск моніторингу
  python system_monitor.py --help      # Показати цю довідку

Функції:
  - Моніторинг CPU, RAM, диску
  - Автоматичне очищення при високому навантаженні
  - Логування топ процесів
  - Статистика роботи

Налаштування в коді:
  - memory_threshold: 80% (поріг RAM)
  - cpu_threshold: 90% (поріг CPU)
  - monitoring_interval: 60s (інтервал моніторингу)
  - cleanup_interval: 300s (інтервал очищення)
            """)
            return
    
    monitor = SystemMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
