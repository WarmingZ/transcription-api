"""
Система управління API токенами
"""
import json
import secrets
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Менеджер для управління API токенами"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.api_keys_file = self.data_dir / "api_keys.json"
        self.master_token_file = self.data_dir / "master_token.txt"
        
        # Ініціалізуємо файли якщо не існують
        self._init_files()
    
    def _init_files(self):
        """Ініціалізує файли якщо вони не існують"""
        # Створюємо master токен якщо не існує
        if not self.master_token_file.exists():
            master_token = secrets.token_urlsafe(32)
            self.master_token_file.write_text(master_token)
            logger.info(f"🔑 Створено master токен: {master_token}")
            logger.info("📋 Збережіть цей токен! Він потрібен для доступу до адмін панелі")
        
        # Створюємо файл API ключів якщо не існує
        if not self.api_keys_file.exists():
            self._save_api_keys({})
            logger.info("📄 Створено файл API ключів")
    
    def _load_api_keys(self) -> Dict[str, Dict]:
        """Завантажує API ключі з файлу"""
        try:
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Помилка завантаження API ключів: {e}")
            return {}
    
    def _save_api_keys(self, api_keys: Dict[str, Dict]):
        """Зберігає API ключі у файл"""
        try:
            with open(self.api_keys_file, 'w', encoding='utf-8') as f:
                json.dump(api_keys, f, ensure_ascii=False, indent=2)
            logger.info(f"Збережено {len(api_keys)} API ключів")
        except Exception as e:
            logger.error(f"Помилка збереження API ключів: {e}")
            raise
    
    def get_master_token(self) -> str:
        """Отримує master токен"""
        return self.master_token_file.read_text().strip()
    
    def verify_master_token(self, token: str) -> bool:
        """Перевіряє master токен"""
        return token == self.get_master_token()
    
    def generate_api_key(self, client_name: str) -> str:
        """Генерує новий API ключ"""
        api_key = secrets.token_urlsafe(32)
        api_keys = self._load_api_keys()
        
        api_keys[api_key] = {
            "client_name": client_name,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "usage_count": 0,
            "last_used": None,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "notes": ""
        }
        
        self._save_api_keys(api_keys)
        logger.info(f"Створено новий API ключ для клієнта: {client_name}")
        return api_key
    
    def delete_api_key(self, api_key: str) -> bool:
        """Видаляє API ключ"""
        api_keys = self._load_api_keys()
        if api_key in api_keys:
            client_name = api_keys[api_key]["client_name"]
            del api_keys[api_key]
            self._save_api_keys(api_keys)
            logger.info(f"Видалено API ключ для клієнта: {client_name}")
            return True
        return False
    
    def verify_api_key(self, api_key: str) -> bool:
        """Перевіряє API ключ"""
        api_keys = self._load_api_keys()
        return api_key in api_keys and api_keys[api_key].get("active", True)
    
    def get_api_key_info(self, api_key: str) -> Optional[Dict]:
        """Отримує інформацію про API ключ"""
        api_keys = self._load_api_keys()
        return api_keys.get(api_key)
    
    def list_api_keys(self) -> List[Dict]:
        """Отримує список всіх API ключів"""
        api_keys = self._load_api_keys()
        result = []
        for key, info in api_keys.items():
            result.append({
                "key": key,
                "client_name": info["client_name"],
                "created_at": info["created_at"],
                "active": info.get("active", True),
                "usage_count": info.get("usage_count", 0),
                "last_used": info.get("last_used"),
                "total_requests": info.get("total_requests", 0),
                "successful_requests": info.get("successful_requests", 0),
                "failed_requests": info.get("failed_requests", 0),
                "total_processing_time": round(info.get("total_processing_time", 0), 2),
                "average_processing_time": round(info.get("total_processing_time", 0) / max(info.get("total_requests", 1), 1), 2),
                "notes": info.get("notes", "")
            })
        return result
    
    def get_stats(self) -> Dict:
        """Отримує статистику API ключів"""
        api_keys = self._load_api_keys()
        active_count = sum(1 for info in api_keys.values() if info.get("active", True))
        total_requests = sum(info.get("total_requests", 0) for info in api_keys.values())
        total_processing_time = sum(info.get("total_processing_time", 0) for info in api_keys.values())
        
        return {
            "total_keys": len(api_keys),
            "active_keys": active_count,
            "inactive_keys": len(api_keys) - active_count,
            "total_requests": total_requests,
            "total_processing_time": round(total_processing_time, 2),
            "average_processing_time": round(total_processing_time / max(total_requests, 1), 2)
        }
    
    def log_api_usage(self, api_key: str, success: bool = True, processing_time: float = 0.0):
        """Логує використання API ключа"""
        api_keys = self._load_api_keys()
        if api_key in api_keys:
            api_keys[api_key]["usage_count"] = api_keys[api_key].get("usage_count", 0) + 1
            api_keys[api_key]["last_used"] = datetime.now().isoformat()
            api_keys[api_key]["total_requests"] = api_keys[api_key].get("total_requests", 0) + 1
            
            if success:
                api_keys[api_key]["successful_requests"] = api_keys[api_key].get("successful_requests", 0) + 1
            else:
                api_keys[api_key]["failed_requests"] = api_keys[api_key].get("failed_requests", 0) + 1
            
            api_keys[api_key]["total_processing_time"] = api_keys[api_key].get("total_processing_time", 0) + processing_time
            
            self._save_api_keys(api_keys)
    
    def update_api_key_notes(self, api_key: str, notes: str) -> bool:
        """Оновлює нотатки для API ключа"""
        api_keys = self._load_api_keys()
        if api_key in api_keys:
            api_keys[api_key]["notes"] = notes
            self._save_api_keys(api_keys)
            return True
        return False
    
    def toggle_api_key_status(self, api_key: str) -> bool:
        """Перемикає статус API ключа (активний/неактивний)"""
        api_keys = self._load_api_keys()
        if api_key in api_keys:
            api_keys[api_key]["active"] = not api_keys[api_key].get("active", True)
            self._save_api_keys(api_keys)
            return True
        return False
    
    def print_startup_info(self):
        """Виводить інформацію про master токен при запуску"""
        master_token = self.get_master_token()
        logger.info("=" * 60)
        logger.info("🔑 MASTER TOKEN для адмін панелі:")
        logger.info(f"   {master_token}")
        logger.info("=" * 60)
        logger.info("📋 Доступ до адмін панелі:")
        logger.info("   • Статична: http://localhost:8000/admin-panel")
        logger.info("   • Динамічна: http://localhost:8000/admin?master_token=TOKEN")
        logger.info("=" * 60)

# Глобальний екземпляр менеджера
api_key_manager = APIKeyManager()
