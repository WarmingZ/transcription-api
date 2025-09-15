"""
Система авторизації для Ukrainian Transcription API
"""

import secrets
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Менеджер API ключів"""
    
    def __init__(self, storage_file: str = "api_keys.json"):
        self.api_keys: Dict[str, Dict] = {}
        self.storage_file = storage_file
        self.load_keys()
    
    def load_keys(self):
        """Завантажуємо API ключі з файлу або створюємо дефолтний"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Конвертуємо дати назад в datetime об'єкти
                    for key, info in data.items():
                        info['created_at'] = datetime.fromisoformat(info['created_at'])
                        if info['last_used']:
                            info['last_used'] = datetime.fromisoformat(info['last_used'])
                        # Додаємо поле is_admin якщо його немає (для старих ключів)
                        if 'is_admin' not in info:
                            info['is_admin'] = False
                    self.api_keys = data
                logger.info(f"✅ Завантажено {len(self.api_keys)} API ключів з {self.storage_file}")
            except Exception as e:
                logger.error(f"❌ Помилка завантаження ключів: {e}")
                self.load_default_keys()
        else:
            self.load_default_keys()
    
    def save_keys(self):
        """Зберігаємо API ключі в файл"""
        try:
            # Конвертуємо datetime в строки для JSON
            data = {}
            for key, info in self.api_keys.items():
                data[key] = {
                    "name": info["name"],
                    "description": info["description"],
                    "created_at": info["created_at"].isoformat(),
                    "last_used": info["last_used"].isoformat() if info["last_used"] else None,
                    "usage_count": info["usage_count"],
                    "is_active": info["is_active"],
                    "permissions": info["permissions"],
                    "is_admin": info.get("is_admin", False)
                }
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Збережено {len(self.api_keys)} API ключів в {self.storage_file}")
        except Exception as e:
            logger.error(f"❌ Помилка збереження ключів: {e}")
    
    def load_default_keys(self):
        """Завантажуємо дефолтні API ключі"""
        # Використовуємо фіксований дефолтний ключ для стабільності
        default_key = "2OqDFTkg5Wa4uLw015EqzGx2BRHhBkUfG89iOsXX1P8"
        self.api_keys[default_key] = {
            "name": "default",
            "description": "Default API Key",
            "created_at": datetime.now(),
            "last_used": None,
            "usage_count": 0,
            "is_active": True,
            "permissions": ["transcribe", "transcribe_with_diarization"],
            "is_admin": False  # Звичайний користувач
        }
        logger.info(f"✅ Завантажено дефолтний API ключ: {default_key}")
        self.save_keys()
    
    def generate_api_key(self, name: str, description: str = "", is_admin: bool = False) -> str:
        """Генерує новий API ключ"""
        # Генеруємо випадковий ключ
        key = secrets.token_urlsafe(32)
        
        # Зберігаємо інформацію про ключ
        self.api_keys[key] = {
            "name": name,
            "description": description,
            "created_at": datetime.now(),
            "last_used": None,
            "usage_count": 0,
            "is_active": True,
            "permissions": ["transcribe", "transcribe_with_diarization"],
            "is_admin": is_admin
        }
        
        # Зберігаємо в файл
        self.save_keys()
        
        logger.info(f"✅ Створено новий API ключ: {key} (admin: {is_admin})")
        return key
    
    def validate_api_key(self, api_key: str, update_stats: bool = True) -> bool:
        """Перевіряє чи валідний API ключ"""
        if api_key not in self.api_keys:
            return False
        
        key_info = self.api_keys[api_key]
        
        # Перевіряємо чи активний
        if not key_info["is_active"]:
            return False
        
        # Оновлюємо статистику тільки якщо потрібно
        if update_stats:
            key_info["last_used"] = datetime.now()
            key_info["usage_count"] += 1
        
        return True
    
    def get_key_info(self, api_key: str) -> Optional[Dict]:
        """Отримує інформацію про API ключ"""
        return self.api_keys.get(api_key)
    
    def is_admin(self, api_key: str) -> bool:
        """Перевіряє чи є ключ адміністраторським"""
        key_info = self.get_key_info(api_key)
        return key_info and key_info.get("is_admin", False) and key_info.get("is_active", False)
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Відкликає API ключ"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = False
            self.save_keys()  # Зберігаємо зміни
            logger.info(f"✅ API ключ відкликано: {api_key}")
            return True
        return False
    
    def activate_api_key(self, api_key: str) -> bool:
        """Активує API ключ"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = True
            self.save_keys()  # Зберігаємо зміни
            logger.info(f"✅ API ключ активовано: {api_key}")
            return True
        return False
    
    def list_api_keys(self) -> List[Dict]:
        """Показує всі API ключі (без самих ключів)"""
        keys_info = []
        for key, info in self.api_keys.items():
            keys_info.append({
                "name": info["name"],
                "description": info["description"],
                "created_at": info["created_at"].isoformat(),
                "last_used": info["last_used"].isoformat() if info["last_used"] else None,
                "usage_count": info["usage_count"],
                "is_active": info["is_active"],
                "permissions": info["permissions"]
            })
        return keys_info
    
    def list_api_keys_with_keys(self) -> List[Dict]:
        """Показує всі API ключі з самими ключами (тільки для адміністратора)"""
        keys_info = []
        for key, info in self.api_keys.items():
            keys_info.append({
                "api_key": key,
                "name": info["name"],
                "description": info["description"],
                "created_at": info["created_at"].isoformat(),
                "last_used": info["last_used"].isoformat() if info["last_used"] else None,
                "usage_count": info["usage_count"],
                "is_active": info["is_active"],
                "permissions": info["permissions"]
            })
        return keys_info

# Глобальний менеджер API ключів
api_key_manager = APIKeyManager()

def get_api_key_manager() -> APIKeyManager:
    """Отримує глобальний менеджер API ключів"""
    return api_key_manager

