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
            "active": True
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
                "active": info.get("active", True)
            })
        return result
    
    def get_stats(self) -> Dict:
        """Отримує статистику API ключів"""
        api_keys = self._load_api_keys()
        active_count = sum(1 for info in api_keys.values() if info.get("active", True))
        return {
            "total_keys": len(api_keys),
            "active_keys": active_count,
            "inactive_keys": len(api_keys) - active_count
        }
    
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
