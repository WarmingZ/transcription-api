#!/usr/bin/env python3
"""
Створення першого адміністраторського ключа
"""

from auth import get_api_key_manager

def create_admin_key():
    """Створює перший адміністраторський ключ"""
    print("🔐 СТВОРЕННЯ АДМІНІСТРАТОРСЬКОГО КЛЮЧА")
    print("=" * 50)
    print("⚠️  ЦЕЙ КЛЮЧ ДАЄ ПОВНИЙ ДОСТУП ДО СИСТЕМИ!")
    print("   Зберігайте його в надійному місці!")
    print()
    
    name = input("📝 Введіть назву адміністратора: ").strip()
    if not name:
        print("❌ Назва не може бути порожньою!")
        return
    
    description = input("📄 Введіть опис (опціонально): ").strip()
    
    # Створюємо адміністраторський ключ
    api_key_manager = get_api_key_manager()
    admin_key = api_key_manager.generate_api_key(name, description, is_admin=True)
    
    print()
    print("✅ АДМІНІСТРАТОРСЬКИЙ КЛЮЧ СТВОРЕНО!")
    print("=" * 50)
    print(f"🔑 Ключ: {admin_key}")
    print(f"📝 Назва: {name}")
    print(f"📄 Опис: {description}")
    print(f"👑 Тип: АДМІНІСТРАТОР")
    print()
    print("💡 Використання:")
    print(f"   curl -H 'Authorization: Bearer {admin_key}' http://localhost:8000/admin/api-keys")
    print()
    print("⚠️  ЗБЕРЕЖІТЬ ЦЕЙ КЛЮЧ В БЕЗПЕЧНОМУ МІСЦІ!")
    print("   Він дає повний доступ до управління API!")
    print("   Без нього ви не зможете створювати нові ключі!")

if __name__ == "__main__":
    create_admin_key()
