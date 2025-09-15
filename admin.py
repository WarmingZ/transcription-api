#!/usr/bin/env python3
"""
Проста система адміністрування для Ukrainian Transcription API
"""

import os
import sys
from auth import get_api_key_manager

class AdminPanel:
    """Панель адміністрування API ключів"""
    
    def __init__(self):
        self.api_manager = get_api_key_manager()
    
    def clear_screen(self):
        """Очищує екран"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def show_header(self):
        """Показує заголовок"""
        print("🔐 АДМІНІСТРАТОР API КЛЮЧІВ")
        print("=" * 40)
        print("Ukrainian Transcription API")
        print()
    
    def show_menu(self):
        """Показує головне меню"""
        print("📋 ГОЛОВНЕ МЕНЮ:")
        print("1. 🔑 Генерувати новий API ключ")
        print("2. 📋 Показати всі ключі")
        print("3. 🚫 Відкликати ключ")
        print("4. ✅ Активувати ключ")
        print("5. 📊 Статистика використання")
        print("6. 💾 Експорт ключів")
        print("7. ❌ Вийти")
        print()
    
    def generate_key(self):
        """Генерує новий API ключ"""
        print("🔑 ГЕНЕРАЦІЯ НОВОГО API КЛЮЧА")
        print("-" * 30)
        
        name = input("📝 Введіть назву ключа: ").strip()
        if not name:
            print("❌ Назва не може бути порожньою!")
            return
        
        description = input("📄 Введіть опис (опціонально): ").strip()
        
        # Питаємо чи це адміністраторський ключ
        is_admin_input = input("👑 Це адміністраторський ключ? (y/N): ").strip().lower()
        is_admin = is_admin_input in ['y', 'yes', 'так']
        
        # Генеруємо ключ
        new_key = self.api_manager.generate_api_key(name, description, is_admin)
        
        print()
        print("✅ НОВИЙ API КЛЮЧ СТВОРЕНО!")
        print("=" * 40)
        print(f"🔑 Ключ: {new_key}")
        print(f"📝 Назва: {name}")
        print(f"📄 Опис: {description}")
        print(f"👑 Тип: {'АДМІНІСТРАТОР' if is_admin else 'ЗВИЧАЙНИЙ КОРИСТУВАЧ'}")
        print()
        print("💡 Використання:")
        print(f"   curl -H 'Authorization: Bearer {new_key}' http://localhost:8000/api")
        print()
        print("⚠️  ЗБЕРЕЖІТЬ ЦЕЙ КЛЮЧ В БЕЗПЕЧНОМУ МІСЦІ!")
        print("   Він більше не буде показаний!")
        
        input("\nНатисніть Enter для продовження...")
    
    def show_keys(self):
        """Показує всі API ключі"""
        print("📋 ВСІ API КЛЮЧІ")
        print("-" * 30)
        
        keys = self.api_manager.list_api_keys_with_keys()
        
        if not keys:
            print("❌ API ключі не знайдені")
            input("\nНатисніть Enter для продовження...")
            return
        
        for i, key_info in enumerate(keys, 1):
            status_icon = "✅" if key_info['is_active'] else "❌"
            admin_icon = "👑" if key_info.get('is_admin', False) else "👤"
            print(f"{i}. {status_icon} {admin_icon} {key_info['name']}")
            print(f"   🔑 Ключ: {key_info['api_key']}")
            print(f"   📄 Опис: {key_info['description']}")
            print(f"   📅 Створено: {key_info['created_at']}")
            print(f"   🔄 Використань: {key_info['usage_count']}")
            print(f"   👑 Тип: {'АДМІНІСТРАТОР' if key_info.get('is_admin', False) else 'ЗВИЧАЙНИЙ КОРИСТУВАЧ'}")
            if key_info['last_used']:
                print(f"   🕒 Останнє використання: {key_info['last_used']}")
            print()
        
        input("Натисніть Enter для продовження...")
    
    def revoke_key(self):
        """Відкликає API ключ"""
        print("🚫 ВІДКЛИКАННЯ API КЛЮЧА")
        print("-" * 30)
        
        # Показуємо активні ключі
        keys = self.api_manager.list_api_keys_with_keys()
        active_keys = [k for k in keys if k['is_active']]
        
        if not active_keys:
            print("❌ Немає активних ключів для відкликання")
            input("\nНатисніть Enter для продовження...")
            return
        
        print("Активні ключі:")
        for i, key_info in enumerate(active_keys, 1):
            print(f"{i}. {key_info['name']} - {key_info['api_key']}")
        
        try:
            choice = int(input("\nВиберіть номер ключа для відкликання: ")) - 1
            if 0 <= choice < len(active_keys):
                selected_key = active_keys[choice]
                
                print(f"\n⚠️  Ви впевнені, що хочете відкликати ключ '{selected_key['name']}'?")
                confirm = input("Введіть 'ТАК' для підтвердження: ").strip()
                
                if confirm.upper() == 'ТАК':
                    if self.api_manager.revoke_api_key(selected_key['api_key']):
                        print("✅ Ключ успішно відкликано!")
                    else:
                        print("❌ Помилка при відкликанні ключа")
                else:
                    print("❌ Операцію скасовано")
            else:
                print("❌ Невірний вибір")
        except ValueError:
            print("❌ Введіть правильний номер")
        
        input("\nНатисніть Enter для продовження...")
    
    def activate_key(self):
        """Активує відкликаний ключ"""
        print("✅ АКТИВАЦІЯ API КЛЮЧА")
        print("-" * 30)
        
        # Показуємо неактивні ключі
        keys = self.api_manager.list_api_keys_with_keys()
        inactive_keys = [k for k in keys if not k['is_active']]
        
        if not inactive_keys:
            print("❌ Немає неактивних ключів для активації")
            input("\nНатисніть Enter для продовження...")
            return
        
        print("Неактивні ключі:")
        for i, key_info in enumerate(inactive_keys, 1):
            print(f"{i}. {key_info['name']} - {key_info['api_key']}")
        
        try:
            choice = int(input("\nВиберіть номер ключа для активації: ")) - 1
            if 0 <= choice < len(inactive_keys):
                selected_key = inactive_keys[choice]
                
                # Активуємо ключ
                self.api_manager.api_keys[selected_key['api_key']]['is_active'] = True
                self.api_manager.save_keys()
                
                print(f"✅ Ключ '{selected_key['name']}' успішно активовано!")
            else:
                print("❌ Невірний вибір")
        except ValueError:
            print("❌ Введіть правильний номер")
        
        input("\nНатисніть Enter для продовження...")
    
    def show_stats(self):
        """Показує статистику використання"""
        print("📊 СТАТИСТИКА ВИКОРИСТАННЯ")
        print("-" * 30)
        
        keys = self.api_manager.list_api_keys_with_keys()
        
        if not keys:
            print("❌ Немає даних для відображення")
            input("\nНатисніть Enter для продовження...")
            return
        
        # Загальна статистика
        total_keys = len(keys)
        active_keys = len([k for k in keys if k['is_active']])
        admin_keys = len([k for k in keys if k.get('is_admin', False)])
        user_keys = total_keys - admin_keys
        total_usage = sum(k['usage_count'] for k in keys)
        
        print(f"📈 Загальна статистика:")
        print(f"   🔑 Всього ключів: {total_keys}")
        print(f"   👑 Адміністраторів: {admin_keys}")
        print(f"   👤 Звичайних користувачів: {user_keys}")
        print(f"   ✅ Активних: {active_keys}")
        print(f"   ❌ Неактивних: {total_keys - active_keys}")
        print(f"   🔄 Загальне використання: {total_usage}")
        print()
        
        # Топ ключів за використанням
        sorted_keys = sorted(keys, key=lambda x: x['usage_count'], reverse=True)
        
        print("🏆 Топ ключів за використанням:")
        for i, key_info in enumerate(sorted_keys[:5], 1):
            status_icon = "✅" if key_info['is_active'] else "❌"
            admin_icon = "👑" if key_info.get('is_admin', False) else "👤"
            print(f"   {i}. {status_icon} {admin_icon} {key_info['name']}: {key_info['usage_count']} використань")
        
        input("\nНатисніть Enter для продовження...")
    
    def export_keys(self):
        """Експортує ключі в файл"""
        print("💾 ЕКСПОРТ КЛЮЧІВ")
        print("-" * 30)
        
        filename = input("📁 Введіть назву файлу (без розширення): ").strip()
        if not filename:
            filename = "api_keys_export"
        
        filename += ".txt"
        
        try:
            keys = self.api_manager.list_api_keys_with_keys()
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("API КЛЮЧІ UKRAINIAN TRANSCRIPTION API\n")
                f.write("=" * 50 + "\n\n")
                
                for key_info in keys:
                    status = "АКТИВНИЙ" if key_info['is_active'] else "НЕАКТИВНИЙ"
                    admin_type = "АДМІНІСТРАТОР" if key_info.get('is_admin', False) else "ЗВИЧАЙНИЙ КОРИСТУВАЧ"
                    f.write(f"Назва: {key_info['name']}\n")
                    f.write(f"Ключ: {key_info['api_key']}\n")
                    f.write(f"Опис: {key_info['description']}\n")
                    f.write(f"Тип: {admin_type}\n")
                    f.write(f"Статус: {status}\n")
                    f.write(f"Створено: {key_info['created_at']}\n")
                    f.write(f"Використань: {key_info['usage_count']}\n")
                    if key_info['last_used']:
                        f.write(f"Останнє використання: {key_info['last_used']}\n")
                    f.write("-" * 30 + "\n\n")
            
            print(f"✅ Ключі експортовано в файл: {filename}")
            
        except Exception as e:
            print(f"❌ Помилка експорту: {e}")
        
        input("\nНатисніть Enter для продовження...")
    
    def run(self):
        """Запускає панель адміністрування"""
        while True:
            self.clear_screen()
            self.show_header()
            self.show_menu()
            
            choice = input("Виберіть опцію (1-7): ").strip()
            
            if choice == "1":
                self.clear_screen()
                self.show_header()
                self.generate_key()
            elif choice == "2":
                self.clear_screen()
                self.show_header()
                self.show_keys()
            elif choice == "3":
                self.clear_screen()
                self.show_header()
                self.revoke_key()
            elif choice == "4":
                self.clear_screen()
                self.show_header()
                self.activate_key()
            elif choice == "5":
                self.clear_screen()
                self.show_header()
                self.show_stats()
            elif choice == "6":
                self.clear_screen()
                self.show_header()
                self.export_keys()
            elif choice == "7":
                print("👋 До побачення!")
                break
            else:
                print("❌ Невірний вибір. Спробуйте ще раз.")
                input("Натисніть Enter для продовження...")

def main():
    """Головна функція"""
    try:
        admin = AdminPanel()
        admin.run()
    except KeyboardInterrupt:
        print("\n👋 До побачення!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Помилка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
