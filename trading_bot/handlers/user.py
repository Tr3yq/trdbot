import logging
from datetime import datetime
from typing import Optional, Dict, List
from telebot import TeleBot
from telebot.types import (
    Message, ReplyKeyboardMarkup, KeyboardButton, 
    InlineKeyboardMarkup, InlineKeyboardButton
)

from config.settings import Config
from database import DatabaseManager, SubscriptionService, SubscriptionType
from utils.helpers import format_datetime, truncate_text


class UserHandlers:
    """Класс обработчиков пользовательских команд"""
    
    def __init__(self, bot: TeleBot, db_manager: DatabaseManager, 
                 config: Config, signal_service):
        self.bot = bot
        self.db = db_manager
        self.config = config
        self.signal_service = signal_service
        self.subscription_service = SubscriptionService(db_manager)
        self.logger = logging.getLogger(__name__)
    
    def register_handlers(self):
        """Регистрация всех пользовательских обработчиков"""
        
        # Команды
        self.bot.message_handler(commands=['start'])(self.cmd_start)
        self.bot.message_handler(commands=['help'])(self.cmd_help)
        # self.bot.message_handler(commands=['subscribe'])(self.cmd_subscribe)
        # self.bot.message_handler(commands=['status'])(self.cmd_status)
        # self.bot.message_handler(commands=['settings'])(self.cmd_settings)
        # self.bot.message_handler(commands=['analyze'])(self.cmd_analyze)
        
        # Текстовые кнопки
        self.bot.message_handler(func=lambda msg: msg.text == "📊 Мой статус")(self.handle_my_status)
        self.bot.message_handler(func=lambda msg: msg.text == "⚙️ Настройки")(self.handle_settings)
        self.bot.message_handler(func=lambda msg: msg.text == "💎 Подписка")(self.handle_subscription)
        self.bot.message_handler(func=lambda msg: msg.text == "📈 Анализ символа")(self.handle_analyze_symbol)
        self.bot.message_handler(func=lambda msg: msg.text == "📚 Обучение")(self.handle_education)
        self.bot.message_handler(func=lambda msg: msg.text == "📋 Тарифы")(self.handle_tariffs)
        self.bot.message_handler(func=lambda msg: msg.text == "🎯 Мои сигналы")(self.handle_my_signals)
        
        # Обработчики настроек
        self.bot.message_handler(func=lambda msg: msg.text and msg.text.startswith("analyze_"))(self.process_symbol_analysis)
        
        self.logger.info("✅ Пользовательские обработчики зарегистрированы")
    
    def create_main_keyboard(self, user_id: int) -> ReplyKeyboardMarkup:
        """Создание основной клавиатуры"""
        keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
        
        # Основные кнопки для всех
        btn_status = KeyboardButton("📊 Мой статус")
        btn_analyze = KeyboardButton("📈 Анализ символа")
        btn_subscription = KeyboardButton("💎 Подписка")
        btn_tariffs = KeyboardButton("📋 Тарифы")
        
        keyboard.add(btn_status, btn_analyze)
        keyboard.add(btn_subscription, btn_tariffs)
        
        # Дополнительные кнопки для активных подписчиков
        can_receive, _, _ = self.subscription_service.can_receive_signals(user_id)
        if can_receive:
            btn_settings = KeyboardButton("⚙️ Настройки")
            btn_signals = KeyboardButton("🎯 Мои сигналы")
            btn_education = KeyboardButton("📚 Обучение")
            
            keyboard.add(btn_settings, btn_signals)
            keyboard.add(btn_education)
        
        return keyboard
    
    def cmd_start(self, message: Message):
        """Команда /start"""
        user_id = message.from_user.id
        
        # Проверяем реферальную ссылку
        referrer_id = None
        if message.text and len(message.text.split()) > 1:
            try:
                ref_code = message.text.split()[1]
                if ref_code.startswith('ref_'):
                    referrer_id = int(ref_code[4:])
            except ValueError:
                pass
        
        # Создаем или обновляем пользователя
        user_data = {
            'user_id': user_id,
            'username': message.from_user.username,
            'first_name': message.from_user.first_name,
            'last_name': message.from_user.last_name,
            'language_code': message.from_user.language_code,
            'referrer_id': referrer_id
        }
        
        user = self.subscription_service.create_or_update_user(user_data)
        if not user:
            self.bot.send_message(user_id, "❌ Ошибка регистрации. Попробуйте позже.")
            return
        
        # Проверяем есть ли активная подписка
        subscription = self.subscription_service.get_active_subscription(user_id)
        
        welcome_text = f"""
🚀 <b>Добро пожаловать в Trading Signals Bot!</b>

Привет, {user.full_name}! 👋

🎯 <b>Что я умею:</b>
• Профессиональные торговые сигналы на основе ИИ
• Технический анализ криптовалютных пар
• Персональные настройки под ваш стиль торговли
• Детальная аналитика и статистика

📊 <b>Ваш статус:</b> {self._get_subscription_status_text(subscription)}

🎁 <b>Новые пользователи получают 7 дней БЕСПЛАТНО!</b>

Выберите действие в меню ниже ⬇️
        """
        
        self.bot.send_message(
            user_id, 
            welcome_text, 
            parse_mode='HTML',
            reply_markup=self.create_main_keyboard(user_id)
        )
    
    def cmd_help(self, message: Message):
        """Команда /help"""
        help_text = """
🆘 <b>Помощь по использованию бота</b>

<b>📋 Основные команды:</b>
/start - Главное меню
/status - Статус подписки
/settings - Персональные настройки
/subscribe - Оформить подписку
/analyze <SYMBOL> - Анализ символа

<b>📊 Функции бота:</b>
🎯 <b>Торговые сигналы</b> - AI-анализ рынка
⚙️ <b>Настройки</b> - Персонализация под вас
📈 <b>Анализ</b> - Технический анализ любой пары
📚 <b>Обучение</b> - Гайды по торговле

<b>🎁 Тарифные планы:</b>
• Trial - 7 дней бесплатно
• Basic - $29.99/месяц (10 сигналов/день)
• Premium - $59.99/месяц (25 сигналов/день) 
• VIP - $99.99/месяц (безлимит)

<b>🔗 Поддержка:</b> @support_username
        """
        
        self.bot.send_message(message.chat.id, help_text, parse_mode='HTML')
    
    def handle_my_status(self, message: Message):
        """Обработчик "Мой статус" """
        user_id = message.from_user.id
        
        user = self.subscription_service.get_user(user_id)
        subscription = self.subscription_service.get_active_subscription(user_id)
        
        if not user:
            self.bot.send_message(user_id, "❌ Пользователь не найден")
            return
        
        # Базовая информация
        status_text = f"""
👤 <b>Информация о пользователе</b>

<b>ID:</b> {user.user_id}
<b>Имя:</b> {user.full_name}
<b>Username:</b> @{user.username or 'не указан'}
<b>Дата регистрации:</b> {format_datetime(user.created_at)}
<b>Последняя активность:</b> {format_datetime(user.last_activity)}

📊 <b>Статус подписки:</b>
{self._get_detailed_subscription_info(subscription)}
        """
        
        if subscription:
            # Информация об использовании
            tariff = self.db.get_tariff_plan(subscription.subscription_type.value)
            if tariff:
                status_text += f"""
📈 <b>Использование сигналов:</b>
• Сегодня использовано: {subscription.signals_used_today}/{tariff.signals_per_day}
• Последний сигнал: {format_datetime(subscription.last_signal_date) if subscription.last_signal_date else 'Не получали'}

⚙️ <b>Доступные функции:</b>
"""
                for feature, enabled in tariff.features.items():
                    status_text += f"• {feature}: {'✅' if enabled else '❌'}\n"
        
        # Реферальная информация
        if user.total_referrals > 0:
            status_text += f"\n👥 <b>Рефералы:</b> {user.total_referrals} человек"
        
        # Кнопки действий
        keyboard = InlineKeyboardMarkup()
        
        if not subscription or subscription.subscription_type.value == 'trial':
            keyboard.add(InlineKeyboardButton("💎 Улучшить подписку", callback_data="upgrade_subscription"))
        
        keyboard.add(InlineKeyboardButton("⚙️ Настройки", callback_data="open_settings"))
        keyboard.add(InlineKeyboardButton("📋 Тарифы", callback_data="show_tariffs"))
        
        self.bot.send_message(user_id, status_text, parse_mode='HTML', reply_markup=keyboard)
    
    def handle_subscription(self, message: Message):
        """Обработчик "Подписка" """
        user_id = message.from_user.id
        
        subscription = self.subscription_service.get_active_subscription(user_id)
        
        if subscription:
            self._show_active_subscription_info(user_id, subscription)
        else:
            self._show_subscription_offers(user_id)
    
    def handle_tariffs(self, message: Message):
        """Обработчик "Тарифы" """
        user_id = message.from_user.id
        
        tariff_text = """
💎 <b>Тарифные планы</b>

🆓 <b>TRIAL</b> (7 дней бесплатно)
• 3 сигнала в день
• Базовая аналитика
• Поддержка в чате

💰 <b>BASIC</b> - $29.99/месяц
• 10 сигналов в день
• Расширенная аналитика
• Приоритетная поддержка
• Скидки: 3 мес (-11%) • 1 год (-17%)

🚀 <b>PREMIUM</b> - $59.99/месяц
• 25 сигналов в день  
• Продвинутая аналитика
• Персональные настройки
• Risk management инструменты
• Скидки: 3 мес (-11%) • 1 год (-17%)

⭐ <b>VIP</b> - $99.99/месяц
• БЕЗЛИМИТНЫЕ сигналы
• Все функции Premium
• Персональный менеджер
• Приоритетный доступ к новинкам
• Скидки: 3 мес (-10%) • 1 год (-17%)

🎁 <b>Бонусы:</b>
• Реферальная программа - 20% с платежей
• Образовательные материалы  
• Telegram-канал с аналитикой
        """
        
        keyboard = InlineKeyboardMarkup()
        
        subscription = self.subscription_service.get_active_subscription(user_id)
        if not subscription:
            keyboard.add(InlineKeyboardButton("🎁 Активировать Trial", callback_data="activate_trial"))
        else:
            keyboard.add(InlineKeyboardButton("💎 Улучшить план", callback_data="upgrade_subscription"))
        
        keyboard.add(InlineKeyboardButton("💳 Купить подписку", callback_data="buy_subscription"))
        keyboard.add(InlineKeyboardButton("👥 Реферальная ссылка", callback_data="get_referral_link"))
        
        self.bot.send_message(user_id, tariff_text, parse_mode='HTML', reply_markup=keyboard)
    
    def handle_settings(self, message: Message):
        """Обработчик "Настройки" """
        user_id = message.from_user.id
        
        # Проверяем доступ к настройкам
        can_receive, reason, tariff = self.subscription_service.can_receive_signals(user_id)
        if not can_receive:
            self.bot.send_message(user_id, f"⚠️ {reason}\nНастройки доступны только активным подписчикам.")
            return
        
        settings = self.subscription_service.get_user_settings(user_id)
        if not settings:
            self.bot.send_message(user_id, "❌ Ошибка загрузки настроек")
            return
        
        settings_text = f"""
⚙️ <b>Персональные настройки</b>

📊 <b>Торговые параметры:</b>
• Уровень риска: {settings.risk_level.title()}
• Мин. уверенность: {settings.min_confidence:.1%}
• Макс. сигналов/день: {settings.max_signals_per_day}

📈 <b>RSI настройки:</b>  
• Уровень перепроданности: {settings.rsi_oversold_level}
• Уровень перекупленности: {settings.rsi_overbought_level}

🔔 <b>Уведомления:</b>
• Сигналы: {'✅' if settings.signal_notifications else '❌'}
• Новости: {'✅' if settings.news_notifications else '❌'}
• Маркетинг: {'✅' if settings.marketing_notifications else '❌'}

🕒 <b>Активные часы:</b> {settings.active_hours_start:02d}:00 - {settings.active_hours_end:02d}:00 ({settings.timezone})

📋 <b>Предпочитаемые символы:</b>
{', '.join(settings.preferred_symbols) if settings.preferred_symbols else 'Все символы'}

🌍 <b>Язык:</b> {settings.language.upper()}
        """
        
        keyboard = InlineKeyboardMarkup(row_width=2)
        
        # Основные настройки
        keyboard.add(
            InlineKeyboardButton("🎯 Торговые настройки", callback_data="settings_trading"),
            InlineKeyboardButton("📊 RSI параметры", callback_data="settings_rsi")
        )
        keyboard.add(
            InlineKeyboardButton("🔔 Уведомления", callback_data="settings_notifications"),
            InlineKeyboardButton("🕒 Время торговли", callback_data="settings_time")
        )
        keyboard.add(
            InlineKeyboardButton("📋 Символы", callback_data="settings_symbols"),
            InlineKeyboardButton("🌍 Язык", callback_data="settings_language")
        )
        
        # Дополнительные функции для Premium+
        if tariff and tariff.custom_settings:
            keyboard.add(
                InlineKeyboardButton("🎛️ Продвинутые", callback_data="settings_advanced"),
                InlineKeyboardButton("📈 Стратегии", callback_data="settings_strategies")
            )
        
        keyboard.add(InlineKeyboardButton("💾 Сохранить по умолчанию", callback_data="settings_save_default"))
        
        self.bot.send_message(user_id, settings_text, parse_mode='HTML', reply_markup=keyboard)
    
    def handle_analyze_symbol(self, message: Message):
        """Обработчик "Анализ символа" """
        user_id = message.from_user.id
        
        msg = self.bot.send_message(
            user_id,
            "📈 <b>Анализ торгового символа</b>\n\n"
            "Введите символ для анализа (например: BTCUSDT, ETHUSDT):\n\n"
            "💡 <i>Доступны все пары Binance Futures</i>",
            parse_mode='HTML'
        )
        
        self.bot.register_next_step_handler(msg, self.process_symbol_analysis)
    
    def process_symbol_analysis(self, message: Message):
        """Обработка ввода символа для анализа"""
        user_id = message.from_user.id
        
        if not message.text:
            self.bot.send_message(user_id, "❌ Введите корректный символ")
            return
        
        symbol = message.text.strip().upper()
        
        # Валидация символа
        if not symbol.endswith('USDT') and not symbol.endswith('BUSD'):
            self.bot.send_message(user_id, "⚠️ Поддерживаются только пары с USDT или BUSD")
            return
        
        # Отправляем сообщение о начале анализа
        processing_msg = self.bot.send_message(
            user_id,
            f"🔄 Анализирую {symbol}...\n\n"
            "⏱️ Это может занять 10-30 секунд",
            parse_mode='HTML'
        )
        
        try:
            # Вызываем сервис анализа (заглушка - будет реализован в services)
            analysis_result = self.signal_service.analyze_single_symbol(symbol)
            
            if analysis_result:
                # Удаляем сообщение о обработке
                self.bot.delete_message(user_id, processing_msg.message_id)
                
                # Отправляем результат
                self.bot.send_message(user_id, analysis_result, parse_mode='HTML')
                
                # Если есть график, отправляем его
                chart_path = f"charts/{symbol}_analysis.png"
                try:
                    with open(chart_path, 'rb') as chart:
                        self.bot.send_photo(user_id, chart)
                except FileNotFoundError:
                    pass
            else:
                self.bot.edit_message_text(
                    f"❌ Не удалось проанализировать {symbol}\n\n"
                    "Возможные причины:\n"
                    "• Символ не найден на Binance\n"
                    "• Недостаточно данных\n"  
                    "• Временная ошибка сервиса",
                    user_id,
                    processing_msg.message_id
                )
        
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа {symbol} для {user_id}: {e}")
            self.bot.edit_message_text(
                f"❌ Произошла ошибка при анализе {symbol}\n\n"
                "Попробуйте позже или обратитесь в поддержку",
                user_id,
                processing_msg.message_id
            )
    
    def handle_education(self, message: Message):
        """Обработчик "Обучение" """
        user_id = message.from_user.id
        
        # Проверяем доступ
        can_receive, reason, _ = self.subscription_service.can_receive_signals(user_id)
        if not can_receive:
            self.bot.send_message(user_id, f"⚠️ {reason}\nОбучающие материалы доступны подписчикам.")
            return
        
        education_text = """
📚 <b>Обучающие материалы</b>

🎓 <b>Доступные курсы:</b>

1️⃣ <b>Основы технического анализа</b>
   • Свечной анализ
   • Поддержки и сопротивления
   • Тренды и паттерны

2️⃣ <b>Индикаторы и осцилляторы</b>
   • RSI, MACD, Stochastic
   • Bollinger Bands, ATR
   • Применение в торговле

3️⃣ <b>Risk Management</b>
   • Управление капиталом
   • Позиционирование
   • Стратегии выхода

4️⃣ <b>Психология трейдинга</b>
   • Дисциплина и эмоции
   • План торговли
   • Ведение журнала

📖 <b>Форматы обучения:</b>
• PDF гайды
• Видео уроки
• Интерактивные примеры
• Тесты и задания
        """
        
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📖 Скачать PDF-гайд", callback_data="download_guide"))
        keyboard.add(InlineKeyboardButton("🎥 Видео курсы", url="https://youtube.com/yourchannel"))
        keyboard.add(InlineKeyboardButton("💬 Telegram канал", url="https://t.me/yourchannel"))
        keyboard.add(InlineKeyboardButton("❓ FAQ", callback_data="show_faq"))
        
        self.bot.send_message(user_id, education_text, parse_mode='HTML', reply_markup=keyboard)
    
    def handle_my_signals(self, message: Message):
        """Обработчик "Мои сигналы" """
        user_id = message.from_user.id
        
        # Проверяем доступ
        can_receive, reason, _ = self.subscription_service.can_receive_signals(user_id)
        if not can_receive:
            self.bot.send_message(user_id, f"⚠️ {reason}")
            return
        
        # Получаем статистику сигналов пользователя (заглушка)
        signals_text = f"""
🎯 <b>Статистика ваших сигналов</b>

📊 <b>За последние 30 дней:</b>
• Всего сигналов: 45
• Прибыльных: 28 (62.2%)
• Убыточных: 17 (37.8%)
• Общая прибыль: +15.6%

📈 <b>По типам:</b>
• LONG сигналы: 24 (58.3% прибыльных)
• SHORT сигналы: 21 (66.7% прибыльных)

⭐ <b>Лучшие символы:</b>
1. BTCUSDT: +8.4% (12 сигналов)
2. ETHUSDT: +4.2% (8 сигналов)  
3. ADAUSDT: +2.1% (5 сигналов)

📉 <b>Средний R:R:</b> 1:2.3
⏱️ <b>Средняя длительность:</b> 4.5 часа

💡 <i>Статистика обновляется в режиме реального времени</i>
        """
        
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📋 Детальная история", callback_data="detailed_signals"))
        keyboard.add(InlineKeyboardButton("📊 Аналитика", callback_data="signals_analytics"))
        keyboard.add(InlineKeyboardButton("📥 Экспорт в CSV", callback_data="export_signals"))
        
        self.bot.send_message(user_id, signals_text, parse_mode='HTML', reply_markup=keyboard)
    
    def _get_subscription_status_text(self, subscription) -> str:
        """Получить текст статуса подписки"""
        if not subscription:
            return "🆓 Нет активной подписки"
        
        if subscription.is_trial:
            days_left = subscription.days_remaining
            return f"🎁 Trial подписка ({days_left} дн. осталось)"
        
        status_map = {
            'basic': '💰 Basic',
            'premium': '🚀 Premium', 
            'vip': '⭐ VIP',
            'lifetime': '♾️ Lifetime'
        }
        
        plan_name = status_map.get(subscription.subscription_type.value, subscription.subscription_type.value)
        
        if subscription.expires_at:
            days_left = subscription.days_remaining
            return f"{plan_name} ({days_left} дн. осталось)"
        else:
            return f"{plan_name} (активна)"
    
    def _get_detailed_subscription_info(self, subscription) -> str:
        """Получить детальную информацию о подписке"""
        if not subscription:
            return """
❌ <b>Подписка не активна</b>

🎁 Хотите попробовать бесплатно?
Активируйте Trial подписку на 7 дней!
            """
        
        info = f"""
✅ <b>{subscription.subscription_type.value.upper()}</b>
• Статус: {subscription.status.value}
• Начало: {format_datetime(subscription.started_at)}
"""
        
        if subscription.expires_at:
            info += f"• Окончание: {format_datetime(subscription.expires_at)}\n"
            days_left = subscription.days_remaining
            if days_left <= 3:
                info += f"⚠️ <b>Осталось {days_left} дней!</b>\n"
        else:
            info += "• Окончание: Бессрочно\n"
        
        if subscription.auto_renewal:
            info += "🔄 Автопродление включено\n"
        
        return info
    
    def _show_active_subscription_info(self, user_id: int, subscription):
        """Показать информацию об активной подписке"""
        tariff = self.db.get_tariff_plan(subscription.subscription_type.value)
        
        info_text = f"""
💎 <b>Ваша подписка</b>

{self._get_detailed_subscription_info(subscription)}

📊 <b>Ваши лимиты:</b>
• Сигналов в день: {tariff.signals_per_day if tariff else 'Неизвестно'}
• Использовано сегодня: {subscription.signals_used_today}

🎯 <b>Доступные функции:</b>
"""
        
        if tariff:
            for feature, enabled in tariff.features.items():
                info_text += f"• {feature.replace('_', ' ').title()}: {'✅' if enabled else '❌'}\n"
        
        keyboard = InlineKeyboardMarkup()
        
        if subscription.subscription_type.value == 'trial':
            keyboard.add(InlineKeyboardButton("💎 Улучшить до Basic", callback_data="upgrade_basic"))
            keyboard.add(InlineKeyboardButton("🚀 Улучшить до Premium", callback_data="upgrade_premium"))
        elif subscription.subscription_type.value == 'basic':
            keyboard.add(InlineKeyboardButton("🚀 Улучшить до Premium", callback_data="upgrade_premium"))
            keyboard.add(InlineKeyboardButton("⭐ Улучшить до VIP", callback_data="upgrade_vip"))
        elif subscription.subscription_type.value == 'premium':
            keyboard.add(InlineKeyboardButton("⭐ Улучшить до VIP", callback_data="upgrade_vip"))
        
        if subscription.auto_renewal:
            keyboard.add(InlineKeyboardButton("🔄 Отключить автопродление", callback_data="disable_auto_renewal"))
        else:
            keyboard.add(InlineKeyboardButton("🔄 Включить автопродление", callback_data="enable_auto_renewal"))
        
        keyboard.add(InlineKeyboardButton("📋 История платежей", callback_data="payment_history"))
        
        self.bot.send_message(user_id, info_text, parse_mode='HTML', reply_markup=keyboard)
    
    def _show_subscription_offers(self, user_id: int):
        """Показать предложения подписок"""
        offers_text = """
🎁 <b>Активируйте подписку!</b>

У вас пока нет активной подписки. Выберите подходящий план:

🆓 <b>TRIAL</b> - Бесплатно на 7 дней
• 3 сигнала в день
• Базовая аналитика

💰 <b>BASIC</b> - $29.99/месяц
• 10 сигналов в день
• Расширенная аналитика

🚀 <b>PREMIUM</b> - $59.99/месяц  
• 25 сигналов в день
• Персональные настройки

⭐ <b>VIP</b> - $99.99/месяц
• БЕЗЛИМИТНЫЕ сигналы
• Персональный менеджер

💡 <i>Начните с бесплатного Trial!</i>
        """
        
        keyboard = InlineKeyboardMarkup()
        
        # Проверяем использовался ли trial
        user = self.subscription_service.get_user(user_id)
        if user:
            # Проверяем был ли уже trial
            try:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT trial_used FROM subscriptions WHERE user_id = %s AND trial_used = TRUE", (user_id,))
                    trial_used = cursor.fetchone()
                
                if not trial_used:
                    keyboard.add(InlineKeyboardButton("🎁 Активировать Trial", callback_data="activate_trial"))
            except:
                pass
        
        keyboard.add(InlineKeyboardButton("💰 Basic - $29.99", callback_data="buy_basic"))
        keyboard.add(InlineKeyboardButton("🚀 Premium - $59.99", callback_data="buy_premium"))  
        keyboard.add(InlineKeyboardButton("⭐ VIP - $99.99", callback_data="buy_vip"))
        keyboard.add(InlineKeyboardButton("📋 Сравнить планы", callback_data="compare_plans"))
        
        self.bot.send_message(user_id, offers_text, parse_mode='HTML', reply_markup=keyboard)


def register_user_handlers(bot: TeleBot, db_manager: DatabaseManager, 
                         config: Config, signal_service):
    """Регистрация пользовательских обработчиков"""
    user_handlers = UserHandlers(bot, db_manager, config, signal_service)
    user_handlers.register_handlers()
    
    return user_handlers