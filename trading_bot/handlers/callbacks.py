import logging
from decimal import Decimal
from telebot import TeleBot
from telebot.types import CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from config.settings import Config
from database import DatabaseManager, SubscriptionService, SubscriptionType
from utils.helpers import format_datetime


class CallbackHandlers:
    """Класс обработчиков callback кнопок"""
    
    def __init__(self, bot: TeleBot, db_manager: DatabaseManager, config: Config):
        self.bot = bot
        self.db = db_manager
        self.config = config
        self.subscription_service = SubscriptionService(db_manager)
        self.logger = logging.getLogger(__name__)
    
    def register_handlers(self):
        """Регистрация всех callback обработчиков"""
        
        # Подписки и платежи
        self.bot.callback_query_handler(func=lambda call: call.data == "activate_trial")(self.activate_trial)
        self.bot.callback_query_handler(func=lambda call: call.data.startswith("buy_"))(self.handle_buy_subscription)
        # self.bot.callback_query_handler(func=lambda call: call.data.startswith("upgrade_"))(self.handle_upgrade)
        
        # Настройки
        self.bot.callback_query_handler(func=lambda call: call.data.startswith("settings_"))(self.handle_settings)
        
        # Информационные
        self.bot.callback_query_handler(func=lambda call: call.data == "show_tariffs")(self.show_tariffs)
        # self.bot.callback_query_handler(func=lambda call: call.data == "compare_plans")(self.compare_plans)
        self.bot.callback_query_handler(func=lambda call: call.data == "get_referral_link")(self.get_referral_link)
        
        # Обучение
        self.bot.callback_query_handler(func=lambda call: call.data == "download_guide")(self.download_guide)
        self.bot.callback_query_handler(func=lambda call: call.data == "show_faq")(self.show_faq)
        
        # Сигналы
        self.bot.callback_query_handler(func=lambda call: call.data == "detailed_signals")(self.detailed_signals)
        self.bot.callback_query_handler(func=lambda call: call.data == "signals_analytics")(self.signals_analytics)
        
        self.logger.info("✅ Callback обработчики зарегистрированы")
    
    def activate_trial(self, call: CallbackQuery):
        """Активация пробной подписки"""
        user_id = call.from_user.id
        
        # Проверяем не использовал ли уже trial
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM subscriptions WHERE user_id = %s AND trial_used = TRUE", (user_id,))
                if cursor.fetchone():
                    self.bot.answer_callback_query(
                        call.id, 
                        "⚠️ Вы уже использовали пробную подписку", 
                        show_alert=True
                    )
                    return
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки trial для {user_id}: {e}")
            self.bot.answer_callback_query(call.id, "❌ Ошибка проверки")
            return
        
        # Создаем trial подписку
        subscription = self.subscription_service.create_trial_subscription(user_id, self.config.TRIAL_PERIOD_DAYS)
        
        if subscription:
            success_text = f"""
🎉 <b>Trial подписка активирована!</b>

✅ <b>Период:</b> {self.config.TRIAL_PERIOD_DAYS} дней
✅ <b>Сигналов в день:</b> 3
✅ <b>Действует до:</b> {format_datetime(subscription.expires_at)}

🚀 <b>Теперь вам доступны:</b>
• Торговые сигналы с ИИ
• Базовая аналитика рынка
• Технический анализ символов
• Обучающие материалы

💡 <i>Не забудьте настроить персональные параметры в разделе "⚙️ Настройки"</i>
            """
            
            keyboard = InlineKeyboardMarkup()
            keyboard.add(InlineKeyboardButton("⚙️ Настроить параметры", callback_data="settings_trading"))
            keyboard.add(InlineKeyboardButton("📈 Анализировать символ", callback_data="analyze_symbol"))
            
            self.bot.edit_message_text(
                success_text,
                call.message.chat.id,
                call.message.message_id,
                parse_mode='HTML',
                reply_markup=keyboard
            )
            
            self.bot.answer_callback_query(call.id, "✅ Trial активирован!")
        else:
            self.bot.answer_callback_query(
                call.id,
                "❌ Ошибка активации Trial",
                show_alert=True
            )
    
    def handle_buy_subscription(self, call: CallbackQuery):
        """Обработка покупки подписки"""
        user_id = call.from_user.id
        
        # Извлекаем тип подписки из callback_data
        subscription_type = call.data.split("_")[1]  # buy_basic -> basic
        
        tariff = self.db.get_tariff_plan(subscription_type)
        if not tariff:
            self.bot.answer_callback_query(call.id, "❌ Тариф не найден")
            return
        
        # Создаем меню выбора периода оплаты
        payment_text = f"""
💳 <b>Оплата подписки {tariff.name}</b>

📋 <b>Выберите период:</b>

💰 <b>1 месяц</b> - ${tariff.price_monthly}
💎 <b>3 месяца</b> - ${tariff.price_quarterly} <i>(экономия 11%)</i>
⭐ <b>1 год</b> - ${tariff.price_yearly} <i>(экономия 17%)</i>

🎯 <b>Что входит в {tariff.name}:</b>
• {tariff.signals_per_day} сигналов в день
• {"Приоритетная поддержка" if tariff.priority_support else "Обычная поддержка"}
• {"Персональные настройки" if tariff.custom_settings else "Стандартные настройки"}
• {"Расширенная аналитика" if tariff.analytics_access else "Базовая аналитика"}

💳 <b>Способы оплаты:</b> Card, Crypto, PayPal
        """
        
        keyboard = InlineKeyboardMarkup()
        keyboard.add(
            InlineKeyboardButton(f"💰 1 мес - ${tariff.price_monthly}", 
                               callback_data=f"pay_{subscription_type}_1")
        )
        keyboard.add(
            InlineKeyboardButton(f"💎 3 мес - ${tariff.price_quarterly}", 
                               callback_data=f"pay_{subscription_type}_3")
        )
        keyboard.add(
            InlineKeyboardButton(f"⭐ 1 год - ${tariff.price_yearly}", 
                               callback_data=f"pay_{subscription_type}_12")
        )
        keyboard.add(InlineKeyboardButton("◀️ Назад", callback_data="show_tariffs"))
        
        self.bot.edit_message_text(
            payment_text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        self.bot.answer_callback_query(call.id)
    
    def handle_settings(self, call: CallbackQuery):
        """Обработка настроек"""
        user_id = call.from_user.id
        setting_type = call.data.split("_")[1]  # settings_trading -> trading
        
        # Проверяем доступ
        can_receive, reason, tariff = self.subscription_service.can_receive_signals(user_id)
        if not can_receive:
            self.bot.answer_callback_query(call.id, f"⚠️ {reason}", show_alert=True)
            return
        
        settings = self.subscription_service.get_user_settings(user_id)
        if not settings:
            self.bot.answer_callback_query(call.id, "❌ Ошибка загрузки настроек")
            return
        
        if setting_type == "trading":
            self._show_trading_settings(call, settings)
        elif setting_type == "rsi":
            self._show_rsi_settings(call, settings)
        elif setting_type == "notifications":
            self._show_notification_settings(call, settings)
        elif setting_type == "time":
            self._show_time_settings(call, settings)
        elif setting_type == "symbols":
            self._show_symbols_settings(call, settings)
        elif setting_type == "language":
            self._show_language_settings(call, settings)
        elif setting_type == "advanced" and tariff.custom_settings:
            self._show_advanced_settings(call, settings)
        else:
            self.bot.answer_callback_query(call.id, "❌ Настройка недоступна")
    
    def _show_trading_settings(self, call: CallbackQuery, settings):
        """Показать торговые настройки"""
        trading_text = f"""
🎯 <b>Торговые настройки</b>

📊 <b>Текущие параметры:</b>
• Уровень риска: <b>{settings.risk_level.title()}</b>
• Мин. уверенность: <b>{settings.min_confidence:.1%}</b>
• Макс. сигналов/день: <b>{settings.max_signals_per_day}</b>

💡 <b>Описание уровней риска:</b>

🛡️ <b>Conservative</b> - Низкий риск
• Мин. уверенность: 80%+
• Только сильные сигналы
• R:R от 1:3

⚖️ <b>Moderate</b> - Умеренный риск  
• Мин. уверенность: 65%+
• Баланс риска и прибыли
• R:R от 1:2

🔥 <b>Aggressive</b> - Высокий риск
• Мин. уверенность: 50%+
• Больше сигналов
• R:R от 1:1.5
        """
        
        keyboard = InlineKeyboardMarkup()
        
        # Уровень риска
        risk_levels = [
            ("🛡️ Conservative", "conservative"),
            ("⚖️ Moderate", "moderate"), 
            ("🔥 Aggressive", "aggressive")
        ]
        
        for name, value in risk_levels:
            marker = "✅" if settings.risk_level == value else ""
            keyboard.add(
                InlineKeyboardButton(
                    f"{marker} {name}",
                    callback_data=f"set_risk_{value}"
                )
            )
        
        # Настройка уверенности
        confidence_levels = [
            ("50%", 0.5), ("60%", 0.6), ("70%", 0.7), ("80%", 0.8), ("90%", 0.9)
        ]
        
        conf_buttons = []
        for name, value in confidence_levels:
            marker = "✅" if abs(settings.min_confidence - value) < 0.05 else ""
            conf_buttons.append(
                InlineKeyboardButton(
                    f"{marker} {name}",
                    callback_data=f"set_confidence_{int(value*100)}"
                )
            )
        
        # Добавляем по 3 кнопки в ряд
        for i in range(0, len(conf_buttons), 3):
            keyboard.add(*conf_buttons[i:i+3])
        
        keyboard.add(InlineKeyboardButton("◀️ К настройкам", callback_data="open_settings"))
        
        self.bot.edit_message_text(
            trading_text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        self.bot.answer_callback_query(call.id)
    
    def _show_rsi_settings(self, call: CallbackQuery, settings):
        """Показать настройки RSI"""
        rsi_text = f"""
📊 <b>Настройки RSI индикатора</b>

📈 <b>Текущие уровни:</b>
• Перепроданность (Long): <b>{settings.rsi_oversold_level}</b>
• Перекупленность (Short): <b>{settings.rsi_overbought_level}</b>

💡 <b>Рекомендуемые значения:</b>

🔵 <b>Консервативные:</b>
• Long: 25-30 (меньше сигналов, выше точность)
• Short: 70-75

⚪ <b>Стандартные:</b>
• Long: 30-35 (баланс)
• Short: 65-70

🟡 <b>Агрессивные:</b>
• Long: 35-40 (больше сигналов)
• Short: 60-65

⚠️ <i>Более экстремальные значения дают меньше ложных сигналов, но могут пропустить возможности</i>
        """
        
        keyboard = InlineKeyboardMarkup()
        
        # RSI Long (перепроданность)
        keyboard.add(InlineKeyboardButton("📈 Настройки для LONG сигналов", callback_data="rsi_section_long"))
        
        long_levels = [20, 25, 30, 35, 40]
        long_buttons = []
        for level in long_levels:
            marker = "✅" if settings.rsi_oversold_level == level else ""
            long_buttons.append(
                InlineKeyboardButton(
                    f"{marker} {level}",
                    callback_data=f"set_rsi_long_{level}"
                )
            )
        
        keyboard.add(*long_buttons)
        
        # RSI Short (перекупленность) 
        keyboard.add(InlineKeyboardButton("📉 Настройки для SHORT сигналов", callback_data="rsi_section_short"))
        
        short_levels = [60, 65, 70, 75, 80]
        short_buttons = []
        for level in short_levels:
            marker = "✅" if settings.rsi_overbought_level == level else ""
            short_buttons.append(
                InlineKeyboardButton(
                    f"{marker} {level}",
                    callback_data=f"set_rsi_short_{level}"
                )
            )
        
        keyboard.add(*short_buttons)
        keyboard.add(InlineKeyboardButton("◀️ К настройкам", callback_data="open_settings"))
        
        self.bot.edit_message_text(
            rsi_text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        self.bot.answer_callback_query(call.id)
    
    def _show_notification_settings(self, call: CallbackQuery, settings):
        """Показать настройки уведомлений"""
        notif_text = f"""
🔔 <b>Настройки уведомлений</b>

📱 <b>Текущие настройки:</b>
• Торговые сигналы: {'✅ Включены' if settings.signal_notifications else '❌ Отключены'}
• Новости рынка: {'✅ Включены' if settings.news_notifications else '❌ Отключены'}
• Маркетинговые: {'✅ Включены' if settings.marketing_notifications else '❌ Отключены'}

💡 <b>Рекомендации:</b>
• <b>Сигналы</b> - всегда включайте для получения торговых возможностей
• <b>Новости</b> - важные события, влияющие на рынок
• <b>Маркетинг</b> - специальные предложения и обновления бота

⏰ <b>Время доставки:</b> {settings.active_hours_start:02d}:00 - {settings.active_hours_end:02d}:00 ({settings.timezone})
        """
        
        keyboard = InlineKeyboardMarkup()
        
        # Переключатели уведомлений
        notifications = [
            ("🎯 Торговые сигналы", "signal_notifications", settings.signal_notifications),
            ("📰 Новости рынка", "news_notifications", settings.news_notifications),
            ("📢 Маркетинговые", "marketing_notifications", settings.marketing_notifications)
        ]
        
        for name, key, value in notifications:
            status = "✅ ВКЛ" if value else "❌ ВЫКЛ"
            keyboard.add(
                InlineKeyboardButton(
                    f"{name}: {status}",
                    callback_data=f"toggle_{key}"
                )
            )
        
        keyboard.add(InlineKeyboardButton("🕒 Настроить время", callback_data="settings_time"))
        keyboard.add(InlineKeyboardButton("◀️ К настройкам", callback_data="open_settings"))
        
        self.bot.edit_message_text(
            notif_text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        self.bot.answer_callback_query(call.id)
    
    def show_tariffs(self, call: CallbackQuery):
        """Показать тарифы"""
        tariffs_text = """
💎 <b>Сравнение тарифных планов</b>

🆓 <b>TRIAL</b> - Бесплатно (7 дней)
├ 3 сигнала в день
├ Базовая аналитика
└ Обычная поддержка

💰 <b>BASIC</b> - $29.99/мес
├ 10 сигналов в день
├ Расширенная аналитика  
├ Приоритетная поддержка
└ Скидки: 3м (-11%) | 1г (-17%)

🚀 <b>PREMIUM</b> - $59.99/мес
├ 25 сигналов в день
├ Продвинутая аналитика
├ Персональные настройки
├ Risk management
└ Скидки: 3м (-11%) | 1г (-17%)

⭐ <b>VIP</b> - $99.99/мес
├ БЕЗЛИМИТНЫЕ сигналы
├ Все функции Premium
├ Персональный менеджер
├ Приоритетный доступ
└ Скидки: 3м (-10%) | 1г (-17%)

🎁 <b>Дополнительно для всех:</b>
• Telegram канал с аналитикой
• Обучающие материалы
• Реферальная программа (20%)
        """
        
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("🎁 Активировать Trial", callback_data="activate_trial"))
        keyboard.add(InlineKeyboardButton("💰 Купить Basic", callback_data="buy_basic"))
        keyboard.add(InlineKeyboardButton("🚀 Купить Premium", callback_data="buy_premium"))
        keyboard.add(InlineKeyboardButton("⭐ Купить VIP", callback_data="buy_vip"))
        keyboard.add(InlineKeyboardButton("👥 Реферальная ссылка", callback_data="get_referral_link"))
        
        try:
            self.bot.edit_message_text(
                tariffs_text,
                call.message.chat.id,
                call.message.message_id,
                parse_mode='HTML',
                reply_markup=keyboard
            )
        except:
            self.bot.send_message(
                call.message.chat.id,
                tariffs_text,
                parse_mode='HTML',
                reply_markup=keyboard
            )
        
        self.bot.answer_callback_query(call.id)
    
    def get_referral_link(self, call: CallbackQuery):
        """Получить реферальную ссылку"""
        user_id = call.from_user.id
        
        # Генерируем реферальную ссылку
        bot_username = self.bot.get_me().username
        referral_link = f"https://t.me/{bot_username}?start=ref_{user_id}"
        
        # Получаем статистику рефералов
        user = self.subscription_service.get_user(user_id)
        
        referral_text = f"""
👥 <b>Ваша реферальная программа</b>

🔗 <b>Ваша ссылка:</b>
<code>{referral_link}</code>

📊 <b>Статистика:</b>
• Приглашено: {user.total_referrals if user else 0} человек
• Заработано: $0.00 (скоро будет доступно)

💰 <b>Как это работает:</b>
• Делитесь ссылкой с друзьями
• Получайте 20% с их первой оплаты
• Выплаты каждые 2 недели

🎯 <b>Советы для привлечения:</b>
• Покажите свои успешные сигналы
• Расскажите о преимуществах
• Поделитесь в соцсетях

💡 <i>Чем больше активных рефералов, тем больше ваш пассивный доход!</i>
        """
        
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📋 Копировать ссылку", url=referral_link))
        keyboard.add(InlineKeyboardButton("📊 История рефералов", callback_data="referral_history"))
        keyboard.add(InlineKeyboardButton("◀️ Назад", callback_data="show_tariffs"))
        
        self.bot.edit_message_text(
            referral_text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        self.bot.answer_callback_query(call.id, "💡 Поделитесь ссылкой и зарабатывайте!")
    
    def download_guide(self, call: CallbackQuery):
        """Скачать обучающий гайд"""
        user_id = call.from_user.id
        
        # Проверяем доступ
        can_receive, reason, _ = self.subscription_service.can_receive_signals(user_id)
        if not can_receive:
            self.bot.answer_callback_query(call.id, f"⚠️ {reason}", show_alert=True)
            return
        
        try:
            # Отправляем файл (путь из конфига)
            with open(self.config.TRAINING_FILE_PATH, "rb") as file:
                self.bot.send_document(
                    user_id,
                    file,
                    caption="""
📚 <b>Полный гайд по техническому анализу</b>

📖 <b>Что внутри:</b>
• Основы технического анализа
• Подробное описание индикаторов
• Стратегии входа и выхода  
• Управление рисками
• Психология трейдинга

💡 <i>Изучите материал для лучшего понимания торговых сигналов!</i>
                    """,
                    parse_mode='HTML'
                )
            
            self.bot.answer_callback_query(call.id, "✅ Гайд отправлен!")
            
        except FileNotFoundError:
            self.bot.answer_callback_query(
                call.id,
                "❌ Файл временно недоступен",
                show_alert=True
            )
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки файла для {user_id}: {e}")
            self.bot.answer_callback_query(call.id, "❌ Ошибка отправки")
    
    def show_faq(self, call: CallbackQuery):
        """Показать FAQ"""
        faq_text = """
❓ <b>Часто задаваемые вопросы</b>

<b>Q: Как работают торговые сигналы?</b>
A: Наш ИИ анализирует технические индикаторы, объемы и рыночные условия для генерации сигналов входа с указанием Stop Loss и Take Profit уровней.

<b>Q: Какая точность сигналов?</b>  
A: Средняя точность 65-75% в зависимости от рыночных условий. Мы фокусируемся на R:R не менее 1:2.

<b>Q: Можно ли настроить параметры под себя?</b>
A: Да! В Premium и VIP планах доступны персональные настройки RSI, уровня риска, предпочитаемых символов и времени торговли.

<b>Q: Поддерживаете ли автоторговлю?</b>
A: Пока нет, но эта функция в разработке. Сейчас сигналы отправляются для ручного исполнения.

<b>Q: Как получить возврат средств?</b>  
A: Возврат возможен в течение 7 дней с момента покупки при определенных условиях. Обращайтесь в поддержку.

<b>Q: В каких таймфреймах работают сигналы?</b>
A: По умолчанию 15m, но в настройках можно выбрать от 5m до 4h в зависимости от вашей стратегии.

<b>Q: Работает ли бот 24/7?</b>
A: Да, анализ рынка ведется круглосуточно, но вы можете настроить часы получения уведомлений.
        """
        
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("💬 Связаться с поддержкой", url="https://t.me/support_username"))
        keyboard.add(InlineKeyboardButton("📚 Обучающие материалы", callback_data="download_guide"))
        keyboard.add(InlineKeyboardButton("◀️ Назад", callback_data="download_guide"))
        
        self.bot.edit_message_text(
            faq_text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        self.bot.answer_callback_query(call.id)
    
    def detailed_signals(self, call: CallbackQuery):
        """Детальная история сигналов"""
        user_id = call.from_user.id
        
        # Проверяем доступ
        can_receive, reason, _ = self.subscription_service.can_receive_signals(user_id)
        if not can_receive:
            self.bot.answer_callback_query(call.id, f"⚠️ {reason}", show_alert=True)
            return
        
        # Получаем последние сигналы (заглушка - будет реализовано в services)
        history_text = """
📋 <b>История ваших сигналов</b>

🟢 <b>#1247 BTCUSDT LONG</b> - 2024-12-15 14:30
├ Вход: $42,150 | SL: $41,200 | TP: $44,050
├ Результат: ✅ TP достигнут (+4.5%)
└ Длительность: 3h 25m

🔴 <b>#1246 ETHUSDT SHORT</b> - 2024-12-15 11:15  
├ Вход: $2,285 | SL: $2,320 | TP: $2,210
├ Результат: ❌ SL сработал (-1.5%)
└ Длительность: 1h 48m

🟢 <b>#1245 ADAUSDT LONG</b> - 2024-12-15 08:45
├ Вход: $0.385 | SL: $0.375 | TP: $0.405  
├ Результат: ✅ TP достигнут (+5.2%)
└ Длительность: 5h 12m

🟢 <b>#1244 SOLUSDT LONG</b> - 2024-12-14 16:20
├ Вход: $98.50 | SL: $95.20 | TP: $105.80
├ Результат: ✅ TP достигнут (+7.4%)
└ Длительность: 8h 35m

🟡 <b>#1243 DOTUSDT SHORT</b> - 2024-12-14 13:10
├ Вход: $7.25 | SL: $7.55 | TP: $6.65
├ Результат: ⏳ Активен
└ Прошло: 1d 4h

📊 <b>За неделю:</b> +11.6% | Win Rate: 75%
        """
        
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📊 Подробная аналитика", callback_data="signals_analytics"))
        keyboard.add(InlineKeyboardButton("📥 Экспорт в CSV", callback_data="export_signals"))
        keyboard.add(InlineKeyboardButton("🎯 Фильтры", callback_data="signals_filters"))
        keyboard.add(InlineKeyboardButton("◀️ К статистике", callback_data="my_signals"))
        
        self.bot.edit_message_text(
            history_text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        self.bot.answer_callback_query(call.id)
    
    def signals_analytics(self, call: CallbackQuery):
        """Аналитика сигналов"""
        user_id = call.from_user.id
        
        analytics_text = """
📈 <b>Детальная аналитика сигналов</b>

📊 <b>Общая производительность (30 дней):</b>
• Всего сигналов: 45
• Прибыльных: 28 (62.2%)
• Убыточных: 17 (37.8%)  
• Общая прибыль: +15.6%
• Средний R:R: 1:2.3

📈 <b>По направлениям:</b>
• LONG: 24 сигнала (58.3% винрейт)
  └ Средняя прибыль: +3.2% за сделку
• SHORT: 21 сигнал (66.7% винрейт)  
  └ Средняя прибыль: +2.8% за сделку

⭐ <b>ТОП символы по прибыли:</b>
1. BTCUSDT: +8.4% (12 сигналов, 75% WR)
2. ETHUSDT: +4.2% (8 сигналов, 62.5% WR)
3. ADAUSDT: +2.1% (5 сигналов, 80% WR)
4. SOLUSDT: +1.8% (6 сигналов, 50% WR)

📉 <b>Худшие символы:</b>
1. XRPUSDT: -2.1% (4 сигнала, 25% WR)
2. DOTUSDT: -1.5% (3 сигнала, 33% WR)

⏱️ <b>По времени удержания:</b>
• < 2h: 65% винрейт (быстрые движения)
• 2-6h: 71% винрейт (основная масса)  
• > 6h: 55% винрейт (долгосрочные)

🎯 <b>Рекомендации:</b>
• Увеличьте позиции на BTCUSDT и ETHUSDT
• Избегайте или уменьшите позиции на XRPUSDT  
• Оптимальное время удержания: 2-6 часов
        """
        
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("📋 История сигналов", callback_data="detailed_signals"))
        keyboard.add(InlineKeyboardButton("⚙️ Настроить символы", callback_data="settings_symbols"))
        keyboard.add(InlineKeyboardButton("📊 Графики", callback_data="analytics_charts"))
        keyboard.add(InlineKeyboardButton("◀️ К статистике", callback_data="my_signals"))
        
        self.bot.edit_message_text(
            analytics_text,
            call.message.chat.id,
            call.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
        
        self.bot.answer_callback_query(call.id)
    
    # Обработчики изменения настроек
    def handle_setting_change(self, call: CallbackQuery):
        """Обработка изменения настроек"""
        user_id = call.from_user.id
        
        # Парсим callback_data
        if call.data.startswith("set_risk_"):
            risk_level = call.data.split("_")[-1]
            self._update_risk_level(user_id, risk_level, call)
            
        elif call.data.startswith("set_confidence_"):
            confidence = int(call.data.split("_")[-1]) / 100
            self._update_confidence(user_id, confidence, call)
            
        elif call.data.startswith("set_rsi_long_"):
            level = int(call.data.split("_")[-1])
            self._update_rsi_long(user_id, level, call)
            
        elif call.data.startswith("set_rsi_short_"):
            level = int(call.data.split("_")[-1])
            self._update_rsi_short(user_id, level, call)
            
        elif call.data.startswith("toggle_"):
            setting_name = call.data.split("toggle_", 1)[1]
            self._toggle_notification(user_id, setting_name, call)
    
    def _update_risk_level(self, user_id: int, risk_level: str, call: CallbackQuery):
        """Обновить уровень риска"""
        success = self.subscription_service.update_user_settings(
            user_id, 
            {"risk_level": risk_level}
        )
        
        if success:
            self.bot.answer_callback_query(
                call.id, 
                f"✅ Уровень риска изменен на {risk_level.title()}"
            )
            # Обновляем сообщение
            settings = self.subscription_service.get_user_settings(user_id)
            if settings:
                self._show_trading_settings(call, settings)
        else:
            self.bot.answer_callback_query(call.id, "❌ Ошибка сохранения")
    
    def _update_confidence(self, user_id: int, confidence: float, call: CallbackQuery):
        """Обновить минимальную уверенность"""
        success = self.subscription_service.update_user_settings(
            user_id, 
            {"min_confidence": confidence}
        )
        
        if success:
            self.bot.answer_callback_query(
                call.id, 
                f"✅ Мин. уверенность изменена на {confidence:.0%}"
            )
            settings = self.subscription_service.get_user_settings(user_id)
            if settings:
                self._show_trading_settings(call, settings)
        else:
            self.bot.answer_callback_query(call.id, "❌ Ошибка сохранения")
    
    def _update_rsi_long(self, user_id: int, level: int, call: CallbackQuery):
        """Обновить RSI уровень для Long"""
        success = self.subscription_service.update_user_settings(
            user_id, 
            {"rsi_oversold_level": level}
        )
        
        if success:
            self.bot.answer_callback_query(
                call.id, 
                f"✅ RSI Long установлен на {level}"
            )
            settings = self.subscription_service.get_user_settings(user_id)
            if settings:
                self._show_rsi_settings(call, settings)
        else:
            self.bot.answer_callback_query(call.id, "❌ Ошибка сохранения")
    
    def _update_rsi_short(self, user_id: int, level: int, call: CallbackQuery):
        """Обновить RSI уровень для Short"""
        success = self.subscription_service.update_user_settings(
            user_id, 
            {"rsi_overbought_level": level}
        )
        
        if success:
            self.bot.answer_callback_query(
                call.id, 
                f"✅ RSI Short установлен на {level}"
            )
            settings = self.subscription_service.get_user_settings(user_id)
            if settings:
                self._show_rsi_settings(call, settings)
        else:
            self.bot.answer_callback_query(call.id, "❌ Ошибка сохранения")
    
    def _toggle_notification(self, user_id: int, setting_name: str, call: CallbackQuery):
        """Переключить настройку уведомлений"""
        settings = self.subscription_service.get_user_settings(user_id)
        if not settings:
            self.bot.answer_callback_query(call.id, "❌ Ошибка загрузки настроек")
            return
        
        current_value = getattr(settings, setting_name, False)
        new_value = not current_value
        
        success = self.subscription_service.update_user_settings(
            user_id, 
            {setting_name: new_value}
        )
        
        if success:
            status = "включены" if new_value else "отключены"
            self.bot.answer_callback_query(
                call.id, 
                f"✅ Уведомления {status}"
            )
            
            # Обновляем настройки
            updated_settings = self.subscription_service.get_user_settings(user_id)
            if updated_settings:
                self._show_notification_settings(call, updated_settings)
        else:
            self.bot.answer_callback_query(call.id, "❌ Ошибка сохранения")


def register_callback_handlers(bot: TeleBot, db_manager: DatabaseManager, config: Config):
    """Регистрация callback обработчиков"""
    callback_handlers = CallbackHandlers(bot, db_manager, config)
    callback_handlers.register_handlers()
    
    # Регистрируем обработчики изменения настроек
    bot.callback_query_handler(
        func=lambda call: call.data.startswith(("set_", "toggle_"))
    )(callback_handlers.handle_setting_change)
    
    return callback_handlers