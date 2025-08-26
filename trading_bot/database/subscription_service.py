import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Tuple
from mysql.connector import Error

from .connection import DatabaseManager
from .models import (
    User, Subscription, Payment, UserSettings, ReferralProgram,
    SubscriptionType, SubscriptionStatus, PaymentStatus, TariffPlan
)


class SubscriptionService:
    """Сервис управления подписками"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
    
    # === ПОЛЬЗОВАТЕЛИ ===
    
    def create_or_update_user(self, user_data: Dict) -> User:
        """Создание или обновление пользователя"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем существует ли пользователь
                cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_data['user_id'],))
                exists = cursor.fetchone()
                
                if exists:
                    # Обновляем существующего
                    cursor.execute("""
                        UPDATE users SET 
                            username = %s, first_name = %s, last_name = %s, 
                            language_code = %s, last_activity = NOW()
                        WHERE user_id = %s
                    """, (
                        user_data.get('username'),
                        user_data.get('first_name'),
                        user_data.get('last_name'),
                        user_data.get('language_code'),
                        user_data['user_id']
                    ))
                else:
                    # Создаем нового
                    cursor.execute("""
                        INSERT INTO users (user_id, username, first_name, last_name, language_code, referrer_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        user_data['user_id'],
                        user_data.get('username'),
                        user_data.get('first_name'), 
                        user_data.get('last_name'),
                        user_data.get('language_code'),
                        user_data.get('referrer_id')
                    ))
                    
                    # Создаем настройки по умолчанию
                    self._create_default_user_settings(user_data['user_id'], cursor)
                
                conn.commit()
                return self.get_user(user_data['user_id'])
                
        except Error as e:
            self.logger.error(f"❌ Ошибка создания/обновления пользователя {user_data['user_id']}: {e}")
            raise
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Получить пользователя по ID"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, username, first_name, last_name, language_code,
                           created_at, last_activity, is_banned, ban_reason, 
                           referrer_id, total_referrals
                    FROM users WHERE user_id = %s
                """, (user_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return User(*row)
                
        except Error as e:
            self.logger.error(f"❌ Ошибка получения пользователя {user_id}: {e}")
            return None
    
    def ban_user(self, user_id: int, reason: str) -> bool:
        """Заблокировать пользователя"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET is_banned = TRUE, ban_reason = %s WHERE user_id = %s
                """, (reason, user_id))
                
                # Деактивируем все подписки
                cursor.execute("""
                    UPDATE subscriptions SET status = 'suspended', suspension_reason = %s 
                    WHERE user_id = %s AND status = 'active'
                """, (f"User banned: {reason}", user_id))
                
                conn.commit()
                self.logger.info(f"🚫 Пользователь {user_id} заблокирован: {reason}")
                return True
                
        except Error as e:
            self.logger.error(f"❌ Ошибка блокировки пользователя {user_id}: {e}")
            return False
    
    def unban_user(self, user_id: int) -> bool:
        """Разблокировать пользователя"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET is_banned = FALSE, ban_reason = NULL WHERE user_id = %s
                """, (user_id,))
                
                # Восстанавливаем активные подписки если не истекли
                cursor.execute("""
                    UPDATE subscriptions SET status = 'active', suspension_reason = NULL
                    WHERE user_id = %s AND status = 'suspended' 
                    AND (expires_at IS NULL OR expires_at > NOW())
                """, (user_id,))
                
                conn.commit()
                self.logger.info(f"✅ Пользователь {user_id} разблокирован")
                return True
                
        except Error as e:
            self.logger.error(f"❌ Ошибка разблокировки пользователя {user_id}: {e}")
            return False
    
    # === ПОДПИСКИ ===
    
    def create_trial_subscription(self, user_id: int, duration_days: int = 7) -> Optional[Subscription]:
        """Создание пробной подписки"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем, не использовал ли уже пробную подписку
                cursor.execute("""
                    SELECT id FROM subscriptions WHERE user_id = %s AND trial_used = TRUE
                """, (user_id,))
                
                if cursor.fetchone():
                    self.logger.warning(f"⚠️ Пользователь {user_id} уже использовал пробную подписку")
                    return None
                
                # Деактивируем другие активные подписки
                cursor.execute("""
                    UPDATE subscriptions SET status = 'canceled' 
                    WHERE user_id = %s AND status = 'active'
                """, (user_id,))
                
                # Создаем пробную подписку
                expires_at = datetime.now() + timedelta(days=duration_days)
                cursor.execute("""
                    INSERT INTO subscriptions (
                        user_id, subscription_type, status, expires_at, trial_used
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (user_id, SubscriptionType.TRIAL.value, SubscriptionStatus.ACTIVE.value, expires_at, True))
                
                subscription_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"✅ Создана пробная подписка для {user_id} на {duration_days} дней")
                return self.get_subscription(subscription_id)
                
        except Error as e:
            self.logger.error(f"❌ Ошибка создания пробной подписки для {user_id}: {e}")
            return None
    
    def create_paid_subscription(self, user_id: int, subscription_type: SubscriptionType, 
                               duration_months: int, payment_id: int) -> Optional[Subscription]:
        """Создание платной подписки"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Деактивируем другие активные подписки
                cursor.execute("""
                    UPDATE subscriptions SET status = 'canceled' 
                    WHERE user_id = %s AND status = 'active'
                """, (user_id,))
                
                # Рассчитываем срок действия
                expires_at = datetime.now() + timedelta(days=30 * duration_months)
                
                # Создаем платную подписку
                cursor.execute("""
                    INSERT INTO subscriptions (
                        user_id, subscription_type, status, expires_at, auto_renewal
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (user_id, subscription_type.value, SubscriptionStatus.ACTIVE.value, expires_at, True))
                
                subscription_id = cursor.lastrowid
                
                # Связываем с платежом
                cursor.execute("""
                    UPDATE payments SET subscription_id = %s WHERE id = %s
                """, (subscription_id, payment_id))
                
                conn.commit()
                
                self.logger.info(f"✅ Создана платная подписка {subscription_type.value} для {user_id}")
                return self.get_subscription(subscription_id)
                
        except Error as e:
            self.logger.error(f"❌ Ошибка создания платной подписки для {user_id}: {e}")
            return None
    
    def get_subscription(self, subscription_id: int) -> Optional[Subscription]:
        """Получить подписку по ID"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, subscription_type, status, started_at, expires_at,
                           auto_renewal, payment_method, created_at, updated_at,
                           signals_used_today, last_signal_date, trial_used,
                           grace_period_until, suspension_reason
                    FROM subscriptions WHERE id = %s
                """, (subscription_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return Subscription(*row)
                
        except Error as e:
            self.logger.error(f"❌ Ошибка получения подписки {subscription_id}: {e}")
            return None
    
    def get_active_subscription(self, user_id: int) -> Optional[Subscription]:
        """Получить активную подписку пользователя"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, subscription_type, status, started_at, expires_at,
                           auto_renewal, payment_method, created_at, updated_at,
                           signals_used_today, last_signal_date, trial_used,
                           grace_period_until, suspension_reason
                    FROM subscriptions 
                    WHERE user_id = %s AND status = 'active'
                    ORDER BY created_at DESC LIMIT 1
                """, (user_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                subscription = Subscription(*row)
                
                # Дополнительная проверка активности
                if subscription.expires_at and subscription.expires_at < datetime.now():
                    self._expire_subscription(subscription.id)
                    return None
                
                return subscription
                
        except Error as e:
            self.logger.error(f"❌ Ошибка получения активной подписки для {user_id}: {e}")
            return None
    
    def can_receive_signals(self, user_id: int) -> Tuple[bool, str, Optional[TariffPlan]]:
        """Проверить может ли пользователь получать сигналы"""
        # Проверяем заблокирован ли пользователь
        user = self.get_user(user_id)
        if not user:
            return False, "Пользователь не найден", None
        
        if user.is_banned:
            return False, "Пользователь заблокирован", None
        
        # Получаем активную подписку
        subscription = self.get_active_subscription(user_id)
        if not subscription:
            return False, "Нет активной подписки", None
        
        # Получаем тарифный план
        tariff = self.db.get_tariff_plan(subscription.subscription_type.value)
        if not tariff:
            return False, "Тарифный план не найден", None
        
        # Проверяем лимиты сигналов
        can_receive, reason = subscription.can_receive_signals(tariff)
        
        return can_receive, reason, tariff
    
    def increment_signals_usage(self, user_id: int) -> bool:
        """Увеличить счетчик использованных сигналов"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                today = datetime.now().date()
                
                # Сбрасываем счетчик если новый день
                cursor.execute("""
                    UPDATE subscriptions SET 
                        signals_used_today = CASE 
                            WHEN last_signal_date != %s THEN 1 
                            ELSE signals_used_today + 1 
                        END,
                        last_signal_date = %s
                    WHERE user_id = %s AND status = 'active'
                """, (today, today, user_id))
                
                conn.commit()
                return True
                
        except Error as e:
            self.logger.error(f"❌ Ошибка обновления счетчика сигналов для {user_id}: {e}")
            return False
    
    def _expire_subscription(self, subscription_id: int) -> bool:
        """Деактивировать истекшую подписку"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE subscriptions SET status = 'expired' WHERE id = %s
                """, (subscription_id,))
                conn.commit()
                
                self.logger.info(f"⏰ Подписка {subscription_id} истекла")
                return True
                
        except Error as e:
            self.logger.error(f"❌ Ошибка деактивации подписки {subscription_id}: {e}")
            return False
    
    def deactivate_expired_subscriptions(self) -> int:
        """Деактивировать все истекшие подписки"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Деактивируем истекшие подписки
                cursor.execute("""
                    UPDATE subscriptions 
                    SET status = 'expired' 
                    WHERE status = 'active' 
                    AND expires_at IS NOT NULL 
                    AND expires_at < NOW()
                """)
                
                expired_count = cursor.rowcount
                
                # Активируем подписки в grace period
                cursor.execute("""
                    UPDATE subscriptions 
                    SET status = 'active'
                    WHERE status = 'expired' 
                    AND grace_period_until IS NOT NULL 
                    AND grace_period_until > NOW()
                """)
                
                reactivated_count = cursor.rowcount
                
                conn.commit()
                
                if expired_count > 0:
                    self.logger.info(f"⏰ Деактивировано {expired_count} истекших подписок")
                if reactivated_count > 0:
                    self.logger.info(f"🔄 Реактивировано {reactivated_count} подписок (grace period)")
                
                return expired_count
                
        except Error as e:
            self.logger.error(f"❌ Ошибка деактивации истекших подписок: {e}")
            return 0
    
    # === ПЛАТЕЖИ ===
    
    def create_payment(self, user_id: int, amount: Decimal, currency: str,
                      payment_method: str, payment_provider: str,
                      metadata: Dict = None) -> Optional[Payment]:
        """Создание записи о платеже"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO payments (
                        user_id, amount, currency, payment_method, 
                        payment_provider, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (user_id, amount, currency, payment_method, payment_provider, 
                      str(metadata or {})))
                
                payment_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"💰 Создан платеж {payment_id} для {user_id}: {amount} {currency}")
                return self.get_payment(payment_id)
                
        except Error as e:
            self.logger.error(f"❌ Ошибка создания платежа для {user_id}: {e}")
            return None
    
    def get_payment(self, payment_id: int) -> Optional[Payment]:
        """Получить платеж по ID"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, subscription_id, amount, currency, status,
                           payment_method, payment_provider, external_payment_id,
                           created_at, completed_at, failed_reason, metadata
                    FROM payments WHERE id = %s
                """, (payment_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Конвертируем metadata из строки обратно в dict
                row_list = list(row)
                row_list[-1] = eval(row_list[-1]) if row_list[-1] else {}
                
                return Payment(*row_list)
                
        except Error as e:
            self.logger.error(f"❌ Ошибка получения платежа {payment_id}: {e}")
            return None
    
    def complete_payment(self, payment_id: int, external_payment_id: str = None) -> bool:
        """Завершить платеж успешно"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE payments SET 
                        status = 'completed', 
                        completed_at = NOW(),
                        external_payment_id = %s
                    WHERE id = %s
                """, (external_payment_id, payment_id))
                
                conn.commit()
                
                self.logger.info(f"✅ Платеж {payment_id} успешно завершен")
                return True
                
        except Error as e:
            self.logger.error(f"❌ Ошибка завершения платежа {payment_id}: {e}")
            return False
    
    def fail_payment(self, payment_id: int, reason: str) -> bool:
        """Отметить платеж как неудачный"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE payments SET 
                        status = 'failed', 
                        failed_reason = %s
                    WHERE id = %s
                """, (reason, payment_id))
                
                conn.commit()
                
                self.logger.info(f"❌ Платеж {payment_id} неудачен: {reason}")
                return True
                
        except Error as e:
            self.logger.error(f"❌ Ошибка обновления статуса платежа {payment_id}: {e}")
            return False
    
    # === НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ ===
    
    def _create_default_user_settings(self, user_id: int, cursor) -> bool:
        """Создать настройки пользователя по умолчанию"""
        try:
            cursor.execute("""
                INSERT INTO user_settings (user_id, preferred_symbols) 
                VALUES (%s, %s)
            """, (user_id, '["BTCUSDT", "ETHUSDT", "ADAUSDT"]'))
            
            return True
            
        except Error as e:
            self.logger.error(f"❌ Ошибка создания настроек для {user_id}: {e}")
            return False
    
    def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        """Получить настройки пользователя"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, notifications_enabled, signal_notifications,
                           news_notifications, marketing_notifications, preferred_symbols,
                           risk_level, min_confidence, max_signals_per_day,
                           rsi_oversold_level, rsi_overbought_level, timezone,
                           active_hours_start, active_hours_end, language,
                           created_at, updated_at
                    FROM user_settings WHERE user_id = %s
                """, (user_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Конвертируем JSON поле
                row_list = list(row)
                row_list[5] = eval(row_list[5]) if row_list[5] else []
                
                return UserSettings(*row_list)
                
        except Error as e:
            self.logger.error(f"❌ Ошибка получения настроек для {user_id}: {e}")
            return None
    
    def update_user_settings(self, user_id: int, settings: Dict) -> bool:
        """Обновить настройки пользователя"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Динамически строим запрос обновления
                set_clauses = []
                values = []
                
                for key, value in settings.items():
                    set_clauses.append(f"{key} = %s")
                    values.append(str(value) if isinstance(value, list) else value)
                
                if not set_clauses:
                    return True
                
                values.append(user_id)
                
                cursor.execute(f"""
                    UPDATE user_settings SET {', '.join(set_clauses)}, updated_at = NOW()
                    WHERE user_id = %s
                """, values)
                
                conn.commit()
                
                self.logger.info(f"✅ Настройки обновлены для {user_id}")
                return True
                
        except Error as e:
            self.logger.error(f"❌ Ошибка обновления настроек для {user_id}: {e}")
            return False
    
    # === СТАТИСТИКА ===
    
    def get_subscription_stats(self) -> Dict:
        """Получить статистику по подпискам"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Общее количество пользователей
                cursor.execute("SELECT COUNT(*) FROM users WHERE NOT is_banned")
                stats['total_users'] = cursor.fetchone()[0]
                
                # Активные подписки по типам
                cursor.execute("""
                    SELECT subscription_type, COUNT(*) 
                    FROM subscriptions 
                    WHERE status = 'active' 
                    GROUP BY subscription_type
                """)
                
                stats['active_by_type'] = dict(cursor.fetchall())
                
                # Доходы по месяцам
                cursor.execute("""
                    SELECT DATE_FORMAT(completed_at, '%Y-%m') as month, 
                           SUM(amount) as total
                    FROM payments 
                    WHERE status = 'completed' 
                    AND completed_at >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
                    GROUP BY month
                    ORDER BY month
                """)
                
                stats['monthly_revenue'] = dict(cursor.fetchall())
                
                return stats
                
        except Error as e:
            self.logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def get_user_list(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Получить список пользователей для админки"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT u.user_id, u.username, u.first_name, u.last_name,
                           u.created_at, u.is_banned, s.subscription_type, s.expires_at
                    FROM users u
                    LEFT JOIN subscriptions s ON u.user_id = s.user_id AND s.status = 'active'
                    ORDER BY u.created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                users = []
                for row in cursor.fetchall():
                    users.append({
                        'user_id': row[0],
                        'username': row[1],
                        'first_name': row[2],
                        'last_name': row[3],
                        'created_at': row[4],
                        'is_banned': row[5],
                        'subscription_type': row[6],
                        'expires_at': row[7]
                    })
                
                return users
                
        except Error as e:
            self.logger.error(f"❌ Ошибка получения списка пользователей: {e}")
            return []