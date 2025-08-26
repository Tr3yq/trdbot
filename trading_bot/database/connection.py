import logging
import mysql.connector
from contextlib import contextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool

from .models import (
    User, Subscription, Payment, TradingSignal, UserSettings,
    SubscriptionType, SubscriptionStatus, PaymentStatus, SignalOutcome,
    TariffPlan, ReferralProgram
)


class DatabaseManager:
    """Менеджер базы данных с пулом соединений"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection_pool: Optional[MySQLConnectionPool] = None
        self.logger = logging.getLogger(__name__)
        
        # Предустановленные тарифные планы
        self.tariff_plans = self._initialize_tariff_plans()
    
    def initialize(self) -> bool:
        """Инициализация базы данных"""
        try:
            # Создание пула соединений
            pool_config = self.db_config.copy()
            pool_config.update({
                'pool_name': 'trading_bot_pool',
                'pool_size': 5,
                'pool_reset_session': True
            })
            
            self.connection_pool = MySQLConnectionPool(**pool_config)
            self.logger.info("✅ Пул соединений с БД создан")
            
            # Создание таблиц
            if self._create_tables():
                self.logger.info("✅ Таблицы БД инициализированы")
                return True
            else:
                self.logger.error("❌ Ошибка создания таблиц")
                return False
                
        except Error as e:
            self.logger.error(f"❌ Ошибка подключения к БД: {e}")
            return False
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для получения соединения из пула"""
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            yield connection
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def _initialize_tariff_plans(self) -> Dict[str, TariffPlan]:
        """Инициализация тарифных планов"""
        return {
            "trial": TariffPlan(
                id="trial",
                name="Trial",
                description="7 дней бесплатно",
                price_monthly=Decimal("0"),
                price_quarterly=Decimal("0"),
                price_yearly=Decimal("0"),
                features={
                    "signals": True,
                    "basic_analytics": True,
                    "support": False,
                    "custom_settings": False
                },
                signals_per_day=3,
                priority_support=False,
                custom_settings=False,
                analytics_access=False,
                is_active=True
            ),
            "basic": TariffPlan(
                id="basic",
                name="Basic",
                description="Базовые торговые сигналы",
                price_monthly=Decimal("29.99"),
                price_quarterly=Decimal("79.99"),
                price_yearly=Decimal("299.99"),
                features={
                    "signals": True,
                    "basic_analytics": True,
                    "support": True,
                    "custom_settings": False
                },
                signals_per_day=10,
                priority_support=False,
                custom_settings=False,
                analytics_access=True,
                is_active=True
            ),
            "premium": TariffPlan(
                id="premium", 
                name="Premium",
                description="Продвинутые сигналы + аналитика",
                price_monthly=Decimal("59.99"),
                price_quarterly=Decimal("159.99"),
                price_yearly=Decimal("599.99"),
                features={
                    "signals": True,
                    "advanced_analytics": True,
                    "priority_support": True,
                    "custom_settings": True,
                    "risk_management": True
                },
                signals_per_day=25,
                priority_support=True,
                custom_settings=True,
                analytics_access=True,
                is_active=True
            ),
            "vip": TariffPlan(
                id="vip",
                name="VIP", 
                description="Безлимитные сигналы + персональная поддержка",
                price_monthly=Decimal("99.99"),
                price_quarterly=Decimal("269.99"),
                price_yearly=Decimal("999.99"),
                features={
                    "unlimited_signals": True,
                    "advanced_analytics": True,
                    "priority_support": True,
                    "custom_settings": True,
                    "risk_management": True,
                    "personal_manager": True
                },
                signals_per_day=999,
                priority_support=True,
                custom_settings=True,
                analytics_access=True,
                is_active=True
            )
        }
    
    def _create_tables(self) -> bool:
        """Создание всех необходимых таблиц"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Таблица пользователей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id BIGINT PRIMARY KEY,
                        username VARCHAR(100),
                        first_name VARCHAR(100),
                        last_name VARCHAR(100),
                        language_code VARCHAR(10),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        is_banned TINYINT(1) DEFAULT 0,
                        ban_reason TEXT,
                        referrer_id BIGINT,
                        total_referrals INT DEFAULT 0,
                        
                        INDEX idx_username (username),
                        INDEX idx_created_at (created_at),
                        INDEX idx_referrer (referrer_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # Таблица подписок
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS subscriptions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        subscription_type ENUM('trial', 'basic', 'premium', 'vip', 'lifetime') NOT NULL,
                        status ENUM('active', 'expired', 'canceled', 'suspended', 'pending_payment') DEFAULT 'active',
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NULL,
                        auto_renewal BOOLEAN DEFAULT FALSE,
                        payment_method VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        
                        signals_used_today INT DEFAULT 0,
                        last_signal_date DATE,
                        trial_used BOOLEAN DEFAULT FALSE,
                        grace_period_until TIMESTAMP NULL,
                        suspension_reason TEXT,
                        
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                        INDEX idx_user_status (user_id, status),
                        INDEX idx_expires (expires_at),
                        INDEX idx_type (subscription_type)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # Таблица платежей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS payments (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        subscription_id INT,
                        amount DECIMAL(10,2) NOT NULL,
                        currency VARCHAR(3) DEFAULT 'USD',
                        status ENUM('pending', 'completed', 'failed', 'refunded', 'canceled') DEFAULT 'pending',
                        payment_method VARCHAR(50) NOT NULL,
                        payment_provider VARCHAR(50) NOT NULL,
                        external_payment_id VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP NULL,
                        failed_reason TEXT,
                        metadata JSON,
                        
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                        FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE SET NULL,
                        INDEX idx_user_payments (user_id),
                        INDEX idx_status (status),
                        INDEX idx_created (created_at),
                        INDEX idx_external (external_payment_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # Таблица торговых сигналов
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        `interval` VARCHAR(10) NOT NULL,
                        is_long BOOLEAN NOT NULL,
                        entry_price DECIMAL(20,8) NOT NULL,
                        stop_loss DECIMAL(20,8) NOT NULL,
                        take_profit DECIMAL(20,8) NOT NULL,
                        risk_reward_ratio DECIMAL(5,2) NOT NULL,
                        confidence_score FLOAT NOT NULL,
                        
                        ml_prediction FLOAT,
                        technical_score FLOAT NOT NULL,
                        indicators_fired JSON,
                        
                        outcome ENUM('pending', 'take_profit', 'stop_loss', 'expired', 'canceled') DEFAULT 'pending',
                        exit_price DECIMAL(20,8),
                        profit_loss_pct FLOAT,
                        duration_hours INT,
                        
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        closed_at TIMESTAMP NULL,
                        market_regime VARCHAR(50),
                        volatility_level VARCHAR(20),
                        
                        INDEX idx_symbol (symbol),
                        INDEX idx_created (created_at),
                        INDEX idx_outcome (outcome),
                        INDEX idx_symbol_created (symbol, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # Таблица настроек пользователей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_settings (
                        user_id BIGINT PRIMARY KEY,
                        
                        notifications_enabled BOOLEAN DEFAULT TRUE,
                        signal_notifications BOOLEAN DEFAULT TRUE,
                        news_notifications BOOLEAN DEFAULT FALSE,
                        marketing_notifications BOOLEAN DEFAULT FALSE,
                        
                        preferred_symbols JSON,
                        risk_level ENUM('conservative', 'moderate', 'aggressive') DEFAULT 'moderate',
                        min_confidence FLOAT DEFAULT 0.6,
                        max_signals_per_day INT DEFAULT 10,
                        
                        rsi_oversold_level INT DEFAULT 30,
                        rsi_overbought_level INT DEFAULT 70,
                        
                        timezone VARCHAR(50) DEFAULT 'UTC',
                        active_hours_start INT DEFAULT 0,
                        active_hours_end INT DEFAULT 23,
                        
                        language VARCHAR(10) DEFAULT 'ru',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # Таблица реферальной программы
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS referrals (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        referrer_id BIGINT NOT NULL,
                        referred_id BIGINT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        first_payment_at TIMESTAMP NULL,
                        commission_earned DECIMAL(10,2) DEFAULT 0.00,
                        commission_paid BOOLEAN DEFAULT FALSE,
                        is_active BOOLEAN DEFAULT TRUE,
                        
                        FOREIGN KEY (referrer_id) REFERENCES users(user_id) ON DELETE CASCADE,
                        FOREIGN KEY (referred_id) REFERENCES users(user_id) ON DELETE CASCADE,
                        UNIQUE KEY unique_referral (referrer_id, referred_id),
                        INDEX idx_referrer (referrer_id),
                        INDEX idx_referred (referred_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # Таблица системных настроек
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_settings (
                        `key` VARCHAR(100) PRIMARY KEY,
                        `value` TEXT NOT NULL,
                        description TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        updated_by BIGINT,
                        
                        FOREIGN KEY (updated_by) REFERENCES users(user_id) ON DELETE SET NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                conn.commit()
                return True
                
        except Error as e:
            self.logger.error(f"❌ Ошибка создания таблиц: {e}")
            return False
    
    def get_tariff_plan(self, plan_id: str) -> Optional[TariffPlan]:
        """Получить тарифный план по ID"""
        return self.tariff_plans.get(plan_id)
    
    def close(self):
        """Закрытие пула соединений"""
        if self.connection_pool:
            # Пул соединений закроется автоматически при завершении программы
            self.logger.info("🔌 База данных отключена")
            
    def deactivate_expired_subscribers(self):
        """
        Заглушка метода деактивации подписок.
        Пока ничего не делает.
        """
        return        
    
    # Далее будут методы для работы с пользователями, подписками и т.д.
    # Это базовая структура - в следующем файле добавим CRUD операции