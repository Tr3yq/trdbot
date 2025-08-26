import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()


class Config:
    """Класс конфигурации приложения"""
    
    def __init__(self):
        # Telegram Bot
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN не установлен")
        
        # Администраторы
        admin_ids = os.getenv("ADMIN_IDS", "").split(",")
        self.ADMIN_IDS = [int(id.strip()) for id in admin_ids if id.strip().isdigit()]
        
        # База данных
        self.DB_CONFIG = {
            "host": os.getenv("DB_HOST", "localhost"),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", ""),
            "database": os.getenv("DB_NAME", "trading_bot"),
            "charset": "utf8mb4",
            "autocommit": True
        }
        
        # Binance API
        self.BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
        self.BINANCE_FUTURES_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        
        # Технический анализ
        self.KLINES_LIMIT = 300
        self.DEFAULT_INTERVAL = "15m"
        self.ANALYSIS_INTERVAL_MINUTES = int(os.getenv("ANALYSIS_INTERVAL_MINUTES", "1"))
        
        # Настройки сигналов
        self.MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD", "2.0"))
        self.MAX_RISK_PERCENT = float(os.getenv("MAX_RISK_PERCENT", "2.0"))
        self.MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.6"))
        
        # RSI настройки (по умолчанию)
        self.DEFAULT_RSI_LONG = 30
        self.DEFAULT_RSI_SHORT = 70
        
        # Авторассылка
        self.AUTO_BROADCAST_ENABLED = os.getenv("AUTO_BROADCAST", "true").lower() == "true"
        
        # Логирование
        self.LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
        self.LOG_FILE = os.getenv("LOG_FILE", "logs/trading_bot.log")
        
        # Файлы моделей ML
        self.ML_PIPELINE_PATH = os.getenv("ML_PIPELINE_PATH", "models/xgb_pipeline.pkl")
        self.ENABLE_ML_VALIDATION = os.getenv("ENABLE_ML", "true").lower() == "true"
        
        # Графики
        self.CHARTS_DIR = "charts/"
        self.DEFAULT_CHART_PATH = "chart.png"
        
        # Подписки
        self.TRIAL_PERIOD_DAYS = int(os.getenv("TRIAL_DAYS", "7"))
        
        # Обучающие материалы
        self.TRAINING_FILE_PATH = os.getenv("TRAINING_FILE", "/path/to/training.pdf")
        
        self._validate_config()
    
    def _validate_config(self):
        """Валидация конфигурации"""
        if not self.ADMIN_IDS:
            raise ValueError("Необходимо указать хотя бы одного администратора в ADMIN_IDS")
        
        if self.ANALYSIS_INTERVAL_MINUTES < 1:
            raise ValueError("ANALYSIS_INTERVAL_MINUTES должен быть >= 1")
        
        if self.MIN_RISK_REWARD_RATIO < 1.0:
            raise ValueError("MIN_RISK_REWARD должен быть >= 1.0")
        
        # Создание директорий
        os.makedirs("logs", exist_ok=True)
        os.makedirs("charts", exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    def is_admin(self, user_id: int) -> bool:
        """Проверка является ли пользователь администратором"""
        return user_id in self.ADMIN_IDS
    
    def get_db_connection_string(self) -> str:
        """Получить строку подключения к БД"""
        return f"mysql://{self.DB_CONFIG['user']}:{self.DB_CONFIG['password']}@{self.DB_CONFIG['host']}/{self.DB_CONFIG['database']}"


class TradingConfig:
    """Конфигурация торговых параметров"""
    
    # Поддерживаемые интервалы
    SUPPORTED_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
    
    # Минимальные требования для сигнала
    MIN_DATA_POINTS = 200
    MIN_INDICATORS_COUNT = 3
    
    # Параметры индикаторов
    RSI_PERIODS = [7, 14]
    EMA_PERIODS = [10, 20, 50, 100, 200]
    
    # Volatility regimes
    HIGH_VOLATILITY_THRESHOLD = 4.0  # ATR %
    LOW_VOLATILITY_THRESHOLD = 1.0   # ATR %
    
    # Market structure
    TREND_STRENGTH_LEVELS = {
        "VERY_WEAK": 1,
        "WEAK": 2, 
        "MODERATE": 3,
        "STRONG": 4,
        "VERY_STRONG": 5
    }