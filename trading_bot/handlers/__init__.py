import logging
from telebot import TeleBot

from config.settings import Config
from database.connection import DatabaseManager
from .user import register_user_handlers
from .callbacks import register_callback_handlers


def register_all_handlers(bot: TeleBot, db_manager: DatabaseManager, 
                         config: Config, signal_service, market_analyzer):
    """Регистрация всех обработчиков команд и сообщений"""
    
    logger = logging.getLogger(__name__)
    
    try:
        # Регистрируем пользовательские обработчики
        user_handlers = register_user_handlers(bot, db_manager, config, signal_service)
        logger.info("✅ Пользовательские обработчики зарегистрированы")
        
        # Регистрируем callback обработчики
        callback_handlers = register_callback_handlers(bot, db_manager, config)
        logger.info("✅ Callback обработчики зарегистрированы")
        
        # Будущие обработчики (в следующих этапах):
        # from .admin import register_admin_handlers
        # register_admin_handlers(bot, db_manager, config, signal_service, market_analyzer)
        
        logger.info("✅ Все обработчики успешно зарегистрированы")
        
        return {
            'user_handlers': user_handlers,
            'callback_handlers': callback_handlers
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка регистрации обработчиков: {e}")
        raise