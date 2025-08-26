import logging
import threading
import time
from typing import Optional

import schedule
from telebot import TeleBot

from config.settings import Config
from database.connection import DatabaseManager
from handlers import register_all_handlers
from services.market_analysis import AdvancedMarketAnalyzer
from services.signal_generator import SignalService
from utils.helpers import setup_logging


class TradingBot:
    """Главный класс торгового бота"""
    
    def __init__(self):
        self.config = Config()
        self.bot: Optional[TeleBot] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.market_analyzer: Optional[AdvancedMarketAnalyzer] = None
        # self.signal_service: Optional[SignalService] = None
        self._running = False
        
    def initialize(self) -> bool:
        """Инициализация всех компонентов бота"""
        try:
            # Настройка логирования
            setup_logging(
                level=self.config.LOG_LEVEL,
                log_file=self.config.LOG_FILE
            )
            
            logging.info("🚀 Запуск торгового бота...")
            
            # Инициализация Telegram бота
            self.bot = TeleBot(
                token=self.config.TELEGRAM_BOT_TOKEN,
                threaded=True
            )
            logging.info("✅ Telegram бот инициализирован")
            
            # Инициализация базы данных
            self.db_manager = DatabaseManager(self.config.DB_CONFIG)
            if not self.db_manager.initialize():
                logging.error("❌ Ошибка инициализации базы данных")
                return False
            logging.info("✅ База данных подключена")
            
            # Инициализация сервисов
            self.market_analyzer = AdvancedMarketAnalyzer(self.config)
            self.signal_service = SignalService(
                self.config, 
                self.db_manager, 
                self.bot
            )
            logging.info("✅ Сервисы инициализированы")
            
            # Регистрация обработчиков
            register_all_handlers(
                bot=self.bot,
                db_manager=self.db_manager,
                config=self.config,
                signal_service=self.signal_service,
                market_analyzer=self.market_analyzer
            )
            logging.info("✅ Обработчики зарегистрированы")
            
            # Настройка планировщика
            self._setup_scheduler()
            logging.info("✅ Планировщик настроен")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    def _setup_scheduler(self):
        """Настройка планировщика задач"""
        # Проверка подписок каждый час
        schedule.every().hour.do(
            self.db_manager.deactivate_expired_subscribers
        )
        
        # Анализ рынка каждые 1-5 минут (настраивается)
        schedule.every(self.config.ANALYSIS_INTERVAL_MINUTES).minutes.do(
            self._scheduled_market_analysis
        )
        
        # Обновление результатов сигналов каждые 15 минут  
        schedule.every(15).minutes.do(
            self.signal_service.update_signal_outcomes
        )
        
        logging.info(f"📅 Планировщик: анализ каждые {self.config.ANALYSIS_INTERVAL_MINUTES} мин")
    
    def _scheduled_market_analysis(self):
        """Запланированный анализ рынка"""
        if not self.config.AUTO_BROADCAST_ENABLED:
            logging.info("⏸ Авторассылка отключена, анализ пропущен")
            return
            
        try:
            logging.info("🔍 Запуск планового анализа рынка")
            self.signal_service.analyze_and_broadcast()
            
        except Exception as e:
            logging.error(f"❌ Ошибка планового анализа: {e}")
    
    def _run_scheduler(self):
        """Запуск планировщика в отдельном потоке"""
        while self._running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Проверка каждые 30 секунд
            except Exception as e:
                logging.error(f"❌ Ошибка планировщика: {e}")
                time.sleep(60)
    
    def start(self):
        """Запуск бота"""
        if not self.initialize():
            logging.error("❌ Не удалось инициализировать бота")
            return False
        
        try:
            self._running = True
            
            # Запуск планировщика в фоновом потоке
            scheduler_thread = threading.Thread(
                target=self._run_scheduler,
                name="Scheduler",
                daemon=True
            )
            scheduler_thread.start()
            logging.info("✅ Планировщик запущен")
            
            # Основной цикл бота
            logging.info("🤖 Бот запущен и готов к работе!")
            logging.info(f"📊 Интервал анализа: {self.config.ANALYSIS_INTERVAL_MINUTES} мин")
            logging.info(f"📈 Автобрассылка: {'включена' if self.config.AUTO_BROADCAST_ENABLED else 'отключена'}")
            
            self.bot.polling(
                none_stop=True,
                interval=0,
                timeout=20,
                long_polling_timeout=20
            )
            
        except KeyboardInterrupt:
            logging.info("🛑 Получен сигнал остановки")
            self.stop()
        except Exception as e:
            logging.error(f"❌ Критическая ошибка: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Остановка бота"""
        logging.info("🛑 Остановка бота...")
        self._running = False
        
        if self.bot:
            self.bot.stop_polling()
        
        if self.db_manager:
            self.db_manager.close()
            
        logging.info("✅ Бот остановлен")


def main():
    """Главная функция"""
    try:
        bot = TradingBot()
        bot.start()
    except Exception as e:
        logging.error(f"❌ Критическая ошибка запуска: {e}")
        exit(1)


if __name__ == "__main__":
    main()