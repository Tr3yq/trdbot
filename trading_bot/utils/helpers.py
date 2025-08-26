import logging
import os
from datetime import datetime
from typing import Optional


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Настройка системы логирования"""
    
    # Создание форматтера
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Очистка существующих обработчиков
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Файловый обработчик (если указан)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Настройка логгеров библиотек
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def format_datetime(dt: datetime, format_str: str = "%d.%m.%Y %H:%M") -> str:
    """Форматирование даты и времени"""
    return dt.strftime(format_str)


def safe_float(value, default: float = 0.0) -> float:
    """Безопасное преобразование в float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Безопасное преобразование в int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 4096) -> str:
    """Обрезка текста до максимальной длины (для Telegram)"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def create_progress_bar(current: int, total: int, length: int = 20) -> str:
    """Создание прогресс-бара"""
    if total == 0:
        return "░" * length
    
    progress = current / total
    filled = int(length * progress)
    bar = "█" * filled + "░" * (length - filled)
    percentage = f"{progress * 100:.1f}%"
    
    return f"[{bar}] {percentage}"


def format_number(number: float, precision: int = 2) -> str:
    """Форматирование числа с разделителями тысяч"""
    return f"{number:,.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Форматирование в проценты"""
    return f"{value:.{precision}f}%"


class Timer:
    """Простой таймер для измерения времени выполнения"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Запуск таймера"""
        self.start_time = datetime.now()
        return self
    
    def stop(self):
        """Остановка таймера"""
        self.end_time = datetime.now()
        return self
    
    def elapsed(self) -> float:
        """Получить время выполнения в секундах"""
        if not self.start_time:
            return 0.0
        
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def elapsed_str(self) -> str:
        """Получить время выполнения в виде строки"""
        seconds = self.elapsed()
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"