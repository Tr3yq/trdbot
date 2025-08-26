from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List
from decimal import Decimal


class SubscriptionType(Enum):
    """Типы подписок"""
    TRIAL = "trial"
    BASIC = "basic"
    PREMIUM = "premium"
    VIP = "vip"
    LIFETIME = "lifetime"


class SubscriptionStatus(Enum):
    """Статусы подписки"""
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELED = "canceled"
    SUSPENDED = "suspended"
    PENDING_PAYMENT = "pending_payment"


class PaymentStatus(Enum):
    """Статусы платежей"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELED = "canceled"


class SignalOutcome(Enum):
    """Результаты сигналов"""
    PENDING = "pending"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    EXPIRED = "expired"
    CANCELED = "canceled"


@dataclass
class TariffPlan:
    """Тарифный план"""
    id: str
    name: str
    description: str
    price_monthly: Decimal
    price_quarterly: Decimal
    price_yearly: Decimal
    features: Dict[str, bool]
    signals_per_day: int
    priority_support: bool
    custom_settings: bool
    analytics_access: bool
    is_active: bool
    
    def get_price(self, duration_months: int) -> Decimal:
        """Получить цену за период"""
        if duration_months == 1:
            return self.price_monthly
        elif duration_months == 3:
            return self.price_quarterly
        elif duration_months == 12:
            return self.price_yearly
        else:
            # Пропорциональный расчет
            return self.price_monthly * duration_months


@dataclass
class User:
    """Модель пользователя"""
    user_id: int
    username: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    language_code: Optional[str]
    created_at: datetime
    last_activity: datetime
    is_banned: bool
    ban_reason: Optional[str]
    referrer_id: Optional[int]
    total_referrals: int
    
    @property
    def full_name(self) -> str:
        """Полное имя пользователя"""
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts) or f"User_{self.user_id}"


@dataclass
class Subscription:
    """Модель подписки"""
    id: int
    user_id: int
    subscription_type: SubscriptionType
    status: SubscriptionStatus
    started_at: datetime
    expires_at: Optional[datetime]
    auto_renewal: bool
    payment_method: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    # Использование лимитов
    signals_used_today: int
    last_signal_date: Optional[datetime]
    
    # Дополнительные поля
    trial_used: bool
    grace_period_until: Optional[datetime]
    suspension_reason: Optional[str]
    
    @property
    def is_active(self) -> bool:
        """Активна ли подписка"""
        if self.status != SubscriptionStatus.ACTIVE:
            return False
        
        if self.expires_at and self.expires_at < datetime.now():
            return False
            
        return True
    
    @property
    def is_trial(self) -> bool:
        """Является ли подписка пробной"""
        return self.subscription_type == SubscriptionType.TRIAL
    
    @property
    def days_remaining(self) -> Optional[int]:
        """Дней до окончания подписки"""
        if not self.expires_at:
            return None
        
        delta = self.expires_at - datetime.now()
        return max(0, delta.days)
    
    def can_receive_signals(self, tariff: TariffPlan) -> tuple[bool, str]:
        """Может ли пользователь получать сигналы"""
        if not self.is_active:
            return False, "Подписка не активна"
        
        # Проверка лимита сигналов в день
        today = datetime.now().date()
        if self.last_signal_date and self.last_signal_date.date() == today:
            if self.signals_used_today >= tariff.signals_per_day:
                return False, f"Достигнут лимит сигналов ({tariff.signals_per_day} в день)"
        
        return True, "OK"


@dataclass
class Payment:
    """Модель платежа"""
    id: int
    user_id: int
    subscription_id: Optional[int]
    amount: Decimal
    currency: str
    status: PaymentStatus
    payment_method: str
    payment_provider: str
    external_payment_id: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    failed_reason: Optional[str]
    metadata: Dict[str, str]


@dataclass
class TradingSignal:
    """Модель торгового сигнала"""
    id: int
    symbol: str
    interval: str
    is_long: bool
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    risk_reward_ratio: Decimal
    confidence_score: float
    
    # ML и анализ
    ml_prediction: Optional[float]
    technical_score: float
    indicators_fired: List[str]
    
    # Результаты
    outcome: SignalOutcome
    exit_price: Optional[Decimal]
    profit_loss_pct: Optional[float]
    duration_hours: Optional[int]
    
    # Метаданные
    created_at: datetime
    closed_at: Optional[datetime]
    market_regime: str
    volatility_level: str
    
    @property
    def is_profitable(self) -> Optional[bool]:
        """Прибыльный ли сигнал"""
        if self.outcome == SignalOutcome.TAKE_PROFIT:
            return True
        elif self.outcome == SignalOutcome.STOP_LOSS:
            return False
        return None
    
    @property
    def actual_rr(self) -> Optional[float]:
        """Фактический R:R"""
        if not self.exit_price or not self.profit_loss_pct:
            return None
        
        risk_pct = abs(self.entry_price - self.stop_loss) / self.entry_price * 100
        if risk_pct == 0:
            return None
            
        return abs(self.profit_loss_pct) / risk_pct


@dataclass
class UserSettings:
    """Настройки пользователя"""
    user_id: int
    
    # Уведомления
    notifications_enabled: bool
    signal_notifications: bool
    news_notifications: bool
    marketing_notifications: bool
    
    # Торговые настройки
    preferred_symbols: List[str]
    risk_level: str  # conservative, moderate, aggressive
    min_confidence: float
    max_signals_per_day: int
    
    # RSI настройки
    rsi_oversold_level: int
    rsi_overbought_level: int
    
    # Временные настройки
    timezone: str
    active_hours_start: int
    active_hours_end: int
    
    # Дополнительно
    language: str
    created_at: datetime
    updated_at: datetime


@dataclass
class ReferralProgram:
    """Реферальная программа"""
    id: int
    referrer_id: int
    referred_id: int
    created_at: datetime
    first_payment_at: Optional[datetime]
    commission_earned: Decimal
    commission_paid: bool
    is_active: bool


@dataclass
class SystemSettings:
    """Системные настройки"""
    key: str
    value: str
    description: str
    updated_at: datetime
    updated_by: int