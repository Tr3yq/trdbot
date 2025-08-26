import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from config.settings import Config, TradingConfig
from utils.helpers import Timer


class MarketRegime(Enum):
    """Рыночные режимы"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    SIDEWAYS_CONSOLIDATION = "sideways_consolidation"
    HIGH_VOLATILITY_EXPANSION = "high_vol_expansion"
    LOW_VOLATILITY_CONTRACTION = "low_vol_contraction"
    BREAKOUT_BULL = "breakout_bull"
    BREAKOUT_BEAR = "breakout_bear"
    REVERSAL_ZONE = "reversal_zone"


class SignalQuality(Enum):
    """Качество сигнала"""
    INSTITUTIONAL = "institutional"  # 90%+ confidence
    PROFESSIONAL = "professional"   # 80-90% confidence
    RETAIL_PLUS = "retail_plus"     # 70-80% confidence
    RETAIL = "retail"               # 60-70% confidence
    NOISE = "noise"                 # <60% confidence


@dataclass
class MarketStructure:
    """Структура рынка"""
    higher_highs: int
    lower_lows: int
    support_levels: List[float]
    resistance_levels: List[float]
    trend_strength: float
    trend_consistency: float
    fractal_dimension: float
    market_efficiency_ratio: float


@dataclass
class VolumeProfile:
    """Профиль объемов"""
    volume_weighted_price: float
    poc_price: float  # Point of Control
    value_area_high: float
    value_area_low: float
    volume_imbalance: float
    institutional_flow: float
    retail_flow: float


@dataclass
class AdvancedSignal:
    """Продвинутый торговый сигнал"""
    symbol: str
    timestamp: datetime
    direction: str  # LONG/SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Качество и уверенность
    confidence_score: float
    signal_quality: SignalQuality
    risk_reward_ratio: float
    
    # Рыночный контекст
    market_regime: MarketRegime
    volatility_percentile: float
    volume_confirmation: float
    
    # Технические факторы
    convergence_score: float  # Схождение индикаторов
    divergence_signals: List[str]
    support_resistance_score: float
    momentum_score: float
    
    # Микроструктура
    order_flow_score: float
    liquidity_score: float
    institutional_bias: float
    
    # ML предсказания
    ml_probability: Optional[float]
    ensemble_score: Optional[float]
    
    # Тайминг
    entry_timing_score: float
    market_hours_factor: float
    session_factor: str  # Asian/European/American
    
    # Дополнительные метрики
    max_adverse_excursion: float
    expected_holding_time: int  # в минутах
    correlation_risk: float


class AdvancedMarketAnalyzer:
    """Профессиональный анализатор рынка"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Настройки для профессионального анализа
        self.min_data_points = 500  # Больше данных для точности
        self.lookback_periods = {
            'short': 20,
            'medium': 50, 
            'long': 200,
            'macro': 500
        }
        
        # Пороги для институциональных сигналов
        self.institutional_thresholds = {
            'min_volume_ratio': 2.5,
            'min_convergence_score': 0.85,
            'min_support_resistance': 0.80,
            'max_correlation_risk': 0.30
        }
        
        # Скейлеры для нормализации
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
        
        # Детектор аномалий
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def get_market_data(self, symbol: str, interval: str = "15m", limit: int = 1000) -> Optional[pd.DataFrame]:
        """Получение расширенных рыночных данных"""
        try:
            timer = Timer().start()
            
            # Binance Futures API
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                return None
            
            # Создаем DataFrame
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades_count", 
                "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ])
            
            # Конвертируем типы
            numeric_columns = ["open", "high", "low", "close", "volume", 
                             "quote_volume", "trades_count", "taker_buy_volume", 
                             "taker_buy_quote_volume"]
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Временные метки
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Дополнительные расчеты
            df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
            df["hl2"] = (df["high"] + df["low"]) / 2
            df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
            
            # Объемные метрики
            df["buy_sell_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-10)
            df["trade_size_avg"] = df["volume"] / (df["trades_count"] + 1)
            
            # Фильтруем аномалии в данных
            df = self._filter_data_anomalies(df)
            
            timer.stop()
            self.logger.info(f"📊 Данные для {symbol} получены за {timer.elapsed_str()}")
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения данных {symbol}: {e}")
            return None
    
    def _filter_data_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрация аномалий в данных"""
        try:
            # Удаляем экстремальные выбросы цен
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                Q1 = df[col].quantile(0.01)
                Q99 = df[col].quantile(0.99)
                df = df[(df[col] >= Q1) & (df[col] <= Q99)]
            
            # Фильтруем аномальные объемы
            volume_median = df["volume"].median()
            df = df[df["volume"] <= volume_median * 50]  # Удаляем супер-спайки
            
            # Проверяем корректность OHLC
            df = df[
                (df["high"] >= df["open"]) & 
                (df["high"] >= df["close"]) &
                (df["low"] <= df["open"]) & 
                (df["low"] <= df["close"])
            ]
            
            return df.reset_index().set_index("timestamp")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка фильтрации аномалий: {e}")
            return df
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет продвинутых технических индикаторов"""
        try:
            timer = Timer().start()
            
            # === БАЗОВЫЕ ИНДИКАТОРЫ ===
            
            # Множественные RSI
            df["rsi_7"] = ta.rsi(df["close"], length=7)
            df["rsi_14"] = ta.rsi(df["close"], length=14)  
            df["rsi_21"] = ta.rsi(df["close"], length=21)
            df["rsi_50"] = ta.rsi(df["close"], length=50)
            
            # Стохастик семейство
            stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]
            
            # Быстрый стохастик
            fast_stoch = ta.stoch(df["high"], df["low"], df["close"], k=5, d=3)
            df["fast_stoch_k"] = fast_stoch["STOCHk_5_3_3"]
            
            # Williams %R
            df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)
            
            # === MACD СЕМЕЙСТВО ===
            
            # Классический MACD
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_histogram"] = macd["MACDh_12_26_9"]
            
            # Быстрый MACD для скальпинга
            fast_macd = ta.macd(df["close"], fast=5, slow=13, signal=5)
            df["macd_fast"] = fast_macd["MACD_5_13_5"]
            df["macd_fast_signal"] = fast_macd["MACDs_5_13_5"]
            
            # MACD на разных источниках
            df["macd_hl2"] = ta.macd(df["hl2"], fast=12, slow=26, signal=9)["MACD_12_26_9"]
            df["macd_typical"] = ta.macd(df["typical_price"], fast=12, slow=26, signal=9)["MACD_12_26_9"]
            
            # === MOVING AVERAGES ===
            
            # EMA семейство
            ema_periods = [8, 13, 21, 34, 55, 89, 144, 200]
            for period in ema_periods:
                df[f"ema_{period}"] = ta.ema(df["close"], length=period)
            
            # SMA для определения долгосрочного тренда
            sma_periods = [50, 100, 200]
            for period in sma_periods:
                df[f"sma_{period}"] = ta.sma(df["close"], length=period)
            
            # Adaptable Moving Average (KAMA)
            df["kama"] = ta.kama(df["close"], length=21)
            
            # Hull Moving Average
            df["hma_21"] = ta.hma(df["close"], length=21)
            df["hma_50"] = ta.hma(df["close"], length=50)
            
            # === BOLLINGER BANDS ===
            
            # Стандартные BB
            bb = ta.bbands(df["close"], length=20, std=2)
            df["bb_upper"] = bb["BBU_20_2.0"]
            df["bb_middle"] = bb["BBM_20_2.0"] 
            df["bb_lower"] = bb["BBL_20_2.0"]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            
            # Bollinger Bands на разных периодах
            bb_50 = ta.bbands(df["close"], length=50, std=2)
            df["bb_upper_50"] = bb_50["BBU_50_2.0"]
            df["bb_lower_50"] = bb_50["BBL_50_2.0"]
            
            # === VOLATILITY INDICATORS ===
            
            # ATR семейство
            df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["atr_21"] = ta.atr(df["high"], df["low"], df["close"], length=21)
            df["atr_50"] = ta.atr(df["high"], df["low"], df["close"], length=50)
            df["atr_pct"] = df["atr_14"] / df["close"] * 100
            
            # Normalized ATR
            df["atr_normalized"] = df["atr_14"] / df["atr_50"]
            
            # df["chaikin_vol"] = ta.chaikinvol(high=df["high"],
            #                      low=df["low"],
            #                      length=14,
            #                      roc_length=10)
            
            # === ADX И DIRECTIONAL MOVEMENT ===
            
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            df["adx"] = adx["ADX_14"]
            df["di_plus"] = adx["DMP_14"]
            df["di_minus"] = adx["DMN_14"]
            
            # Длинный ADX для фильтрации
            adx_50 = ta.adx(df["high"], df["low"], df["close"], length=50)
            df["adx_50"] = adx_50["ADX_50"]
            
            # === MOMENTUM INDICATORS ===
            
            # Rate of Change семейство
            df["roc_1"] = ta.roc(df["close"], length=1)
            df["roc_5"] = ta.roc(df["close"], length=5)
            df["roc_14"] = ta.roc(df["close"], length=14)
            df["roc_21"] = ta.roc(df["close"], length=21)
            
            # Momentum
            df["momentum"] = ta.mom(df["close"], length=14)
            
            # Commodity Channel Index
            df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=14)
            df["cci_50"] = ta.cci(df["high"], df["low"], df["close"], length=50)
            
            # === VOLUME INDICATORS ===
            
            # Volume SMA
            df["volume_sma_20"] = ta.sma(df["volume"], length=20)
            df["volume_sma_50"] = ta.sma(df["volume"], length=50)
            df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
            
            # On Balance Volume
            df["obv"] = ta.obv(df["close"], df["volume"])
            df["obv_ema"] = ta.ema(df["obv"], length=21)
            
            # Volume Price Trend
            # df["vpt"] = ta.vpt(df["close"], df["volume"])
            
            # Accumulation Distribution Line
            df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])
            df["ad_ema"] = ta.ema(df["ad"], length=21)
            
            # Money Flow Index
            df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
            
            # Volume Weighted Average Price
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            
            # === ПРОДВИНУТЫЕ ИНДИКАТОРЫ ===
            
            # Арун
            aroon = ta.aroon(df["high"], df["low"], length=25)
            df["aroon_up"] = aroon["AROONU_25"]
            df["aroon_down"] = aroon["AROOND_25"]
            df["aroon_osc"] = df["aroon_up"] - df["aroon_down"]
            
            # Parabolic SAR
            df['sar'] = ta.psar(high=df['high'], low=df['low'], close=df['close'], af=0.02, max_af=0.2)['PSARl_0.02_0.2']
            
            # Supertrend
            supertrend = ta.supertrend(df["high"], df["low"], df["close"])
            df["supertrend"] = supertrend["SUPERT_7_3.0"]
            df["supertrend_direction"] = supertrend["SUPERTd_7_3.0"]
            
            # Donchian Channels
            donchian = ta.donchian(df["high"], df["low"], lower_length=20, upper_length=20)
            df["donchian_upper"] = donchian["DCU_20_20"]
            df["donchian_lower"] = donchian["DCL_20_20"]
            df["donchian_middle"] = donchian["DCM_20_20"]
            
            # === КАСТОМНЫЕ ИНДИКАТОРЫ ===
            
            # Расчет кастомных метрик
            df = self._calculate_custom_indicators(df)
            
            # Структурные индикаторы
            df = self._calculate_market_structure_indicators(df)
            
            # Микроструктурные индикаторы
            df = self._calculate_microstructure_indicators(df)
            
            timer.stop()
            self.logger.info(f"📈 Индикаторы рассчитаны за {timer.elapsed_str()}")
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка расчета индикаторов: {e}")
            return df
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет кастомных индикаторов уровня hedge-фондов"""
        try:
            # === ADVANCED MOMENTUM ===
            
            # Relative Vigor Index
            num = df["close"] - df["open"]
            den = df["high"] - df["low"]
            df["rvi"] = ta.sma(num, length=10) / ta.sma(den, length=10)
            
            # True Strength Index
            
            tsi_val = ta.tsi(df["close"], fast=25, slow=13)

            # Проверяем тип и присваиваем только первую колонку, если это DataFrame
            if isinstance(tsi_val, pd.DataFrame):
                df["tsi"] = tsi_val.iloc[:, 0]
            else:
                df["tsi"] = tsi_val
            
            # Kaufman Efficiency Ratio
            change = abs(df["close"] - df["close"].shift(14))
            volatility = df["close"].diff().abs().rolling(14).sum()
            df["efficiency_ratio"] = change / volatility
            
            # === ADVANCED VOLATILITY ===
            
            # Historical Volatility (annualized)
            returns = df["close"].pct_change()
            df["hist_vol"] = returns.rolling(window=20).std() * np.sqrt(365 * 24 * 4)  # 15min bars
            
            # Volatility Ratio
            short_vol = returns.rolling(window=10).std()
            long_vol = returns.rolling(window=40).std()  
            df["vol_ratio"] = short_vol / long_vol
            
            # Range Volatility
            df["range_vol"] = (df["high"] - df["low"]) / df["open"]
            df["range_vol_ma"] = df["range_vol"].rolling(20).mean()
            
            # === PRICE ACTION INDICATORS ===
            
            # Inside/Outside Bars
            df["inside_bar"] = (
                (df["high"] < df["high"].shift(1)) & 
                (df["low"] > df["low"].shift(1))
            ).astype(int)
            
            df["outside_bar"] = (
                (df["high"] > df["high"].shift(1)) & 
                (df["low"] < df["low"].shift(1))
            ).astype(int)
            
            # Engulfing Patterns
            df["bullish_engulfing"] = (
                (df["close"] > df["open"]) &  # Current green
                (df["close"].shift(1) < df["open"].shift(1)) &  # Previous red
                (df["open"] < df["close"].shift(1)) &  # Current open < previous close
                (df["close"] > df["open"].shift(1))  # Current close > previous open
            ).astype(int)
            
            df["bearish_engulfing"] = (
                (df["close"] < df["open"]) &  # Current red
                (df["close"].shift(1) > df["open"].shift(1)) &  # Previous green
                (df["open"] > df["close"].shift(1)) &  # Current open > previous close
                (df["close"] < df["open"].shift(1))  # Current close < previous open
            ).astype(int)
            
            # === ADVANCED VOLUME ===
            
            # Volume Rate of Change
            df["volume_roc"] = df["volume"].pct_change(periods=5)
            
            # Ease of Movement
            distance = (df["high"] + df["low"]) / 2 - (df["high"].shift(1) + df["low"].shift(1)) / 2
            box_ratio = df["volume"] / ((df["high"] - df["low"]) * 100000000)
            df["eom"] = distance / box_ratio
            df["eom_ma"] = df["eom"].rolling(14).mean()
            
            # Volume Spread Analysis
            df["spread"] = df["high"] - df["low"]
            df["volume_spread"] = df["volume"] / df["spread"]
            
            # === FRACTAL AND CHAOS ===
            
            # Fractal Dimension
            def fractal_dimension(series, period=20):
                """Расчет фрактальной размерности"""
                if len(series) < period:
                    return np.nan
                
                n = len(series)
                rs_values = []
                
                for i in range(2, min(period, n//2)):
                    chunks = [series[j:j+i] for j in range(0, n-i+1, i)]
                    rs_chunk = []
                    
                    for chunk in chunks:
                        if len(chunk) == i:
                            mean_chunk = np.mean(chunk)
                            deviations = np.cumsum(chunk - mean_chunk)
                            R = np.max(deviations) - np.min(deviations)
                            S = np.std(chunk)
                            if S != 0:
                                rs_chunk.append(R/S)
                    
                    if rs_chunk:
                        rs_values.append((i, np.mean(rs_chunk)))
                
                if len(rs_values) >= 2:
                    x = np.log([r[0] for r in rs_values])
                    y = np.log([r[1] for r in rs_values])
                    slope, _, _, _, _ = stats.linregress(x, y)
                    return slope
                
                return np.nan
            
            # Применяем к скользящему окну
            df["fractal_dim"] = df["close"].rolling(window=50).apply(
                lambda x: fractal_dimension(x.values), raw=False
            )
            
            # === MARKET MICROSTRUCTURE ===
            
            # Bid-Ask Spread Proxy (используем high-low как приближение)
            df["spread_proxy"] = (df["high"] - df["low"]) / ((df["high"] + df["low"]) / 2) * 100
            df["spread_ma"] = df["spread_proxy"].rolling(20).mean()
            
            # Price Impact (как цена реагирует на объем)
            returns_1 = df["close"].pct_change()
            volume_1 = df["volume"] / df["volume"].rolling(20).mean()
            df["price_impact"] = abs(returns_1) / (volume_1 + 1e-10)
            
            # === REGIME DETECTION ===
            
            # Trend Regime Score
            ema_fast = df["ema_21"]
            ema_slow = df["ema_55"]
            df["trend_regime"] = np.where(
                ema_fast > ema_slow, 1,  # Uptrend
                np.where(ema_fast < ema_slow, -1, 0)  # Downtrend or sideways
            )
            
            # Volatility Regime
            vol_percentile = df["hist_vol"].rolling(100).rank(pct=True)
            df["vol_regime"] = np.where(
                vol_percentile > 0.8, 2,  # High vol
                np.where(vol_percentile < 0.2, 0, 1)  # Low vol or normal
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка кастомных индикаторов: {e}")
            return df
    
    def _calculate_market_structure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикаторы структуры рынка"""
        try:
            # === SUPPORT/RESISTANCE LEVELS ===
            
            def find_pivot_points(high, low, close, period=5):
                """Поиск пивотных точек"""
                pivot_highs = []
                pivot_lows = []
                
                for i in range(period, len(high) - period):
                    # Pivot High
                    if all(high[i] >= high[i-j] for j in range(1, period+1)) and \
                       all(high[i] >= high[i+j] for j in range(1, period+1)):
                        pivot_highs.append((i, high[i]))
                    
                    # Pivot Low  
                    if all(low[i] <= low[i-j] for j in range(1, period+1)) and \
                       all(low[i] <= low[i+j] for j in range(1, period+1)):
                        pivot_lows.append((i, low[i]))
                
                return pivot_highs, pivot_lows
            
            # Находим пивоты
            pivot_highs, pivot_lows = find_pivot_points(
                df["high"].values, df["low"].values, df["close"].values
            )
            
            # Создаем серии для пивотов
            df["pivot_high"] = np.nan
            df["pivot_low"] = np.nan
            
            for idx, price in pivot_highs:
                df.iloc[idx, df.columns.get_loc("pivot_high")] = price
            
            for idx, price in pivot_lows:
                df.iloc[idx, df.columns.get_loc("pivot_low")] = price
            
            # === TREND STRUCTURE ===
            
            # Higher Highs / Lower Lows
            pivot_highs_series = df["pivot_high"].dropna()
            pivot_lows_series = df["pivot_low"].dropna()
            
            if len(pivot_highs_series) >= 2:
                higher_highs = (pivot_highs_series.diff() > 0).sum()
                lower_highs = (pivot_highs_series.diff() < 0).sum()
            else:
                higher_highs = lower_highs = 0
            
            if len(pivot_lows_series) >= 2:
                higher_lows = (pivot_lows_series.diff() > 0).sum()
                lower_lows = (pivot_lows_series.diff() < 0).sum()
            else:
                higher_lows = lower_lows = 0
            
            # Trend Structure Score
            bullish_structure = higher_highs + higher_lows
            bearish_structure = lower_highs + lower_lows
            
            if bullish_structure + bearish_structure > 0:
                df["structure_score"] = (bullish_structure - bearish_structure) / (bullish_structure + bearish_structure)
            else:
                df["structure_score"] = 0
                
            df["structure_score"] = df["structure_score"].ffill()
            
            # === BREAKOUT DETECTION ===
            
            # Support/Resistance strength
            lookback = 50
            
            def calculate_sr_strength(price_series, level, tolerance=0.001):
                """Расчет силы уровня поддержки/сопротивления"""
                touches = 0
                bounces = 0
                
                for i in range(1, len(price_series)):
                    if abs(price_series.iloc[i] - level) / level < tolerance:
                        touches += 1
                        
                        # Проверяем отскок
                        if i < len(price_series) - 2:
                            if (price_series.iloc[i-1] > level and price_series.iloc[i+1] > level) or \
                               (price_series.iloc[i-1] < level and price_series.iloc[i+1] < level):
                                bounces += 1
                
                return touches, bounces
            
            # Ближайшие уровни поддержки/сопротивления
            recent_pivots = df[["pivot_high", "pivot_low"]].tail(lookback)
            resistance_levels = recent_pivots["pivot_high"].dropna().values
            support_levels = recent_pivots["pivot_low"].dropna().values
            
            current_price = df["close"].iloc[-1]
            
            # Найти ближайшие уровни
            if len(resistance_levels) > 0:
                nearest_resistance = min(resistance_levels[resistance_levels > current_price], 
                                       key=lambda x: abs(x - current_price), default=np.nan)
            else:
                nearest_resistance = np.nan
                
            if len(support_levels) > 0:
                nearest_support = max(support_levels[support_levels < current_price],
                                    key=lambda x: abs(x - current_price), default=np.nan)  
            else:
                nearest_support = np.nan
            
            # Расстояние до уровней
            if not np.isnan(nearest_resistance):
                df["resistance_distance"] = (nearest_resistance - df["close"]) / df["close"] * 100
            else:
                df["resistance_distance"] = np.nan
                
            if not np.isnan(nearest_support):
                df["support_distance"] = (df["close"] - nearest_support) / df["close"] * 100
            else:
                df["support_distance"] = np.nan
            
            # CONFLUENCE ZONES
            confluence_score = 0
            
            # Схождение с MA
            ma_levels = [df["ema_21"].iloc[-1], df["ema_55"].iloc[-1], df.get("ema_200", pd.Series([current_price])).iloc[-1]]
            for ma_level in ma_levels:
                if not np.isnan(ma_level) and abs(current_price - ma_level) / current_price < 0.005:
                    confluence_score += 1
            
            # Схождение с Bollinger Bands
            if "bb_upper" in df.columns and "bb_lower" in df.columns:
                bb_upper = df["bb_upper"].iloc[-1]
                bb_lower = df["bb_lower"].iloc[-1]
                if (not np.isnan(bb_upper) and abs(current_price - bb_upper) / current_price < 0.003) or \
                   (not np.isnan(bb_lower) and abs(current_price - bb_lower) / current_price < 0.003):
                    confluence_score += 1
            
            # Схождение с пивотами
            all_levels = np.concatenate([resistance_levels, support_levels])
            for level in all_levels:
                if abs(current_price - level) / current_price < 0.005:
                    confluence_score += 1
            
            df["confluence_score"] = confluence_score / 5  # Нормализуем
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка структурных индикаторов: {e}")
            return df
    
    def _calculate_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикаторы микроструктуры рынка"""
        try:
            # ORDER FLOW ANALYSIS
            if "taker_buy_volume" in df.columns:
                df["volume_delta"] = df["taker_buy_volume"] - (df["volume"] - df["taker_buy_volume"])
                df["volume_delta_sma"] = df["volume_delta"].rolling(20).mean()
                df["cumulative_delta"] = df["volume_delta"].cumsum()
            else:
                df["volume_delta"] = 0
                df["volume_delta_sma"] = 0
                df["cumulative_delta"] = 0
            
            # LIQUIDITY INDICATORS
            df["spread_approx"] = (df["high"] - df["low"]) / df["typical_price"] * 10000  # basis points
            
            # Market Impact
            returns_abs = abs(df["close"].pct_change())
            volume_normalized = df["volume"] / df["volume"].rolling(50).mean()
            df["market_impact"] = returns_abs / (volume_normalized + 1e-10)
            df["liquidity_score"] = 1 / (1 + df["market_impact"])
            
            # INSTITUTIONAL FLOW
            volume_z_score = (df["volume"] - df["volume"].rolling(50).mean()) / df["volume"].rolling(50).std()
            df["large_trades"] = (volume_z_score > 2).astype(int)
            
            # Institutional pressure
            green_candle = df["close"] > df["open"]
            high_volume = df["volume"] > df["volume"].rolling(20).mean() * 1.5
            large_body = abs(df["close"] - df["open"]) > df.get("atr_14", df["close"] * 0.02) * 0.5
            
            df["institutional_buying"] = (green_candle & high_volume & large_body).astype(int)
            df["institutional_selling"] = ((~green_candle) & high_volume & large_body).astype(int)
            df["institutional_pressure"] = (df["institutional_buying"] - df["institutional_selling"]).rolling(10).sum()
            
            # TRADE SIZE ANALYSIS
            if "trades_count" in df.columns:
                avg_trade_size = df["volume"] / (df["trades_count"] + 1)
                df["avg_trade_size"] = avg_trade_size
                df["large_trade_ratio"] = (avg_trade_size > avg_trade_size.rolling(50).mean() * 2).astype(int)
                df["trade_intensity"] = df["trades_count"] / df["trades_count"].rolling(20).mean()
            else:
                df["avg_trade_size"] = df["volume"] / 100  # Приближение
                df["large_trade_ratio"] = 0
                df["trade_intensity"] = 1
            
            # SMART MONEY INDICATORS
            price_change = abs(df["close"].pct_change())
            volume_ratio = df["volume"] / df["volume"].rolling(20).mean()
            
            df["smart_money_index"] = np.where(
                (price_change > price_change.rolling(20).quantile(0.8)) & 
                (volume_ratio < 0.8), 1, 0
            )
            
            # VOLATILITY CLUSTERING
            returns = df["close"].pct_change()
            returns_squared = returns ** 2
            df["vol_clustering"] = returns_squared.rolling(5).mean() / returns_squared.rolling(50).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка микроструктурных индикаторов: {e}")
            return df
    
    def analyze_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Определение текущего режима рынка"""
        try:
            latest = df.iloc[-1]
            
            # Получаем ключевые метрики
            atr_pct = latest.get("atr_pct", 2.0)
            adx = latest.get("adx", 20)
            bb_width = latest.get("bb_width", 2.0)
            structure_score = latest.get("structure_score", 0)
            
            # Определяем волатильность
            vol_percentile = df["atr_pct"].rolling(100).rank(pct=True).iloc[-1] if len(df) >= 100 else 0.5
            
            # Определяем тренд
            ema_21 = latest.get("ema_21", latest["close"])
            ema_55 = latest.get("ema_55", latest["close"])
            ema_89 = latest.get("ema_89", latest["close"])
            ema_200 = latest.get("ema_200", latest["close"])
            
            ema_alignment = sum([
                ema_21 > ema_55,
                ema_55 > ema_89,
                ema_89 > ema_200
            ])
            
            # Логика определения режима
            if vol_percentile > 0.9 and atr_pct > 3.0:
                return MarketRegime.HIGH_VOLATILITY_EXPANSION
                
            elif vol_percentile < 0.1 and bb_width < 1.5:
                return MarketRegime.LOW_VOLATILITY_CONTRACTION
                
            elif adx > 30 and ema_alignment >= 3:
                recent_high = df["high"].rolling(20).max().iloc[-1]
                recent_low = df["low"].rolling(20).min().iloc[-1]
                
                if latest["close"] > recent_high * 0.998:
                    return MarketRegime.BREAKOUT_BULL
                elif latest["close"] < recent_low * 1.002:
                    return MarketRegime.BREAKOUT_BEAR
                else:
                    return MarketRegime.TRENDING_BULL
                    
            elif adx > 30 and ema_alignment <= 0:
                return MarketRegime.TRENDING_BEAR
                
            elif structure_score > 0.3:
                return MarketRegime.TRENDING_BULL
            elif structure_score < -0.3:
                return MarketRegime.TRENDING_BEAR
                
            elif abs(structure_score) < 0.1 and adx < 20:
                return MarketRegime.SIDEWAYS_CONSOLIDATION
            
            # Проверка зоны разворота
            rsi_14 = latest.get("rsi_14", 50)
            bb_position = latest.get("bb_position", 0.5)
            
            rsi_oversold = rsi_14 < 25 or rsi_14 > 75
            bb_extreme = bb_position < 0.05 or bb_position > 0.95
            
            if rsi_oversold and bb_extreme:
                return MarketRegime.REVERSAL_ZONE
            
            return MarketRegime.SIDEWAYS_CONSOLIDATION
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка определения режима: {e}")
            return MarketRegime.SIDEWAYS_CONSOLIDATION
    
    def calculate_signal_confidence(self, df: pd.DataFrame, signal_data: Dict) -> Tuple[float, SignalQuality]:
        """Расчет уверенности в сигнале"""
        try:
            latest = df.iloc[-1]
            confidence_factors = []
            
            # TECHNICAL CONVERGENCE
            convergence_score = 0
            total_indicators = 0
            
            # RSI convergence
            rsi_7 = latest.get("rsi_7", 50)
            rsi_14 = latest.get("rsi_14", 50)
            rsi_21 = latest.get("rsi_21", 50)
            
            if signal_data["direction"] == "LONG":
                rsi_signals = [rsi_7 < 30, rsi_14 < 30, rsi_21 < 35]
            else:
                rsi_signals = [rsi_7 > 70, rsi_14 > 70, rsi_21 > 65]
                
            convergence_score += sum(rsi_signals)
            total_indicators += 3
            
            # MACD convergence
            macd = latest.get("macd", 0)
            macd_signal = latest.get("macd_signal", 0)
            
            macd_bullish = macd > macd_signal and macd > 0
            macd_bearish = macd < macd_signal and macd < 0
            
            if (signal_data["direction"] == "LONG" and macd_bullish) or \
               (signal_data["direction"] == "SHORT" and macd_bearish):
                convergence_score += 1
            total_indicators += 1
            
            # Stochastic
            stoch_k = latest.get("stoch_k", 50)
            stoch_d = latest.get("stoch_d", 50)
            
            stoch_oversold = stoch_k < 20 and stoch_d < 20
            stoch_overbought = stoch_k > 80 and stoch_d > 80
            
            if (signal_data["direction"] == "LONG" and stoch_oversold) or \
               (signal_data["direction"] == "SHORT" and stoch_overbought):
                convergence_score += 1
            total_indicators += 1
            
            # Moving averages
            ema_21 = latest.get("ema_21", latest["close"])
            ema_55 = latest.get("ema_55", latest["close"])
            ema_200 = latest.get("ema_200", latest["close"])
            
            ma_bullish = ema_21 > ema_55 > ema_200
            ma_bearish = ema_21 < ema_55 < ema_200
            
            if (signal_data["direction"] == "LONG" and ma_bullish) or \
               (signal_data["direction"] == "SHORT" and ma_bearish):
                convergence_score += 1
            total_indicators += 1
            
            convergence_factor = convergence_score / total_indicators if total_indicators > 0 else 0
            confidence_factors.append(("convergence", convergence_factor, 0.3))
            
            # VOLUME CONFIRMATION
            volume_ratio = latest.get("volume_ratio", 1)
            volume_factor = min(volume_ratio, 3.0) / 3.0
            confidence_factors.append(("volume", volume_factor, 0.2))
            
            # MARKET STRUCTURE
            structure_factor = abs(latest.get("structure_score", 0))
            if signal_data["direction"] == "LONG":
                structure_factor = max(0, latest.get("structure_score", 0))
            else:
                structure_factor = max(0, -latest.get("structure_score", 0))
                
            confidence_factors.append(("structure", structure_factor, 0.15))
            
            # VOLATILITY ENVIRONMENT
            atr_pct = latest.get("atr_pct", 2.0)
            vol_factor = 1.0 if 1.5 <= atr_pct <= 4.0 else max(0, 1 - abs(atr_pct - 2.75) / 2.75)
            confidence_factors.append(("volatility", vol_factor, 0.1))
            
            # MOMENTUM QUALITY
            adx = latest.get("adx", 20)
            roc_5 = latest.get("roc_5", 0)
            
            adx_factor = min(adx / 50, 1.0)
            roc_factor = min(abs(roc_5) / 5.0, 1.0)
            momentum_score = (adx_factor + roc_factor) / 2
            
            confidence_factors.append(("momentum", momentum_score, 0.15))
            
            # INSTITUTIONAL FACTORS
            volume_ratio = latest.get("volume_ratio", 1)
            institutional_pressure = latest.get("institutional_pressure", 0)
            liquidity_score = latest.get("liquidity_score", 0.5)
            
            institutional_factors = []
            
            if volume_ratio > 2.0:
                institutional_factors.append(0.8)
            
            if signal_data["direction"] == "LONG" and institutional_pressure > 0:
                institutional_factors.append(0.7)
            elif signal_data["direction"] == "SHORT" and institutional_pressure < 0:
                institutional_factors.append(0.7)
            
            institutional_factors.append(liquidity_score)
            
            if institutional_factors:
                institutional_score = np.mean(institutional_factors)
                confidence_factors.append(("institutional", institutional_score, 0.1))
            
            # CALCULATE FINAL CONFIDENCE
            weighted_score = sum(factor * weight for _, factor, weight in confidence_factors)
            total_weight = sum(weight for _, _, weight in confidence_factors)
            
            base_confidence = weighted_score / total_weight if total_weight > 0 else 0.5
            
            # Risk penalty
            risk_penalty = 0
            correlation_risk = latest.get("correlation_risk", 0)
            risk_penalty += correlation_risk * 0.2
            
            if atr_pct > 6.0:
                risk_penalty += 0.3
            
            final_confidence = max(0, base_confidence - risk_penalty)
            
            # Determine signal quality
            if final_confidence >= 0.9:
                quality = SignalQuality.INSTITUTIONAL
            elif final_confidence >= 0.8:
                quality = SignalQuality.PROFESSIONAL
            elif final_confidence >= 0.7:
                quality = SignalQuality.RETAIL_PLUS
            elif final_confidence >= 0.6:
                quality = SignalQuality.RETAIL
            else:
                quality = SignalQuality.NOISE
            
            return final_confidence, quality
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка расчета уверенности: {e}")
            return 0.5, SignalQuality.RETAIL
            
            
            
    def _detect_rsi_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Детекция RSI сигналов"""
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        try:
            # Multi-timeframe RSI divergence
            rsi_7 = latest["rsi_7"]
            rsi_14 = latest["rsi_14"]
            rsi_21 = latest["rsi_21"]
            
            # LONG signals
            if rsi_14 < 35 and rsi_7 < 30:
                score = 0.7
                
                # Bonus for divergence
                if rsi_14 > prev["rsi_14"] and latest["close"] < prev["close"]:
                    score += 0.2
                
                # Bonus for multiple timeframe confirmation
                if rsi_21 < 40:
                    score += 0.1
                
                signals.append({
                    "symbol": "unknown",
                    "direction": "LONG", 
                    "score": score,
                    "convergence_score": score * 0.8,
                    "momentum_score": (35 - rsi_14) / 35,
                    "timing_score": 0.8,
                    "type": "RSI_OVERSOLD"
                })
            
            # SHORT signals
            elif rsi_14 > 65 and rsi_7 > 70:
                score = 0.7
                
                # Bonus for divergence
                if rsi_14 < prev["rsi_14"] and latest["close"] > prev["close"]:
                    score += 0.2
                
                # Bonus for multiple timeframe confirmation
                if rsi_21 > 60:
                    score += 0.1
                
                signals.append({
                    "symbol": "unknown",
                    "direction": "SHORT",
                    "score": score,
                    "convergence_score": score * 0.8,
                    "momentum_score": (rsi_14 - 65) / 35,
                    "timing_score": 0.8,
                    "type": "RSI_OVERBOUGHT"
                })
        
        except Exception as e:
            self.logger.error(f"❌ Ошибка детекции RSI: {e}")
        
        return signals  
        
    def _detect_macd_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Детекция MACD сигналов"""
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        try:
            macd = latest["macd"]
            macd_signal = latest["macd_signal"] 
            macd_hist = latest["macd_histogram"]
            
            prev_macd = prev["macd"]
            prev_signal = prev["macd_signal"]
            
            # Bullish crossover
            if macd > macd_signal and prev_macd <= prev_signal:
                score = 0.6
                
                # Bonus for crossover below zero line
                if macd < 0:
                    score += 0.15
                
                # Bonus for histogram increasing
                if macd_hist > 0:
                    score += 0.1
                
                # Bonus for fast MACD confirmation
                fast_macd = latest.get("macd_fast", 0)
                if fast_macd > latest.get("macd_fast_signal", 0):
                    score += 0.1
                
                signals.append({
                    "symbol": "unknown",
                    "direction": "LONG",
                    "score": score,
                    "convergence_score": score * 0.9,
                    "momentum_score": abs(macd_hist) / 100,  # Normalized
                    "timing_score": 0.9,
                    "type": "MACD_BULLISH_CROSS"
                })
            
            # Bearish crossover
            elif macd < macd_signal and prev_macd >= prev_signal:
                score = 0.6
                
                # Bonus for crossover above zero line
                if macd > 0:
                    score += 0.15
                
                # Bonus for histogram decreasing
                if macd_hist < 0:
                    score += 0.1
                
                # Bonus for fast MACD confirmation
                fast_macd = latest.get("macd_fast", 0)
                if fast_macd < latest.get("macd_fast_signal", 0):
                    score += 0.1
                
                signals.append({
                    "symbol": "unknown", 
                    "direction": "SHORT",
                    "score": score,
                    "convergence_score": score * 0.9,
                    "momentum_score": abs(macd_hist) / 100,
                    "timing_score": 0.9,
                    "type": "MACD_BEARISH_CROSS"
                })
        
        except Exception as e:
            self.logger.error(f"❌ Ошибка детекции MACD: {e}")
        
        return signals
        
        
    def _detect_bollinger_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Детекция сигналов Bollinger Bands"""
        signals = []
        latest = df.iloc[-1]
        
        try:
            bb_position = latest["bb_position"]
            bb_width = latest["bb_width"]
            close = latest["close"]
            
            # Bollinger Band squeeze breakout
            if bb_width < df["bb_width"].rolling(50).quantile(0.2).iloc[-1]:
                # Potential breakout setup
                if bb_position > 0.8:  # Near upper band
                    signals.append({
                        "symbol": "unknown",
                        "direction": "LONG",
                        "score": 0.6,
                        "convergence_score": 0.5,
                        "momentum_score": bb_position,
                        "timing_score": 0.7,
                        "type": "BB_SQUEEZE_BREAKOUT_BULL"
                    })
                elif bb_position < 0.2:  # Near lower band
                    signals.append({
                        "symbol": "unknown",
                        "direction": "SHORT", 
                        "score": 0.6,
                        "convergence_score": 0.5,
                        "momentum_score": 1 - bb_position,
                        "timing_score": 0.7,
                        "type": "BB_SQUEEZE_BREAKOUT_BEAR"
                    })
            
            # Mean reversion signals
            elif bb_position < 0.05:  # Extreme oversold
                signals.append({
                    "symbol": "unknown",
                    "direction": "LONG",
                    "score": 0.65,
                    "convergence_score": 0.6,
                    "momentum_score": 0.05 - bb_position,
                    "timing_score": 0.8,
                    "type": "BB_MEAN_REVERSION_BULL"
                })
            elif bb_position > 0.95:  # Extreme overbought
                signals.append({
                    "symbol": "unknown",
                    "direction": "SHORT",
                    "score": 0.65,
                    "convergence_score": 0.6,
                    "momentum_score": 0.95 - bb_position,
                    "timing_score": 0.8,
                    "type": "BB_MEAN_REVERSION_BEAR"
                }) 
                
                
        except Exception as e:
                print(f"Ошибка в Bollinger Bands анализе: {e}")
        
        return signals    
    
    def generate_advanced_signal(self, df: pd.DataFrame) -> Optional[AdvancedSignal]:
        """Генерация продвинутого торгового сигнала"""
        try:
            timer = Timer().start()
            
            if len(df) < self.min_data_points:
                return None
            
            latest = df.iloc[-1]
            
            # SIGNAL DETECTION
            signal_candidates = []
            
            # Multi-timeframe RSI
            rsi_signals = self._detect_rsi_signals(df)
            signal_candidates.extend(rsi_signals)
            
            # MACD signals  
            macd_signals = self._detect_macd_signals(df)
            signal_candidates.extend(macd_signals)
            
            # Bollinger Band signals
            bb_signals = self._detect_bollinger_signals(df)
            signal_candidates.extend(bb_signals)
            
            # # Volume signals
            # volume_signals = self._detect_volume_signals(df)
            # signal_candidates.extend(volume_signals)
            
            # Institutional signals
            # institutional_signals = self._detect_institutional_signals(df)
            # signal_candidates.extend(institutional_signals)
            
            # Structure signals
            # structure_signals = self._detect_structure_signals(df)
            # signal_candidates.extend(structure_signals)
            
            if not signal_candidates:
                return None
            
            # Выбираем лучший сигнал
            best_signal = max(signal_candidates, key=lambda x: x.get("score", 0))
            
            if best_signal.get("score", 0) < 0.6:
                return None
            
            # CONFIDENCE CALCULATION
            confidence, quality = self.calculate_signal_confidence(df, best_signal)
            
            if confidence < 0.6:
                return None
            
            # RISK MANAGEMENT
            entry_price = latest["close"]
            atr = latest.get("atr_14", latest["close"] * 0.02)
            
            atr_multiplier = self._calculate_atr_multiplier(df, best_signal)
            
            if best_signal["direction"] == "LONG":
                stop_loss = entry_price - (atr * atr_multiplier)
                resistance_distance = latest.get("resistance_distance", np.nan)
                if not np.isnan(resistance_distance) and resistance_distance > 0:
                    target_distance = min(resistance_distance * 0.8 * entry_price / 100, atr * atr_multiplier * 3)
                else:
                    target_distance = atr * atr_multiplier * 2.5
                take_profit = entry_price + target_distance
            else:
                stop_loss = entry_price + (atr * atr_multiplier)
                support_distance = latest.get("support_distance", np.nan)
                if not np.isnan(support_distance) and support_distance > 0:
                    target_distance = min(support_distance * 0.8 * entry_price / 100, atr * atr_multiplier * 3)
                else:
                    target_distance = atr * atr_multiplier * 2.5
                take_profit = entry_price - target_distance
            
            # Check R:R ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            if risk_reward_ratio < 1.5:
                return None
            
            # MARKET CONTEXT
            market_regime = self.analyze_market_regime(df)
            volatility_percentile = df["atr_pct"].rolling(100).rank(pct=True).iloc[-1] if len(df) >= 100 else 0.5
            
            # CREATE SIGNAL
            signal = AdvancedSignal(
                symbol="",  # Будет установлен в signal_generator
                timestamp=datetime.now(),
                direction=best_signal["direction"],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                
                confidence_score=confidence,
                signal_quality=quality,
                risk_reward_ratio=risk_reward_ratio,
                
                market_regime=market_regime,
                volatility_percentile=volatility_percentile,
                volume_confirmation=latest.get("volume_ratio", 1),
                
                convergence_score=best_signal.get("convergence_score", 0),
                divergence_signals=best_signal.get("divergences", []),
                support_resistance_score=latest.get("confluence_score", 0),
                momentum_score=best_signal.get("momentum_score", 0),
                
                order_flow_score=latest.get("institutional_pressure", 0),
                liquidity_score=latest.get("liquidity_score", 0.5),
                institutional_bias=latest.get("institutional_pressure", 0),
                
                ml_probability=None,
                ensemble_score=None,
                
                entry_timing_score=best_signal.get("timing_score", 0.5),
                market_hours_factor=self._get_market_hours_factor(),
                session_factor=self._get_trading_session(),
                
                max_adverse_excursion=risk * 0.5,
                expected_holding_time=self._estimate_holding_time(df, best_signal),
                correlation_risk=latest.get("correlation_risk", 0.1)
            )
            
            timer.stop()
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации сигнала: {e}")
            return None


def create_market_analyzer(config: Config) -> AdvancedMarketAnalyzer:
    """Фабричная функция для создания анализатора"""
    return AdvancedMarketAnalyzer(config)
            
            
    def _calculate_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикаторы микроструктуры рынка"""
        try:
            # === ORDER FLOW ANALYSIS ===
            
            # Volume Delta (приближение через taker buy/sell)
            df["volume_delta"] = df["taker_buy_volume"] - (df["volume"] - df["taker_buy_volume"])
            df["volume_delta_sma"] = df["volume_delta"].rolling(20).mean()
            
            # Cumulative Volume Delta
            df["cumulative_delta"] = df["volume_delta"].cumsum()
            
            # Delta Divergence (цена vs накопленная дельта)
            price_direction = np.where(df["close"] > df["close"].shift(1), 1, -1)
            delta_direction = np.where(df["volume_delta"] > 0, 1, -1)
            df["delta_divergence"] = (price_direction != delta_direction).astype(int)
            
            # === LIQUIDITY INDICATORS ===
            
            # Bid-Ask Spread approximation
            df["spread_approx"] = (df["high"] - df["low"]) / df["typical_price"] * 10000  # basis points
            
            # Market Impact (цена движется быстрее объема = низкая ликвидность)
            returns_abs = abs(df["close"].pct_change())
            volume_normalized = df["volume"] / df["volume"].rolling(50).mean()
            df["market_impact"] = returns_abs / (volume_normalized + 1e-10)
            df["liquidity_score"] = 1 / (1 + df["market_impact"])  # Обратная зависимость
            
            # === INSTITUTIONAL FLOW ===
            
            # Large Trade Detection (через аномальные объемы)
            volume_z_score = (df["volume"] - df["volume"].rolling(50).mean()) / df["volume"].rolling(50).std()
            df["large_trades"] = (volume_z_score > 2).astype(int)
            
            # Institutional Buying/Selling Pressure
            # Большие зеленые свечи с высоким объемом = институциональная покупка
            green_candle = df["close"] > df["open"]
            high_volume = df["volume"] > df["volume"].rolling(20).mean() * 1.5
            large_body = abs(df["close"] - df["open"]) > df["atr_14"] * 0.5
            
            df["institutional_buying"] = (green_candle & high_volume & large_body).astype(int)
            df["institutional_selling"] = ((~green_candle) & high_volume & large_body).astype(int)
            
            # Накопленное институциональное давление
            df["institutional_pressure"] = (df["institutional_buying"] - df["institutional_selling"]).rolling(10).sum()
            
            # === TIME & SALES PATTERNS ===
            
            # Trade Size Analysis
            avg_trade_size = df["volume"] / df["trades_count"]
            df["avg_trade_size"] = avg_trade_size
            df["large_trade_ratio"] = (avg_trade_size > avg_trade_size.rolling(50).mean() * 2).astype(int)
            
            # Pace of Trading
            df["trade_intensity"] = df["trades_count"] / df["trades_count"].rolling(20).mean()
            
            # === SMART MONEY INDICATORS ===
            
            # Smart Money Index (большие движения на низком объеме подозрительны)
            price_change = abs(df["close"].pct_change())
            volume_ratio = df["volume"] / df["volume"].rolling(20).mean()
            
            # Когда цена сильно двигается, а объем низкий - возможно накопление/распределение  
            df["smart_money_index"] = np.where(
                (price_change > price_change.rolling(20).quantile(0.8)) & 
                (volume_ratio < 0.8), 1, 0
            )
            
            # === VOLATILITY CLUSTERING ===
            
            # GARCH-like volatility clustering
            returns = df["close"].pct_change()
            returns_squared = returns ** 2
            
            # Простая модель кластеризации волатильности
            df["vol_clustering"] = returns_squared.rolling(5).mean() / returns_squared.rolling(50).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка микроструктурных индикаторов: {e}")
            return df
    
    def analyze_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Определение текущего режима рынка"""
        try:
            latest = df.iloc[-1]
            
            # Получаем ключевые метрики
            atr_pct = latest["atr_pct"]
            adx = latest["adx"]
            bb_width = latest["bb_width"]
            vol_regime = latest.get("vol_regime", 1)
            structure_score = latest.get("structure_score", 0)
            
            # Определяем волатильность
            vol_percentile = df["atr_pct"].rolling(100).rank(pct=True).iloc[-1]
            
            # Определяем тренд
            ema_alignment = sum([
                latest["ema_21"] > latest["ema_55"],
                latest["ema_55"] > latest["ema_89"], 
                latest["ema_89"] > latest["ema_200"]
            ])
            
            # Логика определения режима
            if vol_percentile > 0.9 and atr_pct > 3.0:
                return MarketRegime.HIGH_VOLATILITY_EXPANSION
                
            elif vol_percentile < 0.1 and bb_width < 1.5:
                return MarketRegime.LOW_VOLATILITY_CONTRACTION
                
            elif adx > 30 and ema_alignment >= 3:
                # Проверяем breakout
                recent_high = df["high"].rolling(20).max().iloc[-1]
                recent_low = df["low"].rolling(20).min().iloc[-1]
                
                if latest["close"] > recent_high * 0.998:
                    return MarketRegime.BREAKOUT_BULL
                elif latest["close"] < recent_low * 1.002:
                    return MarketRegime.BREAKOUT_BEAR
                else:
                    return MarketRegime.TRENDING_BULL
                    
            elif adx > 30 and ema_alignment <= 0:
                return MarketRegime.TRENDING_BEAR
                
            elif structure_score > 0.3:
                return MarketRegime.TRENDING_BULL
            elif structure_score < -0.3:
                return MarketRegime.TRENDING_BEAR
                
            elif abs(structure_score) < 0.1 and adx < 20:
                return MarketRegime.SIDEWAYS_CONSOLIDATION
            
            # Проверка зоны разворота
            rsi_oversold = latest["rsi_14"] < 25 or latest["rsi_14"] > 75
            bb_extreme = latest["bb_position"] < 0.05 or latest["bb_position"] > 0.95
            
            if rsi_oversold and bb_extreme:
                return MarketRegime.REVERSAL_ZONE
            
            return MarketRegime.SIDEWAYS_CONSOLIDATION
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка определения режима: {e}")
            return MarketRegime.SIDEWAYS_CONSOLIDATION
    
    def calculate_signal_confidence(self, df: pd.DataFrame, signal_data: Dict) -> Tuple[float, SignalQuality]:
        """Расчет уверенности в сигнале"""
        try:
            latest = df.iloc[-1]
            confidence_factors = []
            
            # === TECHNICAL CONVERGENCE ===
            
            convergence_score = 0
            total_indicators = 0
            
            # RSI convergence
            rsi_signals = [
                latest["rsi_7"] < 30 if signal_data["direction"] == "LONG" else latest["rsi_7"] > 70,
                latest["rsi_14"] < 30 if signal_data["direction"] == "LONG" else latest["rsi_14"] > 70,
                latest["rsi_21"] < 35 if signal_data["direction"] == "LONG" else latest["rsi_21"] > 65
            ]
            convergence_score += sum(rsi_signals)
            total_indicators += 3
            
            # MACD convergence
            macd_bullish = latest["macd"] > latest["macd_signal"] and latest["macd"] > 0
            macd_bearish = latest["macd"] < latest["macd_signal"] and latest["macd"] < 0
            
            if (signal_data["direction"] == "LONG" and macd_bullish) or \
               (signal_data["direction"] == "SHORT" and macd_bearish):
                convergence_score += 1
            total_indicators += 1
            
            # Stochastic convergence
            stoch_oversold = latest["stoch_k"] < 20 and latest["stoch_d"] < 20
            stoch_overbought = latest["stoch_k"] > 80 and latest["stoch_d"] > 80
            
            if (signal_data["direction"] == "LONG" and stoch_oversold) or \
               (signal_data["direction"] == "SHORT" and stoch_overbought):
                convergence_score += 1
            total_indicators += 1
            
            # Moving Average convergence
            ma_bullish = latest["ema_21"] > latest["ema_55"] > latest["ema_200"]
            ma_bearish = latest["ema_21"] < latest["ema_55"] < latest["ema_200"]
            
            if (signal_data["direction"] == "LONG" and ma_bullish) or \
               (signal_data["direction"] == "SHORT" and ma_bearish):
                convergence_score += 1
            total_indicators += 1
            
            convergence_factor = convergence_score / total_indicators
            confidence_factors.append(("convergence", convergence_factor, 0.3))
            
            # === VOLUME CONFIRMATION ===
            
            volume_factor = min(latest["volume_ratio"], 3.0) / 3.0  # Cap at 3x
            confidence_factors.append(("volume", volume_factor, 0.2))
            
            # === MARKET STRUCTURE ===
            
            structure_factor = abs(latest.get("structure_score", 0))
            if signal_data["direction"] == "LONG":
                structure_factor = max(0, latest.get("structure_score", 0))
            else:
                structure_factor = max(0, -latest.get("structure_score", 0))
                
            confidence_factors.append(("structure", structure_factor, 0.15))
            
            # === VOLATILITY ENVIRONMENT ===
            
            atr_pct = latest["atr_pct"]
            vol_factor = 1.0 if 1.5 <= atr_pct <= 4.0 else max(0, 1 - abs(atr_pct - 2.75) / 2.75)
            confidence_factors.append(("volatility", vol_factor, 0.1))
            
            # === MOMENTUM QUALITY ===
            
            momentum_factors = []
            
            # ADX strength
            adx_factor = min(latest["adx"] / 50, 1.0)
            momentum_factors.append(adx_factor)
            
            # Rate of Change
            roc_factor = abs(latest.get("roc_5", 0)) / 5.0  # 5% = 1.0
            momentum_factors.append(min(roc_factor, 1.0))
            
            momentum_score = np.mean(momentum_factors)
            confidence_factors.append(("momentum", momentum_score, 0.15))
            
            # === RISK FACTORS (NEGATIVE) ===
            
            risk_penalty = 0
            
            # High correlation risk
            correlation_risk = latest.get("correlation_risk", 0)
            risk_penalty += correlation_risk * 0.2
            
            # Extreme volatility
            if atr_pct > 6.0:
                risk_penalty += 0.3
            
            # News events proximity (would be implemented with news feed)
            # risk_penalty += news_risk * 0.1
            
            # === INSTITUTIONAL FACTORS ===
            
            institutional_factors = []
            
            # Large volume confirmation
            if latest["volume_ratio"] > 2.0:
                institutional_factors.append(0.8)
            
            # Smart money flow
            smart_money = latest.get("institutional_pressure", 0)
            if signal_data["direction"] == "LONG" and smart_money > 0:
                institutional_factors.append(0.7)
            elif signal_data["direction"] == "SHORT" and smart_money < 0:
                institutional_factors.append(0.7)
            
            # Liquidity
            liquidity_score = latest.get("liquidity_score", 0.5)
            institutional_factors.append(liquidity_score)
            
            if institutional_factors:
                institutional_score = np.mean(institutional_factors)
                confidence_factors.append(("institutional", institutional_score, 0.1))
            
            # === CALCULATE FINAL CONFIDENCE ===
            
            weighted_score = sum(factor * weight for _, factor, weight in confidence_factors)
            total_weight = sum(weight for _, _, weight in confidence_factors)
            
            if total_weight > 0:
                base_confidence = weighted_score / total_weight
            else:
                base_confidence = 0.5
            
            # Apply risk penalty
            final_confidence = max(0, base_confidence - risk_penalty)
            
            # Determine signal quality
            if final_confidence >= 0.9:
                quality = SignalQuality.INSTITUTIONAL
            elif final_confidence >= 0.8:
                quality = SignalQuality.PROFESSIONAL
            elif final_confidence >= 0.7:
                quality = SignalQuality.RETAIL_PLUS
            elif final_confidence >= 0.6:
                quality = SignalQuality.RETAIL
            else:
                quality = SignalQuality.NOISE
            
            return final_confidence, quality
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка расчета уверенности: {e}")
            return 0.5, SignalQuality.RETAIL
    
    def generate_advanced_signal(self, df: pd.DataFrame) -> Optional[AdvancedSignal]:
        """Генерация продвинутого торгового сигнала"""
        try:
            timer = Timer().start()
            
            if len(df) < self.min_data_points:
                self.logger.warning(f"⚠️ Недостаточно данных: {len(df)}")
                return None
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # === SIGNAL DETECTION ===
            
            signal_candidates = []
            
            # Multi-timeframe RSI
            rsi_signals = self._detect_rsi_signals(df)
            signal_candidates.extend(rsi_signals)
            
            # MACD signals  
            macd_signals = self._detect_macd_signals(df)
            signal_candidates.extend(macd_signals)
            
            # Bollinger Band signals
            bb_signals = self._detect_bollinger_signals(df)
            signal_candidates.extend(bb_signals)
            
            # Volume breakout signals
            # volume_signals = self._detect_volume_signals(df)
            # signal_candidates.extend(volume_signals)
            
            # Institutional flow signals
            # institutional_signals = self._detect_institutional_signals(df)
            # signal_candidates.extend(institutional_signals)
            
            # Structure break signals
            # structure_signals = self._detect_structure_signals(df)
            # signal_candidates.extend(structure_signals)
            
            if not signal_candidates:
                return None
            
            # Выбираем лучший сигнал
            best_signal = max(signal_candidates, key=lambda x: x["score"])
            
            if best_signal["score"] < 0.6:  # Минимальный порог
                return None
            
            # === CONFIDENCE CALCULATION ===
            
            confidence, quality = self.calculate_signal_confidence(df, best_signal)
            
            if confidence < 0.6:  # Дополнительная фильтрация
                return None
            
            # === RISK MANAGEMENT ===
            
            entry_price = latest["close"]
            atr = latest["atr_14"]
            
            # Динамический стоп-лосс на основе волатильности и структуры
            atr_multiplier = self._calculate_atr_multiplier(df, best_signal)
            
            if best_signal["direction"] == "LONG":
                stop_loss = entry_price - (atr * atr_multiplier)
                # Take profit на основе ближайшего сопротивления или R:R
                resistance_distance = latest.get("resistance_distance", np.nan)
                if not np.isnan(resistance_distance) and resistance_distance > 0:
                    target_distance = min(resistance_distance * 0.8, atr * atr_multiplier * 3)
                else:
                    target_distance = atr * atr_multiplier * 2.5
                take_profit = entry_price + target_distance
            else:
                stop_loss = entry_price + (atr * atr_multiplier)
                # Take profit на основе ближайшей поддержки или R:R
                support_distance = latest.get("support_distance", np.nan)
                if not np.isnan(support_distance) and support_distance > 0:
                    target_distance = min(support_distance * 0.8, atr * atr_multiplier * 3)
                else:
                    target_distance = atr * atr_multiplier * 2.5
                take_profit = entry_price - target_distance
            
            # Проверяем R:R ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            if risk_reward_ratio < 1.5:  # Минимальный R:R
                return None
            
            # === MARKET CONTEXT ===
            
            market_regime = self.analyze_market_regime(df)
            volatility_percentile = df["atr_pct"].rolling(100).rank(pct=True).iloc[-1]
            
            # === CREATE ADVANCED SIGNAL ===
            
            signal = AdvancedSignal(
                symbol=best_signal["symbol"],
                timestamp=datetime.now(),
                direction=best_signal["direction"],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                
                confidence_score=confidence,
                signal_quality=quality,
                risk_reward_ratio=risk_reward_ratio,
                
                market_regime=market_regime,
                volatility_percentile=volatility_percentile,
                volume_confirmation=latest["volume_ratio"],
                
                convergence_score=best_signal["convergence_score"],
                divergence_signals=best_signal.get("divergences", []),
                support_resistance_score=latest.get("confluence_score", 0),
                momentum_score=best_signal["momentum_score"],
                
                order_flow_score=latest.get("institutional_pressure", 0),
                liquidity_score=latest.get("liquidity_score", 0.5),
                institutional_bias=latest.get("institutional_pressure", 0),
                
                ml_probability=None,  # Will be set by ML service
                ensemble_score=None,
                
                entry_timing_score=best_signal["timing_score"],
                market_hours_factor=self._get_market_hours_factor(),
                session_factor=self._get_trading_session(),
                
                max_adverse_excursion=risk * 0.5,  # Expected MAE
                expected_holding_time=self._estimate_holding_time(df, best_signal),
                correlation_risk=latest.get("correlation_risk", 0.1)
            )
            
            timer.stop()
            self.logger.info(f"🎯 Сигнал сгенерирован за {timer.elapsed_str()}: {signal.direction} {signal.symbol}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации сигнала: {e}")
            return None
    
    
    
    
    
                