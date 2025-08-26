import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import joblib
import os

from config.settings import Config, TradingConfig
from database import DatabaseManager, SubscriptionService, TradingSignal as DBSignal
from .market_analysis import AdvancedMarketAnalyzer, AdvancedSignal, SignalQuality, MarketRegime
# from .ml_model_trainer import AdvancedMLTrainer, create_ml_trainer
from utils.helpers import Timer, format_datetime


class SignalService:
    """Основной сервис генерации и управления торговыми сигналами"""
    
    def __init__(self, config: Config, db_manager: DatabaseManager, bot):
        self.config = config
        self.db = db_manager
        self.bot = bot
        self.subscription_service = SubscriptionService(db_manager)
        self.logger = logging.getLogger(__name__)
        
        # Инициализация компонентов
        self.market_analyzer = AdvancedMarketAnalyzer(config)
        # self.ml_trainer = create_ml_trainer(config)
        
        # Загружаем обученную ML модель
        self.ml_model = None
        self.ml_scalers = None
        self._load_ml_model()
        
        # Кэш данных для оптимизации
        self.data_cache = {}
        self.cache_timeout = 300  # 5 минут
        
        # Статистика
        self.stats = {
            'signals_generated': 0,
            'signals_sent': 0,
            'ml_predictions': 0,
            'analysis_time_total': 0,
            'last_analysis': None
        }
    
    def _load_ml_model(self):
        """Загрузка обученной ML модели"""
        try:
            if os.path.exists(self.config.ML_PIPELINE_PATH):
                model_data = joblib.load(self.config.ML_PIPELINE_PATH)
                self.ml_model = model_data.get('ensemble_model')
                self.ml_scalers = model_data.get('scalers', {})
                self.logger.info("ML модель загружена успешно")
            else:
                self.logger.warning(f"ML модель не найдена: {self.config.ML_PIPELINE_PATH}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки ML модели: {e}")
    
    def get_futures_symbols(self) -> List[str]:
        """Получение списка активных фьючерсных пар"""
        try:
            if 'futures_symbols' in self.data_cache:
                cache_time, symbols = self.data_cache['futures_symbols']
                if (datetime.now() - cache_time).seconds < 3600:  # 1 час кэш
                    return symbols
            
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            symbols = [
                symbol["symbol"] for symbol in data["symbols"] 
                if symbol["status"] == "TRADING" and symbol["symbol"].endswith("USDT")
            ]
            
            # Фильтруем топ символы по объему для оптимизации
            volume_symbols = self._get_top_volume_symbols(symbols[:100])  # Топ 100
            
            self.data_cache['futures_symbols'] = (datetime.now(), volume_symbols)
            
            self.logger.info(f"Получено {len(volume_symbols)} активных символов")
            return volume_symbols
            
        except Exception as e:
            self.logger.error(f"Ошибка получения символов: {e}")
            # Fallback список
            return [
                "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
                "LTCUSDT", "BCHUSDT", "XLMUSDT", "EOSUSDT", "TRXUSDT"
            ]
    
    def _get_top_volume_symbols(self, symbols: List[str]) -> List[str]:
        """Получение символов с наибольшим объемом торгов"""
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            tickers = response.json()
            
            # Фильтруем и сортируем по объему
            usdt_tickers = [
                ticker for ticker in tickers 
                if ticker['symbol'] in symbols and ticker['symbol'].endswith('USDT')
            ]
            
            sorted_tickers = sorted(
                usdt_tickers, 
                key=lambda x: float(x['quoteVolume']), 
                reverse=True
            )
            
            # Возвращаем топ 50 по объему
            return [ticker['symbol'] for ticker in sorted_tickers[:50]]
            
        except Exception as e:
            self.logger.error(f"Ошибка фильтрации по объему: {e}")
            return symbols
    
    def analyze_and_broadcast(self):
        """Основной метод анализа рынка и рассылки сигналов"""
        try:
            timer = Timer().start()
            
            if not self.config.AUTO_BROADCAST_ENABLED:
                self.logger.info("Авторассылка отключена")
                return
            
            self.logger.info("Начинаем анализ рынка для генерации сигналов")
            
            # Получаем список символов
            symbols = self.get_futures_symbols()
            
            # Анализируем каждый символ
            generated_signals = []
            analysis_results = []
            
            for i, symbol in enumerate(symbols, 1):
                try:
                    self.logger.info(f"[{i}/{len(symbols)}] Анализ {symbol}")
                    
                    # Получаем данные
                    df = self.market_analyzer.get_market_data(symbol)
                    if df is None or len(df) < self.market_analyzer.min_data_points:
                        continue
                    
                    # Рассчитываем индикаторы
                    df_with_indicators = self.market_analyzer.calculate_advanced_indicators(df)
                    if df_with_indicators.empty:
                        continue
                    
                    # Генерируем сигнал
                    signal = self.market_analyzer.generate_advanced_signal(df_with_indicators)
                    if signal:
                        # Добавляем ML предсказание
                        if self.config.ENABLE_ML_VALIDATION and self.ml_model:
                            ml_prediction = self._get_ml_prediction(df_with_indicators, signal)
                            signal.ml_probability = ml_prediction
                            
                            # Фильтруем сигналы с низким ML скором
                            if ml_prediction < 0.6:
                                self.logger.info(f"ML отклонила сигнал {symbol}: {ml_prediction:.2%}")
                                continue
                        
                        signal.symbol = symbol
                        generated_signals.append(signal)
                        
                        analysis_results.append({
                            'symbol': symbol,
                            'signal_generated': True,
                            'confidence': signal.confidence_score,
                            'quality': signal.signal_quality.value
                        })
                    else:
                        analysis_results.append({
                            'symbol': symbol,
                            'signal_generated': False,
                            'reason': 'No qualifying signal found'
                        })
                
                except Exception as e:
                    self.logger.error(f"Ошибка анализа {symbol}: {e}")
                    analysis_results.append({
                        'symbol': symbol,
                        'signal_generated': False,
                        'reason': f'Analysis error: {str(e)}'
                    })
            
            # Ранжируем сигналы по качеству
            if generated_signals:
                ranked_signals = self._rank_signals(generated_signals)
                
                # Отправляем лучшие сигналы
                sent_count = self._broadcast_signals(ranked_signals)
                
                self.logger.info(f"Отправлено {sent_count} сигналов из {len(generated_signals)} сгенерированных")
            else:
                self.logger.info("Не найдено качественных сигналов для отправки")
            
            timer.stop()
            
            # Обновляем статистику
            self.stats['signals_generated'] += len(generated_signals)
            self.stats['analysis_time_total'] += timer.elapsed()
            self.stats['last_analysis'] = datetime.now()
            
            self.logger.info(f"Анализ завершен за {timer.elapsed_str()}")
            
        except Exception as e:
            self.logger.error(f"Критическая ошибка анализа: {e}")
    
    def _get_ml_prediction(self, df: pd.DataFrame, signal: AdvancedSignal) -> Optional[float]:
        """Получение ML предсказания для сигнала"""
        try:
            if not self.ml_model:
                return None
            
            # Подготавливаем фичи
            features = self._prepare_ml_features(df, signal)
            if features is None:
                return None
            
            # Масштабирование
            if 'feature_scaler' in self.ml_scalers:
                scaler = self.ml_scalers['feature_scaler']
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            # Предсказание
            if hasattr(self.ml_model, 'predict_proba'):
                # Вероятность успешного сигнала
                probabilities = self.ml_model.predict_proba(features_scaled)
                # Берем вероятность положительного класса
                positive_prob = probabilities[0][1] if probabilities.shape[1] > 1 else probabilities[0][0]
                
                self.stats['ml_predictions'] += 1
                return float(positive_prob)
            else:
                prediction = self.ml_model.predict(features_scaled)
                return float(prediction[0]) if prediction[0] > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Ошибка ML предсказания: {e}")
            return None
    
    def _prepare_ml_features(self, df: pd.DataFrame, signal: AdvancedSignal) -> Optional[np.ndarray]:
        """Подготовка фичей для ML модели"""
        try:
            latest = df.iloc[-1]
            
            # Базовые технические фичи
            features = [
                latest.get("rsi_7", 50),
                latest.get("rsi_14", 50),
                latest.get("rsi_21", 50),
                latest.get("macd", 0),
                latest.get("macd_signal", 0),
                latest.get("bb_position", 0.5),
                latest.get("bb_width", 2),
                latest.get("atr_pct", 2),
                latest.get("adx", 20),
                latest.get("volume_ratio", 1),
                latest.get("stoch_k", 50),
                latest.get("stoch_d", 50)
            ]
            
            # Фичи сигнала
            signal_features = [
                signal.confidence_score,
                signal.risk_reward_ratio,
                signal.volatility_percentile,
                signal.volume_confirmation,
                signal.convergence_score,
                signal.momentum_score,
                signal.entry_timing_score,
                1.0 if signal.direction == "LONG" else 0.0,
                signal.market_hours_factor
            ]
            
            # Режимные фичи
            regime_features = [
                1.0 if signal.market_regime == MarketRegime.TRENDING_BULL else 0.0,
                1.0 if signal.market_regime == MarketRegime.TRENDING_BEAR else 0.0,
                1.0 if signal.market_regime == MarketRegime.SIDEWAYS_CONSOLIDATION else 0.0,
                1.0 if signal.market_regime == MarketRegime.HIGH_VOLATILITY_EXPANSION else 0.0
            ]
            
            all_features = features + signal_features + regime_features
            
            # Проверяем на валидность
            if any(np.isnan(f) or np.isinf(f) for f in all_features):
                return None
            
            return np.array(all_features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки ML фичей: {e}")
            return None
    
    def _rank_signals(self, signals: List[AdvancedSignal]) -> List[AdvancedSignal]:
        """Ранжирование сигналов по качеству"""
        try:
            # Функция оценки качества сигнала
            def signal_score(signal: AdvancedSignal) -> float:
                score = 0.0
                
                # Базовая уверенность (40%)
                score += signal.confidence_score * 0.4
                
                # ML предсказание если есть (25%)
                if signal.ml_probability:
                    score += signal.ml_probability * 0.25
                else:
                    score += signal.confidence_score * 0.25  # Fallback
                
                # Risk/Reward ratio (15%)
                rr_score = min(signal.risk_reward_ratio / 3.0, 1.0)  # Нормализуем к 1.0
                score += rr_score * 0.15
                
                # Объемное подтверждение (10%)
                volume_score = min(signal.volume_confirmation / 3.0, 1.0)
                score += volume_score * 0.10
                
                # Качество сигнала (10%)
                quality_scores = {
                    SignalQuality.INSTITUTIONAL: 1.0,
                    SignalQuality.PROFESSIONAL: 0.9,
                    SignalQuality.RETAIL_PLUS: 0.7,
                    SignalQuality.RETAIL: 0.5,
                    SignalQuality.NOISE: 0.0
                }
                score += quality_scores.get(signal.signal_quality, 0.5) * 0.10
                
                return score
            
            # Сортируем по убыванию оценки
            return sorted(signals, key=signal_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Ошибка ранжирования сигналов: {e}")
            return signals
    
    def _broadcast_signals(self, signals: List[AdvancedSignal]) -> int:
        """Рассылка сигналов подписчикам"""
        try:
            if not signals:
                return 0
            
            sent_count = 0
            
            # Получаем всех активных подписчиков
            active_subscribers = self._get_active_subscribers()
            
            if not active_subscribers:
                self.logger.info("Нет активных подписчиков")
                return 0
            
            # Группируем подписчиков по тарифам для оптимизации
            subscribers_by_tier = self._group_subscribers_by_tier(active_subscribers)
            
            # Отправляем сигналы по тарифам
            for tier, subscribers in subscribers_by_tier.items():
                if not subscribers:
                    continue
                
                # Определяем количество сигналов для тарифа
                signals_for_tier = self._get_signals_for_tier(signals, tier)
                
                if not signals_for_tier:
                    continue
                
                # Отправляем каждому подписчику
                for subscriber_data in subscribers:
                    user_id = subscriber_data['user_id']
                    
                    try:
                        # Проверяем лимиты пользователя
                        can_receive, reason, tariff = self.subscription_service.can_receive_signals(user_id)
                        if not can_receive:
                            continue
                        
                        # Отправляем сигналы
                        for signal in signals_for_tier:
                            if self._send_signal_to_user(user_id, signal):
                                sent_count += 1
                                # Обновляем счетчик использования
                                self.subscription_service.increment_signals_usage(user_id)
                            
                            # Небольшая пауза между отправками
                            import time
                            time.sleep(0.1)
                    
                    except Exception as e:
                        self.logger.error(f"Ошибка отправки сигнала пользователю {user_id}: {e}")
                        continue
            
            self.stats['signals_sent'] += sent_count
            
            return sent_count
            
        except Exception as e:
            self.logger.error(f"Ошибка рассылки сигналов: {e}")
            return 0
    
    def _get_active_subscribers(self) -> List[Dict]:
        """Получение списка активных подписчиков"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.user_id, s.subscription_type, s.signals_used_today, 
                           s.last_signal_date, u.username, u.first_name
                    FROM subscriptions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.status = 'active' 
                    AND u.is_banned = FALSE
                    AND (s.expires_at IS NULL OR s.expires_at > NOW())
                """)
                
                subscribers = []
                for row in cursor.fetchall():
                    subscribers.append({
                        'user_id': row[0],
                        'subscription_type': row[1],
                        'signals_used_today': row[2],
                        'last_signal_date': row[3],
                        'username': row[4],
                        'first_name': row[5]
                    })
                
                return subscribers
                
        except Exception as e:
            self.logger.error(f"Ошибка получения подписчиков: {e}")
            return []
    
    def _group_subscribers_by_tier(self, subscribers: List[Dict]) -> Dict[str, List[Dict]]:
        """Группировка подписчиков по тарифам"""
        grouped = {
            'trial': [],
            'basic': [],
            'premium': [],
            'vip': []
        }
        
        for subscriber in subscribers:
            tier = subscriber['subscription_type']
            if tier in grouped:
                grouped[tier].append(subscriber)
        
        return grouped
    
    def _get_signals_for_tier(self, signals: List[AdvancedSignal], tier: str) -> List[AdvancedSignal]:
        """Получение сигналов для конкретного тарифа"""
        if tier == 'trial':
            # Trial получает только лучшие сигналы
            return [s for s in signals[:2] if s.signal_quality.value in ['institutional', 'professional']]
        elif tier == 'basic':
            # Basic получает качественные сигналы
            return [s for s in signals[:5] if s.signal_quality.value in ['institutional', 'professional', 'retail_plus']]
        elif tier == 'premium':
            # Premium получает больше сигналов
            return signals[:10]
        elif tier == 'vip':
            # VIP получает все сигналы
            return signals
        
        return []
    
    def _send_signal_to_user(self, user_id: int, signal: AdvancedSignal) -> bool:
        """Отправка сигнала конкретному пользователю"""
        try:
            # Форматируем сообщение
            message = self._format_signal_message(signal)
            
            # Создаем график если нужно
            chart_path = None
            try:
                chart_path = self._create_signal_chart(signal)
            except Exception as e:
                self.logger.warning(f"Ошибка создания графика для {signal.symbol}: {e}")
            
            # Отправляем сообщение
            if chart_path and os.path.exists(chart_path):
                with open(chart_path, 'rb') as chart:
                    self.bot.send_photo(user_id, chart, caption=message, parse_mode='HTML')
            else:
                self.bot.send_message(user_id, message, parse_mode='HTML')
            
            # Сохраняем сигнал в базу данных
            self._save_signal_to_db(signal, user_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки сигнала пользователю {user_id}: {e}")
            return False
    
    def _format_signal_message(self, signal: AdvancedSignal) -> str:
        """Форматирование сообщения с сигналом"""
        direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
        quality_emoji = {
            'institutional': '⭐⭐⭐',
            'professional': '⭐⭐',
            'retail_plus': '⭐',
            'retail': '',
            'noise': ''
        }.get(signal.signal_quality.value, '')
        
        message = f"""
{direction_emoji} <b>{signal.direction} {signal.symbol}</b> {quality_emoji}

💰 <b>Вход:</b> ${signal.entry_price:.4f}
🛑 <b>Стоп-лосс:</b> ${signal.stop_loss:.4f}
🎯 <b>Тейк-профит:</b> ${signal.take_profit:.4f}

📊 <b>R:R соотношение:</b> 1:{signal.risk_reward_ratio:.1f}
📈 <b>Уверенность:</b> {signal.confidence_score:.1%}
        """
        
        if signal.ml_probability:
            message += f"🤖 <b>ML оценка:</b> {signal.ml_probability:.1%}\n"
        
        message += f"""
⚡ <b>Режим рынка:</b> {signal.market_regime.value.replace('_', ' ').title()}
📊 <b>Волатильность:</b> {signal.volatility_percentile:.0%} перцентиль
🔊 <b>Объем:</b> {signal.volume_confirmation:.1f}x средний

🕒 <b>Время:</b> {format_datetime(signal.timestamp)}
⏱️ <b>Ожидаемое удержание:</b> {signal.expected_holding_time // 60}ч {signal.expected_holding_time % 60}м

💡 <i>Качество сигнала: {signal.signal_quality.value.replace('_', ' ').title()}</i>
        """
        
        return message.strip()
    
    def _create_signal_chart(self, signal: AdvancedSignal) -> Optional[str]:
        """Создание графика для сигнала"""
        try:
            # Получаем данные для графика
            df = self.market_analyzer.get_market_data(signal.symbol)
            if df is None:
                return None
            
            # Здесь должна быть логика создания графика
            # Пока возвращаем None - график будет реализован отдельно
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка создания графика: {e}")
            return None
    
    def _save_signal_to_db(self, signal: AdvancedSignal, user_id: int):
        """Сохранение сигнала в базу данных"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trading_signals (
                        symbol, `interval`, is_long, entry_price, stop_loss, take_profit,
                        risk_reward_ratio, confidence_score, ml_prediction, technical_score,
                        indicators_fired, outcome, market_regime, volatility_level,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    signal.symbol,
                    self.config.DEFAULT_INTERVAL,
                    signal.direction == "LONG",
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.risk_reward_ratio,
                    signal.confidence_score,
                    signal.ml_probability,
                    signal.momentum_score,
                    str(signal.divergence_signals),
                    'pending',
                    signal.market_regime.value,
                    f"{signal.volatility_percentile:.0%}",
                    signal.timestamp
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Ошибка сохранения сигнала в БД: {e}")
    
    def analyze_single_symbol(self, symbol: str) -> Optional[str]:
        """Анализ одного символа по запросу пользователя"""
        try:
            self.logger.info(f"Ручной анализ символа: {symbol}")
            
            # Получаем данные
            df = self.market_analyzer.get_market_data(symbol)
            if df is None:
                return f"❌ Не удалось получить данные для {symbol}"
            
            if len(df) < 200:
                return f"⚠️ Недостаточно исторических данных для {symbol}"
            
            # Рассчитываем индикаторы
            df_with_indicators = self.market_analyzer.calculate_advanced_indicators(df)
            if df_with_indicators.empty:
                return f"❌ Ошибка расчета индикаторов для {symbol}"
            
            # Анализируем текущее состояние
            latest = df_with_indicators.iloc[-1]
            market_regime = self.market_analyzer.analyze_market_regime(df_with_indicators)
            
            # Пытаемся сгенерировать сигнал
            signal = self.market_analyzer.generate_advanced_signal(df_with_indicators)
            
            # Форматируем ответ
            analysis = f"""
📊 <b>Анализ {symbol}</b>

💰 <b>Текущая цена:</b> ${latest['close']:.4f}
📈 <b>Режим рынка:</b> {market_regime.value.replace('_', ' ').title()}

<b>📈 Технические индикаторы:</b>
• RSI(14): {latest.get('rsi_14', 0):.1f}
• MACD: {latest.get('macd', 0):.4f}
• ADX: {latest.get('adx', 0):.1f}
• ATR: {latest.get('atr_pct', 0):.2f}%

<b>🔊 Объем:</b> {latest.get('volume_ratio', 1):.1f}x средний
<b>📊 Волатильность:</b> {df_with_indicators['atr_pct'].rolling(100).rank(pct=True).iloc[-1]:.0%} перцентиль
            """
            
            if signal:
                analysis += f"""

🎯 <b>ТОРГОВАЯ ВОЗМОЖНОСТЬ</b>
{self._format_signal_message(signal)}
                """
            else:
                analysis += f"""

⏳ <b>Текущий статус:</b> Нет качественных сигналов
💡 <i>Ожидайте лучшую точку входа</i>
                """
            
            return analysis.strip()
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа {symbol}: {e}")
            return f"❌ Произошла ошибка при анализе {symbol}"
    
    def update_signal_outcomes(self):
        """Обновление результатов торговых сигналов"""
        try:
            # Получаем активные сигналы
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, symbol, is_long, entry_price, stop_loss, take_profit, created_at
                    FROM trading_signals 
                    WHERE outcome = 'pending' 
                    AND created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
                """)
                
                pending_signals = cursor.fetchall()
                
                if not pending_signals:
                    return
                
                updated_count = 0
                
                for signal_data in pending_signals:
                    signal_id, symbol, is_long, entry_price, stop_loss, take_profit, created_at = signal_data
                    
                    try:
                        # Получаем текущие данные для проверки результата
                        current_df = self.market_analyzer.get_market_data(symbol, limit=100)
                        if current_df is None:
                            continue
                        
                        # Проверяем результат сигнала
                        outcome, exit_price, profit_pct = self._check_signal_outcome(
                            current_df, is_long, entry_price, stop_loss, take_profit, created_at
                        )
                        
                        if outcome != 'pending':
                            # Обновляем результат в БД
                            duration_hours = (datetime.now() - created_at).total_seconds() / 3600
                            
                            cursor.execute("""
                                UPDATE trading_signals 
                                SET outcome = %s, exit_price = %s, profit_loss_pct = %s, 
                                    duration_hours = %s, closed_at = NOW()
                                WHERE id = %s
                            """, (outcome, exit_price, profit_pct, int(duration_hours), signal_id))
                            
                            updated_count += 1
                    
                    except Exception as e:
                        self.logger.error(f"Ошибка обновления сигнала {signal_id}: {e}")
                        continue
                
                conn.commit()
                
                if updated_count > 0:
                    self.logger.info(f"Обновлено {updated_count} сигналов")
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления результатов сигналов: {e}")
    
    def _check_signal_outcome(self, df: pd.DataFrame, is_long: bool, entry_price: float, 
                             stop_loss: float, take_profit: float, created_at: datetime) -> Tuple[str, Optional[float], Optional[float]]:
        """Проверка результата торгового сигнала"""
        try:
            # Фильтруем данные после создания сигнала
            df_after_signal = df[df.index > created_at]
            
            if df_after_signal.empty:
                return 'pending', None, None
            
            if is_long:
                # LONG позиция
                # Проверяем Stop Loss
                if (df_after_signal['low'] <= stop_loss).any():
                    exit_price = stop_loss
                    profit_pct = (exit_price - entry_price) / entry_price * 100
                    return 'stop_loss', exit_price, profit_pct
                
                # Проверяем Take Profit
                if (df_after_signal['high'] >= take_profit).any():
                    exit_price = take_profit
                    profit_pct = (exit_price - entry_price) / entry_price * 100
                    return 'take_profit', exit_price, profit_pct
            else:
                # SHORT позиция
                # Проверяем Stop Loss
                if (df_after_signal['high'] >= stop_loss).any():
                    exit_price = stop_loss
                    profit_pct = (entry_price - exit_price) / entry_price * 100
                    return 'stop_loss', exit_price, profit_pct
                
                # Проверяем Take Profit
                if (df_after_signal['low'] <= take_profit).any():
                    exit_price = take_profit
                    profit_pct = (entry_price - exit_price) / entry_price * 100
                    return 'take_profit', exit_price, profit_pct
            
            # Проверяем срок действия (7 дней)
            if (datetime.now() - created_at).days >= 7:
                current_price = df_after_signal['close'].iloc[-1]
                if is_long:
                    profit_pct = (current_price - entry_price) / entry_price * 100
                else:
                    profit_pct = (entry_price - current_price) / entry_price * 100
                
                return 'expired', current_price, profit_pct
            
            return 'pending', None, None
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки результата сигнала: {e}")
            return 'pending', None, None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности сигналов"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Общая статистика
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN outcome = 'take_profit' THEN 1 END) as tp_count,
                        COUNT(CASE WHEN outcome = 'stop_loss' THEN 1 END) as sl_count,
                        AVG(CASE WHEN outcome IN ('take_profit', 'stop_loss') THEN profit_loss_pct END) as avg_return,
                        AVG(CASE WHEN outcome = 'take_profit' THEN profit_loss_pct END) as avg_winner,
                        AVG(CASE WHEN outcome = 'stop_loss' THEN profit_loss_pct END) as avg_loser,
                        AVG(duration_hours) as avg_duration
                    FROM trading_signals 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    AND outcome != 'pending'
                """)
                
                stats = cursor.fetchone()
                if not stats:
                    return {}
                
                total, tp_count, sl_count, avg_return, avg_winner, avg_loser, avg_duration = stats
                
                win_rate = (tp_count / (tp_count + sl_count) * 100) if (tp_count + sl_count) > 0 else 0
                
                # Статистика по качеству сигналов
                cursor.execute("""
                    SELECT 
                        confidence_score,
                        COUNT(CASE WHEN outcome = 'take_profit' THEN 1 END) as wins,
                        COUNT(CASE WHEN outcome = 'stop_loss' THEN 1 END) as losses
                    FROM trading_signals 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    AND outcome IN ('take_profit', 'stop_loss')
                    GROUP BY ROUND(confidence_score, 1)
                    ORDER BY confidence_score DESC
                """)
                
                quality_stats = cursor.fetchall()
                
                return {
                    'total_signals': total or 0,
                    'win_rate': round(win_rate, 1),
                    'average_return': round(avg_return or 0, 2),
                    'average_winner': round(avg_winner or 0, 2),
                    'average_loser': round(avg_loser or 0, 2),
                    'average_duration_hours': round(avg_duration or 0, 1),
                    'tp_count': tp_count or 0,
                    'sl_count': sl_count or 0,
                    'quality_breakdown': quality_stats,
                    'service_stats': self.stats.copy()
                }
                
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики: {e}")
            return {}
    
    def retrain_ml_model(self, symbols: List[str] = None, days_back: int = 365):
        """Переобучение ML модели на свежих данных"""
        try:
            self.logger.info("Начинаем переобучение ML модели")
            
            # Используем топ символы если не указаны
            if not symbols:
                symbols = self.get_futures_symbols()[:30]  # Топ 30
            
            # Собираем данные для обучения
            market_data = {}
            for symbol in symbols:
                self.logger.info(f"Получение данных для обучения: {symbol}")
                
                # Получаем больше исторических данных для обучения
                df = self.market_analyzer.get_market_data(symbol, limit=days_back * 24 * 4)  # 15min bars
                if df is not None and len(df) >= 500:
                    market_data[symbol] = df
            
            if len(market_data) < 5:
                self.logger.error("Недостаточно данных для переобучения")
                return False
            
            # Подготавливаем данные
            X, y = self.ml_trainer.prepare_training_data(market_data)
            
            if len(X) < 1000:
                self.logger.error("Недостаточно образцов для обучения")
                return False
            
            # Обучаем новую модель
            model_result = self.ml_trainer.train_ensemble_model(X, y)
            
            # Сохраняем модель
            model_path = self.config.ML_PIPELINE_PATH
            self.ml_trainer.save_model(model_result, model_path)
            
            # Перезагружаем модель
            self._load_ml_model()
            
            self.logger.info(f"ML модель переобучена. Validation Score: {model_result['validation_score']:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка переобучения ML модели: {e}")
            return False


def create_signal_service(config: Config, db_manager: DatabaseManager, bot) -> SignalService:
    """Фабричная функция для создания сервиса сигналов"""
    return SignalService(config, db_manager, bot)
        