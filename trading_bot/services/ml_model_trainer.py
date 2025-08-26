import logging
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

from config.settings import Config
from utils.helpers import Timer


@dataclass
class TrainingResults:
    """Результаты обучения модели"""
    model: Any
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_names: List[str]
    feature_importance: Dict[str, float]
    training_time: float
    samples_count: int


class MLModelTrainer:
    """Основной класс для обучения ML модели"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(f_classif, k=25)
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка фич для ML модели"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Базовые технические индикаторы
            features['rsi_7'] = self._calculate_rsi(df['close'], 7)
            features['rsi_14'] = self._calculate_rsi(df['close'], 14)
            features['rsi_21'] = self._calculate_rsi(df['close'], 21)
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            features['bb_upper'] = sma20 + (std20 * 2)
            features['bb_lower'] = sma20 - (std20 * 2)
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma20 * 100
            
            # ATR и волатильность
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features['atr'] = true_range.rolling(14).mean()
            features['atr_pct'] = features['atr'] / df['close'] * 100
            
            # Объемные индикаторы
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            
            # OBV
            obv = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, 
                   np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
            features['obv'] = obv
            features['obv_sma'] = obv.rolling(20).mean()
            
            # Stochastic
            low_min = df['low'].rolling(14).min()
            high_max = df['high'].rolling(14).max()
            features['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()
            
            # Williams %R
            features['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
            
            # Price momentum
            for period in [5, 10, 20]:
                features[f'price_change_{period}'] = df['close'].pct_change(period) * 100
                features[f'volume_change_{period}'] = df['volume'].pct_change(period) * 100
            
            # Moving averages
            for period in [10, 20, 50]:
                ma = df['close'].rolling(period).mean()
                features[f'price_vs_ma_{period}'] = (df['close'] - ma) / ma * 100
            
            # Trend indicators
            features['price_trend_5'] = (df['close'] > df['close'].shift(5)).astype(int)
            features['price_trend_10'] = (df['close'] > df['close'].shift(10)).astype(int)
            features['volume_trend'] = (df['volume'] > df['volume'].shift(5)).astype(int)
            
            # Market structure
            features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            
            # Volatility features
            returns = df['close'].pct_change()
            features['volatility_5'] = returns.rolling(5).std() * 100
            features['volatility_20'] = returns.rolling(20).std() * 100
            
            return features.dropna()
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки фич: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_labels(self, df: pd.DataFrame, forward_periods: int = 10) -> pd.Series:
        """Создание меток для supervised learning"""
        try:
            labels = pd.Series(0, index=df.index)  # 0 = Hold
            
            for i in range(len(df) - forward_periods):
                current_price = df['close'].iloc[i]
                future_prices = df['close'].iloc[i+1:i+1+forward_periods]
                future_highs = df['high'].iloc[i+1:i+1+forward_periods]
                future_lows = df['low'].iloc[i+1:i+1+forward_periods]
                
                if len(future_prices) == 0:
                    continue
                
                # Максимальная прибыль и убыток
                max_profit = (future_highs.max() - current_price) / current_price
                max_loss = (current_price - future_lows.min()) / current_price
                final_return = (future_prices.iloc[-1] - current_price) / current_price
                
                # Создаем метки на основе профитабельности и риска
                if max_profit > 0.02 and final_return > 0.01 and max_profit > max_loss * 1.5:
                    labels.iloc[i] = 1  # Buy
                elif max_loss > 0.02 and final_return < -0.01 and max_loss > max_profit * 1.5:
                    labels.iloc[i] = -1  # Sell
                # Остальное остается Hold (0)
            
            # Убираем последние периоды
            labels.iloc[-forward_periods:] = np.nan
            
            return labels.dropna()
            
        except Exception as e:
            self.logger.error(f"Ошибка создания меток: {e}")
            return pd.Series(0, index=df.index)
    
    def prepare_training_data(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка всех данных для обучения"""
        try:
            all_features = []
            all_labels = []
            
            for symbol, df in market_data.items():
                self.logger.info(f"Подготовка данных для {symbol}")
                
                # Создаем фичи
                features = self.prepare_features(df)
                if features.empty:
                    continue
                
                # Создаем метки
                labels = self.create_labels(df)
                
                # Синхронизируем по индексу
                common_index = features.index.intersection(labels.index)
                if len(common_index) < 100:  # Минимальный размер
                    continue
                
                features_aligned = features.loc[common_index]
                labels_aligned = labels.loc[common_index]
                
                all_features.append(features_aligned)
                all_labels.append(labels_aligned)
            
            if not all_features:
                raise ValueError("Нет подходящих данных для обучения")
            
            # Объединяем все данные
            X = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_labels, ignore_index=True)
            
            # Очищаем данные
            X, y = self._clean_data(X, y)
            
            self.logger.info(f"Подготовлено {len(X)} образцов с {len(X.columns)} фичами")
            class_counts = y.value_counts().sort_index()
            self.logger.info(f"Распределение классов: {dict(class_counts)}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки данных: {e}")
            raise
    
    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Очистка данных"""
        try:
            # Удаляем строки с NaN
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X.loc[mask].copy()
            y_clean = y.loc[mask].copy()
            
            # Заменяем inf на NaN и заполняем медианой
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            X_clean = X_clean.fillna(X_clean.median())
            
            # Удаляем константные колонки
            constant_cols = X_clean.columns[X_clean.nunique() <= 1]
            if len(constant_cols) > 0:
                X_clean = X_clean.drop(columns=constant_cols)
                self.logger.info(f"Удалено {len(constant_cols)} константных колонок")
            
            return X_clean, y_clean
            
        except Exception as e:
            self.logger.error(f"Ошибка очистки данных: {e}")
            return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> TrainingResults:
        """Обучение модели"""
        try:
            timer = Timer().start()
            
            # Разделяем данные по времени
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Балансируем классы если нужно
            if len(y_train.value_counts()) > 1:
                X_train, y_train = self._balance_classes(X_train, y_train)
            
            # Отбираем лучшие фичи
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            X_test_selected = self.feature_selector.transform(X_test)
            selected_features = X_train.columns[self.feature_selector.get_support()].tolist()
            
            # Масштабируем данные
            X_train_scaled = self.scaler.fit_transform(X_train_selected)
            X_test_scaled = self.scaler.transform(X_test_selected)
            
            # Создаем ансамбль моделей
            models = {
                'xgb': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='mlogloss'
                ),
                'lgb': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                ),
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                'lr': LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42
                )
            }
            
            # Обучаем отдельные модели
            trained_models = []
            model_scores = []
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    score = model.score(X_test_scaled, y_test)
                    trained_models.append((name, model))
                    model_scores.append(score)
                    self.logger.info(f"{name}: accuracy = {score:.4f}")
                except Exception as e:
                    self.logger.warning(f"Ошибка обучения {name}: {e}")
            
            if not trained_models:
                raise ValueError("Не удалось обучить ни одну модель")
            
            # Создаем ансамбль
            if len(trained_models) > 1:
                ensemble = VotingClassifier(
                    estimators=trained_models,
                    voting='soft',
                    weights=model_scores
                )
                ensemble.fit(X_train_scaled, y_train)
                final_model = ensemble
            else:
                final_model = trained_models[0][1]
            
            # Оцениваем финальную модель
            y_pred = final_model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            
            # Важность фич
            feature_importance = self._get_feature_importance(final_model, selected_features)
            
            timer.stop()
            
            # Создаем финальную модель pipeline
            final_pipeline = {
                'model': final_model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'selected_features': selected_features
            }
            
            results = TrainingResults(
                model=final_pipeline,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                feature_names=selected_features,
                feature_importance=feature_importance,
                training_time=timer.elapsed(),
                samples_count=len(X_train)
            )
            
            self.logger.info(f"Обучение завершено за {timer.elapsed_str()}")
            self.logger.info(f"Финальная точность: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            raise
    
    def _balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Балансировка классов"""
        try:
            # Используем SMOTE для oversampling + undersampling
            smote = SMOTE(random_state=42, k_neighbors=3)
            under = RandomUnderSampler(random_state=42)
            
            X_smote, y_smote = smote.fit_resample(X, y)
            X_balanced, y_balanced = under.fit_resample(X_smote, y_smote)
            
            self.logger.info(f"Балансировка: {len(X)} -> {len(X_balanced)} образцов")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.logger.warning(f"Ошибка балансировки классов: {e}")
            return X.values, y.values
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Получение важности фич"""
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                # Для tree-based моделей
                importances = model.feature_importances_
            elif hasattr(model, 'estimators_'):
                # Для ансамблей
                importances = np.zeros(len(feature_names))
                for estimator in model.estimators_:
                    if hasattr(estimator[1], 'feature_importances_'):
                        importances += estimator[1].feature_importances_
                importances /= len(model.estimators_)
            elif hasattr(model, 'coef_'):
                # Для линейных моделей
                importances = np.abs(model.coef_[0])
            else:
                importances = np.ones(len(feature_names))
            
            for i, feature in enumerate(feature_names):
                importance_dict[feature] = float(importances[i])
            
            # Сортируем по важности
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            self.logger.error(f"Ошибка получения важности фич: {e}")
            return {}
    
    def predict(self, df: pd.DataFrame, model_pipeline: Dict) -> Optional[float]:
        """Предсказание для новых данных"""
        try:
            # Подготавливаем фичи
            features = self.prepare_features(df)
            if features.empty:
                return None
            
            # Берем последнюю строку
            latest_features = features.iloc[-1:].copy()
            
            # Применяем feature selection
            feature_selector = model_pipeline['feature_selector']
            features_selected = feature_selector.transform(latest_features)
            
            # Масштабируем
            scaler = model_pipeline['scaler']
            features_scaled = scaler.transform(features_selected)
            
            # Предсказываем
            model = model_pipeline['model']
            
            if hasattr(model, 'predict_proba'):
                # Возвращаем вероятность положительного класса
                probabilities = model.predict_proba(features_scaled)[0]
                if len(probabilities) > 2:  # Многоклассовая классификация
                    # Для классов [-1, 0, 1] берем вероятность класса 1
                    positive_class_idx = list(model.classes_).index(1) if 1 in model.classes_ else -1
                    return float(probabilities[positive_class_idx]) if positive_class_idx >= 0 else 0.5
                else:  # Бинарная
                    return float(probabilities[1])
            else:
                prediction = model.predict(features_scaled)[0]
                return float(prediction) if prediction > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Ошибка предсказания: {e}")
            return None
    
    def save_model(self, results: TrainingResults, filepath: str):
        """Сохранение модели"""
        try:
            model_data = {
                'model_pipeline': results.model,
                'metrics': {
                    'accuracy': results.accuracy,
                    'precision': results.precision,
                    'recall': results.recall,
                    'f1_score': results.f1_score,
                    'training_time': results.training_time,
                    'samples_count': results.samples_count
                },
                'feature_names': results.feature_names,
                'feature_importance': results.feature_importance,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'forward_periods': 10,
                    'selected_features_count': len(results.feature_names)
                }
            }
            
            # Создаем директорию если не существует
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Сохраняем
            joblib.dump(model_data, filepath, compress=3)
            self.logger.info(f"Модель сохранена: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")
    
    def load_model(self, filepath: str) -> Optional[Dict]:
        """Загрузка модели"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Файл модели не существует: {filepath}")
                return None
            
            model_data = joblib.load(filepath)
            self.logger.info(f"Модель загружена: {filepath}")
            
            # Логируем информацию о модели
            metrics = model_data.get('metrics', {})
            self.logger.info(f"Точность модели: {metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"F1-score: {metrics.get('f1_score', 0):.4f}")
            
            return model_data
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            return None


def create_and_train_model(config: Config, market_data: Dict[str, pd.DataFrame], 
                          save_path: str) -> Optional[TrainingResults]:
    """Создание и обучение модели"""
    try:
        trainer = MLModelTrainer(config)
        
        # Подготавливаем данные
        X, y = trainer.prepare_training_data(market_data)
        
        # Проверяем достаточность данных
        if len(X) < 500:
            logging.error(f"Недостаточно данных для обучения: {len(X)}")
            return None
        
        # Обучаем модель
        results = trainer.train_model(X, y)
        
        # Сохраняем
        if save_path:
            trainer.save_model(results, save_path)
        
        return results
        
    except Exception as e:
        logging.error(f"Ошибка создания модели: {e}")
        return None


def load_trained_model(filepath: str) -> Optional[Dict]:
    """Загрузка обученной модели"""
    trainer = MLModelTrainer(Config())
    return trainer.load_model(filepath)