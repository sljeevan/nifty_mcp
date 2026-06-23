import os
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List, Optional

# Add paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

class RegimeClassifier:
    def __init__(self):
        self.model_path = os.path.join(BASE_DIR, "scratch", "regime_lgbm.model")
        self.meta_path = os.path.join(BASE_DIR, "scratch", "regime_meta.json")
        self.model: Optional[lgb.Booster] = None
        self.decision_threshold = 0.50
        
        self.feature_cols = [
            'return_1h', 'return_3h', 'return_5h', 
            'volatility_5h', 'volatility_10h', 
            'atr_ratio', 'rsi_14', 
            'bb_width', 'bb_percent', 
            'macd_hist_ratio'
        ]
        self.load_model_and_meta()
        
    def load_model_and_meta(self):
        if os.path.exists(self.model_path):
            try:
                self.model = lgb.Booster(model_file=self.model_path)
                print(f"[ML Classifier] Successfully loaded LightGBM binary model from {self.model_path}")
            except Exception as e:
                print(f"[ML Classifier] Error loading LightGBM model: {e}")
        else:
            print(f"[ML Classifier] Warning: Model file not found at {self.model_path}. ML classification disabled.")

        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                    self.decision_threshold = meta.get("decision_threshold", 0.50)
                    print(f"[ML Classifier] Loaded Decision Threshold: {self.decision_threshold:.2f}")
            except Exception as e:
                print(f"[ML Classifier] Error loading metadata: {e}")

    def compute_features(self, price_history: List[Dict[str, float]]) -> Optional[pd.DataFrame]:
        """
        Calculate model features from raw price history.
        Expects price_history to contain dicts with 'open', 'high', 'low', 'close', 'timestamp'.
        Requires at least 35 bars for stable EMA/SMA indicator calculations.
        """
        if len(price_history) < 35:
            return None
            
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(price_history)
        
        # Calculate returns
        df['return_1h'] = df['close'].pct_change()
        df['return_3h'] = df['close'].pct_change(3)
        df['return_5h'] = df['close'].pct_change(5)
        
        # Volatility
        df['volatility_5h'] = df['return_1h'].rolling(5).std()
        df['volatility_10h'] = df['return_1h'].rolling(10).std()
        
        # ATR
        high_low = df['high'] - df['low']
        high_cp = (df['high'] - df['close'].shift()).abs()
        low_cp = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma_20'] + 1e-9)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
        
        # Normalized indicators
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist_ratio'] = (macd - macd_signal) / df['close']
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        return df[self.feature_cols].tail(1)

    def predict_regime(self, price_history: List[Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Predict market regime: 0 (CHOPPY), 1 (TRENDING)
        """
        if self.model is None:
            return None
            
        features_df = self.compute_features(price_history)
        if features_df is None or features_df.isnull().values.any():
            return None
            
        # Get binary prediction probability of TRENDING
        prob_trending = float(self.model.predict(features_df)[0])
        
        pred_class = 1 if prob_trending >= self.decision_threshold else 0
        regime_label = "TRENDING" if pred_class == 1 else "CHOPPY"
        
        return {
            "class": pred_class,
            "label": regime_label,
            "probabilities": {
                "CHOPPY": 1.0 - prob_trending,
                "TRENDING": prob_trending
            }
        }
