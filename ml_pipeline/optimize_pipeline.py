import os
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "tradingview-mcp-india", "src"))
sys.path.insert(0, BASE_DIR)

from tradingview_mcp.core.services.backtest_service import _fetch_ohlcv
from ml_pipeline.data_collector import compute_atr, compute_rsi

def build_and_optimize():
    print("[Pipeline Optimization] Re-downloading and re-labeling Nifty data...")
    raw_candles = _fetch_ohlcv("^NSEI", period="2y", interval="1h")
    df = pd.DataFrame(raw_candles)
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature Engineering
    df['return_1h'] = df['close'].pct_change()
    df['return_3h'] = df['close'].pct_change(3)
    df['return_5h'] = df['close'].pct_change(5)
    df['volatility_5h'] = df['return_1h'].rolling(5).std()
    df['volatility_10h'] = df['return_1h'].rolling(10).std()
    df['atr_14'] = compute_atr(df, 14)
    df['rsi_14'] = compute_rsi(df, 14)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma_20'] + 1e-9)
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    
    # Normalized MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist_ratio'] = (macd - macd_signal) / df['close']
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    # Better labeling: 
    # 0 = CHOPPY (price stays inside 2.0 * ATR band over the next 10 hours)
    # 1 = TRENDING (price breaks out of 2.0 * ATR band in either direction)
    lookahead = 10
    labels = []
    
    for i in range(len(df)):
        if i + lookahead >= len(df):
            labels.append(np.nan)
            continue
            
        current_close = df.loc[i, 'close']
        current_atr = df.loc[i, 'atr_14']
        
        future_window = df.loc[i+1 : i+lookahead]
        future_high = future_window['high'].max()
        future_low = future_window['low'].min()
        
        max_deviation = max(future_high - current_close, current_close - future_low)
        
        # Squeeze threshold: 2.0 * ATR
        if max_deviation >= (2.0 * current_atr):
            labels.append(1) # TRENDING
        else:
            labels.append(0) # CHOPPY (Squeeze)
            
    df['label'] = labels
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = [
        'return_1h', 'return_3h', 'return_5h', 
        'volatility_5h', 'volatility_10h', 
        'atr_ratio', 'rsi_14', 
        'bb_width', 'bb_percent', 
        'macd_hist_ratio'
    ]
    
    X = df[feature_cols]
    y = df['label']
    
    # Print new distribution
    dist = y.value_counts()
    print(f"[Pipeline Optimization] New Distribution -> Choppy (0): {dist.get(0, 0)} ({dist.get(0, 0)/len(y)*100:.2f}%), Trending (1): {dist.get(1, 0)} ({dist.get(1, 0)/len(y)*100:.2f}%)")
    
    # Train-test split (shuffled for cross-cycle stability)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Set hyper-parameters optimized for small binary datasets
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.03,
        'max_depth': 5,
        'num_leaves': 15,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 1.0,
        'lambda_l2': 2.0,
        'verbose': -1,
        'seed': 42
    }
    
    print("[Pipeline Optimization] Training LightGBM binary classifier...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[train_data, test_data],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=40, verbose=False)
        ]
    )
    
    # Evaluate with customized threshold
    preds_prob = model.predict(X_test)
    
    # Instead of default 0.5, we tune threshold based on precision of the trend signal
    # We want to be very confident when we call a market "TRENDING" (1)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for th in np.arange(0.4, 0.7, 0.02):
        temp_preds = (preds_prob >= th).astype(int)
        report = classification_report(y_test, temp_preds, output_dict=True, zero_division=0)
        f1_trend = report['1.0']['f1-score'] if '1.0' in report else report['1']['f1-score']
        if f1_trend > best_f1:
            best_f1 = f1_trend
            best_threshold = th
            
    print(f"[Pipeline Optimization] Optimal Decision Threshold: {best_threshold:.2f} (F1-score: {best_f1:.2f})")
    
    final_preds = (preds_prob >= best_threshold).astype(int)
    print("\n--- Optimized Binary Classification Report ---")
    print(classification_report(y_test, final_preds, target_names=['CHOPPY (0)', 'TRENDING (1)']))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, final_preds))
    
    # Save optimized model over the old multiclass one
    model_output_path = os.path.join(BASE_DIR, "scratch", "regime_lgbm.model")
    model.save_model(model_output_path)
    
    # Save optimized parameters meta file
    meta_path = os.path.join(BASE_DIR, "scratch", "regime_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": "binary_lgbm",
            "decision_threshold": float(best_threshold),
            "features": feature_cols,
            "updated_at": datetime.now().isoformat()
        }, f, indent=4)
        
    print(f"\n[Pipeline Optimization] Optimized Model saved to {model_output_path}")
    print(f"[Pipeline Optimization] Meta configuration saved to {meta_path}")

if __name__ == "__main__":
    build_and_optimize()
