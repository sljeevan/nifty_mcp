import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_diagnostics():
    input_path = os.path.join(BASE_DIR, "scratch", "nifty_features.csv")
    if not os.path.exists(input_path):
        print(f"Error: Feature store file not found: {input_path}")
        return
        
    df = pd.read_csv(input_path)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Let's normalize non-stationary features
    print("[Diagnostics] Normalizing non-stationary features...")
    df['atr_ratio'] = df['atr_14'] / df['close']
    df['macd_ratio'] = df['macd'] / df['close']
    df['macd_signal_ratio'] = df['macd_signal'] / df['close']
    df['macd_hist_ratio'] = df['macd_hist'] / df['close']
    
    # Purely stationary feature set (prevents temporal regime leakage)
    feature_cols = [
        'return_1h', 'return_3h', 'return_5h', 
        'volatility_5h', 'volatility_10h', 
        'atr_ratio', 'rsi_14', 
        'bb_width', 'bb_percent', 
        'macd_hist_ratio'
    ]
    
    X = df[feature_cols]
    # Map BULLISH (1) and BEARISH (2) to TRENDING (1)
    y = df['label'].replace(2, 1)
    
    print(f"[Diagnostics] Binary Class Distribution -> Choppy (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.2f}%), Trending (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.2f}%)")
    
    # 1. EVALUATE TEMPORAL SPLIT
    print("\n--- RUNNING TEMPORAL SPLIT EVALUATION ---")
    split_idx = int(len(df) * 0.8)
    X_train_t, X_test_t = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_t, y_test_t = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_data_t = lgb.Dataset(X_train_t, label=y_train_t)
    test_data_t = lgb.Dataset(X_test_t, label=y_test_t, reference=train_data_t)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.02,
        'max_depth': 4,
        'num_leaves': 11,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 1.5,
        'lambda_l2': 3.0,
        'verbose': -1,
        'seed': 42
    }
    
    model_t = lgb.train(
        params,
        train_data_t,
        num_boost_round=300,
        valid_sets=[train_data_t, test_data_t],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=30, verbose=False)
        ]
    )
    
    preds_prob_t = model_t.predict(X_test_t)
    preds_t = (preds_prob_t >= 0.5).astype(int)
    print("\n[Temporal Split] Classification Report:")
    print(classification_report(y_test_t, preds_t, target_names=['CHOPPY (0)', 'TRENDING (1)']))
    
    # 2. EVALUATE SHUFFLED SPLIT
    print("\n--- RUNNING SHUFFLED SPLIT EVALUATION ---")
    from sklearn.model_selection import train_test_split
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    train_data_s = lgb.Dataset(X_train_s, label=y_train_s)
    test_data_s = lgb.Dataset(X_test_s, label=y_test_s, reference=train_data_s)
    
    model_s = lgb.train(
        params,
        train_data_s,
        num_boost_round=300,
        valid_sets=[train_data_s, test_data_s],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=30, verbose=False)
        ]
    )
    
    preds_prob_s = model_s.predict(X_test_s)
    preds_s = (preds_prob_s >= 0.5).astype(int)
    print("\n[Shuffled Split] Classification Report:")
    print(classification_report(y_test_s, preds_s, target_names=['CHOPPY (0)', 'TRENDING (1)']))
    
    # Print Feature Importances of Shuffled Model
    print("\n--- Feature Importances (Gain) for Shuffled Model ---")
    importance = model_s.feature_importance(importance_type='gain')
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    for name, val in feat_imp.items():
        print(f"  {name:20s}: {val:.2f}")

if __name__ == "__main__":
    run_diagnostics()
