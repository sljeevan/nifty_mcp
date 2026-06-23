import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    input_path = os.path.join(BASE_DIR, "scratch", "nifty_features.csv")
    if not os.path.exists(input_path):
        print(f"[Trainer] Error: Feature store file not found: {input_path}")
        sys.exit(1)
        
    print(f"[Trainer] Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Sort chronologically to ensure strict out-of-sample temporal split
    df = df.sort_values('date').reset_index(drop=True)
    
    # Select features
    feature_cols = [
        'return_1h', 'return_3h', 'return_5h', 
        'volatility_5h', 'volatility_10h', 
        'atr_14', 'rsi_14', 
        'bb_width', 'bb_percent', 
        'dist_ema20', 'dist_ema50', 'dist_ema200', 
        'macd', 'macd_signal', 'macd_hist'
    ]
    
    X = df[feature_cols]
    y = df['label']
    
    # Chronological Split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"[Trainer] Training set size: {X_train.shape[0]} rows")
    print(f"[Trainer] Test set size: {X_test.shape[0]} rows")
    
    # Configure LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Multi-class parameters (0: CHOPPY, 1: BULLISH, 2: BEARISH)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    print("[Trainer] Training LightGBM Classifier...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=150,
        valid_sets=[train_data, test_data],
        callbacks=[lgb.log_evaluation(period=25)]
    )
    
    # Evaluate model
    print("\n[Trainer] Evaluating model on out-of-sample test set...")
    preds_prob = model.predict(X_test)
    preds = np.argmax(preds_prob, axis=1)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds, target_names=['CHOPPY (0)', 'BULLISH (1)', 'BEARISH (2)']))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, preds))
    
    # Save Model
    model_output_path = os.path.join(BASE_DIR, "scratch", "regime_lgbm.model")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save_model(model_output_path)
    print(f"\n[Trainer] Model successfully saved to {model_output_path}")

if __name__ == "__main__":
    train_model()
