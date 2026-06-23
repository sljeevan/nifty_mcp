import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add tradingview-mcp src path dynamically to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "tradingview-mcp-india", "src"))
sys.path.insert(0, BASE_DIR)

from tradingview_mcp.core.services.backtest_service import _fetch_ohlcv

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_cp = (df['high'] - df['close'].shift()).abs()
    low_cp = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def collect_data_and_features():
    print("[Pipeline] Downloading Nifty 50 hourly data (2 years)...")
    # Fetch 2y of 1h Nifty Spot data from Yahoo Finance
    raw_candles = _fetch_ohlcv("^NSEI", period="2y", interval="1h")
    print(f"[Pipeline] Download completed. Retrieved {len(raw_candles)} hourly candles.")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(raw_candles)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Feature Engineering
    print("[Pipeline] Computing features...")
    df['return_1h'] = df['close'].pct_change()
    df['return_3h'] = df['close'].pct_change(3)
    df['return_5h'] = df['close'].pct_change(5)
    
    # Volatility
    df['volatility_5h'] = df['return_1h'].rolling(5).std()
    df['volatility_10h'] = df['return_1h'].rolling(10).std()
    
    # Technical Indicators
    df['atr_14'] = compute_atr(df, 14)
    df['rsi_14'] = compute_rsi(df, 14)
    
    # Bollinger Bands
    df['sma_20'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma_20'] + 1e-9)
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    
    # Moving Averages
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Distance to EMAs
    df['dist_ema20'] = (df['close'] - df['ema_20']) / df['close']
    df['dist_ema50'] = (df['close'] - df['ema_50']) / df['close']
    df['dist_ema200'] = (df['close'] - df['ema_200']) / df['close']
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 2. Labeling for supervised classification
    # Look ahead N hours (e.g. 10 hours)
    lookahead = 10
    print(f"[Pipeline] Labeling data using lookahead of {lookahead} hours...")
    
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
        
        up_move = future_high - current_close
        down_move = current_close - future_low
        
        # In option buying, we need a strong one-sided move to beat theta decay.
        # Label 1: Bullish trend (high up-move, limited down-move)
        # Label 2: Bearish trend (high down-move, limited up-move)
        # Label 0: Choppy/Range-bound (neither target hit, or double sweep)
        threshold = 1.3 * current_atr
        
        if up_move > threshold and down_move < (threshold * 0.7):
            labels.append(1) # BULLISH
        elif down_move > threshold and up_move < (threshold * 0.7):
            labels.append(2) # BEARISH
        else:
            labels.append(0) # CHOPPY / RANGE
            
    df['label'] = labels
    
    # Drop rows with NaN features or labels
    df = df.dropna().reset_index(drop=True)
    
    # Save feature store
    output_path = os.path.join(BASE_DIR, "scratch", "nifty_features.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print label distribution
    distribution = df['label'].value_counts()
    print(f"[Pipeline] Feature store created with {len(df)} rows.")
    print(f"[Pipeline] Label distribution:\n  Choppy (0): {distribution.get(0, 0)} ({distribution.get(0, 0)/len(df)*100:.2f}%)\n  Bullish (1): {distribution.get(1, 0)} ({distribution.get(1, 0)/len(df)*100:.2f}%)\n  Bearish (2): {distribution.get(2, 0)} ({distribution.get(2, 0)/len(df)*100:.2f}%)")
    print(f"[Pipeline] Saved to {output_path}")

if __name__ == "__main__":
    collect_data_and_features()
