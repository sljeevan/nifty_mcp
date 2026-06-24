import urllib.request
import json
import pickle
import os
import sys
import numpy as np
from datetime import datetime, timezone

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calc_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down > 0 else 0
    rsi = np.zeros_like(prices, dtype=float)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down > 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi.tolist()

def calc_atr(highs, lows, closes, period=14):
    tr = []
    for i in range(len(closes)):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
            
    atr = [sum(tr[:period]) / period]
    for i in range(period, len(closes)):
        val = (atr[-1] * (period - 1) + tr[i]) / period
        atr.append(val)
    return ([None] * (period - 1)) + atr

def calculate_ema(prices, period=50):
    if len(prices) < period:
        return [None] * len(prices)
    ema = []
    sma = sum(prices[:period]) / period
    ema.extend([None] * (period - 1))
    ema.append(sma)
    multiplier = 2 / (period + 1)
    for price in prices[period:]:
        next_ema = (price - ema[-1]) * multiplier + ema[-1]
        ema.append(next_ema)
    return ema

def train_and_save():
    symbol = "^NSEI"
    interval = "5m"
    period = "60d"
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={period}"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    print("Fetching Nifty 50 5m data from Yahoo Finance...")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
        
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    q = result["indicators"]["quote"][0]
    
    candles = []
    for i, ts in enumerate(timestamps):
        o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, l, c):
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        candles.append({
            "timestamp": ts,
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": v or 0
        })
        
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    
    rsi_14 = calc_rsi(closes, 14)
    atr_14 = calc_atr(highs, lows, closes, 14)
    ema_50 = calculate_ema(closes, 50)
    
    for idx, c in enumerate(candles):
        c["rsi"] = rsi_14[idx]
        c["atr"] = atr_14[idx]
        c["ema"] = ema_50[idx]
        
    days = {}
    for c in candles:
        d = c["date"]
        if d not in days:
            days[d] = []
        days[d].append(c)
        
    sorted_dates = sorted(list(days.keys()))
    
    daily_stats = {}
    for d in sorted_dates:
        h_list = [c["high"] for c in days[d]]
        l_list = [c["low"] for c in days[d]]
        daily_stats[d] = {
            "high": max(h_list),
            "low": min(l_list),
            "open": days[d][0]["open"]
        }
        
    initial_premium = 120.0
    delta = 0.50
    
    all_setups = []
    
    for day_idx in range(1, len(sorted_dates)):
        prev_date = sorted_dates[day_idx - 1]
        curr_date = sorted_dates[day_idx]
        
        prev_high = daily_stats[prev_date]["high"]
        prev_low = daily_stats[prev_date]["low"]
        day_open = daily_stats[curr_date]["open"]
        
        day_candles = days[curr_date]
        
        bull_cond_a = False
        bull_trigger_high = None
        bull_trigger_low = None
        bull_trigger_found = False
        
        bear_cond_a = False
        bear_trigger_high = None
        bear_trigger_low = None
        bear_trigger_found = False
        
        for i, c in enumerate(day_candles):
            if i < 2:
                continue
                
            high, low, close, open_val = c["high"], c["low"], c["close"], c["open"]
            rsi = c["rsi"]
            atr = c["atr"]
            ema = c["ema"]
            
            if None in (rsi, atr, ema):
                continue
                
            def check_outcome(direction, entry_spot, sl_level, tp_level):
                for k in range(i + 1, min(i + 40, len(day_candles))):
                    future_c = day_candles[k]
                    f_high, f_low, f_open = future_c["high"], future_c["low"], future_c["open"]
                    
                    if direction == "CE":
                        if f_low <= sl_level:
                            exit_spot = sl_level
                            if f_open < sl_level and k == i + 1:
                                exit_spot = f_open
                            opt_exit = max(5.0, initial_premium + (exit_spot - entry_spot) * delta)
                            return opt_exit - initial_premium
                        elif f_high >= tp_level:
                            exit_spot = tp_level
                            if f_open > tp_level and k == i + 1:
                                exit_spot = f_open
                            opt_exit = initial_premium + (exit_spot - entry_spot) * delta
                            return opt_exit - initial_premium
                    else: # PE
                        if f_high >= sl_level:
                            exit_spot = sl_level
                            if f_open > sl_level and k == i + 1:
                                exit_spot = f_open
                            opt_exit = max(5.0, initial_premium + (entry_spot - exit_spot) * delta)
                            return opt_exit - initial_premium
                        elif f_low <= tp_level:
                            exit_spot = tp_level
                            if f_open < tp_level and k == i + 1:
                                exit_spot = f_open
                            opt_exit = initial_premium + (entry_spot - exit_spot) * delta
                            return opt_exit - initial_premium
                            
                last_c = day_candles[-1]
                last_spot = last_c["close"]
                if direction == "CE":
                    opt_exit = max(5.0, initial_premium + (last_spot - entry_spot) * delta)
                else:
                    opt_exit = max(5.0, initial_premium + (entry_spot - last_spot) * delta)
                return opt_exit - initial_premium

            features_base = {
                "rsi": rsi,
                "atr": atr,
                "ema_dist": close - ema,
                "intraday_ret": (close - day_open) / day_open * 100,
                "spread": high - low
            }

            # 1. Bullish (CE) Setup Scan
            if not bull_cond_a:
                if low < prev_low:
                    bull_cond_a = True
            
            is_green = close > open_val
            if bull_cond_a and is_green and not bull_trigger_found:
                bull_trigger_high = high
                bull_trigger_low = low
                bull_trigger_found = True
                bull_cond_a = False
                
            elif bull_trigger_found and bull_trigger_high is not None:
                if high > bull_trigger_high:
                    entry_spot = bull_trigger_high
                    if open_val > bull_trigger_high:
                        entry_spot = open_val
                    spot_sl = bull_trigger_low
                    
                    lookback = day_candles[max(0, i-15):i]
                    if lookback:
                        recent_swing_high = max([x["high"] for x in lookback])
                    else:
                        recent_swing_high = prev_high
                    if recent_swing_high <= entry_spot:
                        recent_swing_high = entry_spot + 2 * (entry_spot - spot_sl)
                        
                    pnl_opt = check_outcome("CE", entry_spot, spot_sl, recent_swing_high)
                    all_setups.append({
                        "features": {**features_base, "is_ce": 1},
                        "pnl": pnl_opt,
                        "outcome": 1 if pnl_opt > 0 else 0,
                        "date": curr_date
                    })
                    bull_trigger_high = None
                    bull_trigger_low = None
                    bull_trigger_found = False
                else:
                    bull_trigger_high = None
                    bull_trigger_low = None
                    bull_trigger_found = False

            # 2. Bearish (PE) Setup Scan
            if not bear_cond_a:
                if high > prev_high:
                    bear_cond_a = True
            
            is_red = close < open_val
            if bear_cond_a and is_red and not bear_trigger_found:
                bear_trigger_high = high
                bear_trigger_low = low
                bear_trigger_found = True
                bear_cond_a = False
                
            elif bear_trigger_found and bear_trigger_low is not None:
                if low < bear_trigger_low:
                    entry_spot = bear_trigger_low
                    if open_val < bear_trigger_low:
                        entry_spot = open_val
                    spot_sl = bear_trigger_high
                    
                    lookback = day_candles[max(0, i-15):i]
                    if lookback:
                        recent_swing_low = min([x["low"] for x in lookback])
                    else:
                        recent_swing_low = prev_low
                    if recent_swing_low >= entry_spot:
                        recent_swing_low = entry_spot - 2 * (spot_sl - entry_spot)
                        
                    pnl_opt = check_outcome("PE", entry_spot, spot_sl, recent_swing_low)
                    all_setups.append({
                        "features": {**features_base, "is_ce": 0},
                        "pnl": pnl_opt,
                        "outcome": 1 if pnl_opt > 0 else 0,
                        "date": curr_date
                    })
                    bear_trigger_high = None
                    bear_trigger_low = None
                    bear_trigger_found = False
                else:
                    bear_trigger_high = None
                    bear_trigger_low = None
                    bear_trigger_found = False

    total_setups = len(all_setups)
    print(f"Total Dual-Sweep setups harvested: {total_setups}")
    if total_setups == 0:
        return
        
    X = []
    y = []
    for s in all_setups:
        f = s["features"]
        X.append([f["rsi"], f["atr"], f["ema_dist"], f["intraday_ret"], f["spread"], f["is_ce"]])
        y.append(s["outcome"])
        
    X = np.array(X)
    y = np.array(y)
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=80, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Save the trained model to scratch
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "scratch", "dual_sweep_rf.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"✅ Random Forest model successfully trained on all {total_setups} setups and saved to {model_path}.")

if __name__ == "__main__":
    train_and_save()
