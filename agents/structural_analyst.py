import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from schema import SpotSetup
from config import MACRO_STATE_PATH, MODEL_PATH, BOS_BREAKOUT_WINDOW

class StructuralAnalyst:
    def __init__(self, swing_window: int = 2, data_provider = None):
        self.swing_window = swing_window
        self.price_history: List[Dict[str, float]] = []
        
        if data_provider is None:
            from data_provider import DataProvider
            self.data_provider = DataProvider()
        else:
            self.data_provider = data_provider
            
        # State tracking for setup phases
        self.recent_swing_low: Optional[float] = None
        self.recent_swing_high: Optional[float] = None
        
        # Breakout window parameter
        self.breakout_window = BOS_BREAKOUT_WINDOW
        
        # Bullish (CE) setup tracking
        self.bull_cond_a = False
        self.bull_trigger_high: Optional[float] = None
        self.bull_trigger_low: Optional[float] = None
        self.bull_trigger_found = False
        self.bull_bars_since_trigger = 0
        
        # Bearish (PE) setup tracking
        self.bear_cond_a = False
        self.bear_trigger_high: Optional[float] = None
        self.bear_trigger_low: Optional[float] = None
        self.bear_trigger_found = False
        self.bear_bars_since_trigger = 0
        
        # Daily state tracking
        self.day_open: Optional[float] = None
        self.last_date: Optional[str] = None
        
        # Load the optimized ML model
        self.model = None
        self.model_path = MODEL_PATH
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                print(f"[Analyst] Loaded optimized Dual-Sweep RF model from {self.model_path}.")
            except Exception as e:
                print(f"[Analyst] Warning: Failed to load RF model: {e}")
        else:
            print(f"[Analyst] RF model not found at {self.model_path}. Running without ML filtering.")

    def update_history(self, open_p: float, high: float, low: float, close: float) -> Optional[SpotSetup]:
        """Update historical candles and scan for the weakness -> strength pattern."""
        # Detect new day
        current_date = datetime.now().strftime("%Y-%m-%d")
        if self.last_date != current_date:
            self.day_open = open_p
            self.last_date = current_date
            # Reset daily flags
            self.bull_cond_a = False
            self.bull_trigger_found = False
            self.bear_cond_a = False
            self.bear_trigger_found = False
            print(f"[Analyst] New day started. Day Open: {self.day_open}")

        self.price_history.append({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "timestamp": datetime.now()
        })
        
        # Keep window size efficient (max 250 bars)
        if len(self.price_history) > 250:
            self.price_history.pop(0)
            
        # Compute technical indicators dynamically
        self._calculate_indicators()
            
        return self._scan_structure()

    def _calculate_indicators(self):
        """Calculate technical indicators incrementally for the latest candle."""
        n = len(self.price_history)
        if n < 2:
            return
            
        current = self.price_history[-1]
        previous = self.price_history[-2]
        
        close = current["close"]
        high = current["high"]
        low = current["low"]
        prev_close = previous["close"]
        
        # 1. EMA 50
        if n < 50:
            current["ema_50"] = close
        elif n == 50:
            closes = [c["close"] for c in self.price_history]
            current["ema_50"] = sum(closes) / 50
        else:
            prev_ema = previous.get("ema_50", prev_close)
            ema_multiplier = 2 / (50 + 1)
            current["ema_50"] = (close - prev_ema) * ema_multiplier + prev_ema
            
        # 2. RSI 14
        if n < 15:
            current["rsi_14"] = 50.0
            current["avg_gain_14"] = 0.0
            current["avg_loss_14"] = 0.0
        elif n == 15:
            closes = [c["close"] for c in self.price_history]
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [d if d > 0 else 0.0 for d in deltas]
            losses = [-d if d < 0 else 0.0 for d in deltas]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            current["avg_gain_14"] = avg_gain
            current["avg_loss_14"] = avg_loss
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            current["rsi_14"] = 100 - (100 / (1 + rs)) if avg_loss > 0 else 100.0
        else:
            prev_avg_gain = previous.get("avg_gain_14", 0.0)
            prev_avg_loss = previous.get("avg_loss_14", 0.0)
            delta = close - prev_close
            gain = delta if delta > 0 else 0.0
            loss = -delta if delta < 0 else 0.0
            
            avg_gain = (prev_avg_gain * 13 + gain) / 14
            avg_loss = (prev_avg_loss * 13 + loss) / 14
            current["avg_gain_14"] = avg_gain
            current["avg_loss_14"] = avg_loss
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            current["rsi_14"] = 100 - (100 / (1 + rs)) if avg_loss > 0 else 100.0
            
        # 3. ATR 14
        current_tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if n < 15:
            current["atr_14"] = high - low
        elif n == 15:
            tr_list = []
            for i in range(len(self.price_history)):
                if i == 0:
                    tr_list.append(self.price_history[0]["high"] - self.price_history[0]["low"])
                else:
                    tr_list.append(max(
                        self.price_history[i]["high"] - self.price_history[i]["low"],
                        abs(self.price_history[i]["high"] - self.price_history[i-1]["close"]),
                        abs(self.price_history[i]["low"] - self.price_history[i-1]["close"])
                    ))
            current["atr_14"] = sum(tr_list[:14]) / 14
        else:
            prev_atr = previous.get("atr_14", previous["high"] - previous["low"])
            current["atr_14"] = (prev_atr * 13 + current_tr) / 14

    def _scan_structure(self) -> Optional[SpotSetup]:
        """
        Scan history for structure patterns:
        1. Identify swing levels.
        2. Detect Liquidity Sweep of previous day's high/low.
        3. Detect Break of Structure (BOS).
        """
        if len(self.price_history) < 15:
            return None
            
        # Get swing levels for fallback
        self._calculate_swings()
        
        # Retrieve previous day's high/low
        prev_levels = self.data_provider.get_nifty_prev_day_levels()
        if prev_levels:
            prev_high, prev_low = prev_levels[0], prev_levels[1]
        else:
            # Fallback to swing levels if not available (e.g., in simulation)
            if self.recent_swing_high is not None and self.recent_swing_low is not None:
                prev_high, prev_low = self.recent_swing_high, self.recent_swing_low
            else:
                return None
                
        current_candle = self.price_history[-1]
        high, low, close, open_val = current_candle["high"], current_candle["low"], current_candle["close"], current_candle["open"]
        
        # Get indicators
        rsi = current_candle.get("rsi_14", 50.0)
        atr = current_candle.get("atr_14", high - low)
        ema = current_candle.get("ema_50", close)
        
        # Feature dictionary
        features = {
            "rsi": rsi,
            "atr": atr,
            "ema_dist": close - ema,
            "intraday_ret": ((close - self.day_open) / self.day_open * 100) if self.day_open else 0.0,
            "spread": high - low
        }
        
        setup_triggered = False
        setup_price = 0.0
        invalidation_price = 0.0
        setup_type = ""
        
        # 1. Bullish (CE) Sweep Scan
        if not self.bull_cond_a:
            if low < prev_low:
                self.bull_cond_a = True
                print(f"[Analyst] Bullish Sweep Cond A: Low {low:.2f} swept Prev Low {prev_low:.2f}")
                
        is_green = close > open_val
        if self.bull_cond_a and is_green and not self.bull_trigger_found:
            self.bull_trigger_high = high
            self.bull_trigger_low = low
            self.bull_trigger_found = True
            self.bull_cond_a = False
            self.bull_bars_since_trigger = 0
            print(f"[Analyst] Bullish Trigger Candle found! High: {high:.2f}, Low: {low:.2f}")
            
        elif self.bull_trigger_found and self.bull_trigger_high is not None:
            self.bull_bars_since_trigger += 1
            if high > self.bull_trigger_high:
                setup_price = high if open_val > self.bull_trigger_high else self.bull_trigger_high
                invalidation_price = self.bull_trigger_low
                setup_type = "BULLISH_BOS"
                setup_triggered = True
                
                # Reset flags
                self.bull_trigger_high = None
                self.bull_trigger_low = None
                self.bull_trigger_found = False
                self.bull_bars_since_trigger = 0
            elif self.bull_bars_since_trigger >= self.breakout_window:
                # Reset if window expired
                print(f"[Analyst] Bullish breakout window expired ({self.breakout_window} bars). Resetting trigger.")
                self.bull_trigger_high = None
                self.bull_trigger_low = None
                self.bull_trigger_found = False
                self.bull_bars_since_trigger = 0
                
        # 2. Bearish (PE) Sweep Scan
        if not self.bear_cond_a:
            if high > prev_high:
                self.bear_cond_a = True
                print(f"[Analyst] Bearish Sweep Cond A: High {high:.2f} swept Prev High {prev_high:.2f}")
                
        is_red = close < open_val
        if self.bear_cond_a and is_red and not self.bear_trigger_found:
            self.bear_trigger_high = high
            self.bear_trigger_low = low
            self.bear_trigger_found = True
            self.bear_cond_a = False
            self.bear_bars_since_trigger = 0
            print(f"[Analyst] Bearish Trigger Candle found! High: {high:.2f}, Low: {low:.2f}")
            
        elif self.bear_trigger_found and self.bear_trigger_low is not None:
            self.bear_bars_since_trigger += 1
            if low < self.bear_trigger_low:
                setup_price = low if open_val < self.bear_trigger_low else self.bear_trigger_low
                invalidation_price = self.bear_trigger_high
                setup_type = "BEARISH_BOS"
                setup_triggered = True
                
                # Reset flags
                self.bear_trigger_high = None
                self.bear_trigger_low = None
                self.bear_trigger_found = False
                self.bear_bars_since_trigger = 0
            elif self.bear_bars_since_trigger >= self.breakout_window:
                # Reset if window expired
                print(f"[Analyst] Bearish breakout window expired ({self.breakout_window} bars). Resetting trigger.")
                self.bear_trigger_high = None
                self.bear_trigger_low = None
                self.bear_trigger_found = False
                self.bear_bars_since_trigger = 0
                
        # If setup is triggered, run ML validation
        if setup_triggered:
            is_ce_val = 1 if setup_type == "BULLISH_BOS" else 0
            feature_vector = [features["rsi"], features["atr"], features["ema_dist"], features["intraday_ret"], features["spread"], is_ce_val]
            
            if self.model:
                try:
                    prob = float(self.model.predict_proba([feature_vector])[0][1])
                except Exception as e:
                    print(f"[Analyst] Error predicting probability: {e}")
                    prob = 0.50
            else:
                prob = 0.55
                
            if prob > 0.54:
                confidence_level = "High Confidence"
            elif prob >= 0.51:
                confidence_level = "Moderate Confidence"
            else:
                confidence_level = "Filtered"
                print(f"[Analyst] Setup blocked by Random Forest model (Win Prob: {prob:.2%}).")
                return None
                
            setup = SpotSetup(
                spot_price=setup_price,
                invalidation_price=invalidation_price,
                setup_type=setup_type,
                prob=prob,
                confidence_level=confidence_level
            )
            print(f"[Analyst] Setup passed ML Filter with probability: {prob:.2%} ({setup.confidence_level})")
            return setup
            
        return None

    def _calculate_swings(self):
        """Identify key Swing Highs and Swing Lows in price history (Fallback)."""
        n = len(self.price_history)
        w = self.swing_window
        
        # Calculate recent Swing Lows
        for i in range(n - w - 1, w - 1, -1):
            is_low = True
            for j in range(1, w + 1):
                if self.price_history[i]["low"] > self.price_history[i - j]["low"] or \
                   self.price_history[i]["low"] > self.price_history[i + j]["low"]:
                    is_low = False
                    break
            if is_low:
                self.recent_swing_low = self.price_history[i]["low"]
                break

        # Calculate recent Swing Highs
        for i in range(n - w - 1, w - 1, -1):
            is_high = True
            for j in range(1, w + 1):
                if self.price_history[i]["high"] < self.price_history[i - j]["high"] or \
                   self.price_history[i]["high"] < self.price_history[i + j]["high"]:
                    is_high = False
                    break
            if is_high:
                self.recent_swing_high = self.price_history[i]["high"]
                break
