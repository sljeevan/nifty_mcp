import sys
import math
import os
import csv
import io
import requests
import time
import threading
import json
import websocket
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from config import NIFTY_SYMBOL, INDSTOCKS_TOKEN

# Path configuration for tradingview-mcp services
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, "tradingview-mcp-india", "src"))
from tradingview_mcp.core.services.yahoo_finance_service import get_price

from config import NIFTY_SYMBOL, INDSTOCKS_TOKEN, INSTRUMENTS_CSV_PATH

class DataProvider:
    def __init__(self):
        self._last_spot_price = 23200.0  # Safe default
        self.token = INDSTOCKS_TOKEN
        self.headers = {"Authorization": self.token}
        self.cache_path = INSTRUMENTS_CSV_PATH
        self._instruments_cache = None
        self._cached_daily_ema200 = None
        self._ema_cache_date = None
        self._cached_vix = None
        self._vix_cache_time = None


        # WebSocket live streaming state
        self.live_prices = {}
        self.subscribed_instruments = set()
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
        self.ws_lock = threading.Lock()

        # Determine execution mode (simulation vs live)
        self.is_sim = True
        if "--mode" in sys.argv:
            idx = sys.argv.index("--mode")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "live":
                self.is_sim = False

        if not self.is_sim:
            self.start_websocket()

    def start_websocket(self):
        """Starts the WebSocket connection in a background thread."""
        print("[DataProvider] Starting live WebSocket data feed thread...")
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()

    def _run_websocket_loop(self):
        """Runs the WebSocket event loop with auto-reconnection."""
        url = "wss://api.indstocks.com/ws"
        while not self.is_sim:
            try:
                print(f"[WebSocket] Connecting to {url}...")
                self.ws = websocket.WebSocketApp(
                    url,
                    header={"Authorization": self.token},
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close
                )
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"[WebSocket] Connection loop error: {e}")
            time.sleep(5)

    def _on_ws_open(self, ws):
        print("[WebSocket] Connection established successfully.")
        with self.ws_lock:
            self.ws_connected = True
        
        # Subscribe to Nifty spot and any pre-subscribed options
        to_sub = {"NIDX:40000001"}
        with self.ws_lock:
            to_sub.update(self.subscribed_instruments)
        self._send_subscription(list(to_sub), action="subscribe")

    def _on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
            instrument = data.get("instrument")
            price = data.get("last_price") or data.get("last_traded_price") or data.get("price") or data.get("ltp")
            if instrument and price is not None:
                price_val = float(price)
                inst_str = str(instrument)
                with self.ws_lock:
                    self.live_prices[inst_str] = price_val
                    if ":" in inst_str:
                        token = inst_str.split(":")[1]
                        self.live_prices[token] = price_val
                        seg = inst_str.split(":")[0]
                        self.live_prices[f"{seg}_{token}"] = price_val
                    else:
                        self.live_prices[f"NIDX:{inst_str}"] = price_val
                        self.live_prices[f"NIDX_{inst_str}"] = price_val
                        self.live_prices[f"NFO:{inst_str}"] = price_val
                        self.live_prices[f"NFO_{inst_str}"] = price_val
                
                if "40000001" in inst_str:
                    self._last_spot_price = price_val
        except Exception as e:
            print(f"[WebSocket] Error parsing message: {e}")

    def _on_ws_error(self, ws, error):
        print(f"[WebSocket] Error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg):
        print(f"[WebSocket] Connection closed (code: {close_status_code}, msg: {close_msg})")
        with self.ws_lock:
            self.ws_connected = False

    def _send_subscription(self, instruments: List[str], action: str = "subscribe"):
        """Formats and sends subscription request to INDstocks WebSocket."""
        if not instruments:
            return
        formatted = []
        for inst in instruments:
            if ":" in inst:
                formatted.append(inst)
            elif "_" in inst:
                parts = inst.split("_")
                formatted.append(f"{parts[0]}:{parts[1]}")
            else:
                if inst == "40000001":
                    formatted.append("NIDX:40000001")
                else:
                    formatted.append(f"NFO:{inst}")

        payload = {
            "action": action,
            "mode": "ltp",
            "instruments": formatted
        }
        try:
            with self.ws_lock:
                if self.ws and self.ws_connected:
                    self.ws.send(json.dumps(payload))
                    print(f"[WebSocket] Sent {action} request: {payload}")
                else:
                    print(f"[WebSocket] {action} request buffered (waiting for connection): {payload}")
        except Exception as e:
            print(f"[WebSocket] Error sending subscription: {e}")

    def _load_instruments(self) -> List[Dict[str, str]]:
        """Load F&O instruments list from local cache or download it from INDstocks if expired."""
        if self._instruments_cache is not None:
            return self._instruments_cache

        # Check if cache exists and was modified today
        if os.path.exists(self.cache_path):
            try:
                mtime = os.path.getmtime(self.cache_path)
                mdate = date.fromtimestamp(mtime)
                if mdate == date.today():
                    with open(self.cache_path, "r") as f:
                        self._instruments_cache = list(csv.DictReader(f))
                        return self._instruments_cache
            except Exception as e:
                print(f"[Warning] Failed to read instruments cache: {e}")

        # Download and cache
        print("[DataProvider] Downloading F&O instruments master from INDstocks...")
        try:
            r = requests.get("https://api.indstocks.com/market/instruments?source=fno", headers=self.headers, timeout=20)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "w") as f:
                    f.write(r.text)
                self._instruments_cache = list(csv.DictReader(io.StringIO(r.text)))
                return self._instruments_cache
            else:
                print(f"[Warning] Indstocks instruments download failed: {r.status_code}")
        except Exception as e:
            print(f"[Warning] Failed to fetch instruments from API: {e}")

        # Fallback to expired cache if available
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    self._instruments_cache = list(csv.DictReader(f))
                    return self._instruments_cache
            except Exception:
                pass

        return []

    def get_nifty_spot(self) -> float:
        """Fetch real-time Nifty 50 spot price from live WebSocket feed (with REST fallback)."""
        if not self.is_sim:
            with self.ws_lock:
                price = (self.live_prices.get("NIDX:40000001") or 
                         self.live_prices.get("NIDX_40000001") or 
                         self.live_prices.get("40000001"))
            if price:
                self._last_spot_price = float(price)
                return self._last_spot_price

        # Fallback to REST API
        try:
            url = "https://api.indstocks.com/market/quotes/ltp?scrip-codes=NIDX_40000001"
            r = requests.get(url, headers=self.headers, timeout=5)
            if r.status_code == 200:
                data = r.json().get("data", {})
                price = data.get("NIDX_40000001", {}).get("live_price")
                if price:
                    self._last_spot_price = float(price)
                    return self._last_spot_price
        except Exception as e:
            print(f"[Warning] Failed to fetch Nifty Spot from INDstocks: {e}. Falling back to Yahoo Finance.")

        # Fallback to Yahoo Finance
        try:
            quote = get_price(NIFTY_SYMBOL)
            if quote and "price" in quote and quote["price"] is not None:
                self._last_spot_price = float(quote["price"])
                return self._last_spot_price
        except Exception as e:
            print(f"[Warning] Failed to fetch live spot price from Yahoo Finance: {e}. Using cached: {self._last_spot_price}")
            
        return self._last_spot_price

    def get_options_chain(self, spot_price: float) -> List[Dict[str, Any]]:
        """
        Fetch real-time option contracts and premiums from INDstocks.
        Returns options chain for strikes close to spot price for the next Tuesday.
        """
        instruments = self._load_instruments()
        if not instruments:
            print("[DataProvider] Error: No instruments loaded. Option chain empty.")
            return []

        # Target 5 strikes above and below ATM (steps of 50)
        atm_strike = round(spot_price / 50.0) * 50
        target_strikes = [atm_strike + i * 50 for i in range(-5, 6)]

        # Calculate the next Thursday expiry date
        today = date.today()
        days_ahead = (3 - today.weekday()) % 7
        expiry = today + timedelta(days=days_ahead)
        expiry_date_prefix = expiry.strftime("%m/%d/%Y")  # e.g. "06/18/2026"

        # Filter active contracts close to ATM
        filtered = []
        for row in instruments:
            if row.get("SEM_EXCH_INSTRUMENT_TYPE") == "OPTIDX" and \
               row.get("TRADING_SYMBOL", "").startswith("NIFTY-") and \
               row.get("EXPIRY_DATE", "").startswith(expiry_date_prefix):
                try:
                    strike = float(row.get("STRIKE_PRICE", 0))
                    if int(strike) in target_strikes:
                        filtered.append(row)
                except ValueError:
                    pass

        if not filtered:
            print(f"[DataProvider] Warning: No Nifty contracts found for expiry {expiry_date_prefix} close to ATM.")
            return []

        # Determine execution mode (simulation vs live)
        is_sim = self.is_sim

        # Bulk query prices if in live mode
        live_quotes = {}
        if not is_sim:
            # Manage WebSocket subscriptions dynamically for target options
            target_instruments = {f"NFO:{c['SECURITY_ID']}" for c in filtered}
            target_instruments.add("NIDX:40000001")

            with self.ws_lock:
                new_to_sub = target_instruments - self.subscribed_instruments
                old_to_unsub = self.subscribed_instruments - target_instruments

            if new_to_sub:
                self._send_subscription(list(new_to_sub), action="subscribe")
                with self.ws_lock:
                    self.subscribed_instruments.update(new_to_sub)

            if old_to_unsub:
                old_to_unsub.discard("NIDX:40000001")  # Keep Nifty Spot subscribed
                if old_to_unsub:
                    self._send_subscription(list(old_to_unsub), action="unsubscribe")
                    with self.ws_lock:
                        self.subscribed_instruments.difference_update(old_to_unsub)

            # Retrieve from WebSocket cache
            with self.ws_lock:
                for c in filtered:
                    code = f"NFO_{c['SECURITY_ID']}"
                    sec_id = c['SECURITY_ID']
                    val = (self.live_prices.get(code) or 
                           self.live_prices.get(f"NFO:{sec_id}") or 
                           self.live_prices.get(sec_id))
                    if val is not None:
                        live_quotes[code] = {"live_price": val}

            # REST fallback for any option contracts not yet received via WebSocket
            missing_codes = [f"NFO_{c['SECURITY_ID']}" for c in filtered if f"NFO_{c['SECURITY_ID']}" not in live_quotes]
            if missing_codes:
                scrip_codes_str = ",".join(missing_codes)
                try:
                    url = f"https://api.indstocks.com/market/quotes/ltp?scrip-codes={scrip_codes_str}"
                    r = requests.get(url, headers=self.headers, timeout=10)
                    if r.status_code == 200:
                        rest_data = r.json().get("data", {})
                        for k, v in rest_data.items():
                             live_quotes[k] = v
                             if "live_price" in v:
                                 with self.ws_lock:
                                     self.live_prices[k] = float(v["live_price"])
                    else:
                        print(f"[Warning] Bulk quote query fallback failed with status: {r.status_code}")
                except Exception as e:
                    print(f"[Warning] Failed to fetch live quotes fallback: {e}")

        chain = []
        for c in filtered:
            code = f"NFO_{c['SECURITY_ID']}"
            strike = int(float(c["STRIKE_PRICE"]))
            option_type = c["OPTION_TYPE"]
            
            # Get real premium from Indstocks or calculate fallback if not present
            premium = None
            if not is_sim and code in live_quotes:
                premium = live_quotes[code].get("live_price")
                
            iv_val = self.get_india_vix()
            if premium is None:
                # Fallback to simulated premium (default in sim mode)
                premium = self._calculate_premium(spot_price, strike, option_type, days_to_expiry=float(days_ahead), iv=iv_val)
            
            delta = self._calculate_delta(spot_price, strike, option_type, days_to_expiry=float(days_ahead), iv=iv_val)
            
            chain.append({
                "strike_symbol": code,  # We use NFO_{SECURITY_ID} as symbol
                "strike_name": c.get("CUSTOM_SYMBOL", code),
                "strike": strike,
                "type": option_type,
                "premium": float(premium),
                "delta": delta
            })

        return chain

    def get_option_by_symbol(self, symbol: str, spot_price: float) -> Optional[Dict[str, Any]]:
        """Retrieve contract detail by strike symbol and current spot."""
        chain = self.get_options_chain(spot_price)
        for contract in chain:
            if contract["strike_symbol"] == symbol:
                return contract
        return None

    def _calculate_delta(self, spot: float, strike: int, option_type: str, days_to_expiry: float = 7.0, iv: float = 0.15) -> float:
        """Approximate option delta using normal distribution simulation."""
        try:
            # Use a floor of 0.5 days to avoid division by zero on expiry day
            days = max(0.5, days_to_expiry)
            t = days / 365.0
            
            d1 = (math.log(spot / strike) + (0.5 * iv ** 2) * t) / (iv * math.sqrt(t))
            
            # Normal cumulative distribution approximation (CDF)
            cdf_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
            
            if option_type == "CE":
                return round(cdf_d1, 2)
            else:
                return round(cdf_d1 - 1.0, 2)
        except Exception:
            # Fallback
            if option_type == "CE":
                return 0.5 if spot >= strike else 0.3
            else:
                return -0.5 if spot <= strike else -0.3

    def _calculate_premium(self, spot: float, strike: int, option_type: str, days_to_expiry: float = 7.0, iv: float = 0.15) -> float:
        """Simulate realistic option premium with intrinsic and extrinsic values."""
        # Scale extrinsic value by square root of time remaining and dynamic IV
        extrinsic = 100.0 * (iv / 0.15) * math.sqrt(max(0.5, days_to_expiry) / 7.0)
        distance = abs(spot - strike)
        
        # Extrinsic decays as it moves out-of-the-money
        extrinsic = extrinsic * math.exp(-distance / 150.0)
        
        if option_type == "CE":
            intrinsic = max(0.0, spot - strike)
        else:
            intrinsic = max(0.0, strike - spot)
            
        premium = intrinsic + extrinsic
        return round(max(5.0, premium), 2)  # Limit options floor price at 5.0 INR

    def get_nifty_daily_ema200(self) -> Optional[float]:
        """Fetch daily closes for ^NSEI and compute the 200 EMA with caching."""
        today = date.today()
        if self._cached_daily_ema200 is not None and self._ema_cache_date == today:
            return self._cached_daily_ema200

        try:
            import urllib.request
            import json
            from tradingview_mcp.core.services.proxy_manager import build_opener_with_proxy
            
            symbol = "^NSEI"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=300d"
            req = urllib.request.Request(url, headers={"User-Agent": "tradingview-mcp/0.5.0"})
            opener = build_opener_with_proxy("tradingview-mcp/0.5.0")
            
            with opener.open(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data["chart"]["result"][0]
                closes = result["indicators"]["quote"][0]["close"]
                valid_closes = [c for c in closes if c is not None]
                
                if len(valid_closes) < 200:
                    print("[DataProvider] Error: Not enough data points to compute 200 EMA.")
                    return None
                
                sma_200 = sum(valid_closes[:200]) / 200
                ema = sma_200
                multiplier = 2 / (200 + 1)
                
                for close in valid_closes[200:]:
                    ema = (close - ema) * multiplier + ema
                    
                self._cached_daily_ema200 = float(ema)
                self._ema_cache_date = today
                return self._cached_daily_ema200
        except Exception as e:
            print(f"[DataProvider] Error calculating Daily 200 EMA: {e}")
            return None

    def get_nifty_prev_day_levels(self) -> Optional[List[float]]:
        """Fetch previous day's High and Low for Nifty 50 spot index [high, low]."""
        if self.is_sim:
            return None
        try:
            import urllib.request
            import json
            from tradingview_mcp.core.services.proxy_manager import build_opener_with_proxy
            
            symbol = "^NSEI"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
            req = urllib.request.Request(url, headers={"User-Agent": "tradingview-mcp/0.5.0"})
            opener = build_opener_with_proxy("tradingview-mcp/0.5.0")
            
            with opener.open(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data["chart"]["result"][0]
                highs = result["indicators"]["quote"][0]["high"]
                lows = result["indicators"]["quote"][0]["low"]
                
                valid_highs = [h for h in highs if h is not None]
                valid_lows = [l for l in lows if l is not None]
                
                if len(valid_highs) >= 2:
                    return [float(valid_highs[-2]), float(valid_lows[-2])]
                elif len(valid_highs) == 1:
                    return [float(valid_highs[0]), float(valid_lows[0])]
        except Exception as e:
            print(f"[DataProvider] Error fetching Nifty prev day levels: {e}")
        return None

    def get_india_vix(self) -> float:
        """Fetch real-time India VIX from Yahoo Finance to use as IV with caching."""
        now = datetime.now()
        if self._cached_vix is not None and self._vix_cache_time is not None:
            if now - self._vix_cache_time < timedelta(minutes=15):
                return self._cached_vix
        try:
            from tradingview_mcp.core.services.yahoo_finance_service import get_price
            quote = get_price("^INDIAVIX")
            if quote and "price" in quote and quote["price"] is not None:
                vix = float(quote["price"]) / 100.0
                self._cached_vix = vix
                self._vix_cache_time = now
                print(f"[DataProvider] Dynamic IV updated from India VIX: {vix * 100.0:.2f}%")
                return vix
        except Exception as e:
            print(f"[DataProvider] Error fetching India VIX: {e}")
        return 0.15  # Fallback 15% IV

    def get_historical_5m_candles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch historical 5-minute candles for ^NSEI from Yahoo Finance."""
        try:
            import urllib.request
            import json
            from tradingview_mcp.core.services.proxy_manager import build_opener_with_proxy
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{NIFTY_SYMBOL}?interval=5m&range=5d"
            req = urllib.request.Request(url, headers={"User-Agent": "tradingview-mcp/0.5.0"})
            opener = build_opener_with_proxy("tradingview-mcp/0.5.0")
            
            with opener.open(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data["chart"]["result"][0]
                quotes = result["indicators"]["quote"][0]
                highs = quotes["high"]
                lows = quotes["low"]
                closes = quotes["close"]
                opens = quotes["open"]
                timestamps = result["timestamp"]
                
                candles = []
                for i in range(len(timestamps)):
                    if (opens[i] is not None and highs[i] is not None and 
                        lows[i] is not None and closes[i] is not None):
                        candles.append({
                            "open": float(opens[i]),
                            "high": float(highs[i]),
                            "low": float(lows[i]),
                            "close": float(closes[i]),
                            "timestamp": datetime.fromtimestamp(timestamps[i])
                        })
                return candles[-limit:]
        except Exception as e:
            print(f"[DataProvider] Error fetching historical 5m candles: {e}")
            return []

