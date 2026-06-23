import uuid
import requests
from typing import Dict, Any, Optional
from schema import NiftyTradeState, ExecutionOrder
from datetime import datetime
from config import PAPER_TRADING_MODE, INDSTOCKS_TOKEN
import db

class ExecutionRouter:
    def __init__(self):
        self.active_position: Optional[Dict[str, Any]] = None
        self.trade_id_in_db: Optional[int] = None
        
    def query_order_status(self, order_id: str) -> str:
        """
        Polls the INDstocks API for the status of a specific order.
        Returns one of: 'FILLED', 'PENDING', 'REJECTED', 'CANCELLED', or 'UNKNOWN'
        """
        if PAPER_TRADING_MODE:
            return "FILLED"
            
        headers = {
            "Authorization": INDSTOCKS_TOKEN,
            "Content-Type": "application/json"
        }
        
        params = {
            "order_id": order_id,
            "segment": "DERIVATIVE"
        }
        
        try:
            url = "https://api.indstocks.com/order"
            response = requests.get(url, params=params, json=params, headers=headers, timeout=10)
            if response.status_code == 200:
                res_data = response.json()
                data = res_data.get("data", {})
                status = data.get("status") or res_data.get("status")
                
                if status:
                    status_upper = str(status).upper()
                    if "FILL" in status_upper:
                        return "FILLED"
                    elif "PEND" in status_upper or "OPEN" in status_upper or "ACCEPT" in status_upper:
                        return "PENDING"
                    elif "REJECT" in status_upper:
                        return "REJECTED"
                    elif "CANCEL" in status_upper:
                        return "CANCELLED"
                    else:
                        print(f"[Execution] Unknown order status string: {status}")
                        return "PENDING"  # Safe default to avoid false fills
            else:
                print(f"[Execution] Status query failed: HTTP {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[Execution] Exception querying order status: {e}")
            
        return "UNKNOWN"
        
    def execute_validated_trade(self, state: NiftyTradeState, db_trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Routes order:
        - If PAPER_TRADING_MODE is True: Simulates order fill.
        - If PAPER_TRADING_MODE is False: Places a real order via the INDstocks API and sets to PENDING.
        """
        self.trade_id_in_db = db_trade_id
        
        # Instantiate active position state - initially set status to PENDING
        self.active_position = {
            "strike_selected": state.strike_selected,
            "strike_name": state.strike_name,
            "entry_premium": state.entry_premium,
            "stop_loss_premium": state.stop_loss_premium,
            "target_premium": state.target_premium,
            "lot_size": state.lot_size,
            "qty": state.lot_size,
            "current_premium": state.entry_premium,
            "pnl": 0.0,
            "status": "PENDING",  # Will change to OPEN once filled
            "entry_time": datetime.now(),
            "entry_order_id": None,
            "exit_order_id": None,
            "exit_info": None
        }
        
        # Strip NFO_ prefix if it exists to retrieve security ID
        security_id = state.strike_selected
        if security_id.startswith("NFO_"):
            security_id = security_id.split("_")[1]
            
        if PAPER_TRADING_MODE:
            print(f"[Execution] [PAPER MODE] Order routed for {state.strike_name} ({state.strike_selected}).")
            print(f"[Execution] [PAPER MODE] LIMIT BUY filled at {state.entry_premium:.2f} INR (Quantity: {state.lot_size}).")
            self.active_position["status"] = "OPEN"
            db.update_trade_pnl(db_trade_id, 0.0, "FILLED")
        else:
            print(f"[Execution] [LIVE MODE] Placing order on INDstocks for {state.strike_name} (ID: {security_id})...")
            headers = {
                "Authorization": INDSTOCKS_TOKEN,
                "Content-Type": "application/json"
            }
            payload = {
                "txn_type": "BUY",
                "exchange": "NSE",
                "segment": "DERIVATIVE",
                "product": "MARGIN",
                "order_type": "LIMIT",
                "limit_price": float(state.entry_premium),
                "validity": "DAY",
                "security_id": str(security_id),
                "qty": int(state.lot_size),
                "is_amo": False,
                "algo_id": "99999"
            }
            
            try:
                response = requests.post("https://api.indstocks.com/order", json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    res_data = response.json()
                    order_id = res_data.get("data", {}).get("order_id", "LIVE_ORDER_OK")
                    print(f"[Execution] [LIVE MODE] Order successfully placed! Order ID: {order_id}")
                    self.active_position["entry_order_id"] = order_id
                    db.update_trade_pnl(db_trade_id, 0.0, "PENDING")
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"[Execution] [LIVE MODE] Order placement failed: {error_msg}")
                    db.update_trade_pnl(db_trade_id, 0.0, f"FAILED_PLACEMENT ({error_msg})")
                    self.active_position = None
            except Exception as e:
                print(f"[Execution] [LIVE MODE] Order exception: {e}")
                db.update_trade_pnl(db_trade_id, 0.0, f"FAILED_PLACEMENT_ERR ({e})")
                self.active_position = None
                
        return self.active_position

    def process_tick(self, current_spot: float, current_premium: float) -> Optional[Dict[str, Any]]:
        """
        Update position PnL based on current options premium and check exit rules.
        Exits when stop loss or target triggers.
        """
        if not self.active_position:
            return None
            
        pos = self.active_position
        
        # 1. If status is PENDING, poll entry order status
        if pos["status"] == "PENDING":
            entry_order_id = pos["entry_order_id"]
            if not entry_order_id:
                # In paper trading or fallback, immediately fill
                pos["status"] = "OPEN"
                db.update_trade_pnl(self.trade_id_in_db, 0.0, "FILLED")
                print(f"[Execution] Order filled (no order ID). Position is now OPEN.")
            else:
                status = self.query_order_status(entry_order_id)
                print(f"[Execution] Polled entry order {entry_order_id} status: {status}")
                if status == "FILLED":
                    pos["status"] = "OPEN"
                    db.update_trade_pnl(self.trade_id_in_db, 0.0, "FILLED")
                    print(f"[Execution] 🟢 Order {entry_order_id} FILLED on exchange! Position is now OPEN.")
                elif status in ["REJECTED", "CANCELLED"]:
                    print(f"[Execution] ❌ Order {entry_order_id} was {status}. Cancelling position.")
                    db.update_trade_pnl(self.trade_id_in_db, 0.0, status)
                    exit_info = {
                        "strike": pos["strike_selected"],
                        "strike_name": pos["strike_name"],
                        "exit_reason": f"ENTRY_ORDER_{status}",
                        "exit_premium": 0.0,
                        "realized_pnl": 0.0,
                        "entry_premium": pos["entry_premium"],
                        "pnl_percent": 0.0
                    }
                    self.active_position = None
                    self.trade_id_in_db = None
                    return exit_info
                elif status == "UNKNOWN":
                    # Network issue, wait and try next tick
                    pass
                else:
                    # Still PENDING
                    pass
            return None
            
        # 2. If status is EXITING, poll exit order status
        if pos["status"] == "EXITING":
            exit_order_id = pos["exit_order_id"]
            if not exit_order_id:
                # Immediate fill fallback
                exit_info = pos["exit_info"]
                if self.trade_id_in_db:
                    db.update_trade_pnl(self.trade_id_in_db, exit_info["realized_pnl"], "CLOSED")
                self.active_position = None
                self.trade_id_in_db = None
                print(f"[Execution] Exit order filled (no order ID). Position is now CLOSED.")
                return exit_info
            else:
                status = self.query_order_status(exit_order_id)
                print(f"[Execution] Polled exit order {exit_order_id} status: {status}")
                if status == "FILLED":
                    exit_info = pos["exit_info"]
                    if self.trade_id_in_db:
                        db.update_trade_pnl(self.trade_id_in_db, exit_info["realized_pnl"], "CLOSED")
                    self.active_position = None
                    self.trade_id_in_db = None
                    print(f"[Execution] 🔴 Exit order {exit_order_id} FILLED on exchange! Position is now CLOSED.")
                    return exit_info
                elif status in ["REJECTED", "CANCELLED"]:
                    print(f"[Execution] ⚠️ WARNING: Exit order {exit_order_id} was {status}!")
                    exit_info = pos["exit_info"]
                    exit_info["exit_reason"] = f"EXIT_ORDER_{status}"
                    if self.trade_id_in_db:
                        db.update_trade_pnl(self.trade_id_in_db, exit_info["realized_pnl"], f"CLOSED_{status}")
                    self.active_position = None
                    self.trade_id_in_db = None
                    return exit_info
                else:
                    # Still EXITING (pending fill)
                    pass
            return None

        # 3. Position is OPEN, compute floating PnL and check exit rules
        pos["current_premium"] = current_premium
        points_diff = current_premium - pos["entry_premium"]
        floating_pnl = points_diff * pos["qty"]
        pos["pnl"] = round(floating_pnl, 2)
        
        # Determine exit trigger
        exit_reason = None
        exit_premium = 0.0
        
        if current_premium <= pos["stop_loss_premium"]:
            exit_reason = "STOP_LOSS"
            exit_premium = pos["stop_loss_premium"]
        elif current_premium >= pos["target_premium"]:
            exit_reason = "TARGET"
            exit_premium = pos["target_premium"]
            
        if exit_reason:
            realized_pnl = (exit_premium - pos["entry_premium"]) * pos["qty"]
            if exit_reason == "STOP_LOSS":
                print(f"\n[Execution] 🛑 STOP LOSS TRIGGERED for {pos['strike_name']} ({pos['strike_selected']})!")
            else:
                print(f"\n[Execution] 🎯 TARGET HIT for {pos['strike_name']} ({pos['strike_selected']})!")
                
            security_id = pos["strike_selected"]
            if security_id.startswith("NFO_"):
                security_id = security_id.split("_")[1]
                
            exit_info = {
                "strike": pos["strike_selected"],
                "strike_name": pos["strike_name"],
                "exit_reason": exit_reason,
                "exit_premium": exit_premium,
                "realized_pnl": realized_pnl,
                "entry_premium": pos["entry_premium"],
                "pnl_percent": round((exit_premium - pos["entry_premium"]) / pos["entry_premium"] * 100, 2)
            }
            
            if PAPER_TRADING_MODE:
                print(f"[Execution] [PAPER MODE] Exit order simulated at {exit_premium:.2f} INR.")
                if self.trade_id_in_db:
                    db.update_trade_pnl(self.trade_id_in_db, realized_pnl, "CLOSED")
                self.active_position = None
                self.trade_id_in_db = None
                return exit_info
            else:
                print(f"[Execution] [LIVE MODE] Placing exit order on INDstocks for {pos['strike_name']} at {exit_premium:.2f}...")
                headers = {
                    "Authorization": INDSTOCKS_TOKEN,
                    "Content-Type": "application/json"
                }
                payload = {
                    "txn_type": "SELL",
                    "exchange": "NSE",
                    "segment": "DERIVATIVE",
                    "product": "MARGIN",
                    "order_type": "LIMIT",
                    "limit_price": float(exit_premium),
                    "validity": "DAY",
                    "security_id": str(security_id),
                    "qty": int(pos["qty"]),
                    "is_amo": False,
                    "algo_id": "99999"
                }
                
                try:
                    response = requests.post("https://api.indstocks.com/order", json=payload, headers=headers, timeout=10)
                    if response.status_code == 200:
                        res_data = response.json()
                        order_id = res_data.get("data", {}).get("order_id", "LIVE_EXIT_OK")
                        print(f"[Execution] [LIVE MODE] Exit order placed successfully! Order ID: {order_id}")
                        pos["status"] = "EXITING"
                        pos["exit_order_id"] = order_id
                        pos["exit_info"] = exit_info
                        if self.trade_id_in_db:
                            db.update_trade_pnl(self.trade_id_in_db, realized_pnl, "EXITING")
                    else:
                        print(f"[Execution] [LIVE MODE] Exit placement failed with status {response.status_code}: {response.text}")
                        # Fallback to direct close if placement failed completely
                        if self.trade_id_in_db:
                            db.update_trade_pnl(self.trade_id_in_db, realized_pnl, "CLOSED")
                        self.active_position = None
                        self.trade_id_in_db = None
                        return exit_info
                except Exception as e:
                    print(f"[Execution] [LIVE MODE] Exit exception: {e}")
                    if self.trade_id_in_db:
                        db.update_trade_pnl(self.trade_id_in_db, realized_pnl, "CLOSED")
                    self.active_position = None
                    self.trade_id_in_db = None
                    return exit_info
                    
        return None
