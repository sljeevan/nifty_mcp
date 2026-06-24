import time
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import config
from config import HUMAN_IN_THE_LOOP, BASE_CAPITAL, NIFTY_LOT_SIZE
from data_provider import DataProvider
from schema import SpotSetup, NiftyTradeState
from agents.structural_analyst import StructuralAnalyst
from agents.strike_selector import StrikeSelector
from agents.risk_guardrail import RiskGuardrail
from agents.execution_router import ExecutionRouter
from agents.regime_classifier import RegimeClassifier
from agents.rl_filter import RLFilter
from telegram_client import send_telegram_message, send_telegram_approval_request, wait_for_telegram_approval
import db

class NiftyTradingSystem:
    def __init__(self, human_in_the_loop: bool = HUMAN_IN_THE_LOOP, base_capital: float = BASE_CAPITAL):
        self.data_provider = DataProvider()
        self.analyst = StructuralAnalyst(data_provider=self.data_provider)
        self.selector = StrikeSelector(self.data_provider)
        self.risk_manager = RiskGuardrail(base_capital)
        self.execution = ExecutionRouter()
        
        # Instantiate regime classifier and RL confluence filter
        self.regime_classifier = RegimeClassifier()
        self.rl_filter = RLFilter()
        
        self.human_in_the_loop = human_in_the_loop
        self.capital = base_capital
        self.active_trade_id: Optional[int] = None
        
        # Candle accumulation state for live mode (retained for backward compatibility / fallback)
        self.current_candle_start_time: Optional[float] = None
        self.current_candle_open: Optional[float] = None
        self.current_candle_high: Optional[float] = None
        self.current_candle_low: Optional[float] = None
        self.current_candle_close: Optional[float] = None
        
        # Pre-populate analyst price history for live mode
        if not self.data_provider.is_sim:
            print("[Core] Pre-populating analyst price history with historical 5m candles...")
            candles = self.data_provider.get_historical_5m_candles(limit=100)
            if candles:
                for c in candles:
                    self.analyst.price_history.append(c)
                    self.analyst._calculate_indicators()
                print(f"[Core] Pre-populated {len(self.analyst.price_history)} historical candles.")
            else:
                print("[Core] Warning: Failed to pre-populate historical candles.")
        
    def tick(self, simulated_bar: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """
        Process a single market tick.
        Can ingest simulated bars for backtesting/testing, or fetch live data.
        """
        # 1. Fetch spot price (simulated or real)
        if simulated_bar:
            spot_price = simulated_bar["close"]
            open_p, high, low, close = (
                simulated_bar["open"],
                simulated_bar["high"],
                simulated_bar["low"],
                simulated_bar["close"]
            )
            candle_completed = True
        else:
            spot_price = self.data_provider.get_nifty_spot()
            candle_completed = False
            
            # Check active position exits on EVERY live tick
            if self.execution.active_position:
                contract_symbol = self.execution.active_position["strike_selected"]
                contract_detail = self.data_provider.get_option_by_symbol(contract_symbol, spot_price)
                if contract_detail:
                    current_premium = contract_detail["premium"]
                    exit_info = self.execution.process_tick(spot_price, current_premium)
                    if exit_info:
                        # Update available capital after trade close
                        self.capital += exit_info["realized_pnl"]
                        self.risk_manager.update_balance(self.capital)
                        self.active_trade_id = None
                        
                        # Send Telegram Alert
                        pnl_icon = "🔴" if exit_info["realized_pnl"] < 0 else "🟢"
                        msg = (
                            f"<b>{pnl_icon} [Nifty_mcp] POSITION CLOSED</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━\n"
                            f"<b>Contract:</b> {exit_info.get('strike_name', exit_info['strike'])} ({exit_info['strike']})\n"
                            f"<b>Exit Reason:</b> {exit_info['exit_reason']}\n"
                            f"<b>Exit Premium:</b> {exit_info['exit_premium']:.2f} INR (Entry: {exit_info['entry_premium']:.2f})\n"
                            f"<b>Realized PnL:</b> {exit_info['realized_pnl']:.2f} INR ({exit_info['pnl_percent']}%)\n"
                            f"<b>New Capital:</b> {self.capital:.2f} INR"
                        )
                        send_telegram_message(msg)
                        return {"event": "TRADE_CLOSED", "data": exit_info}
                return None
            
            # Fetch the latest completed 5-minute candles
            latest_candles = self.data_provider.get_historical_5m_candles(limit=5)
            if latest_candles:
                now_dt = datetime.now()
                # A 5m candle with timestamp T is completed when current time >= T + 5 minutes
                completed_candles = [c for c in latest_candles if now_dt - c["timestamp"] >= timedelta(minutes=5)]
                
                if completed_candles:
                    latest_c = completed_candles[-1]
                    last_hist_timestamp = self.analyst.price_history[-1].get("timestamp") if self.analyst.price_history else None
                    
                    if last_hist_timestamp is None or latest_c["timestamp"] > last_hist_timestamp:
                        open_p = latest_c["open"]
                        high = latest_c["high"]
                        low = latest_c["low"]
                        close = latest_c["close"]
                        candle_completed = True
                        print(f"[Core] New completed 5m candle detected: {latest_c['timestamp']} | O: {open_p:.2f} | H: {high:.2f} | L: {low:.2f} | C: {close:.2f}")
                
        # If no completed candle bar is ready to scan, skip structural analysis
        if not candle_completed:
            return None

        # 3. Feed completed candle bar to Analyst (Agent 1) to scan for setups
        setup = self.analyst.update_history(open_p, high, low, close)
        if not setup:
            return None
            
        # 3b. Run ML Regime Classifier check (LightGBM)
        regime = self.regime_classifier.predict_regime(self.analyst.price_history)
        if regime:
            print(f"[Core] Regime Classifier prediction: {regime['label']} (Prob Trending: {regime['probabilities']['TRENDING']:.2%})")
            if regime["class"] == 0:
                print(f"[Core] Setup filtered by LGBM Regime Classifier: Market is CHOPPY.")
                # Send Telegram Alert
                msg = (
                    f"<b>⚠️ [Nifty_mcp] SETUP FILTERED (CHOPPY)</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Setup Type:</b> {setup.setup_type}\n"
                    f"<b>LGBM Regime:</b> CHOPPY (Trending Prob: {regime['probabilities']['TRENDING']:.2%})"
                )
                send_telegram_message(msg)
                return {"event": "TRADE_REJECTED", "reason": "LGBM Regime Classifier: CHOPPY market"}
                
        # 3c. Run RL Confluence Filter check (PPO Model)
        rl_pred = self.rl_filter.predict_action(self.analyst.price_history)
        if rl_pred:
            print(f"[Core] RL Filter prediction: {rl_pred['label']}")
            expected_action = "BUY CALL" if setup.setup_type == "BULLISH_BOS" else "BUY PUT"
            if rl_pred["label"] != expected_action:
                print(f"[Core] Setup filtered by RL confluence check. Setup is {setup.setup_type} but RL suggests {rl_pred['label']}.")
                # Send Telegram Alert
                msg = (
                    f"<b>⚠️ [Nifty_mcp] SETUP FILTERED (RL CONFLUENCE)</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Setup Type:</b> {setup.setup_type}\n"
                    f"<b>RL Action:</b> {rl_pred['label']} (Expected: {expected_action})"
                )
                send_telegram_message(msg)
                return {"event": "TRADE_REJECTED", "reason": f"RL confluence check: expected {expected_action}, got {rl_pred['label']}"}
            
        # Setup detected! Save the signal to DB
        signal_id = db.save_signal(setup)
        print(f"\n[Core] Signal logged in DB (ID: {signal_id}).")
        
        # Send Telegram Signal Alert
        msg = (
            f"<b>⚡ [Nifty_mcp] SIGNAL DETECTED ⚡</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Spot Price:</b> {setup.spot_price:.2f} INR\n"
            f"<b>Invalidation (SL):</b> {setup.invalidation_price:.2f} INR\n"
            f"<b>Setup Type:</b> {setup.setup_type}\n"
            f"<b>ML Win Prob:</b> {getattr(setup, 'prob', 0.0):.2%}\n"
            f"<b>Confidence:</b> {getattr(setup, 'confidence_level', 'Unknown')}\n"
            f"<b>Signal ID:</b> {signal_id}"
        )
        send_telegram_message(msg)
        
        # 4. Route setup to Strike Selector (Agent 2) to compute contract levels
        raw_trade_params = self.selector.select_strike_and_levels(setup, self.capital)
        if not raw_trade_params:
            return None
            
        # 5. Route raw trade parameters to Risk Guardrail (Agent 3) for validation
        is_valid, validated_state, error_msg = self.risk_manager.validate_trade(raw_trade_params)
        if not is_valid or not validated_state:
            print(f"[Core] Trade blocked by Risk Engine: {error_msg}")
            
            # Send Telegram Alert
            msg = (
                f"<b>⚠️ [Nifty_mcp] TRADE REJECTED BY RISK ENGINE</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Reason:</b> {error_msg}\n"
                f"<b>Setup Spot:</b> {setup.spot_price:.2f} INR"
            )
            send_telegram_message(msg)
            return {"event": "TRADE_REJECTED", "reason": error_msg}
            
        # Log approval, now check Human-In-The-Loop Edge (Step 3/Philosophy)
        db_trade_id = db.log_trade_attempt(validated_state, "APPROVED")
        
        if self.human_in_the_loop:
            print("\n" + "=" * 50)
            print("🚨 HUMAN-IN-THE-LOOP CHECKPOINT: EDGE VERIFICATION 🚨")
            print("=" * 50)
            print(f"Contract:      {validated_state.strike_selected}")
            print(f"Option Entry:   {validated_state.entry_premium:.2f} INR")
            print(f"Option Stop:    {validated_state.stop_loss_premium:.2f} INR")
            print(f"Option Target:  {validated_state.target_premium:.2f} INR")
            print(f"Nifty Spot:     {setup.spot_price:.2f} INR")
            print(f"Nifty Stop:     {setup.invalidation_price:.2f} INR")
            print(f"ML Win Prob:    {getattr(setup, 'prob', 0.0):.2%}")
            print(f"Confidence:     {getattr(setup, 'confidence_level', 'Unknown')}")
            print(f"Capital Risk:   {(validated_state.entry_premium - validated_state.stop_loss_premium) * validated_state.lot_size:.2f} INR (Max 2%)")
            print("-" * 50)
            
            # Send Telegram HITL notification
            msg = (
                f"<b>🚨 [Nifty_mcp] PENDING OPERATOR APPROVAL</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Contract:</b> {validated_state.strike_name} ({validated_state.strike_selected})\n"
                f"<b>Entry Premium:</b> {validated_state.entry_premium:.2f} INR\n"
                f"<b>Stop Loss:</b> {validated_state.stop_loss_premium:.2f} INR\n"
                f"<b>Target:</b> {validated_state.target_premium:.2f} INR\n"
                f"<b>Nifty Spot:</b> {setup.spot_price:.2f} INR\n"
                f"<b>ML Win Prob:</b> {getattr(setup, 'prob', 0.0):.2%}\n"
                f"<b>Confidence:</b> {getattr(setup, 'confidence_level', 'Unknown')}\n"
                f"<b>Capital Risk:</b> {(validated_state.entry_premium - validated_state.stop_loss_premium) * validated_state.lot_size:.2f} INR"
            )
            
            # Attempt to send approval request via Telegram
            telegram_msg_id = send_telegram_approval_request(db_trade_id, msg)
            
            if telegram_msg_id is not None:
                print("[Core] Telegram approval request sent. Waiting for operator click via Telegram bot...")
                decision = wait_for_telegram_approval(db_trade_id, telegram_msg_id, msg, timeout_seconds=300)
                
                if decision != "approve":
                    print(f"[Core] Trade cancelled by operator via Telegram ({decision}). Resetting state.")
                    db.update_trade_pnl(db_trade_id, 0.0, "CANCELLED_BY_HUMAN")
                    return {"event": "TRADE_CANCELLED_BY_HUMAN"}
            else:
                # Fallback to interactive console prompt if Telegram is not configured or fails
                print("[Core] Telegram not configured/available. Falling back to console approval.")
                msg_console = msg + "\n\n<i>Please verify in the console to approve or reject.</i>"
                send_telegram_message(msg_console)
                
                try:
                    user_input = input("Approve trade routing to Broker API? (yes/no): ").strip().lower()
                    if user_input not in ["y", "yes"]:
                        print("[Core] Trade cancelled by human operator. Resetting state.")
                        db.update_trade_pnl(db_trade_id, 0.0, "CANCELLED_BY_HUMAN")
                        send_telegram_message("❌ <b>[Nifty_mcp] Trade cancelled by operator.</b>")
                        return {"event": "TRADE_CANCELLED_BY_HUMAN"}
                except Exception as ie:
                    print(f"[Core] Error reading input ({ie}). Auto-rejecting for safety.")
                    db.update_trade_pnl(db_trade_id, 0.0, "CANCELLED_BY_HUMAN")
                    send_telegram_message("❌ <b>[Nifty_mcp] Trade auto-cancelled due to operator timeout/error.</b>")
                    return {"event": "TRADE_CANCELLED_BY_HUMAN"}
                
        # 6. Execute Order (Agent 4)
        active_pos = self.execution.execute_validated_trade(validated_state, db_trade_id)
        if not active_pos:
            print("[Core] Order placement failed. Trade not opened.")
            return {"event": "TRADE_FAILED"}
            
        self.active_trade_id = db_trade_id
        
        # Determine status title for Telegram
        status_title = "POSITION OPENED" if active_pos["status"] == "OPEN" else "POSITION PLACED (PENDING)"
        
        # Send Telegram execution success alert
        msg = (
            f"<b>🟢 [Nifty_mcp] {status_title}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Contract:</b> {active_pos.get('strike_name', active_pos['strike_selected'])} ({active_pos['strike_selected']})\n"
            f"<b>Entry Premium:</b> {active_pos['entry_premium']:.2f} INR\n"
            f"<b>Target Premium:</b> {active_pos['target_premium']:.2f} INR\n"
            f"<b>Stop Loss Premium:</b> {active_pos['stop_loss_premium']:.2f} INR\n"
            f"<b>Quantity:</b> {active_pos['lot_size']} units ({active_pos['lot_size'] // NIFTY_LOT_SIZE} lots)"
        )
        send_telegram_message(msg)
        return {"event": "TRADE_OPENED", "data": active_pos}
