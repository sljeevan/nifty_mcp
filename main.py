import time
import argparse
from datetime import datetime
from core_orchestrator import NiftyTradingSystem
from config import BASE_CAPITAL, NIFTY_SYMBOL, HUMAN_IN_THE_LOOP
import db

def run_simulation():
    print("\n" + "=" * 60)
    print("📈 RUNNING NIFTY OPTIONS TRADER - VALIDATION SIMULATION 📈")
    print("=" * 60)
    print(f"Base Capital: {BASE_CAPITAL:.2f} INR")
    print(f"Risk Per Trade Limit: 2.0% (Max {BASE_CAPITAL * 0.02:.2f} INR)")
    print("Expectancy: Strict 3:1 Reward-to-Risk")
    print("Pattern Scan: Bullish Break of Structure (BOS) after Liquidity Sweep")
    print("=" * 60 + "\n")
    
    # Initialize trading system (Auto-approve to run automated simulation)
    system = NiftyTradingSystem(human_in_the_loop=False, base_capital=BASE_CAPITAL)
    
    # Generated test bars representing:
    # 1. Sideways range (establishing swing high around 23220, swing low around 23200)
    # 2. Liquidity Sweep (dipping to 23190, but closing back above 23200)
    # 3. Strength BOS (breaking above 23220 to close at 23225)
    # 4. Continuation to target (premium rising to target level)
    # Prepend 30 baseline bars to build history for ML/RL checks (min 35 bars required)
    simulated_bars = [{"open": 23200.0, "high": 23205.0, "low": 23195.0, "close": 23200.0} for _ in range(30)] + [
        # Sideways range setup
        {"open": 23200.0, "high": 23210.0, "low": 23200.0, "close": 23205.0},
        {"open": 23205.0, "high": 23215.0, "low": 23202.0, "close": 23212.0},
        {"open": 23212.0, "high": 23220.0, "low": 23208.0, "close": 23215.0}, # Swing high formed (23220)
        {"open": 23215.0, "high": 23218.0, "low": 23204.0, "close": 23208.0},
        {"open": 23208.0, "high": 23212.0, "low": 23200.0, "close": 23203.0}, # Swing low formed (23200)
        {"open": 23203.0, "high": 23208.0, "low": 23202.0, "close": 23206.0},
        {"open": 23206.0, "high": 23210.0, "low": 23203.0, "close": 23205.0},
        
        # Liquidity Sweep bar
        {"open": 23205.0, "high": 23208.0, "low": 23190.0, "close": 23205.0}, # Sweeps low (23190 < 23200), reclaims close
        
        # Build up
        {"open": 23205.0, "high": 23212.0, "low": 23204.0, "close": 23210.0},
        {"open": 23210.0, "high": 23216.0, "low": 23208.0, "close": 23215.0},
        
        # Break of Structure bar (BOS!)
        {"open": 23215.0, "high": 23228.0, "low": 23214.0, "close": 23225.0}, # Closes above Swing High 23220 -> TRIGGER!
        
        # Follow-through (climbing to target)
        {"open": 23225.0, "high": 23245.0, "low": 23222.0, "close": 23240.0},
        {"open": 23240.0, "high": 23270.0, "low": 23238.0, "close": 23265.0},
        {"open": 23265.0, "high": 23310.0, "low": 23260.0, "close": 23305.0},
        {"open": 23305.0, "high": 23360.0, "low": 23300.0, "close": 23355.0}, # Premium should cross target here
    ]
    
    for idx, bar in enumerate(simulated_bars):
        print(f"\n[Time: {idx+1:02d}] Nifty Spot Candle -> O: {bar['open']:.2f} | H: {bar['high']:.2f} | L: {bar['low']:.2f} | C: {bar['close']:.2f}")
        result = system.tick(simulated_bar=bar)
        
        if result:
            event = result.get("event")
            if event == "TRADE_OPENED":
                pos = result["data"]
                print(f"🟢 POSITION OPENED: {pos['strike_selected']} @ {pos['entry_premium']} INR")
            elif event == "TRADE_CLOSED":
                exit_data = result["data"]
                print(f"🔴 POSITION CLOSED: {exit_data['strike']} via {exit_data['exit_reason']} @ {exit_data['exit_premium']} INR")
                print(f"💰 Realized Profit/Loss: {exit_data['realized_pnl']:.2f} INR ({exit_data['pnl_percent']}% return)")
                break
                
        time.sleep(0.5)  # Simulate short block intervals
        
    print("\n" + "=" * 60)
    print("📊 POST-SIMULATION AUDIT REPORT 📊")
    print("=" * 60)
    history = db.get_trades_history()
    for t in history:
        print(f"Trade ID {t['id']}: Strike {t['strike_selected']} | Entry: {t['entry_premium']:.2f} | Stop: {t['stop_loss_premium']:.2f} | Target: {t['target_premium']:.2f}")
        print(f"          Status: {t['status']} | Realized PnL: {t['realized_pnl']:.2f} INR")
        if t['validation_error']:
            print(f"          Risk Exception Msg: {t['validation_error']}")
    print("=" * 60 + "\n")

def run_live():
    print("\n" + "=" * 60)
    print("🟢 RUNNING NIFTY OPTIONS TRADER - LIVE FEED MODE 🟢")
    print("=" * 60)
    print("Subscribing to Nifty 50 live feed...")
    
    system = NiftyTradingSystem(human_in_the_loop=HUMAN_IN_THE_LOOP, base_capital=BASE_CAPITAL)
    
    print("[Live] Active event loop started. Press Ctrl+C to terminate.")
    print("=" * 60 + "\n")
    
    try:
        while True:
            # Query Yahoo Finance real-time spot price
            spot_price = system.data_provider.get_nifty_spot()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Nifty Spot Price: {spot_price:.2f} INR", end="\r")
            
            result = system.tick()
            if result:
                event = result.get("event")
                if event == "TRADE_OPENED":
                    pos = result["data"]
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 POSITION OPENED: {pos['strike_selected']} @ {pos['entry_premium']} INR")
                elif event == "TRADE_CLOSED":
                    exit_data = result["data"]
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔴 POSITION CLOSED: {exit_data['strike']} via {exit_data['exit_reason']} @ {exit_data['exit_premium']} INR")
                    print(f"Realized PnL: {exit_data['realized_pnl']:.2f} INR")
            
            time.sleep(2.0)  # Polling interval
    except KeyboardInterrupt:
        print("\n[Live] Terminated by operator. Closing database handles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty 50 Intraday Options Trader System")
    parser.add_argument("--mode", choices=["live", "sim"], default="sim", help="Execution mode: live or sim (default)")
    args = parser.parse_args()
    
    if args.mode == "sim":
        run_simulation()
    else:
        run_live()
