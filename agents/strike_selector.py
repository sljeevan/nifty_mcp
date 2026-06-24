import os
import json
import math
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from data_provider import DataProvider
from schema import SpotSetup
from config import NIFTY_LOT_SIZE, MACRO_STATE_PATH

class StrikeSelector:
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

    def select_strike_and_levels(self, setup: SpotSetup, base_capital: float) -> Optional[Dict[str, Any]]:
        """
        Receives structural levels and calculates:
        1. ATM option strike contract.
        2. Premium entry, stop-loss and target based on Option Delta.
        """
        spot = setup.spot_price
        invalidation = setup.invalidation_price
        
        # Enforce direction logic (setup_type "BULLISH_BOS" implies buying calls)
        option_type = "CE" if "BULLISH" in setup.setup_type else "PE"
        
        # 1. Fetch options chain
        chain = self.data_provider.get_options_chain(spot)
        
        # 2. Select closest ATM strike
        atm_contract = None
        min_distance = float('inf')
        for contract in chain:
            if contract["type"] == option_type:
                dist = abs(contract["strike"] - spot)
                if dist < min_distance:
                    min_distance = dist
                    atm_contract = contract
                    
        if not atm_contract:
            print("[Selector] Error: Could not find suitable ATM contract in options chain.")
            return None
            
        # 3. Pull contract details
        strike_symbol = atm_contract["strike_symbol"]
        strike_name = atm_contract.get("strike_name", strike_symbol)
        entry_premium = atm_contract["premium"]
        delta = abs(atm_contract["delta"])  # Use absolute delta for premium math
        
        # 4. Mathematically compute stop-loss and target using Delta
        # Option Risk Points = (Spot Entry - Spot Invalidation) * Delta
        spot_risk_points = abs(spot - invalidation)
        option_risk_points = spot_risk_points * delta
        
        # Ensure option risk is positive and reasonable (e.g. minimum 2 points premium risk)
        option_risk_points = max(2.0, option_risk_points)
        
        stop_loss_premium = entry_premium - option_risk_points
        
        # If stop loss goes below zero, floor it at 2.0 INR (liquidation floor)
        if stop_loss_premium <= 0:
            stop_loss_premium = 2.0
            option_risk_points = entry_premium - stop_loss_premium
            
        # Load recommended Reward-to-Risk ratio from Sentinel Cache
        recommended_rr = 3.0 # Default fallback
        if os.path.exists(MACRO_STATE_PATH):
            try:
                with open(MACRO_STATE_PATH, "r") as f:
                    bias_data = json.load(f)
                    updated_at_str = bias_data.get("updated_at")
                    if updated_at_str:
                        updated_at = datetime.fromisoformat(updated_at_str)
                        if updated_at.tzinfo is not None:
                            updated_at = updated_at.replace(tzinfo=None)
                        if datetime.now() - updated_at > timedelta(hours=18):
                            print(f"[Selector] Warning: macro_state.json is stale (>18h, updated at {updated_at_str}). Using default R:R 3.0:1.")
                            recommended_rr = 3.0
                        else:
                            recommended_rr = bias_data.get("recommended_rr", 3.0)
                            print(f"[Selector] Loaded Recommended Reward-to-Risk Ratio: {recommended_rr}:1")
                    else:
                        recommended_rr = bias_data.get("recommended_rr", 3.0)
                        print(f"[Selector] Loaded Recommended Reward-to-Risk Ratio: {recommended_rr}:1")
            except Exception as e:
                print(f"[Selector] Warning: Failed to load recommended R:R ratio: {e}")

        # Target Premium is based on the dynamic recommended R:R ratio
        target_premium = entry_premium + (option_risk_points * recommended_rr)
        
        print(f"[Selector] Selected ATM Strike: {strike_name} ({strike_symbol}) (Spot: {spot}, Strike: {atm_contract['strike']})")
        print(f"[Selector] Option Delta: {delta}")
        print(f"[Selector] Spot Risk: {spot_risk_points:.2f} pts | Option Risk: {option_risk_points:.2f} pts")
        print(f"[Selector] Premium Levels -> Entry: {entry_premium:.2f} | Stop-Loss: {stop_loss_premium:.2f} | Target: {target_premium:.2f}")
        
        # Dynamic option quantity sizing based on ML confidence levels
        qty = NIFTY_LOT_SIZE
        if hasattr(setup, "confidence_level") and setup.confidence_level:
            if setup.confidence_level == "High Confidence":
                qty = 500
            elif setup.confidence_level == "Moderate Confidence":
                qty = 375
                
        # Risk-aware capping: Ensure quantity does not violate the 2% risk budget
        max_allowed_cash_risk = base_capital * 0.02
        max_qty_by_risk = math.floor(max_allowed_cash_risk / option_risk_points)
        
        # Round down to nearest multiple of Nifty Lot Size for exchange compliance
        if qty > max_qty_by_risk:
            scaled_qty = max(NIFTY_LOT_SIZE, (max_qty_by_risk // NIFTY_LOT_SIZE) * NIFTY_LOT_SIZE)
            if scaled_qty < qty:
                print(f"[Selector] Risk-aware Cap: Scaled down quantity from {qty} to {scaled_qty} units to respect 2% risk budget ({max_allowed_cash_risk:.2f} INR).")
                qty = scaled_qty
                
        print(f"[Selector] Dynamic Sizing applied: {qty} units ({qty // NIFTY_LOT_SIZE} lots) based on confidence tier: '{getattr(setup, 'confidence_level', 'Default')}'")
        
        return {
            "strike_selected": strike_symbol,
            "strike_name": strike_name,
            "entry_premium": round(entry_premium, 2),
            "stop_loss_premium": round(stop_loss_premium, 2),
            "target_premium": round(target_premium, 2),
            "lot_size": qty,  # Dynamic sizing mapped to lot_size field
            "base_capital": base_capital,
            "min_rr_ratio": round(recommended_rr, 2)
        }


