from typing import Dict, Any, Tuple, Optional
from schema import NiftyTradeState
from pydantic import ValidationError
import db

class RiskGuardrail:
    def __init__(self, base_capital: float = 500000.0):
        self.base_capital = base_capital

    def update_balance(self, new_balance: float):
        """Update available broker account balance."""
        self.base_capital = new_balance

    def validate_trade(self, raw_trade_params: Dict[str, Any]) -> Tuple[bool, Optional[NiftyTradeState], Optional[str]]:
        """
        Receives parameters from strike selector and runs Pydantic validation.
        Enforces:
        - Strict 3:1 Reward-to-Risk ratio
        - Max 2% capital risk per trade
        """
        # Inject the current base capital to ensure freshness
        raw_trade_params["base_capital"] = self.base_capital
        
        try:
            # Pydantic validates inputs and runs model_validators
            state = NiftyTradeState(**raw_trade_params)
            
            print(f"[Risk] Compliance Check: PASSED. Risk Exposure: {(state.entry_premium - state.stop_loss_premium) * state.lot_size:.2f} INR ({(state.entry_premium - state.stop_loss_premium) * state.lot_size / self.base_capital * 100:.2f}% of capital).")
            
            return True, state, None
            
        except ValidationError as e:
            # Capture specific Pydantic verification failures
            error_msgs = [err["msg"] for err in e.errors()]
            combined_error = "; ".join(error_msgs)
            
            # Log the rejected trade in database for auditing
            # Create a mock/unvalidated object to insert into DB
            try:
                db_state = NiftyTradeState.model_construct(
                    strike_selected=raw_trade_params.get("strike_selected", "UNKNOWN"),
                    entry_premium=raw_trade_params.get("entry_premium", 0.01),
                    stop_loss_premium=raw_trade_params.get("stop_loss_premium", 0.01),
                    target_premium=raw_trade_params.get("target_premium", 0.01),
                    lot_size=raw_trade_params.get("lot_size", 65),
                    base_capital=self.base_capital
                )
                db.log_trade_attempt(db_state, "REJECTED", combined_error)
            except Exception as dbe:
                print(f"[Risk] DB Logging warning: {dbe}")
                
            print(f"[Risk] Compliance Check: REJECTED! Reason: {combined_error}")
            return False, None, combined_error
