from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from datetime import datetime
from config import NIFTY_LOT_SIZE

class NiftyTradeState(BaseModel):
    strike_selected: str = Field(description="Nifty strike contract, e.g., NIFTY26JUN23200CE")
    strike_name: str = Field(default="", description="Human-readable strike name, e.g., NIFTY 23 JUN 18600 PE")
    entry_premium: float = Field(gt=0, description="Option entry premium price")
    stop_loss_premium: float = Field(gt=0, description="Calculated hard stop-loss premium")
    target_premium: float = Field(gt=0, description="Calculated 3:1 reward premium target")
    lot_size: int = Field(default=NIFTY_LOT_SIZE, description="Fixed Nifty 50 lot size for 2026")
    base_capital: float = Field(description="Current available broker account balance")
    min_rr_ratio: float = Field(default=3.0, description="Minimum allowed Reward-to-Risk ratio")

    @model_validator(mode='after')
    def verify_expectancy_and_risk(self):
        # Rule 1: Rigidly enforce the Expectancy Math
        expected_risk = self.entry_premium - self.stop_loss_premium
        expected_reward = self.target_premium - self.entry_premium
        
        # Ensure entry is greater than stop loss (valid trade orientation)
        if expected_risk <= 0:
            raise ValueError(f"Trade rejected: Entry premium ({self.entry_premium}) must be strictly greater than stop-loss premium ({self.stop_loss_premium}).")
            
        if round(expected_reward, 4) < round(expected_risk * self.min_rr_ratio, 4):
            actual_ratio = expected_reward / expected_risk if expected_risk > 0 else 0
            raise ValueError(f"Trade rejected: Does not meet strict {self.min_rr_ratio:.2f}:1 Reward-to-Risk criteria. Expected Reward/Risk: {actual_ratio:.2f}:1")
            
        # Rule 2: Rigidly enforce capital survival limit (Max 2% Risk)
        total_cash_at_risk = expected_risk * self.lot_size
        max_allowed_risk = self.base_capital * 0.02
        if total_cash_at_risk > max_allowed_risk:
            raise ValueError(f"Trade rejected: Total cash at risk ({total_cash_at_risk:.2f} INR) exceeds 2% risk budget ({max_allowed_risk:.2f} INR) of base capital.")
            
        return self

class SpotSetup(BaseModel):
    spot_price: float = Field(gt=0, description="Current Nifty 50 Spot price")
    invalidation_price: float = Field(gt=0, description="Index invalidation/stop-loss spot price")
    setup_type: str = Field(description="Type of trade setup detected (e.g. BULLISH_BOS)")
    prob: Optional[float] = Field(default=None, description="ML probability score for trade win")
    confidence_level: Optional[str] = Field(default=None, description="Dynamic sizing confidence tier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of setup detection")

class ExecutionOrder(BaseModel):
    order_id: str
    symbol: str
    order_type: str  # LIMIT, STOP_LIMIT, TARGET
    side: str  # BUY or SELL
    price: float
    trigger_price: Optional[float] = None
    quantity: int
    status: str  # PENDING, FILLED, CANCELLED, REJECTED
    created_at: datetime = Field(default_factory=datetime.now)
