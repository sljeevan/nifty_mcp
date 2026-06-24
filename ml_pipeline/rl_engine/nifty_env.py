import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from config import NIFTY_LOT_SIZE

class NiftyTradingEnv(gym.Env):
    """
    Gymnasium Environment for training an RL agent to trade Nifty 50 ATM Options.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, csv_path: str = None, capital: float = 500000.0, lot_size: int = NIFTY_LOT_SIZE):
        super(NiftyTradingEnv, self).__init__()
        
        self.capital = capital
        self.lot_size = lot_size
        self.commission = 20.0 # Standard flat brokerage per order in INR
        
        # Load dataset
        if csv_path is None:
            csv_path = os.path.join(BASE_DIR, "scratch", "nifty_features.csv")
        self.df = pd.read_csv(csv_path)
        
        # Calculate stationary feature ratios
        self.df['atr_ratio'] = self.df['atr_14'] / self.df['close']
        self.df['macd_hist_ratio'] = (self.df['macd'] - self.df['macd_signal']) / self.df['close']
        
        self.feature_cols = [
            'return_1h', 'return_3h', 'return_5h', 
            'volatility_5h', 'volatility_10h', 
            'atr_ratio', 'rsi_14', 
            'bb_width', 'bb_percent', 
            'macd_hist_ratio'
        ]
        
        # State space: 10 stationary features + 3 trade status dimensions = 13 dimensions
        # 3 status dims:
        # - in_position (0 or 1)
        # - position_type (0 = CALL, 1 = PUT, -1 = FLAT)
        # - relative_floating_pnl (float representing unrealized P&L relative to capital)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # Actions: 0 = HOLD, 1 = BUY CALL, 2 = BUY PUT, 3 = CLOSE
        self.action_space = spaces.Discrete(4)
        
        # Initialize internal variables
        self.current_idx = 0
        self.max_idx = len(self.df) - 12 # Reserve room for lookahead
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_idx = np.random.randint(200, self.max_idx - 500) # Random starting point in training
        self.account_balance = self.capital
        self.in_position = 0
        self.position_type = -1 # -1 = FLAT, 0 = CALL, 1 = PUT
        self.entry_spot = 0.0
        self.entry_premium = 0.0
        self.current_premium = 0.0
        self.option_risk = 0.0
        self.holding_steps = 0
        
        obs = self._get_observation()
        info = {}
        return obs, info
        
    def _get_observation(self):
        row = self.df.iloc[self.current_idx]
        features = row[self.feature_cols].values.astype(np.float32)
        
        relative_pnl = 0.0
        if self.in_position:
            floating_pnl = (self.current_premium - self.entry_premium) * self.lot_size
            relative_pnl = floating_pnl / self.capital
            
        status = np.array([
            float(self.in_position),
            float(self.position_type),
            relative_pnl
        ], dtype=np.float32)
        
        return np.concatenate([features, status])
        
    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        
        row = self.df.iloc[self.current_idx]
        spot = row['close']
        atr = row['atr_14']
        
        # Option Pricing Sensitivity (Delta)
        delta = 0.50 # ATM delta is roughly 0.50
        
        # 1. Update position price if already in trade
        if self.in_position:
            self.holding_steps += 1
            prev_spot = self.df.iloc[self.current_idx - 1]['close']
            spot_diff = spot - prev_spot
            
            # CE gains as spot goes up, PE gains as spot goes down
            direction = 1 if self.position_type == 0 else -1
            premium_change = spot_diff * delta * direction
            
            # Apply Theta Decay (0.015 * entry premium per hour)
            decay = self.entry_premium * 0.015
            self.current_premium += (premium_change - decay)
            self.current_premium = max(2.0, self.current_premium) # option floor
            
            # Calculate current OCO stops
            option_risk_points = self.option_risk
            stop_loss = self.entry_premium - option_risk_points
            target = self.entry_premium + (option_risk_points * 3.0)
            
            # Exit Check: Stop-Loss or Target hit
            if self.current_premium <= stop_loss:
                pnl = (stop_loss - self.entry_premium) * self.lot_size - self.commission
                self.account_balance += pnl
                reward = pnl / self.capital
                self.in_position = 0
                self.position_type = -1
            elif self.current_premium >= target:
                pnl = (target - self.entry_premium) * self.lot_size - self.commission
                self.account_balance += pnl
                reward = pnl / self.capital
                self.in_position = 0
                self.position_type = -1
            # Time exit: Force close if held > 15 hours to avoid decay
            elif self.holding_steps >= 15:
                pnl = (self.current_premium - self.entry_premium) * self.lot_size - self.commission
                self.account_balance += pnl
                reward = pnl / self.capital
                self.in_position = 0
                self.position_type = -1
                
        # 2. Process Action
        # Action: 0 = HOLD, 1 = BUY CALL, 2 = BUY PUT, 3 = CLOSE
        if action == 1 and not self.in_position: # BUY CALL
            self.in_position = 1
            self.position_type = 0
            self.entry_spot = spot
            self.option_risk = (1.5 * atr) * delta # 1.5 * Spot ATR converted to Option Risk Points
            self.entry_premium = max(15.0, 0.005 * spot) # Mock premium pricing (0.5% of spot)
            self.current_premium = self.entry_premium
            self.holding_steps = 0
            reward = -self.commission / self.capital # Entry fee penalty
            
        elif action == 2 and not self.in_position: # BUY PUT
            self.in_position = 1
            self.position_type = 1
            self.entry_spot = spot
            self.option_risk = (1.5 * atr) * delta
            self.entry_premium = max(15.0, 0.005 * spot)
            self.current_premium = self.entry_premium
            self.holding_steps = 0
            reward = -self.commission / self.capital
            
        elif action == 3 and self.in_position: # Manual CLOSE
            pnl = (self.current_premium - self.entry_premium) * self.lot_size - self.commission
            self.account_balance += pnl
            reward = pnl / self.capital
            self.in_position = 0
            self.position_type = -1
            
        # Small penalty per step in position to discourage lazy trades
        if self.in_position:
            reward -= 0.0002
            
        # Move step index forward
        self.current_idx += 1
        if self.current_idx >= self.max_idx:
            terminated = True
            
        obs = self._get_observation()
        info = {"balance": self.account_balance}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
