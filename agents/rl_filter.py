import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from stable_baselines3 import PPO

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from agents.regime_classifier import RegimeClassifier

class RLFilter:
    def __init__(self):
        self.model_path = os.path.join(BASE_DIR, "scratch", "ppo_nifty.zip")
        self.model: Optional[PPO] = None
        self.classifier_helper = RegimeClassifier()  # Reuse feature computation logic
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path)
                print(f"[RL Filter] Successfully loaded PPO RL agent from {self.model_path}")
            except Exception as e:
                print(f"[RL Filter] Error loading PPO model: {e}")
        else:
            print(f"[RL Filter] Warning: PPO model file not found at {self.model_path}. RL confluence filter disabled.")

    def predict_action(self, price_history: List[Dict[str, float]]) -> Optional[Dict[str, Any]]:
        """
        Predict trading action using the trained PPO agent:
        0 = HOLD, 1 = BUY CALL, 2 = BUY PUT, 3 = CLOSE
        """
        if self.model is None:
            return None
            
        # 1. Compute the 10 stationary features
        features_df = self.classifier_helper.compute_features(price_history)
        if features_df is None or features_df.isnull().values.any():
            return None
            
        features = features_df.values[0].astype(np.float32)
        
        # 2. Append flat trade status dimensions (in_position=0, position_type=-1, relative_pnl=0.0)
        status = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        obs = np.concatenate([features, status])
        
        # 3. Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        
        action_labels = {
            0: "HOLD",
            1: "BUY CALL",
            2: "BUY PUT",
            3: "CLOSE"
        }
        
        return {
            "action": action,
            "label": action_labels.get(action, "UNKNOWN")
        }
