import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Compatibility hack: Map numpy.core and its submodules to numpy._core for loading models saved on numpy 2.x in a numpy 1.x environment
import numpy.core as core
sys.modules['numpy._core'] = core
for sub in ['numeric', 'multiarray', 'umath', 'records', 'scalarmath', 'function_base']:
    try:
        sub_mod = __import__(f'numpy.core.{sub}', fromlist=[sub])
        sys.modules[f'numpy._core.{sub}'] = sub_mod
    except ImportError:
        pass

# Map numpy.random._pcg64 to numpy.random.pcg64
try:
    import numpy.random.pcg64 as pcg64
    sys.modules['numpy.random._pcg64'] = pcg64
except ImportError:
    pass

# Monkey-patch numpy's bit generator unpickler to handle class object arguments
try:
    import numpy.random._pickle as _pickle
    orig_ctor = _pickle.__bit_generator_ctor
    def patched_ctor(bit_generator_name='MT19937'):
        name_str = str(bit_generator_name)
        for known in ['PCG64DXSM', 'PCG64', 'MT19937', 'Philox', 'SFC64']:
            if known in name_str:
                return orig_ctor(known)
        return orig_ctor(bit_generator_name)
    _pickle.__bit_generator_ctor = patched_ctor
except Exception as e:
    print(f"[RL Filter] Warning: Failed to apply BitGenerator patch: {e}")

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
                import gymnasium as gym
                custom_objects = {
                    "observation_space": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
                    "action_space": gym.spaces.Discrete(4)
                }
                self.model = PPO.load(self.model_path, custom_objects=custom_objects)
                print(f"[RL Filter] Successfully loaded PPO RL agent from {self.model_path}")
            except Exception as e:
                print(f"[RL Filter] Error loading PPO model: {e}")
        else:
            print(f"[RL Filter] Warning: PPO model file not found at {self.model_path}. RL confluence filter disabled.")

    def predict_action(self, price_history: List[Dict[str, float]], in_position: float = 0.0, position_type: float = -1.0, relative_pnl: float = 0.0) -> Optional[Dict[str, Any]]:
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
        
        # 2. Append trade status dimensions (defaults to FLAT: in_position=0.0, position_type=-1.0, relative_pnl=0.0)
        status = np.array([in_position, position_type, relative_pnl], dtype=np.float32)
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
