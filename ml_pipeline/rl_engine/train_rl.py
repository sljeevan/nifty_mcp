import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "ml_pipeline", "rl_engine"))

from nifty_env import NiftyTradingEnv

def train_agent():
    print("[RL Trainer] Initializing Gymnasium environment...")
    csv_path = os.path.join(BASE_DIR, "scratch", "nifty_features.csv")
    env = NiftyTradingEnv(csv_path=csv_path)
    
    # Run sanity checks
    check_env(env, warn=True)
    
    # Configure PPO neural network agent
    # We use a Multi-Layer Perceptron (MLP) policy network
    print("[RL Trainer] Building PPO neural network agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Entropy coefficient (encourages exploration)
        verbose=1,
        seed=42
    )
    
    # Train policy
    print("[RL Trainer] Training reinforcement learning model for 250,000 steps...")
    model.learn(total_timesteps=250000)
    
    # Save the model
    model_output_path = os.path.join(BASE_DIR, "scratch", "ppo_nifty.zip")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    print(f"[RL Trainer] Trained RL agent saved successfully to {model_output_path}")
    
    # Run out-of-sample backtest
    print("\n[RL Trainer] Running backtest evaluation on out-of-sample test set...")
    df = pd.read_csv(csv_path)
    split_idx = int(len(df) * 0.8)
    
    # Evaluate over test partition index range
    test_env = NiftyTradingEnv(csv_path=csv_path)
    test_env.current_idx = split_idx
    test_env.max_idx = len(df) - 15
    
    obs, info = test_env.reset()
    test_env.current_idx = split_idx # Enforce start index override
    
    done = False
    trades_pnl = []
    capital_curve = [test_env.account_balance]
    
    trade_count = 0
    wins = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        # Track balances
        capital_curve.append(test_env.account_balance)
        done = terminated or truncated
        
        # Track realized trades (when balance increases or decreases from previous step)
        if len(capital_curve) >= 2:
            diff = capital_curve[-1] - capital_curve[-2]
            # Since stepping subtracts commission on trade opening, check for exit returns
            if abs(diff) > 21.0: # Exceeds flat commission cost
                trades_pnl.append(diff)
                trade_count += 1
                if diff > 0:
                    wins += 1
                    
    total_return = (test_env.account_balance - test_env.capital) / test_env.capital * 100
    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0.0
    
    print("\n" + "=" * 50)
    print("📊 PPO RL AGENT BACKTEST REPORT 📊")
    print("=" * 50)
    print(f"Total Return:     {total_return:.2f}%")
    print(f"Ending Balance:   {test_env.account_balance:.2f} INR")
    print(f"Total Trades:     {trade_count}")
    print(f"Win Rate:         {win_rate:.2f}%")
    
    if len(trades_pnl) > 1:
        # Annualized Sharpe ratio of trades
        avg_trade = np.mean(trades_pnl)
        std_trade = np.std(trades_pnl)
        sharpe = (avg_trade / (std_trade + 1e-9)) * np.sqrt(252) # assuming 252 trade sequences
        print(f"Trade Sharpe:     {sharpe:.2f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    train_agent()
