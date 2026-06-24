import os
from dotenv import load_dotenv

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_PATH = os.path.join(BASE_DIR, "trading_state.db")
MACRO_STATE_PATH = os.path.join(BASE_DIR, "macro_state.json")

# Trading parameters
NIFTY_SYMBOL = "^NSEI"  # Nifty 50 spot index on Yahoo Finance
NIFTY_LOT_SIZE = int(os.environ.get("NIFTY_LOT_SIZE", "75"))    # Nifty 50 lot size
EXPIRY_DAY_OF_WEEK = int(os.environ.get("EXPIRY_DAY_OF_WEEK", "1"))  # Weekday index for weekly options expiry (0=Mon, 1=Tue, 2=Wed, 3=Thu, etc.)

# Risk configuration
BASE_CAPITAL = 500000.0  # Base capital in INR (default: 5 Lakhs)
MAX_CAPITAL_RISK_PCT = 0.02  # Maximum risk per trade (2%)
MIN_REWARD_TO_RISK_RATIO = 3.0  # Strict 3:1 reward-to-risk ratio

# Paper Trading & Simulation Settings
PAPER_TRADING_MODE = True
SIMULATION_TICK_INTERVAL_SEC = 1.0  # Time between ticks in paper trading loop

# Human-In-The-Loop Checkpoint settings
HUMAN_IN_THE_LOOP = False
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
INDSTOCKS_TOKEN = os.environ.get("INDSTOCKS_TOKEN", "")

# Shared file paths
INSTRUMENTS_CSV_PATH = os.environ.get("INSTRUMENTS_CSV_PATH", os.path.join(BASE_DIR, "scratch", "instruments_fno.csv"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "scratch", "dual_sweep_rf.pkl"))

# Advanced Agent configurations
BOS_BREAKOUT_WINDOW = 3       # Break of Structure breakout window (in bars)
CANDLE_DURATION_SEC = 300     # Candle accumulation period in live mode (5 minutes)

