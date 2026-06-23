import os
from dotenv import load_dotenv

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_PATH = os.path.join(BASE_DIR, "trading_state.db")
MACRO_STATE_PATH = os.path.join(BASE_DIR, "macro_state.json")

# Trading parameters
NIFTY_SYMBOL = "^NSEI"  # Nifty 50 spot index on Yahoo Finance
NIFTY_LOT_SIZE = 65    # Nifty 50 lot size for 2026

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
INDSTOCKS_TOKEN = os.environ.get("INDSTOCKS_TOKEN", "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRJRCI6IjdON0lPIiwiZXhwIjoxNzgxMTQxNDAwLCJpYXQiOjE3ODEwODk4MDMsImlzcyI6ImluZG1vbmV5IiwicGFydG5lcklEIjoxMDQ5NiwidG9rZW5JRCI6Mjc0NjZ9.GPfqdj-JlTOxOPB7-PySNKJTvwVP8jzyQirvCrBo6iXMiOMWKw3ZDeOSJgPVal_JbJYgfSaxTEt7GKCsqHGoNw")
