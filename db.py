import sqlite3
import os
import json
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from config import DB_PATH
from schema import NiftyTradeState, SpotSetup

# Global lock and persistent connection for thread safety
_db_lock = threading.Lock()
_conn = None

def get_connection():
    global _conn
    with _db_lock:
        if _conn is None:
            _conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
        return _conn

def init_db():
    conn = get_connection()
    with _db_lock:
        cursor = conn.cursor()
        
        # Signals table (Agent 1)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spot_price REAL NOT NULL,
            invalidation_price REAL NOT NULL,
            setup_type TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        ''')
        
        # Trade history & compliance logs (Agents 2, 3, 4)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strike_selected TEXT NOT NULL,
            strike_name TEXT,
            entry_premium REAL NOT NULL,
            stop_loss_premium REAL NOT NULL,
            target_premium REAL NOT NULL,
            lot_size INTEGER NOT NULL,
            base_capital REAL NOT NULL,
            status TEXT NOT NULL,  -- APPROVED, REJECTED, FILLED, CLOSED
            validation_error TEXT, -- Pydantic error msg if rejected
            realized_pnl REAL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            closed_at TEXT
        )
        ''')
        
        # Run dynamic migration in case column is missing in an existing database
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN strike_name TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        conn.commit()
     
def save_signal(setup: SpotSetup) -> int:
    conn = get_connection()
    with _db_lock:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO signals (spot_price, invalidation_price, setup_type, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (setup.spot_price, setup.invalidation_price, setup.setup_type, setup.timestamp.isoformat()))
        signal_id = cursor.lastrowid
        conn.commit()
        return signal_id
     
def log_trade_attempt(state: NiftyTradeState, status: str, error_msg: Optional[str] = None) -> int:
    conn = get_connection()
    with _db_lock:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO trades (strike_selected, strike_name, entry_premium, stop_loss_premium, target_premium, lot_size, base_capital, status, validation_error, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            state.strike_selected,
            state.strike_name,
            state.entry_premium,
            state.stop_loss_premium,
            state.target_premium,
            state.lot_size,
            state.base_capital,
            status,
            error_msg,
            datetime.now().isoformat()
        ))
        trade_id = cursor.lastrowid
        conn.commit()
        return trade_id

def update_trade_pnl(trade_id: int, pnl: float, status: str = "CLOSED"):
    conn = get_connection()
    with _db_lock:
        cursor = conn.cursor()
        cursor.execute('''
        UPDATE trades 
        SET realized_pnl = ?, status = ?, closed_at = ?
        WHERE id = ?
        ''', (pnl, status, datetime.now().isoformat(), trade_id))
        conn.commit()

def get_trades_history() -> List[Dict[str, Any]]:
    conn = get_connection()
    with _db_lock:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM trades ORDER BY id DESC')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# Auto-initialize database on import
init_db()
