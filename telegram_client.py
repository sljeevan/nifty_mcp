import time
import requests
from typing import Optional
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_message(text: str) -> bool:
    """
    Sends a formatted message to the configured Telegram chat.
    Fails gracefully if the token or chat ID is missing or if the API request fails.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # Silent pass if not configured to prevent logs pollution, 
        # but print locally to terminal.
        print("[Telegram] Telegram not configured (Bot Token or Chat ID is missing in .env).")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=8)
        if response.status_code == 200:
            return True
        else:
            print(f"[Telegram] Failed to send message (HTTP {response.status_code}): {response.text}")
            return False
    except Exception as e:
        print(f"[Telegram] Connection error sending message: {e}")
        return False

def send_telegram_approval_request(trade_id: int, text: str) -> Optional[int]:
    """
    Sends a trade approval request with inline buttons (Approve / Reject).
    Returns the message ID if successful.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Telegram not configured (Bot Token or Chat ID is missing in .env).")
        return None
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": {
            "inline_keyboard": [
                [
                    {"text": "Approve ✅", "callback_data": f"approve_{trade_id}"},
                    {"text": "Reject ❌", "callback_data": f"reject_{trade_id}"}
                ]
            ]
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=8)
        if response.status_code == 200:
            res_data = response.json()
            return res_data.get("result", {}).get("message_id")
        else:
            print(f"[Telegram] Failed to send approval request (HTTP {response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"[Telegram] Connection error sending approval request: {e}")
        return None

def wait_for_telegram_approval(trade_id: int, message_id: int, original_text: str, timeout_seconds: int = 300) -> str:
    """
    Polls getUpdates for the callback query indicating approval or rejection.
    Updates the telegram message based on the user's action and returns "approve", "reject", or "timeout".
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Telegram not configured.")
        return "reject"
        
    # Get current latest update ID to avoid processing past button clicks
    start_time = time.time()
    offset = 0
    
    try:
        r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            offset = r["result"][-1]["update_id"] + 1
    except Exception as e:
        print(f"[Telegram] Error getting initial update offset: {e}")
        
    print(f"[Telegram] Waiting for Telegram response on trade {trade_id} (timeout {timeout_seconds}s)...")
    
    while time.time() - start_time < timeout_seconds:
        elapsed = time.time() - start_time
        remaining = int(timeout_seconds - elapsed)
        if remaining <= 0:
            break
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        params = {
            "offset": offset,
            "timeout": min(10, remaining)
        }
        
        try:
            resp = requests.get(url, params=params, timeout=min(15, remaining + 5)).json()
            if not resp.get("ok"):
                print(f"[Telegram] Error during polling: {resp.get('description')}")
                time.sleep(2)
                continue
                
            updates = resp.get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                
                # Check for callback query
                cb_query = update.get("callback_query")
                if not cb_query:
                    continue
                    
                cb_data = cb_query.get("data", "")
                cb_id = cb_query.get("id")
                
                if cb_data == f"approve_{trade_id}":
                    # 1. Answer callback query
                    try:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery", json={"callback_query_id": cb_id, "text": "Trade Approved!"}, timeout=5)
                    except Exception as ac_err:
                        print(f"[Telegram] Error answering callback: {ac_err}")
                        
                    # 2. Edit telegram message
                    updated_text = original_text + "\n\n✅ <b>[Nifty_mcp] Approved by operator via Telegram.</b>"
                    try:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText", json={
                            "chat_id": TELEGRAM_CHAT_ID,
                            "message_id": message_id,
                            "text": updated_text,
                            "parse_mode": "HTML",
                            "reply_markup": {"inline_keyboard": []}
                        }, timeout=5)
                    except Exception as edit_err:
                        print(f"[Telegram] Error editing message: {edit_err}")
                    return "approve"
                    
                elif cb_data == f"reject_{trade_id}":
                    # 1. Answer callback query
                    try:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery", json={"callback_query_id": cb_id, "text": "Trade Rejected!"}, timeout=5)
                    except Exception as ac_err:
                        print(f"[Telegram] Error answering callback: {ac_err}")
                        
                    # 2. Edit telegram message
                    updated_text = original_text + "\n\n❌ <b>[Nifty_mcp] Rejected by operator via Telegram.</b>"
                    try:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText", json={
                            "chat_id": TELEGRAM_CHAT_ID,
                            "message_id": message_id,
                            "text": updated_text,
                            "parse_mode": "HTML",
                            "reply_markup": {"inline_keyboard": []}
                        }, timeout=5)
                    except Exception as edit_err:
                        print(f"[Telegram] Error editing message: {edit_err}")
                    return "reject"
                    
        except Exception as e:
            print(f"[Telegram] Error during update poll: {e}")
            time.sleep(2)
            
    # Timeout occurred
    print(f"[Telegram] Approval request for trade {trade_id} timed out.")
    try:
        updated_text = original_text + "\n\n⌛ <b>[Nifty_mcp] Auto-cancelled due to operator timeout.</b>"
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText", json={
            "chat_id": TELEGRAM_CHAT_ID,
            "message_id": message_id,
            "text": updated_text,
            "parse_mode": "HTML",
            "reply_markup": {"inline_keyboard": []}
        }, timeout=5)
    except Exception as e:
        print(f"[Telegram] Error editing message on timeout: {e}")
        
    return "timeout"

