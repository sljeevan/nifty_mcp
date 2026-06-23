import os
import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add tradingview-mcp src path dynamically to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "tradingview-mcp-india", "src"))
sys.path.insert(0, BASE_DIR)

from config import MACRO_STATE_PATH
from data_provider import DataProvider

class MacroSentinel:
    def __init__(self):
        self.data_provider = DataProvider()
        
    def fetch_market_context(self) -> Dict[str, Any]:
        """
        Fetch fresh market sentiment and news feeds.
        """
        # Dynamic imports from tradingview_mcp library
        try:
            from tradingview_mcp.core.services.news_service import fetch_news
            from tradingview_mcp.core.services.sentiment_service import analyze_sentiment
        except ImportError as e:
            print(f"[Sentinel] Error importing news/sentiment services: {e}")
            sys.exit(1)
            
        print("[Sentinel] Fetching live financial news feeds...")
        news_data = fetch_news(symbol="Nifty", category="india", limit=10)
        
        print("[Sentinel] Analyzing Reddit sentiment for Nifty...")
        reddit_data = analyze_sentiment(symbol="Nifty", category="india", limit=20)
        
        print("[Sentinel] Querying Nifty spot index prices...")
        try:
            spot_price = self.data_provider.get_nifty_spot()
        except Exception as e:
            print(f"[Sentinel] Warning: Could not fetch spot price, defaulting: {e}")
            spot_price = 23225.0
            
        return {
            "news": news_data,
            "reddit": reddit_data,
            "spot_price": spot_price,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_bias_rules(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback heuristic analyzer when no LLM API keys are configured.
        Uses keyword-based sentiment weighting and Reddit score thresholds.
        """
        print("[Sentinel] Running rule-based sentiment heuristics...")
        
        pos_keywords = {"gain", "rise", "rally", "up", "bullish", "jump", "positive", "growth", "high", "rebound", "soar", "record"}
        neg_keywords = {"fall", "drop", "decline", "down", "bearish", "slump", "negative", "loss", "low", "crash", "worry", "fear"}
        
        news_score = 0.0
        news_items = context.get("news", [])
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            pos_count = sum(1 for w in pos_keywords if w in text)
            neg_count = sum(1 for w in neg_keywords if w in text)
            news_score += (pos_count - neg_count)
            
        if news_items:
            news_score /= len(news_items)
            
        reddit_data = context.get("reddit", {})
        reddit_score = reddit_data.get("sentiment_score", 0.0) # -1.0 to 1.0
        
        # Combine normalized scores (60% weight Reddit, 40% weight news)
        # Normalize news_score roughly (capping between -1.5 and 1.5 then rescaling)
        capped_news = max(-1.5, min(1.5, news_score)) / 1.5
        combined_score = (reddit_score * 0.6) + (capped_news * 0.4)
        
        # Determine bias and recommended Reward-to-Risk ratio
        if combined_score > 0.3:
            bias = "BULLISH"
            recommended_rr = 3.5
        elif combined_score > 0.08:
            bias = "BULLISH"
            recommended_rr = 3.0
        elif combined_score < -0.3:
            bias = "BEARISH"
            recommended_rr = 3.5
        elif combined_score < -0.08:
            bias = "BEARISH"
            recommended_rr = 3.0
        else:
            bias = "CHOPPY"
            recommended_rr = 2.0
            
        return {
            "bias": bias,
            "score": round(combined_score, 3),
            "confidence": 0.7,
            "recommended_rr": recommended_rr,
            "reasoning": f"Rule-based consensus. News sentiment: {capped_news:.2f}, Reddit sentiment: {reddit_score:.2f}.",
            "updated_at": datetime.now().isoformat()
        }

    def analyze_bias_llm(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Advanced LLM analysis using Gemini, OpenRouter, or OpenAI APIs if keys exist.
        """
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not (gemini_key or openai_key or openrouter_key):
            print("[Sentinel] No AI API keys found. Skipping LLM sentiment analysis.")
            return None

        # Build prompt
        news_summaries = []
        for i, item in enumerate(context["news"]):
            news_summaries.append(f"{i+1}. {item['title']} - {item['summary'][:150]}")
            
        reddit_summary = (
            f"Reddit Mood: {context['reddit'].get('sentiment_label', 'Neutral')} "
            f"(Score: {context['reddit'].get('sentiment_score', 0.0)}, "
            f"Bullish posts: {context['reddit'].get('bullish_count', 0)}, "
            f"Bearish posts: {context['reddit'].get('bearish_count', 0)})"
        )
        
        prompt = f"""You are an expert financial sentiment analyst specializing in the Indian Stock Market (Nifty 50).
Analyze the following Nifty 50 News Headlines, Reddit Sentiment, and Spot Price.
Determine the overall daily market bias as one of the following:
- BULLISH (favorable conditions for long trades / buying calls)
- BEARISH (favorable conditions for short trades / buying puts)
- CHOPPY (sideways, volatile, or uncertain conditions; not suitable for trading)

---
MARKET CONTEXT:
Nifty Spot Price: {context['spot_price']} INR

---
REDDIT DISCUSSIONS SUMMARY:
{reddit_summary}

---
NEWS HEADLINES & SUMMARIES:
{chr(10).join(news_summaries)}

---
Provide your response strictly in the following JSON format without markdown blocks:
{{
  "bias": "BULLISH" | "BEARISH" | "CHOPPY",
  "score": <float from -1.0 to 1.0 representing sentiment polarity>,
  "confidence": <float from 0.0 to 1.0 representing your confidence level>,
  "recommended_rr": <float from 2.0 to 4.0 representing recommended Reward-to-Risk ratio, e.g., 3.5 for strong trend, 3.0 for normal, 2.0 for range/choppy>,
  "reasoning": "<short description of your reasoning>"
}}"""

        try:
            if gemini_key:
                print("[Sentinel] Querying Google Gemini API...")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
                body = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "responseMimeType": "application/json"
                    }
                }
                
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    text_out = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(text_out.strip())
                    parsed["updated_at"] = datetime.now().isoformat()
                    return parsed
                    
            elif openrouter_key:
                print("[Sentinel] Querying OpenRouter API (Google Gemini 3 Flash)...")
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openrouter_key}"
                }
                body = {
                    "model": "google/gemini-2.5-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
                
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    text_out = res_json["choices"][0]["message"]["content"]
                    parsed = json.loads(text_out.strip())
                    parsed["updated_at"] = datetime.now().isoformat()
                    return parsed
                    
            elif openai_key:
                print("[Sentinel] Querying OpenAI API (GPT-4o-mini)...")
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}"
                }
                body = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
                
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    text_out = res_json["choices"][0]["message"]["content"]
                    parsed = json.loads(text_out.strip())
                    parsed["updated_at"] = datetime.now().isoformat()
                    return parsed
                    
        except Exception as e:
            print(f"[Sentinel] LLM Request failed ({e}). Falling back to rule-based analysis.")
            return None

    def run_analysis(self) -> Dict[str, Any]:
        """
        Main run loop: fetch context, analyze using LLM (if possible) or heuristics,
        and cache results to MACRO_STATE_PATH.
        """
        context = self.fetch_market_context()
        
        # Try LLM first, fall back to rule-based heuristics
        state = self.analyze_llm_with_fallback(context)
        
        # Write state to file
        try:
            with open(MACRO_STATE_PATH, "w") as f:
                json.dump(state, f, indent=4)
            print(f"[Sentinel] Daily Bias updated: {state['bias']} (Confidence: {state['confidence']}, Score: {state['score']})")
            print(f"[Sentinel] Reasoning: {state['reasoning']}")
            print(f"[Sentinel] Cached to {MACRO_STATE_PATH}")
        except Exception as e:
            print(f"[Sentinel] Error caching state: {e}")
            
        return state

    def analyze_llm_with_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        state = self.analyze_bias_llm(context)
        if not state:
            state = self.analyze_bias_rules(context)
        return state

if __name__ == "__main__":
    sentinel = MacroSentinel()
    sentinel.run_analysis()
