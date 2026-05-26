"""
News sentiment fetcher.
Sources:
  - CryptoPanic free API  → BTC, ETH
  - Alpha Vantage (optional free key) → BTC, ETH, XAU/Gold
Sentiment scored from API vote counts + headline keyword matching.
Results are cached for NEWS_CACHE_MINUTES to avoid hammering free-tier limits.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol mappings
# ---------------------------------------------------------------------------

# CryptoPanic uses lowercase currency codes
CRYPTOPANIC_CODES = {
    "BTCUSD":  "BTC",
    "ETHUSD":  "ETH",
    "XAUTUSD": None,   # gold — handled by Alpha Vantage
}

# Alpha Vantage ticker format
ALPHAVANTAGE_TICKERS = {
    "BTCUSD":  "CRYPTO:BTC",
    "ETHUSD":  "CRYPTO:ETH",
    "XAUTUSD": "FOREX:XAU",
}

# ---------------------------------------------------------------------------
# Sentiment keywords
# ---------------------------------------------------------------------------

BULLISH_WORDS = {
    "surge", "rally", "bull", "bullish", "rise", "rising", "gain", "gains",
    "pump", "breakout", "adoption", "partnership", "etf", "approval", "upgrade",
    "high", "all-time", "accumulate", "buy", "long", "support", "recovery",
    "rebound", "upside", "outperform", "strong", "positive", "optimistic",
    "institutional", "hodl", "inflow", "halving", "launch", "milestone",
}

BEARISH_WORDS = {
    "crash", "dump", "bear", "bearish", "fall", "falling", "drop", "dropping",
    "plunge", "plunging", "hack", "hacked", "ban", "banned", "sell", "warning",
    "fear", "liquidation", "liquidated", "correction", "concern", "sec",
    "lawsuit", "fraud", "scam", "collapse", "crisis", "outflow", "loss",
    "losses", "decline", "weakness", "overvalued", "sell-off", "risk",
}


def _score_headline(text: str) -> float:
    """Return a score in [-1, +1]: positive = bullish, negative = bearish."""
    words = set(text.lower().split())
    bull = len(words & BULLISH_WORDS)
    bear = len(words & BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    label:      str           # "BULLISH" | "BEARISH" | "NEUTRAL"
    score:      float         # -1.0 … +1.0
    bull_count: int           # headlines with bullish lean
    bear_count: int           # headlines with bearish lean
    total:      int           # total headlines scanned
    headlines:  list[str] = field(default_factory=list)   # top 3
    source:     str = ""


def _label(score: float) -> str:
    if score >= 0.2:
        return "BULLISH"
    if score <= -0.2:
        return "BEARISH"
    return "NEUTRAL"


NEUTRAL_RESULT = SentimentResult(
    label="NEUTRAL", score=0.0, bull_count=0, bear_count=0, total=0,
    headlines=[], source="unavailable",
)

# ---------------------------------------------------------------------------
# Cache  { symbol: (timestamp, SentimentResult) }
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, SentimentResult]] = {}


def _cached(symbol: str) -> Optional[SentimentResult]:
    entry = _cache.get(symbol)
    if entry is None:
        return None
    ts, result = entry
    if time.time() - ts < config.NEWS_CACHE_MINUTES * 60:
        return result
    return None


def _store(symbol: str, result: SentimentResult) -> None:
    _cache[symbol] = (time.time(), result)


# ---------------------------------------------------------------------------
# CryptoPanic  (no registration needed for basic feed)
# ---------------------------------------------------------------------------

CRYPTOPANIC_URL = "https://cryptopanic.com/api/free/v1/posts/"


def _fetch_cryptopanic(currency_code: str) -> Optional[SentimentResult]:
    try:
        params = {
            "auth_token": "free",
            "currencies": currency_code,
            "filter":     "hot",
            "public":     "true",
        }
        resp = requests.get(
            CRYPTOPANIC_URL, params=params,
            timeout=config.REQUEST_TIMEOUT,
            headers={"User-Agent": "DeltaSignalBot/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("CryptoPanic fetch failed (%s): %s", currency_code, exc)
        return None

    posts = data.get("results", [])
    if not posts:
        return None

    # Limit to recent posts (up to NEWS_SENTIMENT_HOURS)
    cutoff = time.time() - config.NEWS_SENTIMENT_HOURS * 3600
    scores, top_headlines = [], []
    bull_count = bear_count = 0

    for post in posts[:30]:
        title = post.get("title", "")
        if not title:
            continue

        # Use CryptoPanic's own vote counts where available
        votes = post.get("votes") or {}
        api_positive = int(votes.get("positive", 0) or 0)
        api_negative = int(votes.get("negative", 0) or 0)

        kw_score = _score_headline(title)

        # Blend: API votes weighted more heavily if present
        if api_positive + api_negative > 0:
            api_score = (api_positive - api_negative) / (api_positive + api_negative)
            score = 0.4 * kw_score + 0.6 * api_score
        else:
            score = kw_score

        scores.append(score)
        if score > 0.05:
            bull_count += 1
        elif score < -0.05:
            bear_count += 1

        if len(top_headlines) < 3:
            top_headlines.append(title[:100])

    if not scores:
        return None

    avg_score = sum(scores) / len(scores)
    return SentimentResult(
        label=_label(avg_score),
        score=round(avg_score, 3),
        bull_count=bull_count,
        bear_count=bear_count,
        total=len(scores),
        headlines=top_headlines,
        source="CryptoPanic",
    )


# ---------------------------------------------------------------------------
# Alpha Vantage  (optional; requires free API key in config)
# ---------------------------------------------------------------------------

ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"


def _fetch_alphavantage(ticker: str) -> Optional[SentimentResult]:
    key = getattr(config, "ALPHA_VANTAGE_KEY", "")
    if not key:
        return None

    try:
        params = {
            "function":     "NEWS_SENTIMENT",
            "tickers":      ticker,
            "limit":        "30",
            "apikey":       key,
        }
        resp = requests.get(
            ALPHAVANTAGE_URL, params=params,
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Alpha Vantage fetch failed (%s): %s", ticker, exc)
        return None

    feed = data.get("feed", [])
    if not feed:
        return None

    scores, top_headlines = [], []
    bull_count = bear_count = 0

    for item in feed[:30]:
        title = item.get("title", "")
        # Alpha Vantage provides overall_sentiment_score in [-1, +1]
        try:
            av_score = float(item.get("overall_sentiment_score", 0))
        except (TypeError, ValueError):
            av_score = _score_headline(title)

        scores.append(av_score)
        if av_score > 0.05:
            bull_count += 1
        elif av_score < -0.05:
            bear_count += 1
        if len(top_headlines) < 3:
            top_headlines.append(title[:100])

    if not scores:
        return None

    avg_score = sum(scores) / len(scores)
    return SentimentResult(
        label=_label(avg_score),
        score=round(avg_score, 3),
        bull_count=bull_count,
        bear_count=bear_count,
        total=len(scores),
        headlines=top_headlines,
        source="AlphaVantage",
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_sentiment(symbol: str) -> SentimentResult:
    """
    Return news sentiment for symbol.
    Tries cache first, then CryptoPanic, then Alpha Vantage, then neutral fallback.
    Never raises — trading signals must not break due to news fetch failures.
    """
    cached = _cached(symbol)
    if cached is not None:
        return cached

    result = None

    cp_code = CRYPTOPANIC_CODES.get(symbol)
    if cp_code:
        result = _fetch_cryptopanic(cp_code)

    if result is None:
        av_ticker = ALPHAVANTAGE_TICKERS.get(symbol)
        if av_ticker:
            result = _fetch_alphavantage(av_ticker)

    if result is None:
        logger.info("No news data for %s — returning neutral", symbol)
        result = NEUTRAL_RESULT

    _store(symbol, result)
    return result


def sentiment_emoji(label: str) -> str:
    return {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(label, "⚪")


def sentiment_conflicts(direction: str, sentiment: SentimentResult) -> bool:
    """True if news strongly opposes the trade direction."""
    if direction == "LONG"  and sentiment.score <= -0.4:
        return True
    if direction == "SHORT" and sentiment.score >= 0.4:
        return True
    return False
