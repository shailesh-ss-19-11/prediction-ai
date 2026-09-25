"""
End-to-end check of the Jev integration as main.py actually uses it:
news_fetcher.get_sentiment() -> news_fetcher.assess_urgent_risk().

Unlike jev_live_check.py (which hits jev_client directly with a canned
example), this pulls real current headlines for a symbol and asks Jev
about them — the exact call path check_symbol() makes before opening a
trade.

Usage:
    export JEV_AI_API_KEY=...   # or put it in .env
    python scripts/jev_integration_check.py BTCUSD LONG
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import config           # noqa: E402
import news_fetcher      # noqa: E402


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSD"
    direction = sys.argv[2] if len(sys.argv) > 2 else "LONG"

    if not os.environ.get("JEV_AI_API_KEY"):
        print("JEV_AI_API_KEY is not set (checked environment and .env). Aborting.")
        sys.exit(1)

    # Force the gate open for this check regardless of config.py's current value.
    config.USE_AI_ENGINE = True

    print(f"Fetching current news sentiment for {symbol} ...")
    sentiment = news_fetcher.get_sentiment(symbol)
    print(f"  source={sentiment.source} label={sentiment.label} score={sentiment.score}")
    print(f"  headlines ({len(sentiment.headlines)}):")
    for h in sentiment.headlines:
        print(f"    - {h}")

    if not sentiment.headlines:
        print("\nNo headlines available right now — Jev has nothing to assess "
              "(assess_urgent_risk returns None in this case by design).")
        return

    print(f"\nAsking Jev: does this news block a {direction} trade on {symbol}?")
    urgent_prob = news_fetcher.assess_urgent_risk(symbol, direction, sentiment)

    if urgent_prob is None:
        print("Jev call failed or was skipped — check the warning logged above.")
        return

    print(f"\nanswers.urgent.noul = {urgent_prob:.3f}")
    print(f"config.JEV_URGENCY_THRESHOLD = {config.JEV_URGENCY_THRESHOLD}")
    if urgent_prob >= config.JEV_URGENCY_THRESHOLD:
        print(f"=> main.py WOULD SKIP this {direction} {symbol} trade right now.")
    else:
        print(f"=> main.py would NOT block this {direction} {symbol} trade on news grounds.")


if __name__ == "__main__":
    main()
