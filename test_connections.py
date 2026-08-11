"""
Connectivity diagnostic only.

Checks:
- OpenAI API + configured model
- Zerodha authentication
- NSE instrument dump
- One full quote request
- One historical 5-minute candle request

NO ORDERS are placed.
NO WebSocket broad-universe trading loop is started.

The diagnostic is deliberately guarded by ``main()`` so importing this module
during unittest or pytest discovery has no side effects and makes no API calls.
"""

import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv
from kiteconnect import KiteConnect
from openai import OpenAI


def main() -> int:
    load_dotenv()

    print("=" * 72)
    print("CONNECTION TEST - NO ORDERS WILL BE PLACED")
    print("=" * 72)

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-5.6").strip()

    if not openai_key:
        print("OPENAI_API_KEY missing")
        return 1

    try:
        client = OpenAI(api_key=openai_key)
        response = client.responses.create(
            model=model,
            input="Reply with exactly: OPENAI_OK",
        )

        print("OpenAI API works")
        print(f"   Requested model: {model}")
        print(f"   Returned model : {response.model}")
        print(f"   Response       : {response.output_text}")
    except Exception as exc:
        print(f"OpenAI failed: {type(exc).__name__}: {exc}")
        return 1

    api_key = os.getenv("KITE_API_KEY", "").strip()
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()

    if not api_key or not access_token:
        print("\nKITE_API_KEY or KITE_ACCESS_TOKEN missing.")
        print("Run python login.py first.")
        return 1

    try:
        kite = KiteConnect(api_key=api_key, timeout=10)
        kite.set_access_token(access_token)
        profile = kite.profile()

        print("\nZerodha authentication works")
        print("   User:", profile.get("user_name", profile.get("user_id")))

        instruments = kite.instruments("NSE")
        eq = [
            row
            for row in instruments
            if (
                row.get("segment") == "NSE"
                and row.get("instrument_type") == "EQ"
            )
        ]

        print(f"NSE instrument API works ({len(eq):,} EQ instruments)")

        sbin = next(
            (
                row
                for row in eq
                if row.get("tradingsymbol") == "SBIN"
            ),
            None,
        )

        if not sbin:
            raise RuntimeError("Could not find SBIN in NSE instruments.")

        quote = kite.quote(["NSE:SBIN"])
        print(
            "Full quote API works: "
            f"SBIN LTP={quote['NSE:SBIN'].get('last_price')}"
        )

        now = datetime.now()
        candles = kite.historical_data(
            instrument_token=sbin["instrument_token"],
            from_date=now - timedelta(days=1),
            to_date=now,
            interval="5minute",
            continuous=False,
            oi=False,
        )

        print(
            f"Historical API works "
            f"({len(candles)} five-minute candles returned)"
        )
    except Exception as exc:
        print(f"Zerodha failed: {type(exc).__name__}: {exc}")
        return 1

    print("\nOpenAI and Zerodha Kite Connect are working.")
    print("This diagnostic placed ZERO orders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
