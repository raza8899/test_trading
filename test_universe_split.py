"""Read-only diagnostic for the production NSE universe WebSocket split.

Importing this module is side-effect free; API access occurs only via main().
"""

import os
import sys

from dotenv import load_dotenv
from kiteconnect import KiteConnect


# Keep this identical to bot.py's production universe exclusion list.
BLOCKED_SERIES_SUFFIXES = (
    "-BE",
    "-BZ",
    "-BL",
    "-BT",
    "-SM",
    "-ST",
    "-MT",
    "-SG",
    "-GB",
    "-GS",
)
MAX_WS_CONNECTIONS = 3


def production_equities(rows: list[dict]) -> list[dict]:
    """Apply the same NSE EQ and blocked-series filters as production."""
    instruments: dict[str, dict] = {}

    for row in rows:
        symbol = str(row.get("tradingsymbol", "")).strip()
        segment = str(row.get("segment", "")).upper()
        instrument_type = str(row.get("instrument_type", "")).upper()

        if segment != "NSE" or instrument_type != "EQ" or not symbol:
            continue

        if any(symbol.endswith(suffix) for suffix in BLOCKED_SERIES_SUFFIXES):
            continue

        instruments[symbol] = row

    return list(instruments.values())


def main() -> int:
    load_dotenv()

    api_key = os.getenv("KITE_API_KEY", "").strip()
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    per_socket = int(
        os.getenv("WS_MAX_INSTRUMENTS_PER_CONNECTION", "2800")
    )

    if not api_key or not access_token:
        print("Missing KITE_API_KEY/KITE_ACCESS_TOKEN. Run login.py first.")
        return 1

    kite = KiteConnect(api_key=api_key, timeout=10)
    kite.set_access_token(access_token)
    eq = production_equities(kite.instruments("NSE"))
    groups = [eq[i:i + per_socket] for i in range(0, len(eq), per_socket)]

    print(f"Filtered production NSE EQ instruments: {len(eq):,}")
    print(f"Configured capacity/socket: {per_socket:,}")
    print(f"WebSockets required: {len(groups)}")

    for index, group in enumerate(groups, 1):
        print(f"  WS{index}: {len(group):,} instruments")

    if len(groups) > MAX_WS_CONNECTIONS:
        print(
            f"ERROR: More than {MAX_WS_CONNECTIONS} WebSockets "
            "would be required."
        )
        return 1

    print(
        f"\nOK: universe fits within Kite's "
        f"{MAX_WS_CONNECTIONS}-connection architecture."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
