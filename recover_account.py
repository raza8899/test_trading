"""Emergency reconciliation utility for a dedicated Zerodha bot account.

This tool intentionally touches only NSE/MIS orders and positions.  It is meant
for operator-invoked recovery after the main bot has halted.  It never enables
new entries and requires an explicit command-line confirmation phrase.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import bot
from trading_core import OrderSnapshot

CONFIRMATION = "FLATTEN_DEDICATED_ACCOUNT"


def _active_nse_mis_orders(broker: bot.KiteBroker) -> list[OrderSnapshot]:
    result: list[OrderSnapshot] = []
    for payload in broker.orders():
        snapshot = OrderSnapshot.from_payload(payload)
        if (
            not snapshot.terminal
            and snapshot.exchange == "NSE"
            and snapshot.product == "MIS"
        ):
            result.append(snapshot)
    return result


def _nonzero_nse_mis_positions(broker: bot.KiteBroker) -> dict[str, int]:
    return {
        symbol: qty
        for symbol, qty in bot.broker_mis_position_quantities(broker).items()
        if qty != 0
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cancel active NSE/MIS orders and flatten NSE/MIS positions "
        "on an account explicitly dedicated to this bot."
    )
    parser.add_argument("--confirm", required=True)
    parser.add_argument(
        "--reset-state-after-flat",
        action="store_true",
        help="Back up data/bot_state.json and create fresh state only after the "
        "broker is verified flat with no active NSE/MIS orders.",
    )
    args = parser.parse_args()

    if args.confirm != CONFIRMATION:
        raise SystemExit(f"Refusing recovery: --confirm must equal {CONFIRMATION}")
    if not bot.LIVE_TRADING:
        raise SystemExit("Refusing recovery: LIVE_TRADING must be true.")
    if bot.LIVE_TRADING_CONFIRM != "I_UNDERSTAND_REAL_MONEY":
        raise SystemExit("Refusing recovery: LIVE_TRADING_CONFIRM is not set.")
    if not bot.DEDICATED_BOT_ACCOUNT:
        raise SystemExit("Refusing recovery: DEDICATED_BOT_ACCOUNT must be true.")

    broker: bot.KiteBroker | None = None
    try:
        broker = bot.KiteBroker()
        positions = _nonzero_nse_mis_positions(broker)
        active_orders = _active_nse_mis_orders(broker)
        print(f"Open NSE/MIS positions before recovery: {positions}")
        print(
            "Active NSE/MIS orders before recovery: "
            + str([(o.order_id, o.symbol, o.status, o.order_type) for o in active_orders])
        )

        symbols = sorted(
            set(positions)
            | {order.symbol for order in active_orders if order.symbol}
        )
        for symbol in symbols:
            flat, recovery_ids = bot.dedicated_force_flatten_symbol(
                broker,
                symbol,
                "OPERATOR_DEDICATED_RECOVERY",
            )
            print(
                f"{symbol}: flat={flat}; recovery_exit_order_ids={recovery_ids}"
            )
            if not flat:
                raise RuntimeError(f"Recovery did not verify {symbol} flat.")

        remaining_positions = _nonzero_nse_mis_positions(broker)
        remaining_orders = _active_nse_mis_orders(broker)
        if remaining_positions or remaining_orders:
            raise RuntimeError(
                "Recovery incomplete: positions="
                f"{remaining_positions}; active_orders="
                f"{[(o.order_id, o.symbol, o.status) for o in remaining_orders]}"
            )

        print("Broker verification complete: no open NSE/MIS position/order remains.")

        if args.reset_state_after_flat:
            state_path = Path(bot.STATE_FILE)
            if state_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = state_path.with_name(
                    f"{state_path.stem}.pre_recovery_{timestamp}{state_path.suffix}"
                )
                shutil.copy2(state_path, backup)
                print(f"Backed up old state to: {backup}")
            bot.atomic_write_json(state_path, bot.fresh_state())
            print(f"Fresh verified-flat bot state written to: {state_path}")

        return 0
    finally:
        if broker is not None:
            broker.close()


if __name__ == "__main__":
    raise SystemExit(main())
