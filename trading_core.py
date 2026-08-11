"""Broker-independent safety and accounting primitives for the trading bot.

This module deliberately has no Kite or OpenAI dependency.  It can therefore be
used by the live adapter and exercised with deterministic unit tests without
connecting to a broker.
"""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass, fields
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping


TERMINAL_ORDER_STATUSES = frozenset({"COMPLETE", "CANCELLED", "REJECTED"})


def _normalise_status(value: Any) -> str:
    status = " ".join(str(value or "").replace("_", " ").split()).upper()
    if status == "CANCELED":
        return "CANCELLED"
    return status or "UNKNOWN"


def _nonnegative_int(value: Any, *, field: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")

    try:
        number = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc

    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f"{field} must be a non-negative integer")
    return int(number)


def _nonnegative_float(value: Any, *, field: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite non-negative number")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc

    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    """Immutable, normalised view of a Kite order/order-update payload."""

    order_id: str
    status: str
    qty: int
    filled: int
    pending: int
    avg: float
    message: str
    order_type: str = ""
    transaction_type: str = ""
    symbol: str = ""
    exchange: str = ""
    product: str = ""
    tag: str = ""
    trigger_price: float = 0.0

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OrderSnapshot":
        if not isinstance(payload, Mapping):
            raise TypeError("order payload must be a mapping")

        qty = _nonnegative_int(
            payload.get("quantity", payload.get("qty")),
            field="quantity",
        )
        filled = _nonnegative_int(
            payload.get("filled_quantity", payload.get("filled")),
            field="filled_quantity",
        )

        pending_value = payload.get("pending_quantity")
        if pending_value is None:
            pending_value = payload.get("unfilled_quantity")
        if pending_value is None:
            cancelled = _nonnegative_int(
                payload.get("cancelled_quantity"),
                field="cancelled_quantity",
            )
            pending_value = max(qty - filled - cancelled, 0)

        pending = _nonnegative_int(
            pending_value,
            field="pending_quantity",
        )
        if qty and filled > qty:
            raise ValueError("filled_quantity cannot exceed quantity")

        message = payload.get("status_message")
        if message in (None, ""):
            message = payload.get("status_message_raw", payload.get("message", ""))

        return cls(
            order_id=str(payload.get("order_id") or "").strip(),
            status=_normalise_status(payload.get("status")),
            qty=qty,
            filled=filled,
            pending=pending,
            avg=_nonnegative_float(
                payload.get("average_price", payload.get("avg")),
                field="average_price",
            ),
            message=str(message or "").strip(),
            order_type=str(payload.get("order_type") or "").strip().upper(),
            transaction_type=str(
                payload.get("transaction_type") or ""
            ).strip().upper(),
            symbol=str(
                payload.get("tradingsymbol", payload.get("symbol", "")) or ""
            ).strip().upper(),
            exchange=str(payload.get("exchange") or "").strip().upper(),
            product=str(payload.get("product") or "").strip().upper(),
            tag=str(payload.get("tag") or "").strip(),
            trigger_price=_nonnegative_float(
                payload.get("trigger_price"),
                field="trigger_price",
            ),
        )

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_ORDER_STATUSES

    @property
    def stop_armed(self) -> bool:
        return self.status == "TRIGGER PENDING" and self.pending > 0

    def is_terminal(self) -> bool:
        return self.terminal

    def is_stop_armed(self) -> bool:
        return self.stop_armed


class StateFileError(RuntimeError):
    """Raised when a durable JSON state file cannot be loaded safely."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def load_json_strict(
    path: str | os.PathLike[str],
    *,
    expected_type: type | tuple[type, ...] | None = dict,
) -> Any:
    """Load standards-compliant JSON without silently replacing bad state."""

    state_path = Path(path)
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StateFileError(f"could not load valid state from {state_path}") from exc

    if expected_type is not None and not isinstance(value, expected_type):
        raise StateFileError(
            f"state in {state_path} has unexpected top-level type "
            f"{type(value).__name__}"
        )
    return value


strict_load_json = load_json_strict


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    mode: int = 0o600,
) -> None:
    """Atomically replace a JSON file after flushing its contents to disk."""

    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    fd = -1
    temporary_path: Path | None = None
    try:
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            dir=state_path.parent,
        )
        temporary_path = Path(raw_path)
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, state_path)
        temporary_path = None

        directory_fd = os.open(state_path.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                    raise
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


class InstanceAlreadyRunningError(RuntimeError):
    """Raised when another owner already holds the process lock."""


class SingleInstanceLock:
    """Non-blocking POSIX advisory lock held for the object's lifetime."""

    _registry_lock = threading.Lock()
    _owned_paths: set[Path] = set()

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).resolve()
        self._fd: int | None = None

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    def acquire(self) -> "SingleInstanceLock":
        if self.acquired:
            return self

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._registry_lock:
            if self.path in self._owned_paths:
                raise InstanceAlreadyRunningError(
                    f"another bot instance holds {self.path}"
                )
            self._owned_paths.add(self.path)

        fd: int | None = None
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise InstanceAlreadyRunningError(
                    f"another bot instance holds {self.path}"
                ) from exc

            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
            self._fd = fd
            return self
        except Exception:
            if fd is not None:
                os.close(fd)
            with self._registry_lock:
                self._owned_paths.discard(self.path)
            raise

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            with self._registry_lock:
                self._owned_paths.discard(self.path)

    def __enter__(self) -> "SingleInstanceLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def _decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite non-negative number")
    try:
        number = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


@dataclass(frozen=True, slots=True)
class NSEEquityIntradayRates:
    """Configurable Zerodha NSE equity-intraday rates as of 2026-08-11."""

    brokerage_rate: Decimal = Decimal("0.0003")
    brokerage_cap_per_order: Decimal = Decimal("20")
    stt_sell_rate: Decimal = Decimal("0.00025")
    exchange_transaction_rate: Decimal = Decimal("0.0000307")
    sebi_rate: Decimal = Decimal("0.000001")
    stamp_buy_rate: Decimal = Decimal("0.00003")
    ipft_rate: Decimal = Decimal("0.000000001")
    gst_rate: Decimal = Decimal("0.18")
    round_stt_to_rupee: bool = True

    def __post_init__(self) -> None:
        for field_info in fields(self):
            if field_info.name == "round_stt_to_rupee":
                continue
            value = _decimal(getattr(self, field_info.name), field=field_info.name)
            object.__setattr__(self, field_info.name, value)


DEFAULT_NSE_EQUITY_INTRADAY_RATES = NSEEquityIntradayRates()


@dataclass(frozen=True, slots=True)
class IntradayCostEstimate:
    buy_turnover: Decimal
    sell_turnover: Decimal
    brokerage: Decimal
    stt: Decimal
    exchange_transaction_charges: Decimal
    sebi_charges: Decimal
    stamp_duty: Decimal
    ipft_charges: Decimal
    gst: Decimal
    total: Decimal

    @property
    def turnover(self) -> Decimal:
        return self.buy_turnover + self.sell_turnover


def _turnovers(value: Any, *, field: str) -> tuple[Decimal, ...]:
    if isinstance(value, (str, bytes, Decimal, int, float)):
        values: Iterable[Any] = (value,)
    else:
        try:
            values = iter(value)
        except TypeError as exc:
            raise ValueError(f"{field} must be a turnover or iterable") from exc
    return tuple(_decimal(item, field=field) for item in values)


def _paise(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def estimate_nse_equity_intraday_cost(
    buy_order_turnovers: Any,
    sell_order_turnovers: Any,
    *,
    rates: NSEEquityIntradayRates = DEFAULT_NSE_EQUITY_INTRADAY_RATES,
) -> IntradayCostEstimate:
    """Estimate charges from executed order turnovers.

    A scalar represents one executed order.  An iterable represents multiple
    orders, allowing the brokerage cap to be applied to every order correctly.
    Contract notes remain the final authority for actual charges.
    """

    buys = _turnovers(buy_order_turnovers, field="buy_order_turnovers")
    sells = _turnovers(sell_order_turnovers, field="sell_order_turnovers")
    buy_turnover = sum(buys, Decimal("0"))
    sell_turnover = sum(sells, Decimal("0"))
    turnover = buy_turnover + sell_turnover

    brokerage = sum(
        (
            min(order_turnover * rates.brokerage_rate, rates.brokerage_cap_per_order)
            for order_turnover in (*buys, *sells)
            if order_turnover > 0
        ),
        Decimal("0"),
    )
    brokerage = _paise(brokerage)

    raw_stt = sell_turnover * rates.stt_sell_rate
    stt = (
        raw_stt.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        if rates.round_stt_to_rupee
        else _paise(raw_stt)
    )
    exchange_charges = _paise(turnover * rates.exchange_transaction_rate)
    sebi_charges = _paise(turnover * rates.sebi_rate)
    stamp_duty = _paise(buy_turnover * rates.stamp_buy_rate)
    ipft_charges = _paise(turnover * rates.ipft_rate)
    gst = _paise(
        (
            brokerage
            + exchange_charges
            + sebi_charges
            + ipft_charges
        )
        * rates.gst_rate
    )
    total = _paise(
        brokerage
        + stt
        + exchange_charges
        + sebi_charges
        + stamp_duty
        + ipft_charges
        + gst
    )

    return IntradayCostEstimate(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        brokerage=brokerage,
        stt=stt,
        exchange_transaction_charges=exchange_charges,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
        ipft_charges=ipft_charges,
        gst=gst,
        total=total,
    )


def gross_pnl(
    side: str,
    entry_price: Any,
    exit_price: Any,
    quantity: Any,
) -> Decimal:
    """Return signed gross P&L for a long/buy or short/sell position."""

    normalised_side = str(side).strip().upper()
    entry = _decimal(entry_price, field="entry_price")
    exit_ = _decimal(exit_price, field="exit_price")
    qty = _decimal(quantity, field="quantity")

    if normalised_side in {"LONG", "BUY"}:
        return (exit_ - entry) * qty
    if normalised_side in {"SHORT", "SELL"}:
        return (entry - exit_) * qty
    raise ValueError("side must be LONG/BUY or SHORT/SELL")


def directional_slippage_per_share(
    expected_price: Any,
    actual_price: Any,
    transaction_type: str,
) -> Decimal:
    """Return adverse slippage per share; positive is worse execution."""

    expected = _decimal(expected_price, field="expected_price")
    actual = _decimal(actual_price, field="actual_price")
    transaction = str(transaction_type).strip().upper()
    if expected <= 0:
        raise ValueError("expected_price must be greater than zero")
    if transaction == "BUY":
        return actual - expected
    if transaction == "SELL":
        return expected - actual
    raise ValueError("transaction_type must be BUY or SELL")


def directional_slippage_bps(
    expected_price: Any,
    actual_price: Any,
    transaction_type: str,
) -> Decimal:
    """Return adverse execution slippage in basis points."""

    expected = _decimal(expected_price, field="expected_price")
    per_share = directional_slippage_per_share(
        expected,
        actual_price,
        transaction_type,
    )
    return per_share / expected * Decimal("10000")

