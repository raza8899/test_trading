"""Pure deterministic setup rules shared by live trading and offline replay.

This module deliberately has no broker, environment, wall-clock, network, or
filesystem dependencies.  Callers are responsible for constructing every
input from data that was available at their own causal cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Literal


Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True, slots=True)
class SetupRuleConfig:
    """Frozen thresholds and score coefficients for one strategy trial."""

    min_rvol: float = 1.35
    min_opening_rvol: float = 1.0
    min_atr_pct: float = 0.0025
    max_atr_pct: float = 0.025
    max_vwap_distance_atr: float = 1.8
    min_breakout_distance_atr: float = 0.04
    max_breakout_distance_atr: float = 0.80
    min_body_ratio: float = 0.30
    long_min_rsi: float = 50.0
    long_max_rsi: float = 76.0
    short_min_rsi: float = 24.0
    short_max_rsi: float = 50.0
    long_min_close_location: float = 0.58
    short_max_close_location: float = 0.45
    long_min_day_change_pct: float = 0.30
    short_max_day_change_pct: float = -0.20
    max_spread_bps: float = 18.0

    stock_score_multiplier: float = 0.28
    stock_score_cap: float = 28.0
    rvol_score_cap: float = 20.0
    rvol_score_span: float = 2.0
    body_score_cap: float = 12.0
    ideal_breakout_min_atr: float = 0.10
    ideal_breakout_max_atr: float = 0.55
    acceptable_breakout_max_atr: float = 0.70
    ideal_breakout_score: float = 12.0
    acceptable_breakout_score: float = 8.0
    extended_breakout_score: float = 4.0
    vwap_score_cap: float = 10.0
    aligned_regime_score: float = 10.0
    neutral_regime_score: float = 6.0
    spread_score_cap: float = 8.0

    def __post_init__(self) -> None:
        numeric_values = tuple(getattr(self, item.name) for item in fields(self))
        if any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in numeric_values
        ):
            raise ValueError("setup-rule configuration must be finite")
        if not 0 < self.min_atr_pct <= self.max_atr_pct:
            raise ValueError("ATR percentage bounds are invalid")
        if self.min_rvol < 0 or self.min_opening_rvol < 0:
            raise ValueError("relative-volume thresholds cannot be negative")
        if not 0 <= self.min_breakout_distance_atr <= self.max_breakout_distance_atr:
            raise ValueError("breakout-distance bounds are invalid")
        if (
            self.max_vwap_distance_atr <= 0
            or not 0 < self.max_spread_bps <= 1000
        ):
            raise ValueError("VWAP and spread limits must be positive")
        if self.rvol_score_span <= 0:
            raise ValueError("rvol_score_span must be positive")
        score_names = (
            "stock_score_multiplier",
            "stock_score_cap",
            "rvol_score_cap",
            "body_score_cap",
            "ideal_breakout_score",
            "acceptable_breakout_score",
            "extended_breakout_score",
            "vwap_score_cap",
            "aligned_regime_score",
            "neutral_regime_score",
            "spread_score_cap",
        )
        if any(getattr(self, name) < 0 for name in score_names):
            raise ValueError("score coefficients cannot be negative")
        if not (
            0 <= self.ideal_breakout_min_atr
            <= self.ideal_breakout_max_atr
            <= self.acceptable_breakout_max_atr
        ):
            raise ValueError("breakout score bands are invalid")
        if not (
            0 <= self.short_min_rsi <= self.short_max_rsi <= 100
            and 0 <= self.long_min_rsi <= self.long_max_rsi <= 100
        ):
            raise ValueError("RSI bounds are invalid")
        if not (
            0 <= self.short_max_close_location <= 1
            and 0 <= self.long_min_close_location <= 1
        ):
            raise ValueError("close-location bounds are invalid")


@dataclass(frozen=True, slots=True)
class SetupRuleInput:
    """Point-in-time scalar inputs produced by the causal data adapter."""

    price: float
    candle_open: float
    prev_close: float
    opening_high: float
    opening_low: float
    ema9: float
    ema20: float
    vwap: float
    rsi: float
    atr: float
    rvol: float
    opening_rvol: float
    body_ratio: float
    close_location: float
    nifty_regime: str
    stock_in_play_score: float
    spread_bps: float
    prior_post_opening_max_close: float | None = None
    prior_post_opening_min_close: float | None = None


@dataclass(frozen=True, slots=True)
class SetupRuleResult:
    """Deterministic result; rejected inputs retain auditable derived values."""

    accepted: bool
    side: Side | None
    reason: str
    day_change_pct: float = 0.0
    atr_pct: float = 0.0
    long_breakout_atr: float = 0.0
    short_breakout_atr: float = 0.0
    breakout_distance_atr: float = 0.0
    vwap_distance_atr: float = 0.0
    technical_score: float = 0.0


def _rejected(reason: str, **values: float) -> SetupRuleResult:
    return SetupRuleResult(False, None, reason, **values)


def evaluate_setup_rules(
    inputs: SetupRuleInput,
    config: SetupRuleConfig,
) -> SetupRuleResult:
    """Evaluate the current ORB feature gates without observing future state."""

    optional_values = (
        inputs.prior_post_opening_max_close,
        inputs.prior_post_opening_min_close,
    )
    required_values = (
        inputs.price,
        inputs.candle_open,
        inputs.prev_close,
        inputs.opening_high,
        inputs.opening_low,
        inputs.ema9,
        inputs.ema20,
        inputs.vwap,
        inputs.rsi,
        inputs.atr,
        inputs.rvol,
        inputs.opening_rvol,
        inputs.body_ratio,
        inputs.close_location,
        inputs.stock_in_play_score,
        inputs.spread_bps,
    )
    if any(
        isinstance(value, bool) or not math.isfinite(float(value))
        for value in required_values
    ) or any(
        value is not None
        and (isinstance(value, bool) or not math.isfinite(float(value)))
        for value in optional_values
    ):
        return _rejected("NONFINITE_INPUT")
    if inputs.nifty_regime not in {"BULL", "BEAR", "NEUTRAL"}:
        return _rejected("NIFTY_REGIME_UNAVAILABLE")
    if (
        inputs.price <= 0
        or inputs.candle_open <= 0
        or inputs.prev_close <= 0
        or inputs.opening_high <= 0
        or inputs.opening_low <= 0
        or inputs.opening_low > inputs.opening_high
        or inputs.ema9 <= 0
        or inputs.ema20 <= 0
        or inputs.atr <= 0
        or inputs.vwap <= 0
    ):
        return _rejected("INVALID_PRICE_GEOMETRY")
    if not (
        0 <= inputs.rsi <= 100
        and inputs.rvol >= 0
        and inputs.opening_rvol >= 0
        and 0 <= inputs.body_ratio <= 1
        and 0 <= inputs.close_location <= 1
        and 0 <= inputs.stock_in_play_score <= 100
        and inputs.spread_bps >= 0
    ):
        return _rejected("INVALID_FEATURE_RANGE")

    day_change_pct = (
        (inputs.price - inputs.prev_close) / inputs.prev_close * 100
    )
    atr_pct = inputs.atr / inputs.price
    long_breakout_atr = (
        inputs.price - inputs.opening_high
    ) / inputs.atr
    short_breakout_atr = (
        inputs.opening_low - inputs.price
    ) / inputs.atr
    vwap_distance_atr = abs(inputs.price - inputs.vwap) / inputs.atr
    if not all(
        math.isfinite(value)
        for value in (
            day_change_pct,
            atr_pct,
            long_breakout_atr,
            short_breakout_atr,
            vwap_distance_atr,
        )
    ):
        return _rejected("NONFINITE_DERIVED_VALUE")
    derived = {
        "day_change_pct": day_change_pct,
        "atr_pct": atr_pct,
        "long_breakout_atr": long_breakout_atr,
        "short_breakout_atr": short_breakout_atr,
        "vwap_distance_atr": vwap_distance_atr,
    }

    if not config.min_atr_pct <= atr_pct <= config.max_atr_pct:
        return _rejected("ATR_PCT_OUT_OF_RANGE", **derived)
    if inputs.rvol < config.min_rvol:
        return _rejected("RVOL_BELOW_MINIMUM", **derived)
    if inputs.opening_rvol < config.min_opening_rvol:
        return _rejected("OPENING_RVOL_BELOW_MINIMUM", **derived)
    if inputs.body_ratio < config.min_body_ratio:
        return _rejected("BODY_RATIO_BELOW_MINIMUM", **derived)
    if vwap_distance_atr > config.max_vwap_distance_atr:
        return _rejected("VWAP_DISTANCE_ABOVE_MAXIMUM", **derived)
    if inputs.spread_bps < 0 or inputs.spread_bps > config.max_spread_bps:
        return _rejected("SPREAD_OUT_OF_RANGE", **derived)

    long_fresh = (
        inputs.prior_post_opening_max_close is None
        or inputs.prior_post_opening_max_close <= inputs.opening_high
    )
    short_fresh = (
        inputs.prior_post_opening_min_close is None
        or inputs.prior_post_opening_min_close >= inputs.opening_low
    )
    long_setup = (
        long_fresh
        and inputs.price > inputs.candle_open
        and config.min_breakout_distance_atr
        <= long_breakout_atr
        <= config.max_breakout_distance_atr
        and inputs.price > inputs.vwap
        and inputs.ema9 > inputs.ema20
        and config.long_min_rsi <= inputs.rsi <= config.long_max_rsi
        and inputs.close_location >= config.long_min_close_location
        and day_change_pct >= config.long_min_day_change_pct
        and inputs.nifty_regime != "BEAR"
    )
    short_setup = (
        short_fresh
        and inputs.price < inputs.candle_open
        and config.min_breakout_distance_atr
        <= short_breakout_atr
        <= config.max_breakout_distance_atr
        and inputs.price < inputs.vwap
        and inputs.ema9 < inputs.ema20
        and config.short_min_rsi <= inputs.rsi <= config.short_max_rsi
        and inputs.close_location <= config.short_max_close_location
        and day_change_pct <= config.short_max_day_change_pct
        and inputs.nifty_regime != "BULL"
    )
    if not long_setup and not short_setup:
        return _rejected("DIRECTIONAL_SETUP_NOT_CONFIRMED", **derived)

    side: Side = "LONG" if long_setup else "SHORT"
    breakout_distance_atr = (
        long_breakout_atr if side == "LONG" else short_breakout_atr
    )
    score = min(
        config.stock_score_cap,
        inputs.stock_in_play_score * config.stock_score_multiplier,
    )
    score += min(
        config.rvol_score_cap,
        max(
            0.0,
            (inputs.rvol - 1.0)
            / config.rvol_score_span
            * config.rvol_score_cap,
        ),
    )
    score += min(config.body_score_cap, inputs.body_ratio * config.body_score_cap)
    if (
        config.ideal_breakout_min_atr
        <= breakout_distance_atr
        <= config.ideal_breakout_max_atr
    ):
        score += config.ideal_breakout_score
    elif breakout_distance_atr <= config.acceptable_breakout_max_atr:
        score += config.acceptable_breakout_score
    else:
        score += config.extended_breakout_score
    score += max(
        0.0,
        config.vwap_score_cap
        * (1.0 - min(vwap_distance_atr, config.max_vwap_distance_atr)
           / config.max_vwap_distance_atr),
    )
    aligned = (
        side == "LONG" and inputs.nifty_regime == "BULL"
    ) or (
        side == "SHORT" and inputs.nifty_regime == "BEAR"
    )
    score += (
        config.aligned_regime_score if aligned else config.neutral_regime_score
    )
    score += max(
        0.0,
        config.spread_score_cap
        * (1.0 - inputs.spread_bps / config.max_spread_bps),
    )
    score = min(100.0, score)

    return SetupRuleResult(
        accepted=True,
        side=side,
        reason="OK",
        day_change_pct=day_change_pct,
        atr_pct=atr_pct,
        long_breakout_atr=long_breakout_atr,
        short_breakout_atr=short_breakout_atr,
        breakout_distance_atr=breakout_distance_atr,
        vwap_distance_atr=vwap_distance_atr,
        technical_score=score,
    )
