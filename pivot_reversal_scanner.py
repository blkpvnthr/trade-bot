#!/usr/bin/env python3
"""
pivot_reversal_scanner.py

Universe discovery via TradingView Screener:
  - price > $1, volume > 1M shares/day, daily volatility > 5%
  - SMA50 > SMA200 (golden-cross uptrend)
  - 1-week performance > 0%
  - relative volume > 1.3x  →  sorted by rel vol descending

Entry / exit: LuxAlgo "Pivot Points High Low & Missed Reversal Levels"
  Missed pivot lows  → horizontal ghost support zones  → long entries
  Missed pivot highs → horizontal ghost resistance zones → exit targets

Output files (same contract as scanner.py):
  system-data/YYYYMMDD/YYYYMMDD_pivot_reversal_rebalance_orders.csv
  system-data/YYYYMMDD/YYYYMMDD_pivot_reversal_exit_orders.csv
  system-data/YYYYMMDD/YYYYMMDD_pivot_reversal_summary.json

Env vars:
  PIVOT_LENGTH              default 10   (bars left+right for pivot confirmation)
  PIVOT_SUPPORT_ZONE_PCT    default 0.02 (entry if within 2% above support)
  PIVOT_RESISTANCE_ZONE_PCT default 0.01 (exit signal if within 1% of resistance)
  PIVOT_MIN_RR              default 2.0
  PIVOT_MAX_SYMBOLS         default 30
  PIVOT_LOOKBACK_DAYS       default 365
  PIVOT_MIN_CONFIDENCE      default 0.60
  PIVOT_ORDER_NOTIONAL      default 2000.0
  TV_MIN_PRICE              default 1.0
  TV_MIN_VOLUME             default 1000000
  TV_MIN_VOLATILITY         default 5.0
  TV_MIN_PERF_W             default 0.0
  TV_MIN_REL_VOLUME         default 1.3
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

PROJECT_ROOT = Path(__file__).parent.resolve()
TODAY_TAG = datetime.now(ET).strftime("%Y%m%d")
OUT_DIR = PROJECT_ROOT / "system-data" / TODAY_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PIVOT_LENGTH = int(os.getenv("PIVOT_LENGTH", "10"))
PIVOT_SUPPORT_ZONE_PCT = float(os.getenv("PIVOT_SUPPORT_ZONE_PCT", "0.02"))
PIVOT_RESISTANCE_ZONE_PCT = float(os.getenv("PIVOT_RESISTANCE_ZONE_PCT", "0.01"))
PIVOT_MIN_RR = float(os.getenv("PIVOT_MIN_RR", "2.0"))
PIVOT_MAX_SYMBOLS = int(os.getenv("PIVOT_MAX_SYMBOLS", "30"))
PIVOT_LOOKBACK_DAYS = int(os.getenv("PIVOT_LOOKBACK_DAYS", "365"))
PIVOT_MIN_CONFIDENCE = float(os.getenv("PIVOT_MIN_CONFIDENCE", "0.60"))
PIVOT_ORDER_NOTIONAL = float(os.getenv("PIVOT_ORDER_NOTIONAL", "2000.0"))
PIVOT_BAR_FETCH_DELAY = float(os.getenv("PIVOT_BAR_FETCH_DELAY", "0.25"))

TV_MIN_PRICE = float(os.getenv("TV_MIN_PRICE", "1.0"))
TV_MIN_VOLUME = float(os.getenv("TV_MIN_VOLUME", "1000000"))
TV_MIN_VOLATILITY = float(os.getenv("TV_MIN_VOLATILITY", "5.0"))
TV_MIN_PERF_W = float(os.getenv("TV_MIN_PERF_W", "0.0"))
TV_MIN_REL_VOLUME = float(os.getenv("TV_MIN_REL_VOLUME", "1.3"))

OUTPUT_REBALANCE = OUT_DIR / f"{TODAY_TAG}_pivot_reversal_rebalance_orders.csv"
OUTPUT_EXITS = OUT_DIR / f"{TODAY_TAG}_pivot_reversal_exit_orders.csv"
OUTPUT_SUMMARY = OUT_DIR / f"{TODAY_TAG}_pivot_reversal_summary.json"


# ===========================================================================
# TradingView Screener
# ===========================================================================

def run_tradingview_screen() -> pd.DataFrame:
    """
    Screen for stocks matching user criteria, sorted by relative volume.
    Returns DataFrame with columns: name, close, volume, Volatility.D,
    SMA50, SMA200, Perf.W, relative_volume_10d_calc
    """
    try:
        from tradingview_screener import Query, col  # type: ignore
    except ImportError:
        print("[pivot-tv] tradingview-screener not installed; run: pip install tradingview-screener")
        return pd.DataFrame()

    try:
        _count, df = (
            Query()
            .select(
                "name",
                "close",
                "volume",
                "Volatility.D",
                "SMA50",
                "SMA200",
                "Perf.W",
                "relative_volume_10d_calc",
                "exchange",
            )
            .where(
                col("close") > TV_MIN_PRICE,
                col("volume") > TV_MIN_VOLUME,
                col("Volatility.D") > TV_MIN_VOLATILITY,
                col("SMA50") > col("SMA200"),
                col("Perf.W") > TV_MIN_PERF_W,
                col("relative_volume_10d_calc") > TV_MIN_REL_VOLUME,
            )
            .order_by("relative_volume_10d_calc", ascending=False)
            .limit(PIVOT_MAX_SYMBOLS)
            .get_scanner_data()
        )
        print(
            f"[pivot-tv] screener returned {len(df)} symbols "
            f"(price>${TV_MIN_PRICE} vol>{TV_MIN_VOLUME/1e6:.0f}M "
            f"volat>{TV_MIN_VOLATILITY}% sma50>sma200 "
            f"perfW>{TV_MIN_PERF_W}% relvol>{TV_MIN_REL_VOLUME})"
        )
        return df
    except Exception as exc:
        print(f"[pivot-tv] screener error: {exc}")
        return pd.DataFrame()


# ===========================================================================
# Pivot Detection — faithful Python translation of LuxAlgo Pine Script
# ===========================================================================

def _pivot_high(highs: np.ndarray, length: int) -> np.ndarray:
    """Bar i is a pivot high if highs[i] > all highs in [i-length, i-1]
    AND > all highs in [i+1, i+length]. Returns NaN elsewhere."""
    n = len(highs)
    out = np.full(n, np.nan)
    for i in range(length, n - length):
        c = highs[i]
        if c > highs[i - length : i].max() and c > highs[i + 1 : i + length + 1].max():
            out[i] = c
    return out


def _pivot_low(lows: np.ndarray, length: int) -> np.ndarray:
    """Bar i is a pivot low if lows[i] < all lows in [i-length, i-1]
    AND < all lows in [i+1, i+length]. Returns NaN elsewhere."""
    n = len(lows)
    out = np.full(n, np.nan)
    for i in range(length, n - length):
        c = lows[i]
        if c < lows[i - length : i].min() and c < lows[i + 1 : i + length + 1].min():
            out[i] = c
    return out


def compute_pivot_structure(
    highs: np.ndarray,
    lows: np.ndarray,
    length: int = 10,
) -> dict[str, list[tuple[int, float]]]:
    """
    Replicate LuxAlgo "Pivot Points High Low & Missed Reversal Levels".

    Returns:
      regular_highs / regular_lows  — confirmed pivot extremes
      missed_highs  / missed_lows   — ghost (👻) reversal levels
                                      → these extend horizontally as S/R
    """
    n = len(highs)
    ph_arr = _pivot_high(highs, length)
    pl_arr = _pivot_low(lows, length)

    # var declarations from Pine (mirrored exactly)
    max_val: float = 0.0;   max_x1: int = 0
    min_val: float = 0.0;   min_x1: int = 0   # Pine starts at 0.
    follow_max: float = 0.0; follow_max_x1: int = 0
    follow_min: float = 0.0; follow_min_x1: int = 0
    os: int = 0  # 0 = last was low; 1 = last was high

    regular_highs: list[tuple[int, float]] = []
    regular_lows:  list[tuple[int, float]] = []
    missed_highs:  list[tuple[int, float]] = []
    missed_lows:   list[tuple[int, float]] = []

    for i in range(n):
        h = float(highs[i])
        lo = float(lows[i])

        prev_max = max_val
        prev_min = min_val
        prev_follow_max = follow_max
        prev_follow_min = follow_min
        prev_os = os

        # Pine: max := math.max(high[length], max) — monotone increase
        max_val = max(h, max_val)
        # Pine: min := math.min(low[length], min)  — monotone decrease
        #   Since Pine starts min at 0. and prices > 0, math.min(price,0)=0 until first pivot reset.
        #   We replicate this exactly: first non-zero price wins if min_val==0
        if min_val == 0.0:
            min_val = lo
        else:
            min_val = min(lo, min_val)

        if max_val > prev_max:
            max_x1 = i
            follow_min = lo     # reset follow_min when new running max
        if min_val < prev_min or (prev_min == 0.0 and min_val > 0.0):
            min_x1 = i
            follow_max = h      # reset follow_max when new running min

        if follow_min == 0.0 or lo < follow_min:
            follow_min = lo
            follow_min_x1 = i
        if h > follow_max:
            follow_max = h
            follow_max_x1 = i

        if i < length:
            continue

        ph = ph_arr[i]
        pl = pl_arr[i]

        if not np.isnan(ph):
            # --- PIVOT HIGH confirmed ---
            if prev_os == 1:
                # Was already in "high" state → found two highs without a low between them.
                # The running minimum between the two highs is a missed (ghost) pivot low.
                if min_x1 > 0:
                    missed_lows.append((min_x1, min_val))
            elif ph < max_val:
                # New pivot high is LOWER than the running max since last pivot →
                # a missed pivot high occurred at the running-max bar,
                # and a missed pivot low occurred at the follow_min bar.
                if max_x1 > 0:
                    missed_highs.append((max_x1, max_val))
                if follow_min_x1 > 0:
                    missed_lows.append((follow_min_x1, follow_min))

            regular_highs.append((i, float(ph)))

            # Reset state (Pine: os:=1, max:=ph, min:=ph)
            os = 1
            max_val = float(ph); min_val = float(ph)
            max_x1 = i;          min_x1 = i
            follow_max = float(ph); follow_min = float(ph)
            follow_max_x1 = i;   follow_min_x1 = i

        if not np.isnan(pl):
            # --- PIVOT LOW confirmed ---
            if prev_os == 0:
                # Was already in "low" state → found two lows without a high between them.
                # The running maximum between them is a missed (ghost) pivot high.
                if max_x1 > 0:
                    missed_highs.append((max_x1, max_val))
            elif pl > min_val:
                # New pivot low is HIGHER than running min → higher-low structure.
                # Missed pivot low at running-min bar, missed pivot high at follow_max bar.
                if min_x1 > 0:
                    missed_lows.append((min_x1, min_val))
                if follow_max_x1 > 0:
                    missed_highs.append((follow_max_x1, follow_max))

            regular_lows.append((i, float(pl)))

            # Reset state (Pine: os:=0, max:=pl, min:=pl)
            os = 0
            max_val = float(pl); min_val = float(pl)
            max_x1 = i;          min_x1 = i
            follow_max = float(pl); follow_min = float(pl)
            follow_max_x1 = i;   follow_min_x1 = i

    return {
        "regular_highs": regular_highs,
        "regular_lows":  regular_lows,
        "missed_highs":  missed_highs,
        "missed_lows":   missed_lows,
    }


# ===========================================================================
# Signal Generation
# ===========================================================================

def generate_signal(
    symbol: str,
    bars: pd.DataFrame,
    structure: dict[str, list[tuple[int, float]]],
    current_price: float,
    rel_volume: float = 1.0,
    volatility_d: float = 0.0,
    perf_w: float = 0.0,
) -> dict[str, Any] | None:
    """
    Entry: price near a ghost/pivot support with bounce and R/R ≥ PIVOT_MIN_RR.
    Exit:  price within PIVOT_RESISTANCE_ZONE_PCT of nearest ghost/pivot resistance.

    Ghost missed-pivot levels have equal weight to regular pivots — both extend
    horizontally and act as live S/R until price breaks through them.
    """
    # Collect all support and resistance levels (regular + missed)
    supports = sorted(
        set(
            [p for _, p in structure["regular_lows"]]
            + [p for _, p in structure["missed_lows"]]
        ),
        reverse=True,  # highest first so nearest_support = [0]
    )
    resistances = sorted(
        set(
            [p for _, p in structure["regular_highs"]]
            + [p for _, p in structure["missed_highs"]]
        )
    )

    # Filter: only levels on the correct side of price
    supports_below = [s for s in supports if s < current_price * 1.005]
    resistances_above = [r for r in resistances if r > current_price * 0.995]

    if not supports_below or not resistances_above:
        return None

    nearest_support = supports_below[0]
    nearest_resistance = resistances_above[0]

    dist_to_support = (current_price - nearest_support) / max(nearest_support, 1e-9)
    dist_to_resistance = (nearest_resistance - current_price) / max(current_price, 1e-9)
    risk_pct = max(dist_to_support, 1e-4)
    rr_ratio = dist_to_resistance / risk_pct

    stop_price = nearest_support * 0.995   # 0.5% below support
    target_price = nearest_resistance

    # --- Bounce detection on daily bars ---
    last = bars.iloc[-1]
    day_range = last["high"] - last["low"]
    close_position = (last["close"] - last["low"]) / max(day_range, 1e-6)
    is_bullish_close = close_position > 0.50   # closed in upper half
    is_green_bar = last["close"] > last["open"]
    is_bouncing = is_bullish_close or is_green_bar

    # Short-term upward momentum: 5-bar EMA slope
    closes = bars["close"].values
    momentum_positive = False
    if len(closes) >= 8:
        ema5 = pd.Series(closes).ewm(span=5, adjust=False).mean().values
        momentum_positive = bool(ema5[-1] > ema5[-3])

    # Volume expansion on last bar
    vol_avg = bars["volume"].rolling(10).mean().iloc[-1]
    vol_expansion = last["volume"] > vol_avg * 1.1 if vol_avg > 0 else False

    # --- Confidence score ---
    # How close to support (0 = at support, 1 = far away)
    support_proximity = max(0.0, 1.0 - dist_to_support / PIVOT_SUPPORT_ZONE_PCT)
    # R/R quality (0.5 at exactly MIN_RR, 1.0 at 3×MIN_RR)
    rr_quality = min(1.0, max(0.0, (rr_ratio - PIVOT_MIN_RR) / (2.0 * PIVOT_MIN_RR) + 0.5))
    rel_vol_quality = min(1.0, (rel_volume - 1.0) / 2.0)
    momentum_bonus = 1.0 if momentum_positive else 0.4
    bounce_bonus = 1.0 if is_bouncing else 0.5

    confidence = (
        0.30 * support_proximity
        + 0.25 * rr_quality
        + 0.20 * rel_vol_quality
        + 0.15 * momentum_bonus
        + 0.10 * bounce_bonus
    )

    # --- Determine signal ---
    is_near_support = dist_to_support <= PIVOT_SUPPORT_ZONE_PCT
    is_near_resistance = dist_to_resistance <= PIVOT_RESISTANCE_ZONE_PCT
    has_good_rr = rr_ratio >= PIVOT_MIN_RR

    signal: str | None = None
    signal_type: str | None = None

    # Exit takes priority: if price is at resistance, flag sell regardless
    if is_near_resistance:
        signal = "sell"
        signal_type = "pivot_resistance_target"
    elif is_near_support and has_good_rr and is_bouncing and confidence >= PIVOT_MIN_CONFIDENCE:
        signal = "buy"
        signal_type = "pivot_ghost_support_bounce" if structure["missed_lows"] else "pivot_support_bounce"

    return {
        "symbol": symbol,
        "signal": signal,
        "signal_type": signal_type,
        "current_price": round(current_price, 4),
        "nearest_support": round(nearest_support, 4),
        "nearest_resistance": round(nearest_resistance, 4),
        "dist_to_support_pct": round(dist_to_support, 5),
        "dist_to_resistance_pct": round(dist_to_resistance, 5),
        "risk_reward_ratio": round(rr_ratio, 2),
        "stop_price": round(stop_price, 4),
        "target_price": round(target_price, 4),
        "confidence": round(confidence, 4),
        "rel_volume": round(rel_volume, 2),
        "volatility_d": round(volatility_d, 2),
        "perf_w": round(perf_w, 4),
        "is_near_support": is_near_support,
        "is_near_resistance": is_near_resistance,
        "is_bouncing": is_bouncing,
        "momentum_positive": momentum_positive,
        "vol_expansion": vol_expansion,
        "support_levels_count": len(supports_below),
        "resistance_levels_count": len(resistances_above),
        "regular_highs_count": len(structure["regular_highs"]),
        "regular_lows_count": len(structure["regular_lows"]),
        "missed_highs_count": len(structure["missed_highs"]),
        "missed_lows_count": len(structure["missed_lows"]),
    }


# ===========================================================================
# Bar Download
# ===========================================================================

def fetch_daily_bars(symbol: str, lookback_days: int = PIVOT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Download daily OHLCV bars via yfinance."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        print("[pivot] yfinance not installed; run: pip install yfinance")
        return pd.DataFrame()

    try:
        end = datetime.now(ET)
        start = end - timedelta(days=lookback_days)
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return pd.DataFrame()
        return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    except Exception as exc:
        print(f"[pivot] {symbol}: bar fetch failed: {exc}")
        return pd.DataFrame()


# ===========================================================================
# Order Building
# ===========================================================================

def build_orders(
    signals: list[dict[str, Any]],
    equity: float = 100_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert signals → (rebalance_orders_df, exit_orders_df)."""
    buy_rows: list[dict] = []
    sell_rows: list[dict] = []

    for sig in signals:
        if not sig or not sig.get("signal"):
            continue

        symbol = sig["symbol"]
        price = float(sig["current_price"])
        confidence = float(sig["confidence"])
        rr = float(sig["risk_reward_ratio"])

        if sig["signal"] == "buy":
            notional = min(PIVOT_ORDER_NOTIONAL, equity * 0.06)  # max 6% of equity
            qty = max(1, int(notional / max(price, 0.01)))
            actual_notional = qty * price
            buy_rows.append(
                {
                    "symbol": symbol,
                    "side": "buy",
                    "order_qty": qty,
                    "price": price,
                    "capped_dollars": actual_notional,
                    "abs_delta_dollars": actual_notional,
                    "delta_dollars": actual_notional,
                    "target_weight": actual_notional / max(equity, 1.0),
                    "current_weight": 0.0,
                    "delta_weight": actual_notional / max(equity, 1.0),
                    "confidence_used": confidence,
                    "trade_confidence": confidence,
                    "risk_reward_ratio": rr,
                    "nearest_support": sig["nearest_support"],
                    "nearest_resistance": sig["nearest_resistance"],
                    "stop_price": sig["stop_price"],
                    "target_price": sig["target_price"],
                    "entry_price": price,
                    "stop_loss": sig["stop_price"],
                    "take_profit": sig["target_price"],
                    "dist_to_support_pct": sig["dist_to_support_pct"],
                    "dist_to_resistance_pct": sig["dist_to_resistance_pct"],
                    "rel_volume": sig["rel_volume"],
                    "volatility_d": sig["volatility_d"],
                    "perf_w": sig["perf_w"],
                    "pivot_support_bounce": True,
                    "reason": f"pivot_reversal_{sig['signal_type']}",
                    "entry_type": sig["signal_type"],
                    "source": "pivot_reversal_scanner",
                    "execution_alpha_score": min(1.0, confidence * 1.1),
                    "cross_sectional_score": confidence,
                    "time_series_score": confidence,
                    "planned_at_utc": datetime.now(UTC).isoformat(),
                }
            )

        elif sig["signal"] == "sell":
            sell_rows.append(
                {
                    "symbol": symbol,
                    "side": "sell",
                    "reason": f"pivot_reversal_{sig['signal_type']}",
                    "source": "pivot_reversal_scanner",
                    "nearest_resistance": sig["nearest_resistance"],
                    "dist_to_resistance_pct": sig["dist_to_resistance_pct"],
                    "planned_at_utc": datetime.now(UTC).isoformat(),
                }
            )

    buy_df = (
        pd.DataFrame(buy_rows)
        .sort_values("confidence_used", ascending=False)
        .reset_index(drop=True)
        if buy_rows
        else pd.DataFrame()
    )
    sell_df = pd.DataFrame(sell_rows) if sell_rows else pd.DataFrame()
    return buy_df, sell_df


# ===========================================================================
# Main
# ===========================================================================

def main() -> dict[str, Any]:
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")
    print("=" * 70)
    print(f"[pivot-reversal] {ts}")
    print(
        f"[pivot-reversal] pivot_length={PIVOT_LENGTH} "
        f"support_zone={PIVOT_SUPPORT_ZONE_PCT:.1%} "
        f"resistance_zone={PIVOT_RESISTANCE_ZONE_PCT:.1%} "
        f"min_rr={PIVOT_MIN_RR} "
        f"min_conf={PIVOT_MIN_CONFIDENCE}"
    )
    print("=" * 70)

    def _write_empty(msg: str) -> dict[str, Any]:
        summary = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": msg,
            "buy_count": 0,
            "sell_count": 0,
        }
        OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2))
        pd.DataFrame().to_csv(OUTPUT_REBALANCE, index=False)
        pd.DataFrame().to_csv(OUTPUT_EXITS, index=False)
        return summary

    # 1. Screen universe via TradingView
    screen_df = run_tradingview_screen()
    if screen_df.empty:
        print("[pivot-reversal] screener returned no symbols")
        return _write_empty("no_symbols_from_screener")

    symbols = screen_df["name"].tolist()[:PIVOT_MAX_SYMBOLS]
    tv_meta = {str(row["name"]): row.to_dict() for _, row in screen_df.iterrows()}
    print(f"[pivot-reversal] {len(symbols)} symbols to process")

    # 2. Per-symbol: download bars → compute pivots → generate signal
    signals: list[dict[str, Any]] = []
    skipped: list[str] = []

    for symbol in symbols:
        try:
            meta = tv_meta.get(symbol, {})
            rel_vol = float(meta.get("relative_volume_10d_calc") or 1.0)
            volat = float(meta.get("Volatility.D") or 0.0)
            perf_w = float(meta.get("Perf.W") or 0.0)
            tv_price = float(meta.get("close") or 0.0)

            bars = fetch_daily_bars(symbol)
            min_bars = PIVOT_LENGTH * 4
            if len(bars) < min_bars:
                skipped.append(f"{symbol}:only_{len(bars)}_bars<{min_bars}")
                time.sleep(PIVOT_BAR_FETCH_DELAY)
                continue

            highs = bars["high"].values
            lows = bars["low"].values
            current_price = float(bars["close"].iloc[-1])
            if tv_price > 0 and abs(tv_price - current_price) / max(current_price, 1e-6) < 0.10:
                current_price = tv_price  # prefer live TV price if within 10%

            structure = compute_pivot_structure(highs, lows, length=PIVOT_LENGTH)

            total_pivots = (
                len(structure["regular_highs"])
                + len(structure["regular_lows"])
                + len(structure["missed_highs"])
                + len(structure["missed_lows"])
            )
            if total_pivots < 4:
                skipped.append(f"{symbol}:insufficient_pivots({total_pivots})")
                time.sleep(PIVOT_BAR_FETCH_DELAY)
                continue

            sig = generate_signal(
                symbol=symbol,
                bars=bars,
                structure=structure,
                current_price=current_price,
                rel_volume=rel_vol,
                volatility_d=volat,
                perf_w=perf_w,
            )

            if sig:
                signals.append(sig)
                action = sig.get("signal") or "none"
                print(
                    f"[pivot] {symbol:6s} px={current_price:.3f} "
                    f"sup={sig['nearest_support']:.3f}({sig['dist_to_support_pct']:.1%}) "
                    f"res={sig['nearest_resistance']:.3f}({sig['dist_to_resistance_pct']:.1%}) "
                    f"rr={sig['risk_reward_ratio']:.1f} "
                    f"conf={sig['confidence']:.3f} "
                    f"relvol={rel_vol:.1f}x "
                    f"→ {action}"
                )
            else:
                print(
                    f"[pivot] {symbol:6s} px={current_price:.3f} "
                    f"no S/R bracket available — skip"
                )

            time.sleep(PIVOT_BAR_FETCH_DELAY)

        except Exception as exc:
            print(f"[pivot] {symbol}: exception: {exc}")
            traceback.print_exc()
            skipped.append(f"{symbol}:error")

    # 3. Build orders
    buy_df, sell_df = build_orders(signals)

    # 4. Write outputs
    buy_df.to_csv(OUTPUT_REBALANCE, index=False)
    sell_df.to_csv(OUTPUT_EXITS, index=False)

    buy_symbols = buy_df["symbol"].tolist() if not buy_df.empty else []
    sell_symbols = sell_df["symbol"].tolist() if not sell_df.empty else []

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "ok",
        "symbols_screened": len(symbols),
        "signals_generated": len(signals),
        "buy_count": len(buy_df),
        "sell_count": len(sell_df),
        "skipped_count": len(skipped),
        "buy_symbols": buy_symbols,
        "sell_symbols": sell_symbols,
        "skipped": skipped[:20],
        "config": {
            "pivot_length": PIVOT_LENGTH,
            "support_zone_pct": PIVOT_SUPPORT_ZONE_PCT,
            "resistance_zone_pct": PIVOT_RESISTANCE_ZONE_PCT,
            "min_rr": PIVOT_MIN_RR,
            "min_confidence": PIVOT_MIN_CONFIDENCE,
            "order_notional": PIVOT_ORDER_NOTIONAL,
            "tv_min_price": TV_MIN_PRICE,
            "tv_min_volume": TV_MIN_VOLUME,
            "tv_min_volatility": TV_MIN_VOLATILITY,
            "tv_min_perf_w": TV_MIN_PERF_W,
            "tv_min_rel_volume": TV_MIN_REL_VOLUME,
        },
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))

    print(
        f"[pivot-reversal] done | "
        f"buy={len(buy_df)} {buy_symbols} | "
        f"sell={len(sell_df)} {sell_symbols} | "
        f"skipped={len(skipped)}"
    )
    print(f"[pivot-reversal] orders → {OUTPUT_REBALANCE}")
    return summary


if __name__ == "__main__":
    main()
