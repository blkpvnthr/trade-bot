"""Trade-outcome learning dataset: features + leakage control.

Each TradingView trade becomes one labeled example. Features are split into
hard-separated groups so an entry model can never see the future:

  pre_entry_features   : ONLY bars with timestamp <= entry_time.
  during_trade_features: the realized path between entry and exit.
  exit_features        : exit quality (how well the exit captured the move).
  context_features     : per-strategy metadata (family, confidence, human_bucket...).
  target_labels        : normalized-PnL outcome (the objective truth).

human_bucket (WINS/LOSSES/FLAT folders) is preserved as context metadata only --
outcomes come solely from normalized Net PnL. Bar-derived features are NaN when
15Min OHLCV bars are not cached.
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from .tv_import import _col, _num, _parse_dt, REFERENCE_QTY

LOOKBACK = 50          # minimum bars <= entry needed for pre-entry features
BAR_MINUTES = 15

PRE_ENTRY_FEATURES = [
    # trend
    "close_vs_sma20", "close_vs_sma50", "sma20_slope", "sma50_slope",
    "ema9_vs_ema21", "higher_high_higher_low_flag", "distance_from_recent_high",
    "distance_from_recent_low",
    # momentum
    "rsi14", "stoch_position", "macd_histogram", "return_5_bars",
    "return_10_bars", "return_20_bars", "consecutive_green_bars",
    "consecutive_red_bars",
    # volume
    "relative_volume_20", "volume_spike_flag", "volume_trend_20", "dollar_volume",
    # volatility
    "atr14", "atr_pct", "candle_range_pct", "realized_vol_20",
    "gap_pct_from_prev_close",
    # vwap / mean reversion
    "distance_from_vwap", "close_above_vwap", "vwap_reclaim_flag", "zscore_from_sma20",
    # support / resistance
    "distance_to_previous_day_high", "distance_to_previous_day_low",
    "distance_to_opening_range_high", "distance_to_opening_range_low",
    # entry quality
    "entry_distance_from_vwap", "entry_distance_from_sma20", "entry_bar_volume_spike",
    "entry_bar_body_pct", "entry_bar_upper_wick_pct", "entry_bar_lower_wick_pct",
]
DURING_TRADE_FEATURES = [
    "max_favorable_excursion", "max_adverse_excursion", "mfe_pct", "mae_pct",
    "bars_to_mfe", "bars_to_mae", "went_green_before_red", "went_red_before_green",
    "hold_bars",
]
EXIT_FEATURES = [
    "profit_capture_ratio", "giveback_from_mfe", "exit_efficiency",
    "exit_near_trade_high", "exit_near_trade_low", "exit_after_momentum_fade",
    "exit_after_vwap_loss",
]
CONTEXT_FEATURES = [
    "strategy_family", "strategy_confidence", "verification_status",
    "independence_score", "avg_correlation", "composite_score", "rl_eligible",
    "human_bucket",
]
TARGET_LABELS = ["normalized_pnl", "outcome_win", "outcome_loss", "outcome_flat",
                 "return_pct"]
META_COLS = ["strategy_id", "symbol", "timeframe", "entry_time", "exit_time",
             "entry_price", "exit_price", "volume", "direction", "raw_pnl_usd",
             "source", "split", "has_pre_features"]

FEATURE_GROUPS = {
    "pre_entry_features": PRE_ENTRY_FEATURES,
    "during_trade_features": DURING_TRADE_FEATURES,
    "exit_features": EXIT_FEATURES,
    "context_features": CONTEXT_FEATURES,
    "target_labels": TARGET_LABELS,
    "meta": META_COLS,
}


def _sd(a, b):
    try:
        return float(a) / float(b) if b else np.nan
    except (TypeError, ZeroDivisionError):
        return np.nan


# -- read trades with intraday context from a TradingView CSV --------------
def read_trades_with_context(csv_path) -> list:
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    h = {k: (k or "").strip().lower() for k in rows[0]}
    c_type = _col(h, ["type"]); c_num = _col(h, ["trade"]) or _col(h, ["#"])
    c_dt = _col(h, ["date"]) or _col(h, ["time"]); c_price = _col(h, ["price"])
    c_qty = _col(h, ["qty"]) or _col(h, ["size"], ["value", "%"]) or _col(h, ["contracts"])
    c_pnl = _col(h, ["profit"], ["%", "cum"]) or _col(h, ["p&l"], ["%", "cum"]) or _col(h, ["pnl"], ["%", "cum"])
    c_pct = _col(h, ["profit", "%"], ["cum"]) or _col(h, ["p&l", "%"], ["cum"]) or _col(h, ["pnl", "%"], ["cum"])
    c_mfe = _col(h, ["favorable"], ["%"]) or _col(h, ["run-up"], ["%"])
    c_mfp = _col(h, ["favorable", "%"]) or _col(h, ["run-up", "%"])
    c_mae = _col(h, ["adverse"], ["%"]) or _col(h, ["drawdown"], ["%", "cum"])
    c_map = _col(h, ["adverse", "%"]) or _col(h, ["drawdown", "%"], ["cum"])
    if c_type is None or c_dt is None or c_price is None:
        return []
    groups: dict = {}
    for i, r in enumerate(rows):
        key = (r.get(c_num) or "").strip() if c_num else f"_seq{i // 2}"
        groups.setdefault(key, []).append(r)
    out = []
    for grp in groups.values():
        en = next((r for r in grp if "entry" in (r.get(c_type) or "").lower()), None)
        ex = next((r for r in grp if "exit" in (r.get(c_type) or "").lower()), None)
        if en is None or ex is None:
            continue
        e_dt, x_dt = _parse_dt(en[c_dt]), _parse_dt(ex[c_dt])
        e_px, x_px = _num(en[c_price]), _num(ex[c_price])
        if pd.isna(e_dt) or pd.isna(x_dt) or e_px is None or x_px is None:
            continue
        if x_dt < e_dt:
            e_dt, x_dt, e_px, x_px = x_dt, e_dt, x_px, e_px
        pct = _num(ex.get(c_pct)) if c_pct else None
        out.append({
            "entry_ts": e_dt, "exit_ts": x_dt, "entry_price": float(e_px),
            "exit_price": float(x_px),
            "qty": abs(_num(ex.get(c_qty)) or _num(en.get(c_qty)) or 0.0) if c_qty else 0.0,
            "raw_pnl": float(_num(ex.get(c_pnl))) if (c_pnl and _num(ex.get(c_pnl)) is not None) else np.nan,
            "return_pct": (pct / 100.0) if pct is not None else (x_px - e_px) / e_px,
            "mfe_usd": _num(ex.get(c_mfe)) if c_mfe else np.nan,
            "mfe_pct": (_num(ex.get(c_mfp)) or 0.0) / 100.0 if c_mfp else np.nan,
            "mae_usd": _num(ex.get(c_mae)) if c_mae else np.nan,
            "mae_pct": (_num(ex.get(c_map)) or 0.0) / 100.0 if c_map else np.nan,
            "is_long": "short" not in (en.get(c_type) or "").lower(),
        })
    out.sort(key=lambda t: t["entry_ts"])
    return out


def _rsi(c, n=14):
    d = np.diff(c)
    up = d.clip(min=0)[-n:].mean(); dn = (-d.clip(max=0))[-n:].mean()
    return 100.0 if dn == 0 and up > 0 else (50.0 if dn == 0 else float(100 - 100 / (1 + up / dn)))


# -- A. pre-entry features (bars with timestamp <= entry_time ONLY) ---------
def extract_pre_entry(bars: pd.DataFrame, entry_ts, is_long: bool = True):
    pre = bars[bars["timestamp"] <= entry_ts]
    if len(pre) < LOOKBACK:
        return None
    c = pre["close"].to_numpy(float); o = pre["open"].to_numpy(float)
    hi = pre["high"].to_numpy(float); lo = pre["low"].to_numpy(float)
    v = pre["volume"].to_numpy(float)
    last = c[-1]
    sma20, sma50 = c[-20:].mean(), c[-50:].mean()
    cs = pd.Series(c)
    ema9 = cs.ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = cs.ewm(span=21, adjust=False).mean().iloc[-1]
    macd_line = cs.ewm(span=12, adjust=False).mean() - cs.ewm(span=26, adjust=False).mean()
    macd_hist = float(macd_line.iloc[-1] - macd_line.ewm(span=9, adjust=False).mean().iloc[-1])
    tr = np.maximum(hi[1:] - lo[1:], np.maximum(abs(hi[1:] - c[:-1]), abs(lo[1:] - c[:-1])))
    atr = tr[-14:].mean()
    rets = np.diff(c[-21:]) / c[-21:-1]
    green = c > o
    cg = 0
    for x in reversed(green):
        if x:
            cg += 1
        else:
            break
    cr = 0
    for x in reversed(green):
        if not x:
            cr += 1
        else:
            break
    rng = hi[-1] - lo[-1]
    # session vwap + opening range (entry's calendar day, up to entry)
    d = pre["timestamp"].dt.date.to_numpy()
    day = d[-1]
    sess = pre[pre["timestamp"].dt.date == day]
    tp = (sess["high"] + sess["low"] + sess["close"]) / 3.0
    vwap = _sd((tp * sess["volume"]).sum(), sess["volume"].sum()) or last
    orb = sess.iloc[:2]
    or_hi = float(orb["high"].max()) if len(orb) else np.nan
    or_lo = float(orb["low"].min()) if len(orb) else np.nan
    prev_days = sorted({x for x in d if x < day})
    if prev_days:
        pdb = pre[pre["timestamp"].dt.date == prev_days[-1]]
        pdh, pdl = float(pdb["high"].max()), float(pdb["low"].min())
    else:
        pdh = pdl = np.nan
    std20 = c[-20:].std()
    return {
        "close_vs_sma20": _sd(last - sma20, sma20),
        "close_vs_sma50": _sd(last - sma50, sma50),
        "sma20_slope": _sd(c[-20:].mean() - c[-25:-5].mean(), c[-25:-5].mean()),
        "sma50_slope": _sd(c[-50:].mean() - c[-55:-5].mean(), c[-55:-5].mean()) if len(c) >= 55 else np.nan,
        "ema9_vs_ema21": _sd(ema9 - ema21, ema21),
        "higher_high_higher_low_flag": float(hi[-1] > hi[-2] > hi[-3] and lo[-1] > lo[-2] > lo[-3]),
        "distance_from_recent_high": _sd(last - hi[-20:].max(), hi[-20:].max()),
        "distance_from_recent_low": _sd(last - lo[-20:].min(), lo[-20:].min()),
        "rsi14": _rsi(c, 14),
        "stoch_position": _sd(last - lo[-14:].min(), hi[-14:].max() - lo[-14:].min()),
        "macd_histogram": macd_hist,
        "return_5_bars": _sd(last, c[-6]) - 1, "return_10_bars": _sd(last, c[-11]) - 1,
        "return_20_bars": _sd(last, c[-21]) - 1,
        "consecutive_green_bars": float(cg), "consecutive_red_bars": float(cr),
        "relative_volume_20": _sd(v[-1], v[-20:].mean()),
        "volume_spike_flag": float(v[-1] > 2 * v[-20:].mean()),
        "volume_trend_20": _sd(v[-10:].mean() - v[-20:-10].mean(), v[-20:-10].mean()),
        "dollar_volume": float(v[-1] * last),
        "atr14": float(atr), "atr_pct": _sd(atr, last),
        "candle_range_pct": _sd(rng, last),
        "realized_vol_20": float(np.std(rets)),
        "gap_pct_from_prev_close": _sd(o[-1] - c[-2], c[-2]),
        "distance_from_vwap": _sd(last - vwap, vwap),
        "close_above_vwap": float(last > vwap),
        "vwap_reclaim_flag": float(c[-2] < vwap <= last),
        "zscore_from_sma20": _sd(last - sma20, std20),
        "distance_to_previous_day_high": _sd(last - pdh, pdh),
        "distance_to_previous_day_low": _sd(last - pdl, pdl),
        "distance_to_opening_range_high": _sd(last - or_hi, or_hi),
        "distance_to_opening_range_low": _sd(last - or_lo, or_lo),
        "entry_distance_from_vwap": _sd(last - vwap, vwap),
        "entry_distance_from_sma20": _sd(last - sma20, sma20),
        "entry_bar_volume_spike": float(v[-1] > 2 * v[-20:].mean()),
        "entry_bar_body_pct": _sd(abs(c[-1] - o[-1]), rng),
        "entry_bar_upper_wick_pct": _sd(hi[-1] - max(c[-1], o[-1]), rng),
        "entry_bar_lower_wick_pct": _sd(min(c[-1], o[-1]) - lo[-1], rng),
    }


# -- B. during-trade features ---------------------------------------------
def extract_during(trade: dict, bars=None) -> dict:
    hold_bars = round((trade["exit_ts"] - trade["entry_ts"]).total_seconds() / 60.0 / BAR_MINUTES)
    out = {
        "max_favorable_excursion": trade.get("mfe_usd", np.nan),
        "max_adverse_excursion": trade.get("mae_usd", np.nan),
        "mfe_pct": trade.get("mfe_pct", np.nan), "mae_pct": trade.get("mae_pct", np.nan),
        "bars_to_mfe": np.nan, "bars_to_mae": np.nan,
        "went_green_before_red": np.nan, "went_red_before_green": np.nan,
        "hold_bars": float(hold_bars),
    }
    if bars is not None and not bars.empty:
        path = bars[(bars["timestamp"] >= trade["entry_ts"]) &
                    (bars["timestamp"] <= trade["exit_ts"])]
        if len(path):
            ep = trade["entry_price"]
            r = (path["close"].to_numpy(float) - ep) / ep
            if not trade["is_long"]:
                r = -r
            out["bars_to_mfe"] = float(int(np.argmax(r)))
            out["bars_to_mae"] = float(int(np.argmin(r)))
            out["went_green_before_red"] = float(np.argmax(r) < np.argmin(r))
            out["went_red_before_green"] = float(np.argmin(r) < np.argmax(r))
    return out


# -- C. exit-quality features (from realized MFE/MAE/return) ----------------
def extract_exit_quality(trade: dict) -> dict:
    ret = trade["return_pct"]; mfe = trade.get("mfe_pct", np.nan); mae = trade.get("mae_pct", np.nan)
    rng = (mfe - mae) if (not np.isnan(mfe) and not np.isnan(mae)) else np.nan
    return {
        "profit_capture_ratio": _sd(ret, mfe) if (not np.isnan(mfe) and mfe > 0) else np.nan,
        "giveback_from_mfe": (mfe - ret) if not np.isnan(mfe) else np.nan,
        "exit_efficiency": _sd(ret - mae, rng) if (rng and not np.isnan(rng)) else np.nan,
        "exit_near_trade_high": float(ret >= 0.9 * mfe) if (not np.isnan(mfe) and mfe > 0) else np.nan,
        "exit_near_trade_low": float((ret - mae) <= 0.1 * rng) if (rng and not np.isnan(rng)) else np.nan,
        "exit_after_momentum_fade": np.nan,   # needs bars
        "exit_after_vwap_loss": np.nan,        # needs bars
    }


# -- D/E. context + labels + split ----------------------------------------
def assign_split(exit_ts) -> str:
    d = pd.Timestamp(exit_ts).date().isoformat()
    if d < "2024-01-01":
        return "train"
    if d < "2025-01-01":
        return "validation"
    return "oos"


def labels(trade: dict, reference_qty: float = REFERENCE_QTY) -> dict:
    q = trade.get("qty", 0.0) or 0.0
    sf = reference_qty / q if q > reference_qty else 1.0
    raw = trade["raw_pnl"]
    norm = raw * sf if not np.isnan(raw) else np.nan
    win = int(norm > 0) if not np.isnan(norm) else np.nan
    loss = int(norm < 0) if not np.isnan(norm) else np.nan
    flat = int(norm == 0) if not np.isnan(norm) else np.nan
    return {"normalized_pnl": round(norm, 4) if not np.isnan(norm) else np.nan,
            "raw_pnl_usd": round(raw, 2) if not np.isnan(raw) else np.nan,
            "return_pct": round(trade["return_pct"], 6),
            "outcome_win": win, "outcome_loss": loss, "outcome_flat": flat}


def build_row(trade: dict, bars, meta: dict, context: dict,
              reference_qty: float = REFERENCE_QTY) -> dict:
    pre = extract_pre_entry(bars, trade["entry_ts"], trade["is_long"]) \
        if (bars is not None and not bars.empty) else None
    bar_vol = np.nan
    if bars is not None and not bars.empty:
        eb = bars[bars["timestamp"] <= trade["entry_ts"]]
        bar_vol = float(eb["volume"].iloc[-1]) if len(eb) else np.nan
    row = {
        "strategy_id": meta["strategy_id"], "symbol": meta["symbol"],
        "timeframe": meta["timeframe"], "entry_time": trade["entry_ts"].isoformat(),
        "exit_time": trade["exit_ts"].isoformat(), "entry_price": trade["entry_price"],
        "exit_price": trade["exit_price"], "volume": bar_vol,
        "direction": "long" if trade["is_long"] else "short",
        "source": "tradingview_verified", "split": assign_split(trade["exit_ts"]),
        "has_pre_features": pre is not None,
    }
    row.update({k: np.nan for k in PRE_ENTRY_FEATURES})
    row.update(pre or {})
    row.update(extract_during(trade, bars))
    row.update(extract_exit_quality(trade))
    row.update(labels(trade, reference_qty))
    row.update({k: context.get(k, np.nan) for k in CONTEXT_FEATURES})
    return row


def _demo():
    n = 80
    idx = pd.date_range("2025-03-03 09:30", periods=n, freq="15min", tz="UTC")
    c = 100 + np.cumsum(np.sin(np.linspace(0, 9, n))) + np.linspace(0, 3, n)
    bars = pd.DataFrame({"timestamp": idx, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": np.full(n, 1e5)})
    entry = idx[60]
    f1 = extract_pre_entry(bars, entry)
    b2 = bars.copy()
    b2.loc[b2["timestamp"] > entry, ["open", "high", "low", "close", "volume"]] = -1.0
    assert f1 == extract_pre_entry(b2, entry), "pre-entry must ignore post-entry bars"
    assert set(f1) == set(PRE_ENTRY_FEATURES)
    print("trade_features._demo OK (%d pre features, leak-safe)" % len(f1))


if __name__ == "__main__":
    _demo()
