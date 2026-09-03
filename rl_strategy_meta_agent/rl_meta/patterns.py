"""Chart-pattern and candle-pattern recognition for the trade-outcome dataset.

For a trade entered at bar e, all detectors read ONLY bars[:e+1] (the 50-100
candles up to and including the entry candle). No post-entry bar, and no trade
result (realized profit/loss, excursions, or any post-entry exit data) is ever
used to assign a pattern -- that would be label leakage. Session scaffolding
(VWAP, prior-day high/low,
opening range, own trend regime) is reused from market_context.build_context_frame,
which is point-in-time and unit-tested.

Each detector yields (present, strength in [0,1], direction, family). The flat
column set required by the dataset is assembled in detect_patterns(); the
strengths/directions also drive the summary (primary_pattern, pattern_count,
bullish/bearish/neutral scores).

  # ponytail: deterministic price-action heuristics, not a trained pattern net.
  #           Thresholds are eyeballed; the report tells us if they carry signal.

Research-only: no broker, no orders, no live path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rl_meta.market_context import build_context_frame

WINDOW = 100
MIN_HISTORY = 20            # fewer bars before entry -> "unknown", never a fake label
DOJI_BODY = 0.10
PIN_WICK = 0.55
PIN_BODY = 0.35
REJ_WICK = 0.45
NEAR_LEVEL = 0.006
EXT_VWAP = 0.02

FAMILY = {
    "doji": "candle_structure", "pin_bar": "candle_structure",
    "rejection_wick": "reversal", "engulfing": "reversal",
    "inside_bar": "candle_structure", "large_body_momentum": "candle_structure",
    "higher_low_sequence": "trend", "lower_high_sequence": "trend",
    "trend_continuation": "trend", "bull_flag": "trend", "bear_flag": "trend",
    "opening_range_breakout": "breakout", "opening_range_breakdown": "breakout",
    "pdh_retest": "breakout", "pdl_retest": "breakout",
    "failed_breakout": "breakout", "failed_breakdown": "breakout",
    "failed_reclaim": "breakout",
    "vwap_reclaim": "mean_reversion", "vwap_rejection": "reversal",
    "vwap_pullback": "mean_reversion", "extended_from_vwap": "mean_reversion",
    "mean_reversion_setup": "mean_reversion",
    "range_compression": "compression", "inside_bar_compression": "compression",
    "volatility_expansion": "volume",
}

# flat dataset columns produced for every trade (besides primary_pattern str)
FLAG_COLS = [
    "candle_body_pct", "upper_wick_pct", "lower_wick_pct", "close_location",
    "doji_flag", "pin_bar_flag", "rejection_wick_flag", "engulfing_flag",
    "inside_bar_flag", "large_body_momentum_flag",
    "higher_low_sequence", "lower_high_sequence", "trend_continuation_flag",
    "bull_flag_strength", "bear_flag_strength",
    "opening_range_breakout_flag", "opening_range_breakdown_flag",
    "pdh_retest_flag", "pdl_retest_flag", "failed_breakout_flag",
    "failed_breakdown_flag", "failed_reclaim_flag",
    "vwap_reclaim_flag", "vwap_rejection_flag", "vwap_pullback_flag",
    "extended_from_vwap_flag", "mean_reversion_setup_flag",
    "range_compression_flag", "volatility_expansion_flag",
    "inside_bar_compression_flag",
    "primary_pattern_strength", "pattern_count",
    "bullish_pattern_score", "bearish_pattern_score", "neutral_pattern_score",
]
ALL_COLS = FLAG_COLS + ["primary_pattern"]


def _clamp(x) -> float:
    return float(min(1.0, max(0.0, x)))


def _consec(arr, higher: bool) -> int:
    """Consecutive strictly-monotonic steps ending at the last element."""
    cnt = 0
    for i in range(len(arr) - 1, 0, -1):
        if (arr[i] > arr[i - 1]) if higher else (arr[i] < arr[i - 1]):
            cnt += 1
        else:
            break
    return cnt


def unknown_row() -> dict:
    row = {c: np.nan for c in FLAG_COLS}
    row["primary_pattern"] = "unknown"
    return row


def detect_patterns(bars: pd.DataFrame, e: int, ctx: pd.DataFrame | None = None,
                    window: int = WINDOW, min_history: int = MIN_HISTORY) -> dict:
    """All pattern columns for a trade entered at absolute bar index e.

    Reads bars[:e+1] only. Returns unknown_row() if history is insufficient."""
    n = len(bars)
    if e < min_history or e >= n:
        return unknown_row()
    if ctx is None:
        ctx = build_context_frame(bars)
    lo = max(0, e - window + 1)
    o = bars["open"].to_numpy(float)[lo:e + 1]
    h = bars["high"].to_numpy(float)[lo:e + 1]
    l = bars["low"].to_numpy(float)[lo:e + 1]
    c = bars["close"].to_numpy(float)[lo:e + 1]
    cw = ctx.iloc[lo:e + 1].reset_index(drop=True)
    last = len(c) - 1

    # ---- candle structure (entry candle, + prev for 2-bar patterns) ----
    rng = h[last] - l[last]
    body = abs(c[last] - o[last])
    if rng > 0:
        body_pct = body / rng
        up_wick = (h[last] - max(o[last], c[last])) / rng
        lo_wick = (min(o[last], c[last]) - l[last]) / rng
        close_loc = (c[last] - l[last]) / rng
    else:
        body_pct = up_wick = lo_wick = 0.0
        close_loc = 0.5
    bull = c[last] >= o[last]
    tr = h - l
    r20 = float(tr[-20:].mean()) if len(tr) >= 20 else float(tr.mean())
    r5 = float(tr[-5:].mean())

    p = {}  # name -> (present, strength, direction)

    p["doji"] = (body_pct < DOJI_BODY, _clamp((DOJI_BODY - body_pct) / DOJI_BODY), "neutral")
    pin_bull = lo_wick >= PIN_WICK and body_pct <= PIN_BODY and up_wick <= 0.2
    pin_bear = up_wick >= PIN_WICK and body_pct <= PIN_BODY and lo_wick <= 0.2
    p["pin_bar"] = (pin_bull or pin_bear, _clamp(max(lo_wick, up_wick)),
                    "bullish" if pin_bull else "bearish" if pin_bear else "neutral")
    rej_bull = lo_wick >= REJ_WICK and lo_wick > up_wick
    rej_bear = up_wick >= REJ_WICK and up_wick > lo_wick
    p["rejection_wick"] = (rej_bull or rej_bear, _clamp(max(lo_wick, up_wick)),
                           "bullish" if rej_bull else "bearish" if rej_bear else "neutral")
    prev_body = abs(c[last - 1] - o[last - 1])
    eng_bull = bull and c[last - 1] < o[last - 1] and c[last] >= o[last - 1] and o[last] <= c[last - 1]
    eng_bear = (not bull) and c[last - 1] > o[last - 1] and c[last] <= o[last - 1] and o[last] >= c[last - 1]
    p["engulfing"] = (eng_bull or eng_bear,
                      _clamp(body / prev_body if prev_body > 0 else 0.0),
                      "bullish" if eng_bull else "bearish" if eng_bear else "neutral")
    inside = h[last] < h[last - 1] and l[last] > l[last - 1]
    prev_rng = h[last - 1] - l[last - 1]
    p["inside_bar"] = (inside, _clamp(1 - rng / prev_rng) if prev_rng > 0 else 0.0, "neutral")
    lbm = body_pct > 0.70 and rng > 1.3 * r20
    p["large_body_momentum"] = (lbm, _clamp(body_pct), "bullish" if bull else "bearish")

    # ---- trend structure ----
    hl_seq = _consec(l, higher=True)
    lh_seq = _consec(h, higher=False)
    p["higher_low_sequence"] = (hl_seq >= 3, _clamp(hl_seq / 6), "bullish")
    p["lower_high_sequence"] = (lh_seq >= 3, _clamp(lh_seq / 6), "bearish")
    regime = cw["ctx_own_regime"].iloc[-1]
    ret_w = (c[last] - c[0]) / c[0] if c[0] > 0 else 0.0
    p["trend_continuation"] = (regime in ("uptrend", "downtrend"),
                               _clamp(abs(ret_w) / 0.05),
                               "bullish" if regime == "uptrend" else
                               "bearish" if regime == "downtrend" else "neutral")
    # flag = strong impulse then shallow consolidation into entry
    split = max(3, int(len(c) * 0.6))
    imp = (c[split] - c[0]) / c[0] if c[0] > 0 else 0.0
    pull = (c[last] - c[split]) / c[split] if c[split] > 0 else 0.0
    bull_flag = imp > 0.03 and -0.02 < pull < 0.01
    bear_flag = imp < -0.03 and -0.01 < pull < 0.02
    bull_flag_s = _clamp(imp / 0.06) * _clamp(1 - abs(pull) / 0.02) if bull_flag else 0.0
    bear_flag_s = _clamp(-imp / 0.06) * _clamp(1 - abs(pull) / 0.02) if bear_flag else 0.0
    p["bull_flag"] = (bull_flag, bull_flag_s, "bullish")
    p["bear_flag"] = (bear_flag, bear_flag_s, "bearish")

    # ---- breakout structure (ctx at entry) ----
    row = cw.iloc[-1]
    orh, orl = row["ctx_or_dist_high"], row["ctx_or_dist_low"]
    mins = row["ctx_minutes_since_open"]
    intraday = pd.notna(mins) and mins <= 180
    p["opening_range_breakout"] = (pd.notna(orh) and orh > 0 and intraday,
                                   _clamp(orh / 0.01) if pd.notna(orh) else 0.0, "bullish")
    p["opening_range_breakdown"] = (pd.notna(orl) and orl < 0 and intraday,
                                    _clamp(-orl / 0.01) if pd.notna(orl) else 0.0, "bearish")
    pdh, pdl = row["ctx_pdh_dist"], row["ctx_pdl_dist"]
    p["pdh_retest"] = (pd.notna(pdh) and abs(pdh) < NEAR_LEVEL,
                       _clamp((NEAR_LEVEL - abs(pdh)) / NEAR_LEVEL) if pd.notna(pdh) else 0.0, "neutral")
    p["pdl_retest"] = (pd.notna(pdl) and abs(pdl) < NEAR_LEVEL,
                       _clamp((NEAR_LEVEL - abs(pdl)) / NEAR_LEVEL) if pd.notna(pdl) else 0.0, "neutral")
    res = float(h[:-3].max()) if len(h) > 3 else np.nan        # structure excludes last 3 bars
    sup = float(l[:-3].min()) if len(l) > 3 else np.nan
    poked_up = len(h) > 3 and float(h[-3:].max()) > res
    poked_dn = len(l) > 3 and float(l[-3:].min()) < sup
    fbo = poked_up and c[last] < res
    fbd = poked_dn and c[last] > sup
    p["failed_breakout"] = (fbo, _clamp((res - c[last]) / (0.01 * c[last])) if fbo else 0.0, "bearish")
    p["failed_breakdown"] = (fbd, _clamp((c[last] - sup) / (0.01 * c[last])) if fbd else 0.0, "bullish")

    # ---- VWAP / mean reversion ----
    vd = cw["ctx_vwap_dist"].to_numpy(float)
    vlast = vd[-1]
    recl = len(vd) >= 4 and np.nanmin(vd[-4:-1]) < 0 and vlast > 0
    rej = len(vd) >= 4 and np.nanmax(vd[-4:-1]) > 0 and vlast < 0
    p["vwap_reclaim"] = (bool(recl), _clamp(abs(vlast) / 0.01) if recl else 0.0, "bullish")
    p["vwap_rejection"] = (bool(rej), _clamp(abs(vlast) / 0.01) if rej else 0.0, "bearish")
    pull_vwap = regime == "uptrend" and pd.notna(vlast) and 0 < vlast < 0.01
    p["vwap_pullback"] = (bool(pull_vwap), _clamp((0.01 - vlast) / 0.01) if pull_vwap else 0.0, "bullish")
    ext = pd.notna(vlast) and abs(vlast) > EXT_VWAP
    p["extended_from_vwap"] = (bool(ext), _clamp(abs(vlast) / 0.05) if ext else 0.0,
                               "bearish" if (ext and vlast > 0) else "bullish" if ext else "neutral")
    mr = ext and ((vlast > 0 and rej_bear) or (vlast < 0 and rej_bull))
    p["mean_reversion_setup"] = (bool(mr), _clamp(abs(vlast) / 0.05) if mr else 0.0,
                                 "bearish" if (mr and vlast > 0) else "bullish" if mr else "neutral")
    # failed reclaim: reclaimed VWAP then lost it again
    frc = len(vd) >= 5 and np.nanmin(vd[-5:-2]) < 0 and np.nanmax(vd[-3:-1]) > 0 and vlast < 0
    p["failed_reclaim"] = (bool(frc), _clamp(abs(vlast) / 0.01) if frc else 0.0, "bearish")

    # ---- compression / volatility ----
    comp = r20 > 0 and r5 < 0.6 * r20
    vexp = r20 > 0 and r5 > 1.5 * r20
    p["range_compression"] = (bool(comp), _clamp(1 - r5 / r20) if r20 > 0 else 0.0, "neutral")
    p["volatility_expansion"] = (bool(vexp), _clamp((r5 / r20 - 1) / 2) if r20 > 0 else 0.0, "neutral")
    p["inside_bar_compression"] = (inside and comp, _clamp((1 - rng / prev_rng) if prev_rng > 0 else 0.0), "neutral")

    return _assemble(p, body_pct, up_wick, lo_wick, close_loc, hl_seq, lh_seq,
                     bull_flag_s, bear_flag_s)


def _assemble(p, body_pct, up_wick, lo_wick, close_loc, hl_seq, lh_seq,
              bull_flag_s, bear_flag_s) -> dict:
    def fl(name):
        return 1.0 if p[name][0] else 0.0
    row = {
        "candle_body_pct": body_pct, "upper_wick_pct": up_wick,
        "lower_wick_pct": lo_wick, "close_location": close_loc,
        "doji_flag": fl("doji"), "pin_bar_flag": fl("pin_bar"),
        "rejection_wick_flag": fl("rejection_wick"), "engulfing_flag": fl("engulfing"),
        "inside_bar_flag": fl("inside_bar"), "large_body_momentum_flag": fl("large_body_momentum"),
        "higher_low_sequence": float(hl_seq), "lower_high_sequence": float(lh_seq),
        "trend_continuation_flag": fl("trend_continuation"),
        "bull_flag_strength": bull_flag_s, "bear_flag_strength": bear_flag_s,
        "opening_range_breakout_flag": fl("opening_range_breakout"),
        "opening_range_breakdown_flag": fl("opening_range_breakdown"),
        "pdh_retest_flag": fl("pdh_retest"), "pdl_retest_flag": fl("pdl_retest"),
        "failed_breakout_flag": fl("failed_breakout"),
        "failed_breakdown_flag": fl("failed_breakdown"),
        "failed_reclaim_flag": fl("failed_reclaim"),
        "vwap_reclaim_flag": fl("vwap_reclaim"), "vwap_rejection_flag": fl("vwap_rejection"),
        "vwap_pullback_flag": fl("vwap_pullback"),
        "extended_from_vwap_flag": fl("extended_from_vwap"),
        "mean_reversion_setup_flag": fl("mean_reversion_setup"),
        "range_compression_flag": fl("range_compression"),
        "volatility_expansion_flag": fl("volatility_expansion"),
        "inside_bar_compression_flag": fl("inside_bar_compression"),
    }
    present = [(name, s, d) for name, (pr, s, d) in p.items() if pr]
    row["pattern_count"] = float(len(present))
    row["bullish_pattern_score"] = round(sum(s for _, s, d in present if d == "bullish"), 4)
    row["bearish_pattern_score"] = round(sum(s for _, s, d in present if d == "bearish"), 4)
    row["neutral_pattern_score"] = round(sum(s for _, s, d in present if d == "neutral"), 4)
    if present:
        name, s, _ = max(present, key=lambda x: x[1])
        row["primary_pattern"], row["primary_pattern_strength"] = name, round(s, 4)
    else:
        row["primary_pattern"], row["primary_pattern_strength"] = "none", 0.0
    return row


def _demo():
    n = 80
    idx = pd.date_range("2024-03-04 13:30", periods=n, freq="15min", tz="UTC")
    rng = np.linspace(100, 112, n)
    b = pd.DataFrame({"timestamp": idx, "open": rng, "high": rng + 0.4,
                      "low": rng - 0.4, "close": rng + 0.2, "volume": np.full(n, 1000.0)})
    ctx = build_context_frame(b)
    e = 60
    r1 = detect_patterns(b, e, ctx)
    assert set(ALL_COLS) <= set(r1), "missing required columns"
    # no-lookahead: wreck bars after e, labels must not move
    bad = b.copy(); bad.loc[e + 1:, ["open", "high", "low", "close", "volume"]] *= 6
    r2 = detect_patterns(bad, e)            # recompute ctx from corrupted bars
    for k in ALL_COLS:
        a, c = r1[k], r2[k]
        assert a == c or (isinstance(a, float) and np.isnan(a) and np.isnan(c)), f"lookahead in {k}"
    # too little history -> unknown, not a fake label
    assert detect_patterns(b, 5)["primary_pattern"] == "unknown"
    print("patterns._demo OK  primary =", r1["primary_pattern"],
          "| count", r1["pattern_count"], "| cols", len(ALL_COLS))


if __name__ == "__main__":
    _demo()
