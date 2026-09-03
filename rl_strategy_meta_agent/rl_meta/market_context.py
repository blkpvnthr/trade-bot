"""Leak-free market context, pattern, and regime features for trade management.

Every value at bar k uses ONLY bars[:k+1] plus prior-session aggregates -- it is
strictly point-in-time, so these columns are safe to read at any mid-trade
decision bar. Sessions are keyed on US/Eastern calendar days; VWAP and the
opening range reset each session. Reference-series trends (SPY / QQQ / sector
ETF) are joined with a BACKWARD as-of merge, so no future bar can leak in.

Patterns and the regime label are deterministic heuristics on price structure
(NOT a trained classifier). They are descriptive, overlapping setups collapsed
to one primary label by priority.
  # ponytail: heuristic thresholds tuned by eye; upgrade to data-fit cutoffs or
  #           a labeled pattern set only if the descriptive tables justify it.

Research-only: no broker, no orders, no live path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"
OPEN_MIN = 9 * 60 + 30        # 09:30 ET in minutes
OR_BARS = 2                   # opening range = first 2 bars (30min on 15Min data)
TREND_LOOKBACK = 20           # bars for SPY / QQQ / sector momentum
RVOL_WIN = 20                 # bars for the relative-volume baseline
SMA_WIN = 20

# pattern thresholds
NEAR_LEVEL = 0.006            # within 0.6% counts as a PDH/PDL retest
BREAKOUT_LB = 20             # bars of structure for breakout / failed breakout
IMPULSE_W = 10               # impulse window for flags
FLAG_IMPULSE = 0.03          # >=3% prior thrust to call it a flag
FLAG_PULLBACK = 0.015        # <=1.5% counter-move during the flag
MR_VWAP = 0.02               # >=2% from VWAP to call it a mean-reversion entry
REGIME_SLOPE = 0.001         # own-trend SMA slope cutoff
MKT_TREND = 0.005            # SPY 20-bar return cutoff for bull/bear

# best-effort static symbol -> sector ETF map; unmapped -> NaN sector strength.
# ponytail: hand-mapped for this 60-symbol universe, not a GICS lookup service.
SECTOR_ETF = {
    "AAPL": "XLK", "AMAT": "XLK", "AMD": "XLK", "ASML": "XLK", "AVGO": "XLK",
    "FORM": "XLK", "KLAC": "XLK", "MU": "XLK", "NVDA": "XLK", "ORCL": "XLK",
    "SMCI": "XLK", "SNDK": "XLK", "TSM": "XLK", "TTMI": "XLK", "ARM": "XLK",
    "AMSC": "XLK", "SOXL": "XLK", "PLTR": "XLK", "IONQ": "XLK", "QBTS": "XLK",
    "QUBT": "XLK", "RGTI": "XLK", "HSAI": "XLK", "GWRE": "XLK", "IREN": "XLK",
    "GOOG": "XLC", "GOOGL": "XLC", "META": "XLC", "MGNI": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "LULU": "XLY", "WING": "XLY", "COCO": "XLY",
    "YOU": "XLY",
    "HOOD": "XLF", "COIN": "XLF", "BGC": "XLF", "RYAN": "XLF", "MSTR": "XLF",
    "CRCL": "XLF", "XP": "XLF",
    "UNH": "XLV", "ENSG": "XLV", "TGTX": "XLV", "FIGS": "XLV", "WGS": "XLV",
    "CLOV": "XLV",
    "KTOS": "XLI", "RKLB": "XLI", "ASTS": "XLI", "RYCEY": "XLI", "ONDS": "XLI",
    "SHLS": "XLI",
    "CE": "XLB", "CRML": "XLB", "SLV": "XLB",
}
REF_SYMBOLS = ["SPY", "QQQ"] + sorted(set(SECTOR_ETF.values()))

PATTERNS = ["breakout", "failed_breakout", "bull_flag", "bear_flag",
            "vwap_reclaim", "pdh_retest", "pdl_retest", "trend_continuation",
            "mean_reversion", "opening_range_expansion", "none"]
CTX_NUMERIC = ["ctx_vwap_dist", "ctx_pdh_dist", "ctx_pdl_dist", "ctx_or_dist_high",
               "ctx_or_dist_low", "ctx_minutes_since_open", "ctx_rvol_accel",
               "ctx_spy_trend", "ctx_qqq_trend", "ctx_sector_strength",
               "ctx_is_first_hour", "ctx_is_last_hour"]


def _ref_momentum(ref: pd.DataFrame, ts: pd.Series) -> np.ndarray:
    """TREND_LOOKBACK-bar return of `ref`, backward as-of aligned to `ts`."""
    if ref is None or ref.empty:
        return np.full(len(ts), np.nan)
    r = ref.sort_values("timestamp")
    rts = pd.to_datetime(r["timestamp"], utc=True)
    mom = r["close"].astype(float).pct_change(TREND_LOOKBACK)
    right = pd.DataFrame({"timestamp": rts.to_numpy(), "m": mom.to_numpy()}).dropna()
    left = pd.DataFrame({"timestamp": ts.to_numpy()})
    return pd.merge_asof(left, right, on="timestamp", direction="backward")["m"].to_numpy()


def build_context_frame(bars: pd.DataFrame, spy=None, qqq=None, sector=None) -> pd.DataFrame:
    """Per-bar leak-free context, aligned 0..n-1 to `bars` (must be sorted)."""
    b = bars.reset_index(drop=True)
    ts = pd.to_datetime(b["timestamp"], utc=True)
    et = ts.dt.tz_convert(ET)
    sess = et.dt.normalize()
    bis = b.groupby(sess).cumcount()
    mins = (et.dt.hour * 60 + et.dt.minute) - OPEN_MIN

    hi, lo, cl, vol = (b["high"].astype(float), b["low"].astype(float),
                       b["close"].astype(float), b["volume"].astype(float))

    # session VWAP (expanding within the day, typical price)
    tp = (hi + lo + cl) / 3.0
    cum_pv = (tp * vol).groupby(sess).cumsum()
    cum_v = vol.groupby(sess).cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v

    # opening range: high/low of the first OR_BARS bars; NaN until it has formed
    orh = hi.where(bis < OR_BARS).groupby(sess).cummax().groupby(sess).ffill()
    orl = lo.where(bis < OR_BARS).groupby(sess).cummin().groupby(sess).ffill()
    formed = bis >= OR_BARS
    orh, orl = orh.where(formed), orl.where(formed)

    # prior-day high/low (prior session is fully known at today's open)
    sh = hi.groupby(sess).max()
    sl = lo.groupby(sess).min()
    pdh = sess.map(sh.shift(1))
    pdl = sess.map(sl.shift(1))

    # relative-volume acceleration (backward rolling baseline)
    rvol = vol / vol.rolling(RVOL_WIN, min_periods=5).mean()
    rvol_accel = rvol.diff()

    # own trend regime
    sma = cl.rolling(SMA_WIN, min_periods=5).mean()
    slope = (sma - sma.shift(5)) / sma.shift(5)
    pvs = (cl - sma) / sma
    own = np.where((slope > REGIME_SLOPE) & (pvs > 0), "uptrend",
                   np.where((slope < -REGIME_SLOPE) & (pvs < 0), "downtrend", "chop"))

    spy_mom = _ref_momentum(spy, ts)
    qqq_mom = _ref_momentum(qqq, ts)
    sector_mom = _ref_momentum(sector, ts)
    mkt = np.where(spy_mom > MKT_TREND, "bull",
                   np.where(spy_mom < -MKT_TREND, "bear", "chop"))

    return pd.DataFrame({
        "ctx_vwap_dist": (cl - vwap) / vwap,
        "ctx_pdh_dist": (cl - pdh) / pdh,
        "ctx_pdl_dist": (cl - pdl) / pdl,
        "ctx_or_dist_high": (cl - orh) / orh,
        "ctx_or_dist_low": (cl - orl) / orl,
        "ctx_minutes_since_open": mins.astype(float),
        "ctx_is_first_hour": ((mins >= 0) & (mins < 60)).astype(float),
        "ctx_is_last_hour": ((mins >= 330) & (mins <= 390)).astype(float),
        "ctx_rvol_accel": rvol_accel,
        "ctx_spy_trend": spy_mom,
        "ctx_qqq_trend": qqq_mom,
        "ctx_sector_strength": sector_mom - spy_mom,
        "ctx_own_regime": own,
        "ctx_mkt_regime": mkt,
    })


def classify_pattern(high, low, close, ctx: pd.DataFrame, e: int, is_long: bool) -> str:
    """Primary entry setup using ONLY bars[:e+1]. Returns one of PATTERNS."""
    if e < 1:
        return "none"
    lb0 = max(0, e - BREAKOUT_LB)
    rh = float(high[lb0:e].max()) if e > lb0 else np.nan      # structure excludes entry bar
    rl = float(low[lb0:e].min()) if e > lb0 else np.nan
    c = float(close[e])
    poke_hi = e >= 3 and float(high[max(0, e - 3):e + 1].max()) > rh
    poke_lo = e >= 3 and float(low[max(0, e - 3):e + 1].min()) < rl
    ret_w = (c - float(close[e - IMPULSE_W])) / float(close[e - IMPULSE_W]) if e >= IMPULSE_W else 0.0
    ret_r = (c - float(close[e - 3])) / float(close[e - 3]) if e >= 3 else 0.0

    row = ctx.iloc[e]
    vwd = row["ctx_vwap_dist"]
    vwd_prev = ctx["ctx_vwap_dist"].iloc[e - 1]
    near_pdh = pd.notna(row["ctx_pdh_dist"]) and abs(row["ctx_pdh_dist"]) < NEAR_LEVEL
    near_pdl = pd.notna(row["ctx_pdl_dist"]) and abs(row["ctx_pdl_dist"]) < NEAR_LEVEL
    or_formed = pd.notna(row["ctx_or_dist_high"])

    # flags evaluated in the trade's direction (long given; short mirrors)
    if is_long:
        breakout = pd.notna(rh) and c > rh
        failed = poke_hi and c < rh
        flag = ret_w > FLAG_IMPULSE and abs(ret_r) < FLAG_PULLBACK
        vwap_recl = (vwd > 0) and (vwd_prev < 0)
        or_exp = or_formed and row["ctx_or_dist_high"] > 0
        trend = row["ctx_own_regime"] == "uptrend"
        mean_rev = pd.notna(vwd) and vwd < -MR_VWAP
        flag_name = "bull_flag"
    else:
        breakout = pd.notna(rl) and c < rl
        failed = poke_lo and c > rl
        flag = ret_w < -FLAG_IMPULSE and abs(ret_r) < FLAG_PULLBACK
        vwap_recl = (vwd < 0) and (vwd_prev > 0)
        or_exp = or_formed and row["ctx_or_dist_low"] < 0
        trend = row["ctx_own_regime"] == "downtrend"
        mean_rev = pd.notna(vwd) and vwd > MR_VWAP
        flag_name = "bear_flag"

    # priority: most specific structure first
    if failed:
        return "failed_breakout"
    if or_exp and pd.notna(row["ctx_minutes_since_open"]) and row["ctx_minutes_since_open"] <= 180:
        return "opening_range_expansion"
    if breakout:
        return "breakout"
    if vwap_recl:
        return "vwap_reclaim"
    if near_pdh:
        return "pdh_retest"
    if near_pdl:
        return "pdl_retest"
    if flag:
        return flag_name
    if mean_rev:
        return "mean_reversion"
    if trend:
        return "trend_continuation"
    return "none"


def _demo():
    # 3 ET sessions of 15Min bars (rising), volume constant
    idx = pd.date_range("2024-03-04 13:30", periods=60, freq="15min", tz="UTC")
    base = pd.DataFrame({
        "timestamp": idx,
        "open": np.linspace(100, 112, 60), "high": np.linspace(100.5, 112.5, 60),
        "low": np.linspace(99.5, 111.5, 60), "close": np.linspace(100, 112, 60),
        "volume": np.full(60, 1000.0)})
    ctx = build_context_frame(base)
    # PIT check: corrupting bars AFTER k must not change ctx at k
    k = 30
    corrupt = base.copy()
    corrupt.loc[k + 1:, ["high", "low", "close", "volume"]] *= 5
    ctx2 = build_context_frame(corrupt)
    for col in ["ctx_vwap_dist", "ctx_pdh_dist", "ctx_or_dist_high", "ctx_rvol_accel"]:
        a, b2 = ctx[col].iloc[k], ctx2[col].iloc[k]
        assert (pd.isna(a) and pd.isna(b2)) or abs(a - b2) < 1e-9, f"future leaked into {col}"
    # pattern only reads bars up to e
    p = classify_pattern(base["high"].to_numpy(), base["low"].to_numpy(),
                         base["close"].to_numpy(), ctx, 40, is_long=True)
    assert p in PATTERNS
    print("market_context._demo OK  pattern@40 =", p, "| ctx cols", len(ctx.columns))


if __name__ == "__main__":
    _demo()
