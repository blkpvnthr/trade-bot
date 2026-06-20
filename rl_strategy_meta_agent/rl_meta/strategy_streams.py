"""Daily return streams for the new formulaic-alpha stack, so the OLPS allocator
can route capital across REAL deployable candidates instead of legacy families.

Streams (each leak-free: signals use bar t, returns are forward):
  alpha_mined_ensemble   cross-sectional long-short on the registry-accepted alpha
                         composite (long low-giveback-risk, short high)
  giveback_exit_model    long book with exposure scaled down by predicted giveback
  ORB_relvol             opening-range breakout confirmed by relative volume
  *_baseline             the validated family's own daily trade returns

HARD RULE: the retired/invalid families (martingale, pin_bar_magic_v1,
sniper_strategy) may NEVER appear as an asset -- validate_no_retired() raises.
Research-only: no broker / order / allocator-execution / live path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RETIRED = {"martingale", "pin_bar_magic_v1", "sniper_strategy"}
BASELINE_FAMILIES = {"linear_regression", "mean_reversion"}


def validate_no_retired(names) -> None:
    bad = [n for n in names for r in RETIRED if r in str(n)]
    if bad:
        raise ValueError(f"retired/invalid strategies must never be assets: {sorted(set(bad))}")


def _cs_z(s):
    tl = s.index.get_level_values(1)
    return (s - s.groupby(tl).transform("mean")) / (s.groupby(tl).transform("std") + 1e-9)


def _cs_rank(s):
    return s.groupby(s.index.get_level_values(1)).rank(pct=True)


def _to_daily(per_bar):
    """Sum per-bar portfolio returns within each US/Eastern day -> daily return."""
    d = pd.to_datetime(per_bar.index.get_level_values(1), utc=True).tz_convert("America/New_York").normalize()
    return per_bar.groupby(d).sum()


def composite_signal(panel, node_values):
    """Mean of cross-sectionally z-scored accepted-alpha values -> giveback signal
    (higher = higher predicted severe-giveback risk)."""
    z = [_cs_z(v) for v in node_values]
    return pd.concat(z, axis=1).mean(axis=1)


def long_short_stream(panel, signal, fwd="fut_ret_1"):
    """Market-neutral cross-sectional long-short on `signal` (higher signal = long)."""
    w = _cs_z(signal)
    tl = w.index.get_level_values(1)
    num = (w * panel[fwd]).groupby(tl).sum()
    den = w.abs().groupby(tl).sum() + 1e-12
    per_ts = num / den                                      # indexed by timestamp
    dd = pd.to_datetime(per_ts.index, utc=True).tz_convert("America/New_York").normalize()
    return per_ts.groupby(dd).sum()


def exposure_stream(panel, giveback_signal, fwd="fut_ret_1"):
    """Long book, per-name exposure = 1 - rank(giveback) in [0,1] (cut risk names)."""
    expo = 1.0 - _cs_rank(giveback_signal)
    per_bar_sym = expo * panel[fwd]
    tl = per_bar_sym.index.get_level_values(1)
    per_ts = per_bar_sym.groupby(tl).mean()                 # equal-weight across names
    dd = pd.to_datetime(per_ts.index, utc=True).tz_convert("America/New_York").normalize()
    return per_ts.groupby(dd).sum()


def orb_relvol_stream(panel, relvol_min=1.5):
    """Opening-range breakout + relative volume. On the first bar a name closes above
    its opening-range high with rel_volume>=relvol_min, go long to the day's close."""
    df = pd.DataFrame({"or": panel["or_high_dist"], "rv": panel["rel_volume"],
                       "ret": panel["ret"]})
    df["sym"] = df.index.get_level_values(0)
    df["day"] = pd.to_datetime(df.index.get_level_values(1), utc=True).tz_convert("America/New_York").normalize()
    out = {}
    for (sym, day), g in df.groupby(["sym", "day"], sort=False):
        trig = g[(g["or"] > 0) & (g["rv"] >= relvol_min)]
        if trig.empty:
            continue
        first = g.index.get_loc(trig.index[0])
        out.setdefault(day, []).append(float(g["ret"].iloc[first:].sum()))   # entry->close
    days = sorted(out)
    return pd.Series([np.mean(out[d]) for d in days], index=pd.DatetimeIndex(days))


def baseline_stream(dataset, family, min_days=50):
    if family in RETIRED:
        raise ValueError(f"{family} is retired")
    df = pd.read_parquet(dataset, columns=["strategy_id", "exit_time", "return_pct"])
    df = df[df["strategy_id"] == family]
    df["day"] = pd.to_datetime(df["exit_time"], utc=True).dt.tz_convert("America/New_York").dt.normalize()
    daily = df.groupby("day")["return_pct"].mean()
    return daily if len(daily) >= min_days else None


def _demo():
    validate_no_retired(["alpha_mined_ensemble", "linear_regression_baseline"])
    try:
        validate_no_retired(["sniper_strategy_daily"])
        raise AssertionError("should have failed")
    except ValueError:
        pass
    print("strategy_streams._demo OK  retired-guard fires; builders importable")


if __name__ == "__main__":
    _demo()
