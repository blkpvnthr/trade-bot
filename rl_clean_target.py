#!/usr/bin/env python3
"""
rl_clean_target.py — Re-target realized_R with a LEAK-FREE forward return.
==========================================================================

Why: validation (docs/rl_psc_validation_findings.md) showed the offline `realized_R` target was
predictable from `probability_of_profit` (= setup_score/100) at AUC 0.92 — partly because the
dataset builder FABRICATED realized_R from that feature (now fixed), and partly because the
intraday bracket outcome is entangled with the setup detector. Either way, a target that a model
input can (near-)determine is not a valid test of predictive edge.

This module replaces `realized_R` with a genuinely INDEPENDENT label: the **forward H-day return
of the underlying**, computed from REAL cached daily bars (read-only), keyed by (symbol, date_tag).
The entry's setup_score cannot mechanically determine tomorrow's underlying return, so if a feature
still predicts it, that is genuine ex-ante signal — exactly the honest test we want.

  realized_R_clean = clip( forward_return / RISK_PCT , −R_CLIP, +R_CLIP )

scaled to the same R-multiple convention the reward functions expect (RISK_PCT=0.5% ⇒ a +1.5%
forward move ≈ +3R, matching the backfill's 3:1 bracket). Rows whose forward bar is unavailable
(the most recent H days) are DROPPED — never fabricated.

Honest limitations (documented, not hidden):
  * Direction proxy: forward UNDERLYING return, not the option's realized R-multiple. Directionally
    aligned for long-call entries; not a perfect P&L.
  * Shared label: all rows on the same (symbol, date_tag) get the same forward return → fewer
    independent labels than rows. Reported.
  * ~6-week span: still one regime (a data limitation, see the findings doc).

OFFLINE, read-only. Uses the existing BarCache (already-cached daily bars; no live orders).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

import rl_common as rc

logger = logging.getLogger("rl_clean_target")

RISK_PCT = float(os.getenv("RL_CLEAN_RISK_PCT", "0.005"))   # R scaling (matches backfill bracket)
HORIZON_DAYS = int(os.getenv("RL_CLEAN_HORIZON", "2"))      # forward trading days
R_CLIP = float(os.getenv("RL_CLEAN_R_CLIP", "8.0"))


def _daily_closes(symbols):
    """{SYM: pd.Series(close indexed by naive date)} from the read-only BarCache (cached bars)."""
    import pandas as pd
    from rl_alpaca_bar_enrichment import RegimeContext
    ctx = RegimeContext()
    try:
        ctx.prefetch(sorted({str(s).upper() for s in symbols}))
    except Exception as exc:
        logger.warning("daily prefetch failed (%s) — clean target unavailable", exc)
    out = {}
    for sym, df in ctx._daily.items():
        if df is None or len(df) == 0:
            continue
        try:
            s = df["close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            out[sym] = s[~s.index.duplicated(keep="last")].sort_index()
        except Exception:
            continue
    return out


def _forward_return(closes, date_tag: str, h: int) -> Optional[float]:
    """(close[t+h] − close[t]) / close[t] using the first trading day ≥ date_tag and h days later."""
    import pandas as pd
    try:
        t = pd.Timestamp(date_tag).normalize()
    except Exception:
        return None
    idx = closes.index
    pos = idx.searchsorted(t)                       # first bar on/after the entry date
    if pos >= len(idx) or pos + h >= len(idx):
        return None
    c0 = float(closes.iloc[pos]); c1 = float(closes.iloc[pos + h])
    return (c1 - c0) / c0 if c0 > 0 else None


def retarget(dataset: str = "alpha_entry_dataset", write: bool = True) -> Dict[str, Any]:
    """Replace realized_R with the leak-free forward-return R. Returns a summary; writes the
    cleaned dataset (archiving the prior one) when write=True."""
    import pandas as pd
    p = rc.RL_DATA_DIR / f"{dataset}.parquet"
    if not p.exists():
        return {"error": f"{p} not found"}
    df = pd.read_parquet(p)
    syms = df["symbol"].astype(str).str.upper()
    closes = _daily_closes(syms.unique())
    fwd = np.full(len(df), np.nan)
    for i, (sym, tag) in enumerate(zip(syms.values, df["date_tag"].astype(str).values)):
        s = closes.get(sym)
        if s is None:
            continue
        r = _forward_return(s, tag, HORIZON_DAYS)
        if r is not None:
            fwd[i] = r
    have = ~np.isnan(fwd)
    df = df.copy()
    df["realized_R_bracket"] = df["realized_R"]                       # keep the old (entangled) label
    df["forward_return"] = fwd
    df["realized_R"] = np.clip(fwd / RISK_PCT, -R_CLIP, R_CLIP)       # leak-free R-scaled target
    df["realized_R_source"] = np.where(have, "forward_bars", "no_forward_data")
    clean = df[have].reset_index(drop=True)                           # drop rows w/o forward data

    summary = {
        "generated_at_utc": rc.now_iso(), "dataset": dataset,
        "rows_in": int(len(df)), "rows_with_forward": int(have.sum()),
        "rows_dropped_no_forward": int((~have).sum()),
        "symbols_with_bars": len(closes), "horizon_days": HORIZON_DAYS, "risk_pct": RISK_PCT,
        "unique_symbol_date_labels": int(clean.groupby(["symbol", "date_tag"]).ngroups),
        "clean_win_rate": round(float((clean["realized_R"] > 0).mean()), 4) if len(clean) else 0.0,
        "clean_mean_R": round(float(clean["realized_R"].mean()), 4) if len(clean) else 0.0,
    }
    if write and len(clean):
        arch = rc.RL_DATA_DIR / "_pre_clean_target_archive"
        arch.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(p, arch / f"{dataset}.parquet")
        clean.to_parquet(p, index=False)
        summary["written"] = str(p)
        summary["archived_prior"] = str(arch / f"{dataset}.parquet")
    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "rl_clean_target_summary.json", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = retarget()
    if "error" in s:
        print(s["error"]); raise SystemExit(0)
    print("CLEAN TARGET re-applied (leak-free forward return):")
    for k in ("rows_in", "rows_with_forward", "rows_dropped_no_forward", "symbols_with_bars",
              "horizon_days", "unique_symbol_date_labels", "clean_win_rate", "clean_mean_R",
              "written", "archived_prior"):
        if k in s:
            print(f"  {k}: {s[k]}")


if __name__ == "__main__":
    main()
