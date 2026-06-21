#!/usr/bin/env python3
"""APPEND-ONLY daily forward-shadow recorder for the FROZEN earnings_within_7d sleeve.

Run once per trading day after the close. Each run appends exactly ONE new complete
session -- the most recent trading day for which a realized next-day return exists --
and is idempotent (a date already in the log is never re-recorded). The accumulating log
is the REAL forward shadow; status stays IN_PROGRESS until enough real forward days pass.

Records per session: target weights, earnings data freshness (age), coverage, outage,
rank spread, realized net PnL, turnover, IC sign; plus a running realized net Sharpe / IC
sign / gate verdict over the accumulated log.

FAIL CLOSED (earnings age > 3d, coverage < 95%, source unavailable) -> session is a flat
outage. NO capital allocation, NO live orders, NO change to the frozen sleeve/signal.

    python experiments/orthogonal_features/shadow_daily.py [--log-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[1]
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(HERE))

import paper_sleeve as PS
import shadow as SH
import earnings_calendar as EC
import daily_bars as DB
from run_extended_validation import SYMS

DEFAULT_LOG_DIR = Path.home() / "Library" / "Application Support" / "qq-shadow"   # TCC-exempt
COST_BPS = 5.0


def _recent_bars():
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    return DB.fetch(SYMS, start, end, force=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    args = ap.parse_args()
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "shadow_log.jsonl"

    bars = _recent_bars()
    dates = np.sort(pd.unique(pd.to_datetime(bars.index.get_level_values(1))))
    if len(dates) < 2:
        print("not enough bar history yet"); return
    D, Dn = pd.Timestamp(dates[-2]), pd.Timestamp(dates[-1])      # D's book, realized D->D+1

    seen = {json.loads(l)["date"] for l in log.read_text().splitlines() if l.strip()} if log.exists() else set()
    if D.strftime("%Y-%m-%d") in seen:
        _status(log, "up_to_date", D)
        return

    # WIDE window so coverage means "have calendar data for the universe" (every quarterly
    # reporter appears within ~90d); the 7d signal is derived from the (D, D+7] subset. A
    # week with no scheduled earnings is then a legit flat day, not a data outage.
    cal = EC.get_upcoming_earnings((D - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                                   (D + pd.Timedelta(days=95)).strftime("%Y-%m-%d"), SYMS, force=True)
    sig, dq = EC.earnings_within_signal(cal, D.strftime("%Y-%m-%d"), SYMS, horizon_days=7)
    weights, diag = PS.target_weights(sig, dq)                    # frozen sleeve; fails closed

    # realized next-day returns D -> D+1 (per symbol)
    px_D = bars.xs(D, level=1)["close"]; px_Dn = bars.xs(Dn, level=1)["close"]
    nr = {s: float(px_Dn[s] / px_D[s] - 1.0) for s in px_D.index if s in px_Dn.index}

    prev_w = _last_weights(log)
    turnover = sum(abs(weights.get(k, 0.0) - prev_w.get(k, 0.0)) for k in set(weights) | set(prev_w))
    longs = [s for s, w in weights.items() if w > 0]; shorts = [s for s, w in weights.items() if w < 0]
    lr = np.nanmean([nr.get(s, np.nan) for s in longs]) if longs else np.nan
    sr = np.nanmean([nr.get(s, np.nan) for s in shorts]) if shorts else np.nan
    row = {
        "date": D.strftime("%Y-%m-%d"), "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "blocked": bool(diag["blocked"]), "reason": diag["reason"], "data_outage": bool(diag["blocked"]),
        "calendar_source": cal.source, "coverage": round(cal.coverage, 4),
        "earnings_age_days": cal.age_days, "gross": diag.get("gross", 0.0), "net": diag.get("net", 0.0),
        "n_long": diag.get("n_long", 0), "n_short": diag.get("n_short", 0),
        "weights": {k: round(v, 5) for k, v in weights.items()}, "turnover": round(turnover, 5),
        "rank_spread": _r(lr - sr), "daily_pnl": _r(sum(w * nr.get(s, 0.0) for s, w in weights.items())),
        "daily_ic": _r(_xs_ic(sig, nr)), "ic_sign": int(np.sign(_xs_ic(sig, nr))) if weights else 0,
    }
    with log.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    _status(log, "appended", D)


def _last_weights(log):
    if not log.exists():
        return {}
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    return rows[-1].get("weights", {}) if rows else {}


def _status(log, action, D):
    df = SH.ShadowTracker(log).read()
    rep = SH.compare_to_research(df, research_ic=0.02,
                                 research_turnover=PS.SLEEVE["validated_metrics"]["turnover_per_day"]) if not df.empty \
        else {"verdict": "SHADOW_NOT_STARTED", "days": 0}
    out = {"action": action, "session_date": D.strftime("%Y-%m-%d"), "status": "IN_PROGRESS",
           "sleeve_status": PS.SLEEVE["status"], "shadow": rep}
    (log.parent / "shadow_status.json").write_text(json.dumps(out, indent=2, default=str))
    v = rep.get("verdict")
    print(f"[{action}] session {D.date()} | shadow days={rep.get('days')} verdict={v} "
          f"outages={rep.get('operational_failures')} realized_net_sharpe={rep.get('realized_net_sharpe')} "
          f"ic_sign_consistent={rep.get('ic_sign_consistent')} | status=IN_PROGRESS (no capital, no live orders)")


def _xs_ic(signal, rets):
    keys = [k for k in signal if k in rets and np.isfinite(rets[k])]
    if len(keys) < 3:
        return np.nan
    s = pd.Series([signal[k] for k in keys]); r = pd.Series([rets[k] for k in keys])
    return float(s.rank().corr(r.rank())) if s.nunique() > 1 and r.nunique() > 1 else np.nan


def _r(x, n=6):
    return round(float(x), n) if x is not None and np.isfinite(x) else None


if __name__ == "__main__":
    main()
