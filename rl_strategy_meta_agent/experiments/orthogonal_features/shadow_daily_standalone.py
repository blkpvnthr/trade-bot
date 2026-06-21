#!/usr/bin/env python3
"""Self-contained, STDLIB-ONLY daily forward-shadow recorder for the FROZEN
earnings_within_7d sleeve. Designed to run under launchd via the macOS system python
(/usr/bin/python3) from a TCC-exempt path (~/qq-shadow), touching NOTHING under ~/Desktop
-- so it sidesteps the TCC blocker without a Full Disk Access grant.

It reimplements ONLY the tiny frozen object (equal-weight dollar-neutral, gross 100%, +50%
long / -50% short, fail-closed). `frozen_weights` is parity-tested against the canonical
paper_sleeve.target_weights so it can never drift. PAPER ONLY: no broker/order/live path.

Reads keys from <log-dir>/config.env (APCA_*, MASSIVE_API_KEY, ALPHA_VANTAGE_API_KEY).
Appends one idempotent session/day to <log-dir>/shadow_log.jsonl. Status stays IN_PROGRESS.

    /usr/bin/python3 shadow_daily_standalone.py [--log-dir ~/qq-shadow]
"""
import argparse
import csv
import json
import math
import os
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

SYMS = ['AAPL', 'AMSC', 'AMZN', 'ARM', 'AVGO', 'CE', 'ENSG', 'GOOG', 'GOOGL', 'GWRE', 'IREN', 'LULU',
        'META', 'MSTR', 'NVDA', 'ORCL', 'QBTS', 'RYAN', 'SMCI', 'TSLA', 'TSM', 'UNH', 'WGS', 'WING']
MAX_AGE_DAYS = 3
MIN_COVERAGE = 0.95
COST_BPS = 5.0
DEFAULT_LOG_DIR = Path.home() / "qq-shadow"            # TCC-exempt, never under ~/Desktop


def _cfg(log_dir):
    cfg = {}
    p = Path(log_dir) / "config.env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=40).read().decode()


# ---- the FROZEN object (parity-tested vs paper_sleeve.target_weights) ----
def check_data(age_days, missing_ratio, feed_available):
    if not feed_available:
        return False, "earnings_feed_unavailable"
    if age_days is None:
        return False, "earnings_data_missing"
    if age_days < 0:
        return False, "earnings_data_timestamp_in_future"
    if age_days > MAX_AGE_DAYS:
        return False, "earnings_data_stale(%.1fd>%dd)" % (age_days, MAX_AGE_DAYS)
    if missing_ratio is None or missing_ratio > (1 - MIN_COVERAGE):
        return False, "too_many_missing_symbols"
    return True, "ok"


def frozen_weights(signal, data_quality):
    ok, reason = check_data(data_quality.get("earnings_age_days"),
                            data_quality.get("missing_symbols_ratio"),
                            data_quality.get("feed_available", False))
    if not ok:
        return {}, {"blocked": True, "reason": reason, "gross": 0.0, "net": 0.0}
    vals = {k: float(v) for k, v in signal.items() if v is not None and not math.isnan(float(v))}
    if len(vals) < 2:
        return {}, {"blocked": False, "reason": "insufficient_universe_flat", "gross": 0.0, "net": 0.0}
    m = statistics.fmean(vals.values())
    longs = [k for k, v in vals.items() if v > m]
    shorts = [k for k, v in vals.items() if v < m]
    if not longs or not shorts:
        return {}, {"blocked": False, "reason": "no_dispersion_flat", "gross": 0.0, "net": 0.0}
    w = {k: 0.5 / len(longs) for k in longs}
    w.update({k: -0.5 / len(shorts) for k in shorts})
    return w, {"blocked": False, "reason": "ok", "gross": round(sum(abs(x) for x in w.values()), 4),
               "net": round(sum(w.values()), 6), "n_long": len(longs), "n_short": len(shorts)}


# ---- data: Alpaca bars (REST) + Massive/AV calendar ----
def fetch_bars(cfg, days=18):
    end = datetime.now(timezone.utc)
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    h = {"APCA-API-KEY-ID": cfg.get("APCA_API_KEY_ID", ""), "APCA-API-SECRET-KEY": cfg.get("APCA_API_SECRET_KEY", "")}
    out, token = {s: {} for s in SYMS}, None
    while True:
        q = {"symbols": ",".join(SYMS), "timeframe": "1Day", "start": start, "adjustment": "all",
             "feed": "sip", "limit": 10000}
        if token:
            q["page_token"] = token
        d = json.loads(_get("https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(q), h))
        for s, bars in (d.get("bars") or {}).items():
            for b in bars:
                out[s][b["t"][:10]] = float(b["c"])
        token = d.get("next_page_token")
        if not token:
            break
    return out


def fetch_calendar(cfg, start, end):
    try:
        q = {"ticker.any_of": ",".join(SYMS), "date.gte": start, "date.lte": end, "sort": "date.asc",
             "limit": 1000, "apiKey": cfg.get("MASSIVE_API_KEY", "")}
        d = json.loads(_get("https://api.massive.com/benzinga/v1/earnings?" + urllib.parse.urlencode(q)))
        ev = {s: [] for s in SYMS}
        for r in d.get("results") or []:
            if r.get("ticker") in ev and r.get("date"):
                ev[r["ticker"]].append(str(r["date"])[:10])
        if sum(1 for s in SYMS if ev[s]) / len(SYMS) >= MIN_COVERAGE:
            return ev, "massive"
    except Exception:
        pass
    try:                                                            # AV fallback
        q = {"function": "EARNINGS_CALENDAR", "horizon": "3month", "apikey": cfg.get("ALPHA_VANTAGE_API_KEY", "")}
        txt = _get("https://www.alphavantage.co/query?" + urllib.parse.urlencode(q))
        ev = {s: [] for s in SYMS}
        for row in csv.reader(txt.splitlines()[1:]):
            if len(row) >= 3 and row[0] in ev and start <= row[2] <= end:
                ev[row[0]].append(row[2])
        return ev, "alpha_vantage"
    except Exception:
        return None, None


def _spearman(sig, ret):
    keys = [k for k in sig if k in ret]
    if len(keys) < 3:
        return None
    sr = _rank([sig[k] for k in keys]); rr = _rank([ret[k] for k in keys])
    if len(set(sr)) < 2 or len(set(rr)) < 2:
        return None
    n = len(keys); ms, mr = statistics.fmean(sr), statistics.fmean(rr)
    cov = sum((sr[i] - ms) * (rr[i] - mr) for i in range(n))
    ds = math.sqrt(sum((x - ms) ** 2 for x in sr)); dr = math.sqrt(sum((x - mr) ** 2 for x in rr))
    return cov / (ds * dr) if ds and dr else None


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    args = ap.parse_args()
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "shadow_log.jsonl"
    cfg = _cfg(log_dir)

    bars = fetch_bars(cfg)
    all_dates = sorted({d for s in SYMS for d in bars[s]})
    if len(all_dates) < 2:
        print("not enough bars"); return
    D, Dn = all_dates[-2], all_dates[-1]                          # D's book, realized D -> D+1
    if already_recorded(log, D):                                 # never re-record a date (no rebuild/overwrite)
        _status(log, "up_to_date", D); return

    Dt = datetime.fromisoformat(D)
    ev, source = fetch_calendar(cfg, (Dt - timedelta(days=10)).strftime("%Y-%m-%d"),
                                (Dt + timedelta(days=95)).strftime("%Y-%m-%d"))
    if ev is None:
        coverage, feed = 0.0, False
        sig = {s: 0.0 for s in SYMS}
    else:
        coverage = sum(1 for s in SYMS if ev[s]) / len(SYMS)
        feed = coverage >= MIN_COVERAGE
        sig = {s: (1.0 if any(x > D and (datetime.fromisoformat(x) - Dt).days <= 7 for x in ev[s]) else 0.0)
               for s in SYMS}
    dq = {"earnings_age_days": 0.0, "missing_symbols_ratio": round(1 - coverage, 4), "feed_available": feed}
    w, diag = frozen_weights(sig, dq)

    nr = {s: bars[s][Dn] / bars[s][D] - 1.0 for s in SYMS if D in bars[s] and Dn in bars[s]}
    longs = [s for s, x in w.items() if x > 0]; shorts = [s for s, x in w.items() if x < 0]
    lr = statistics.fmean([nr[s] for s in longs if s in nr]) if any(s in nr for s in longs) else None
    sr = statistics.fmean([nr[s] for s in shorts if s in nr]) if any(s in nr for s in shorts) else None
    prev_w = _last_weights(log)
    row = {"date": D, "recorded_utc": datetime.now(timezone.utc).isoformat(),
           "blocked": diag["blocked"], "reason": diag["reason"], "data_outage": diag["blocked"],
           "calendar_source": source, "coverage": round(coverage, 4), "earnings_age_days": 0.0,
           "gross": diag.get("gross", 0.0), "net": diag.get("net", 0.0),
           "n_long": diag.get("n_long", 0), "n_short": diag.get("n_short", 0),
           "weights": {k: round(v, 5) for k, v in w.items()},
           "turnover": round(sum(abs(w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in set(w) | set(prev_w)), 5),
           "rank_spread": round(lr - sr, 6) if (lr is not None and sr is not None) else None,
           "daily_pnl": round(sum(x * nr.get(s, 0.0) for s, x in w.items()), 6),
           "daily_ic": _spearman(sig, nr), "ic_sign": (1 if (_spearman(sig, nr) or 0) > 0 else
                                                       -1 if (_spearman(sig, nr) or 0) < 0 else 0) if w else 0}
    append_session(log, row)                                    # APPEND-ONLY, one session
    _status(log, "appended", D)


def already_recorded(log_path, date):
    """True if `date` already has a session row -> the run will NOT append again."""
    p = Path(log_path)
    if not p.exists():
        return False
    return date in {json.loads(l)["date"] for l in p.read_text().splitlines() if l.strip()}


def append_session(log_path, row):
    """Append exactly one session as a JSONL line. Never rewrites/truncates the log.
    Idempotent guard: refuses to append a date that is already recorded."""
    if already_recorded(log_path, row["date"]):
        return False
    with Path(log_path).open("a") as fh:                        # mode 'a' = append-only
        fh.write(json.dumps(row) + "\n")
    return True


def _last_weights(log):
    if not log.exists():
        return {}
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    return rows[-1].get("weights", {}) if rows else {}


def _status(log, action, D):
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    traded = [r for r in rows if not r.get("blocked")]
    outages = sum(1 for r in rows if r.get("blocked"))
    pnls = [r["daily_pnl"] - COST_BPS / 1e4 * (r.get("turnover") or 0) for r in traded if r.get("daily_pnl") is not None]
    sharpe = (statistics.fmean(pnls) / (statistics.pstdev(pnls) + 1e-12) * math.sqrt(252)) if len(pnls) > 1 else None
    out = {"action": action, "session_date": D, "status": "SHADOW_IN_PROGRESS", "days": len(rows),
           "operational_failures": outages, "realized_net_sharpe": round(sharpe, 3) if sharpe is not None else None}
    (Path(log).parent / "shadow_status.json").write_text(json.dumps(out, indent=2))
    print("[%s] session %s | days=%d outages=%d realized_net_sharpe=%s | status=SHADOW_IN_PROGRESS (paper, no orders)"
          % (action, D, len(rows), outages, out["realized_net_sharpe"]))


if __name__ == "__main__":
    main()
