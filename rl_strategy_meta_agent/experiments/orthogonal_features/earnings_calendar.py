"""Forward earnings-calendar adapter -- supplies FUTURE scheduled earnings events so the
frozen earnings_within_7d sleeve can produce a live book (instead of relying on historical
reportedDate, which thins out near 'today' and caused fail-safe outages).

Single source-agnostic interface:  get_upcoming_earnings(start_date, end_date, symbols)

Primary source : Massive Benzinga earnings (api.massive.com/benzinga/v1/earnings) -- has
                 confirmed + projected dates with full coverage.
Fallback       : Alpha Vantage EARNINGS_CALENDAR (CSV).
Local cache    : backtest_cache/earnings_calendar/ with fetched_at freshness metadata.
FAIL CLOSED    : ok=False (-> sleeve goes flat) if earnings age > 3 days, coverage < 95%,
                 or no source is available. Research only; read-only market data.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TRADER_ROOT = Path(__file__).resolve().parents[3]
CACHE = TRADER_ROOT / "backtest_cache" / "earnings_calendar"
CACHE.mkdir(parents=True, exist_ok=True)
MASSIVE = "https://api.massive.com/benzinga/v1/earnings"
AV = "https://www.alphavantage.co/query"
MAX_AGE_DAYS = 3
MIN_COVERAGE = 0.95


def _key(name):
    if name not in os.environ and (TRADER_ROOT / ".env").exists():
        for line in (TRADER_ROOT / ".env").read_text().splitlines():
            if line.startswith(name + "="):
                os.environ[name] = line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(name, "")


@dataclass
class CalendarResult:
    events: dict = field(default_factory=dict)      # symbol -> sorted list of "YYYY-MM-DD"
    fetched_at: str | None = None
    source: str | None = None
    coverage: float = 0.0
    age_days: float | None = None
    ok: bool = False
    reason: str = ""


def _fetch_massive(symbols, start, end):
    try:
        q = {"ticker.any_of": ",".join(symbols), "date.gte": start, "date.lte": end,
             "sort": "date.asc", "limit": 1000, "apiKey": _key("MASSIVE_API_KEY")}
        d = json.loads(urllib.request.urlopen(f"{MASSIVE}?{urllib.parse.urlencode(q)}", timeout=30).read().decode())
        ev = {s: [] for s in symbols}
        for r in d.get("results") or []:
            t = r.get("ticker")
            if t in ev and r.get("date"):
                ev[t].append(str(r["date"])[:10])
        return {s: sorted(set(v)) for s, v in ev.items()}
    except Exception:
        return None


def _fetch_av(symbols, start, end):
    try:
        q = {"function": "EARNINGS_CALENDAR", "horizon": "3month", "apikey": _key("ALPHA_VANTAGE_API_KEY")}
        txt = urllib.request.urlopen(f"{AV}?{urllib.parse.urlencode(q)}", timeout=30).read().decode()
        ev = {s: [] for s in symbols}
        for line in txt.splitlines()[1:]:
            p = line.split(",")
            if len(p) >= 3 and p[0] in ev and start <= p[2] <= end:
                ev[p[0]].append(p[2])
        return {s: sorted(set(v)) for s, v in ev.items()}
    except Exception:
        return None


def _coverage(events, symbols):
    return sum(1 for s in symbols if events.get(s)) / max(len(symbols), 1)


def assess(events, fetched_at, now, symbols, max_age_days=MAX_AGE_DAYS, min_coverage=MIN_COVERAGE):
    """Pure fail-closed assessment (testable without network)."""
    if not events or fetched_at is None:
        return {"ok": False, "reason": "source_unavailable", "coverage": 0.0, "age_days": None}
    age = (datetime.fromisoformat(str(now)) - datetime.fromisoformat(str(fetched_at))).total_seconds() / 86400.0
    cov = _coverage(events, symbols)
    if age < 0:
        return {"ok": False, "reason": "fetched_in_future", "coverage": cov, "age_days": age}
    if age > max_age_days:
        return {"ok": False, "reason": f"stale({age:.1f}d>{max_age_days}d)", "coverage": cov, "age_days": age}
    if cov < min_coverage:
        return {"ok": False, "reason": f"coverage<{min_coverage}({cov:.2f})", "coverage": cov, "age_days": age}
    return {"ok": True, "reason": "ok", "coverage": cov, "age_days": age}


def get_upcoming_earnings(start_date, end_date, symbols, now=None, force=False,
                          max_age_days=MAX_AGE_DAYS, min_coverage=MIN_COVERAGE) -> CalendarResult:
    now = now or datetime.now(timezone.utc).isoformat()
    cf = CACHE / f"cal_{start_date}_{end_date}.json"
    events, source, fetched_at = None, None, None
    if cf.exists() and not force:
        c = json.loads(cf.read_text())
        if assess(c["events"], c["fetched_at"], now, symbols, max_age_days, min_coverage)["ok"]:
            events, source, fetched_at = c["events"], c["source"], c["fetched_at"]   # fresh cache
    if events is None:                                              # fetch live: primary -> fallback
        em = _fetch_massive(symbols, start_date, end_date)
        if em is not None and _coverage(em, symbols) >= min_coverage:
            events, source = em, "massive"
        else:
            ea = _fetch_av(symbols, start_date, end_date)
            best = max([e for e in (em, ea) if e is not None], key=lambda e: _coverage(e, symbols), default=None)
            events, source = best, ("massive" if best is em else "alpha_vantage" if best is ea else None)
        if events is not None:
            fetched_at = now
            cf.write_text(json.dumps({"events": events, "fetched_at": fetched_at, "source": source}))
        elif cf.exists():                                          # both sources down -> stale cache
            c = json.loads(cf.read_text()); events, source, fetched_at = c["events"], c["source"] + "(stale_cache)", c["fetched_at"]
    a = assess(events, fetched_at, now, symbols, max_age_days, min_coverage)
    return CalendarResult(events or {}, fetched_at, source, a["coverage"], a["age_days"], a["ok"], a["reason"])


def earnings_within_signal(result: CalendarResult, as_of, symbols, horizon_days=7):
    """Frozen signal + data_quality for the sleeve. signal[sym]=1 if an earnings date falls in
    (as_of, as_of+horizon]. The sleeve consumes data_quality and fails closed itself."""
    from datetime import date as _d
    a = datetime.fromisoformat(str(as_of)[:10]).date()
    sig = {}
    for s in symbols:
        nxt = [datetime.fromisoformat(x).date() for x in result.events.get(s, []) if x > str(a)]
        sig[s] = 1.0 if nxt and (min(nxt) - a).days <= horizon_days else 0.0
    dq = {"earnings_age_days": result.age_days, "missing_symbols_ratio": round(1 - result.coverage, 4),
          "feed_available": result.ok}
    return sig, dq


def _demo():
    now = "2026-06-18T00:00:00+00:00"
    ev = {"AAA": ["2026-06-20"], "BBB": ["2026-07-30"], "CCC": ["2026-06-22"]}
    # fresh + full coverage -> ok; stale -> fail; missing -> coverage fail
    assert assess(ev, "2026-06-17T00:00:00+00:00", now, ["AAA", "BBB", "CCC"])["ok"]
    assert not assess(ev, "2026-06-10T00:00:00+00:00", now, ["AAA", "BBB", "CCC"])["ok"]   # stale
    assert not assess({"AAA": ["2026-06-20"]}, "2026-06-17T00:00:00+00:00", now,
                      ["AAA", "BBB", "CCC", "DDD"])["ok"]                                   # coverage 25%
    r = CalendarResult(ev, "2026-06-17T00:00:00+00:00", "massive", 1.0, 1.0, True, "ok")
    sig, dq = earnings_within_signal(r, "2026-06-18", ["AAA", "BBB", "CCC"])
    assert sig == {"AAA": 1.0, "BBB": 0.0, "CCC": 1.0} and dq["feed_available"]            # within 7d
    print("earnings_calendar._demo OK  fail-closed assess + within-7d signal verified")


if __name__ == "__main__":
    _demo()
