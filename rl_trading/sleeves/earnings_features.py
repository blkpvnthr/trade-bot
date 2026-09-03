"""earnings_features — the integrated `earnings_tracker.py` logic, as a project module.

Two strictly separated tiers:

  PIT-SAFE (drift/liquidity)  : prices/volume from cached Alpaca daily bars. Used for variant-B
                                ranking AND historical validation (run_earnings_variants.py).
  LIVE_ONLY_NON_PIT (fundamentals/beat) : Alpha Vantage SNAPSHOT endpoints (OVERVIEW, EARNINGS,
                                INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW, SHARES_OUTSTANDING,
                                EARNINGS_ESTIMATES). These are CURRENT snapshots with no historical
                                as-of, so they are NEVER PIT-backtestable, NEVER a signal source,
                                NEVER used in canonical validation or promotion. Variant-C only.

THE PIT FIREWALL (enforced): `earnings_within_7d` is set ONLY by the forward scheduled calendar.
Drift and fundamentals may rank/size a name ONLY after it is already calendar-flagged. The
fundamentals helpers raise if asked to act as a signal source.

yfinance is fully removed: prices come from rl_trading.data (cached Alpaca adjusted bars).
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
AV_CACHE_DIR = _REPO_ROOT / "data" / "cache" / "earnings_tracker_av"
CALENDAR_DIR = _REPO_ROOT / "backtest_cache" / "earnings_calendar"

LOOKAHEAD_DAYS = 7
MIN_PRICE = 5.0
MIN_ADV_USD = 5_000_000.0

# Hard label: fundamentals are current snapshots, not historical as-of features.
LIVE_ONLY_NON_PIT_BACKTESTABLE = True
FUNDAMENTALS_WARNING = (
    "LIVE_ONLY_NON_PIT_BACKTESTABLE: Alpha Vantage fundamentals/beat fields are CURRENT snapshots, "
    "not historical as-of features. They may rank/size a calendar-flagged name LIVE only; they are "
    "never a signal source, never PIT-backtested, never used for promotion."
)

# Variants. A and B are PIT-safe and promotable; C is live-only metadata, disabled by default.
VARIANTS = ("A", "B", "C")
DEFAULT_VARIANT = "A"  # validated winner (run_earnings_variants.py: B did not beat A net@10bps)


# ── PIT-safe drift / liquidity (cached Alpaca bars) ───────────────────────────────
def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def drift_features(universe: Sequence[str], asof) -> Dict[str, Dict[str, Optional[float]]]:
    """{symbol: price/return/vol/ADV features} causal as of `asof`, from cached Alpaca adj bars.

    PIT-safe: only past prices/volume. Mirrors the uploaded tracker's score_drift_candidates but
    sourced from rl_trading.data (no yfinance). The earnings flag/days_to is NOT computed here — it
    comes from the forward calendar in the engine (firewall).
    """
    try:
        from .. import data as rl_data
    except Exception:  # pragma: no cover - import guard
        return {s: _empty_feat() for s in universe}
    cutoff = _as_date(asof).isoformat() + "T23:59:59+00:00"
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for sym in universe:
        df = rl_data.load_daily(sym, source="adj")
        if df is None or df.empty:
            out[sym] = _empty_feat()
            continue
        d = df.loc[df.index <= cutoff]
        if len(d) < 21:
            out[sym] = _empty_feat()
            continue
        close = d["close"]
        price = float(close.iloc[-1])
        ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0)
        ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        vol_20d = float(close.pct_change().tail(20).std())
        adv20 = float((close * d["volume"]).tail(20).mean())
        out[sym] = {"price": price, "ret_5d": ret_5d, "ret_20d": ret_20d,
                    "vol_20d": vol_20d, "avg_dollar_vol_20d": adv20}
    return out


def _empty_feat() -> Dict[str, Optional[float]]:
    return {"price": None, "ret_5d": None, "ret_20d": None, "vol_20d": None,
            "avg_dollar_vol_20d": None}


def drift_score(feat: Mapping[str, Optional[float]], days_to: Optional[int],
                lookahead: int = LOOKAHEAD_DAYS) -> float:
    """PIT drift score (uploaded tracker formula). Higher = stronger pre-earnings drift candidate."""
    if not feat or feat.get("price") is None or days_to is None:
        return 0.0
    ret_5d = feat.get("ret_5d") or 0.0
    ret_20d = feat.get("ret_20d") or 0.0
    adv = feat.get("avg_dollar_vol_20d") or 0.0
    vol_20d = feat.get("vol_20d") or 0.0
    return float(
        100 * max(ret_5d, 0.0)
        + 35 * max(ret_20d, 0.0)
        + min(adv / 100_000_000, 3)
        + max(0, lookahead - days_to) * 0.15
        - min(vol_20d * 20, 2)
    )


def liquid(feat: Mapping[str, Optional[float]], min_price: float = MIN_PRICE,
           min_adv: float = MIN_ADV_USD) -> bool:
    p, adv = feat.get("price"), feat.get("avg_dollar_vol_20d")
    return p is not None and p >= min_price and adv is not None and adv >= min_adv


# ── ranking ───────────────────────────────────────────────────────────────────────
def rank_score(symbol: str, feat: Mapping, days_to: Optional[int], variant: str,
               fundamentals: Optional[Mapping] = None) -> float:
    """Selection score among ALREADY-calendar-flagged names.

    A: 0 (equal weight; selection is arbitrary/alphabetical upstream).
    B: PIT drift score.
    C: drift + AV fundamentals/beat (LIVE_ONLY_NON_PIT_BACKTESTABLE).
    """
    if variant == "A":
        return 0.0
    d = drift_score(feat, days_to)
    if variant == "B":
        return d
    if variant == "C":
        f = fundamentals or {}
        beat = float(f.get("beat_probability", 0.5))
        quality = float(f.get("quality_score", 0.0))
        # mirrors tracker final_score weighting; fundamentals only nudge an already-flagged name
        return 0.70 * d + 0.30 * (beat * 10) + 0.15 * quality
    raise ValueError(f"unknown variant {variant!r}")


# ── LIVE-ONLY fundamentals (AV snapshots) — fenced off the signal path ─────────────
def assert_not_signal_source() -> None:
    """Guard: fundamentals must never produce the earnings_within_7d signal."""
    raise RuntimeError(
        "fundamentals cannot create the earnings_within_7d signal; it comes only from the forward "
        "scheduled calendar. " + FUNDAMENTALS_WARNING
    )


def _safe_float(x, default=None):
    try:
        if x in (None, "", "None", "NaN", "nan"):
            return default
        return float(x)
    except Exception:
        return default


def score_company_quality(overview: Mapping) -> dict:
    """Port of the tracker's quality score. LIVE_ONLY_NON_PIT_BACKTESTABLE."""
    if not overview or "Symbol" not in overview:
        return {"quality_score": 0.0, "quality_bucket": "UNKNOWN"}
    cap = _safe_float(overview.get("MarketCapitalization"), 0) or 0
    pm = _safe_float(overview.get("ProfitMargin"))
    om = _safe_float(overview.get("OperatingMarginTTM"))
    roe = _safe_float(overview.get("ReturnOnEquityTTM"))
    rg = _safe_float(overview.get("QuarterlyRevenueGrowthYOY"))
    eg = _safe_float(overview.get("QuarterlyEarningsGrowthYOY"))
    fpe = _safe_float(overview.get("ForwardPE"))
    peg = _safe_float(overview.get("PEGRatio"))
    beta = _safe_float(overview.get("Beta"))
    s = 0
    s += 2 if cap >= 10e9 else 1 if cap >= 2e9 else -2
    s += 1 if pm is not None and pm > 0 else -1
    s += 1 if om is not None and om > 0.05 else 0
    s += 1 if roe is not None and roe > 0.10 else 0
    s += 1 if rg is not None and rg > 0 else 0
    s += 1 if eg is not None and eg > 0 else 0
    s += 1 if fpe is not None and 0 < fpe < 60 else 0
    s += 1 if peg is not None and 0 < peg < 3 else 0
    s += 1 if beta is not None and beta <= 2.0 else -1 if beta is not None and beta > 2.0 else 0
    bucket = "A" if s >= 6 else "B" if s >= 3 else "C" if s >= 1 else "D"
    return {"quality_score": float(s), "quality_bucket": bucket}


def fundamentals_features(symbol: str, cache_dir: Path = AV_CACHE_DIR) -> dict:
    """LIVE-ONLY beat/quality features for an already-flagged symbol, from cached AV snapshots.

    Reads only on-disk cache (no network here); returns neutral values when absent. NEVER call this
    in a backtest or to define a signal. Output is tagged with the non-PIT warning.
    """
    out = {"quality_score": 0.0, "quality_bucket": "UNKNOWN", "beat_probability": 0.5,
           "surprise_score": 50.0, "revenue_score": 50.0, "cashflow_score": 50.0,
           "balance_score": 50.0, "buyback_score": 50.0, "revision_score": 50.0,
           "_warning": FUNDAMENTALS_WARNING, "_pit_safe": False}
    ov = _read_av_cache(symbol, "OVERVIEW", cache_dir)
    if ov:
        out.update(score_company_quality(ov))
    earn = _read_av_cache(symbol, "EARNINGS", cache_dir)
    if earn:
        q = earn.get("quarterlyEarnings", []) if isinstance(earn, dict) else []
        sp = [_safe_float(r.get("surprisePercentage")) for r in q[:8]]
        sp = [v for v in sp if v is not None]
        if sp:
            beat_rate = sum(v > 0 for v in sp) / len(sp)
            out["surprise_score"] = float(min(max(50 + 70 * (beat_rate - 0.5), 0), 100))
            out["beat_probability"] = float(1 / (1 + pow(2.718281828, -(out["surprise_score"] - 50) / 10)))
    return out


def _read_av_cache(symbol: str, function: str, cache_dir: Path) -> Optional[dict]:
    p = Path(cache_dir) / f"{function.lower()}_{symbol}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


# ── forward earnings calendar fetch (AV EARNINGS_CALENDAR -> cal_*.json) ───────────
def fetch_av_earnings_calendar(universe: Sequence[str], asof, lookahead: int = LOOKAHEAD_DAYS,
                               out_dir: Path = CALENDAR_DIR, horizon: str = "3month") -> Path:
    """Fetch FORWARD scheduled earnings dates from Alpha Vantage EARNINGS_CALENDAR and write them in
    the project's cal_*.json shape ({events:{sym:[dates]}, fetched_at, source}). Forward-only:
    EARNINGS_CALENDAR returns scheduled future report dates. yfinance-free. Network required."""
    import requests  # local import; offline tests never call this

    key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise RuntimeError("set ALPHAVANTAGE_API_KEY to fetch the forward earnings calendar")
    asof_d = _as_date(asof)
    r = requests.get("https://www.alphavantage.co/query",
                     params={"function": "EARNINGS_CALENDAR", "horizon": horizon, "apikey": key},
                     timeout=30)
    r.raise_for_status()
    import csv
    rows = list(csv.DictReader(io.StringIO(r.text)))
    uni = {s.upper() for s in universe}
    end = asof_d + timedelta(days=lookahead)
    events: Dict[str, List[str]] = {s: [] for s in universe}
    for row in rows:
        sym = (row.get("symbol") or "").upper()
        if sym not in uni:
            continue
        d = (row.get("reportDate") or "").strip()
        try:
            dd = date.fromisoformat(d)
        except ValueError:
            continue
        if asof_d <= dd <= end:
            events[sym].append(dd.isoformat())
    for s in events:
        events[s] = sorted(set(events[s]))
    payload = {"events": events, "fetched_at": datetime.now(timezone.utc).isoformat(),
               "source": "alpha_vantage_earnings_calendar"}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cal_{asof_d.isoformat()}_{end.isoformat()}.json"
    path.write_text(json.dumps(payload))
    return path
