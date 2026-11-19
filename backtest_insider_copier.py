#!/usr/bin/env python3
"""
backtest_insider_copier.py — event-study backtest of the run.py insider trade copier.

What it tests
    The copier in run.py mirrors large, recent open-market insider BUYS as paper BUYs:
        * open-market buy,
        * aggregate value within a trailing window > INSIDER_COPY_MIN_VALUE ($1,000,000),
        * latest insider transaction within INSIDER_COPY_MAX_AGE_DAYS (2) days.
    This harness replays those exact rules over the locally-cached Alpha Vantage
    INSIDER_TRANSACTIONS history and measures forward returns on cached daily bars.

Faithfulness / no-lookahead
    * Open-market-buy classification reuses the PRODUCTION logic
      (insider_transactions_av.normalize_transaction) — the same code that decides what
      the live copier copies.
    * A signal fires on the transaction date, but entry is delayed by FILING_LAG_DAYS
      (default 2 — the SEC Form 4 filing deadline) and executed at the NEXT trading day's
      OPEN. The strategy can never trade on information it would not yet have had.
    * While a copied position is open, the same symbol cannot re-signal (mirrors the
      live "not already held" gate).

Data (no network)
    insider:  data/cache/alpha_vantage/insider_transactions/{SYM}_{YYYYMMDD}.json
    prices :  data/cache/alpaca_bars/{SYM}_day.parquet   (cols: open/high/low/close/volume)
    bench  :  SPY_day.parquet (market-excess return)

Outputs
    backtest_outputs/insider_copier_backtest.json   (full machine-readable result)
    backtest_outputs/insider_copier_backtest.md     (human summary)

Usage
    python backtest_insider_copier.py
    python backtest_insider_copier.py --min-value 1000000 --window-days 2 \
        --filing-lag 2 --horizons 5,10,21,63 --copy-dollars 1500 --max-names 5
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
INSIDER_DIR = PROJECT_ROOT / "data" / "cache" / "alpha_vantage" / "insider_transactions"
BARS_DIR = PROJECT_ROOT / "data" / "cache" / "alpaca_bars"
OUT_DIR = PROJECT_ROOT / "backtest_outputs"

# Reuse the production open-market-buy classifier so the backtest copies exactly what
# the live copier would. Fall back to a local re-implementation if the import fails.
try:
    from insider_transactions_av import normalize_transaction
except Exception:  # pragma: no cover
    normalize_transaction = None  # type: ignore


# ── data loading ──────────────────────────────────────────────────────────────────
def _latest_insider_cache(symbol: str) -> Optional[Path]:
    files = sorted(INSIDER_DIR.glob(f"{symbol}_*.json"))
    return files[-1] if files else None


def load_open_market_buys(symbol: str) -> List[Tuple[date, float]]:
    """Return [(transaction_date, value_usd), ...] of OPEN-MARKET BUYS only."""
    path = _latest_insider_cache(symbol)
    if path is None:
        return []
    try:
        rows = json.loads(path.read_text()).get("data", []) or []
    except Exception:
        return []
    buys: List[Tuple[date, float]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        norm = normalize_transaction(raw) if normalize_transaction else _fallback_norm(raw)
        if norm.get("transaction_class") != "open_market_buy":
            continue
        d = _to_date(norm.get("transaction_date"))
        v = float(norm.get("value") or 0.0)
        if d is not None and v > 0:
            buys.append((d, v))
    buys.sort(key=lambda x: x[0])
    return buys


def _fallback_norm(raw: Dict[str, Any]) -> Dict[str, Any]:
    aod = str(raw.get("acquisition_or_disposal") or "").upper().strip()
    sec = str(raw.get("security_type") or "").lower()
    shares = float(str(raw.get("shares") or 0).replace(",", "") or 0)
    price = float(str(raw.get("share_price") or 0).replace(",", "") or 0)
    is_common = any(k in sec for k in ("common stock", "common share", "ordinary share"))
    is_comp = any(k in sec for k in ("rsu", "restricted", "award", "unit"))
    cls = "open_market_buy" if (is_common and not is_comp and aod == "A" and price > 0) else "other"
    return {"transaction_class": cls, "transaction_date": raw.get("transaction_date"),
            "value": shares * price}


def _to_date(v: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(v)[:10])
    except Exception:
        return None


SEC_EVENTS_PATH = PROJECT_ROOT / "data" / "cache" / "sec_insider_events.json"
PIT_EVENTS_PATH = PROJECT_ROOT / "data" / "cache" / "pit_insider_events.json"


def load_event_file(path: Path) -> Dict[str, List[Tuple[date, float]]]:
    """Load a {symbol: [[iso_date, value], ...]} event file into typed tuples.

    sec  : mined from data/sec_cache/*.cache (recent live snapshot, small/mid-cap).
    pit  : POINT-IN-TIME, built by build_pit_insider_universe.py from SEC quarterly
           Form 345 data sets — every Form 4 filed each quarter, delisted names included
           (no name-selection survivorship bias).
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    out: Dict[str, List[Tuple[date, float]]] = {}
    for sym, evs in raw.items():
        rows = []
        for item in evs or []:
            d = _to_date(item[0]); v = float(item[1] or 0.0)
            if d is not None and v > 0:
                rows.append((d, v))
        rows.sort(key=lambda x: x[0])
        if rows:
            out[str(sym).upper()] = rows
    return out


def load_sec_buy_events() -> Dict[str, List[Tuple[date, float]]]:
    return load_event_file(SEC_EVENTS_PATH)


def load_pit_buy_events() -> Dict[str, List[Tuple[date, float]]]:
    return load_event_file(PIT_EVENTS_PATH)


def load_bars(symbol: str) -> Optional[pd.DataFrame]:
    f = BARS_DIR / f"{symbol}_day.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
        df = df[["open", "close"]].copy()
        df.index = pd.to_datetime(df.index).tz_convert("UTC").date  # type: ignore
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        return df
    except Exception:
        return None


# ── signal generation (mirrors the copier rules) ────────────────────────────────────
def generate_signals(buys: List[Tuple[date, float]], min_value: float,
                     window_days: int, hold_days: int, trading_dates: List[date],
                     filing_lag: int) -> List[date]:
    """
    A signal date d fires when the trailing `window_days`-day open-market buy value
    (inclusive of d) first exceeds `min_value` AND no copied position is currently open.
    Returns the list of transaction dates that trigger a copy.
    """
    signals: List[date] = []
    open_until: Optional[date] = None  # symbol can't re-signal while held
    for i, (d, _v) in enumerate(buys):
        if open_until is not None and d <= open_until:
            continue
        window_lo = d - timedelta(days=window_days)
        window_val = sum(v for (dd, v) in buys if window_lo <= dd <= d)
        if window_val <= min_value:
            continue
        # Determine the realised holding window to set the re-signal lockout.
        entry_dt = _next_trading_on_or_after(d + timedelta(days=filing_lag), trading_dates)
        if entry_dt is None:
            continue
        exit_dt = _shift_trading_days(entry_dt, hold_days, trading_dates)
        signals.append(d)
        open_until = exit_dt or d
    return signals


def _has_cluster(evs: List[Tuple[date, float]], min_value: float, window_days: int) -> bool:
    """True if any trailing `window_days`-day open-market buy value exceeds min_value."""
    rows = sorted(evs)
    for i, (d, _v) in enumerate(rows):
        lo = d - timedelta(days=window_days)
        if sum(v for dd, v in rows if lo <= dd <= d) > min_value:
            return True
    return False


def _next_trading_on_or_after(d: date, trading_dates: List[date]) -> Optional[date]:
    import bisect
    i = bisect.bisect_left(trading_dates, d)
    return trading_dates[i] if i < len(trading_dates) else None


def _shift_trading_days(d: date, n: int, trading_dates: List[date]) -> Optional[date]:
    import bisect
    i = bisect.bisect_left(trading_dates, d)
    j = i + n
    return trading_dates[j] if 0 <= j < len(trading_dates) else None


# ── trade simulation ────────────────────────────────────────────────────────────────
def simulate_symbol(symbol: str, buys: List[Tuple[date, float]], bars: pd.DataFrame,
                    spy: Optional[pd.DataFrame], min_value: float, window_days: int,
                    filing_lag: int, horizons: List[int]) -> List[Dict[str, Any]]:
    trading_dates = list(bars.index)
    trades: List[Dict[str, Any]] = []
    for hd in horizons:
        sig_dates = generate_signals(buys, min_value, window_days, hd, trading_dates, filing_lag)
        for sd in sig_dates:
            entry_dt = _next_trading_on_or_after(sd + timedelta(days=filing_lag), trading_dates)
            if entry_dt is None:
                continue
            exit_dt = _shift_trading_days(entry_dt, hd, trading_dates)
            if exit_dt is None:
                continue
            entry_px = float(bars.loc[entry_dt, "open"])
            exit_px = float(bars.loc[exit_dt, "close"])
            if entry_px <= 0 or exit_px <= 0:
                continue
            ret = exit_px / entry_px - 1.0
            mkt = _spy_return(spy, entry_dt, exit_dt)
            trades.append({
                "symbol": symbol, "horizon": hd, "signal_date": sd.isoformat(),
                "entry_date": entry_dt.isoformat(), "exit_date": exit_dt.isoformat(),
                "entry_px": round(entry_px, 4), "exit_px": round(exit_px, 4),
                "ret": ret, "mkt_ret": mkt,
                "excess_ret": (ret - mkt) if mkt is not None else None,
            })
    return trades


def _spy_return(spy: Optional[pd.DataFrame], entry_dt: date, exit_dt: date) -> Optional[float]:
    if spy is None:
        return None
    try:
        if entry_dt in spy.index and exit_dt in spy.index:
            e = float(spy.loc[entry_dt, "open"]); x = float(spy.loc[exit_dt, "close"])
            if e > 0 and x > 0:
                return x / e - 1.0
    except Exception:
        pass
    return None


# ── aggregation / stats ─────────────────────────────────────────────────────────────
def _stats(rets: List[float]) -> Dict[str, Any]:
    if not rets:
        return {"n": 0}
    n = len(rets)
    mean = statistics.mean(rets)
    sd = statistics.pstdev(rets) if n > 1 else 0.0
    wins = [r for r in rets if r > 0]
    t_stat = (mean / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return {
        "n": n,
        "win_rate": round(len(wins) / n, 4),
        "mean_ret": round(mean, 5),
        "median_ret": round(statistics.median(rets), 5),
        "stdev": round(sd, 5),
        "best": round(max(rets), 5),
        "worst": round(min(rets), 5),
        "t_stat": round(t_stat, 3),
    }


def aggregate(trades: List[Dict[str, Any]], horizons: List[int], copy_dollars: float,
              max_names: int) -> Dict[str, Any]:
    by_h: Dict[str, Any] = {}
    for hd in horizons:
        rows = [t for t in trades if t["horizon"] == hd]
        rets = [t["ret"] for t in rows]
        exc = [t["excess_ret"] for t in rows if t["excess_ret"] is not None]
        s = _stats(rets)
        s_exc = _stats(exc)
        # Map to the live copier's economics: fixed $copy_dollars per name.
        mean_dollar = (s.get("mean_ret", 0.0) or 0.0) * copy_dollars
        # Signal cadence → implied $/trading-day at max_names concurrent copies.
        span_days = _calendar_span(rows)
        trades_per_day = (len(rows) / span_days) if span_days else 0.0
        by_h[str(hd)] = {
            "raw": s,
            "market_excess": s_exc,
            "mean_dollar_per_trade@%d" % int(copy_dollars): round(mean_dollar, 2),
            "signals_per_calendar_day": round(trades_per_day, 4),
            "implied_dollar_per_day@maxnames%d" % max_names:
                round(mean_dollar * min(trades_per_day, max_names), 2),
        }
    return by_h


def _calendar_span(rows: List[Dict[str, Any]]) -> int:
    ds = [date.fromisoformat(r["signal_date"]) for r in rows]
    if len(ds) < 2:
        return 0
    return max(1, (max(ds) - min(ds)).days)


# ── driver ──────────────────────────────────────────────────────────────────────────
def run(min_value: float, window_days: int, filing_lag: int, horizons: List[int],
        copy_dollars: float, max_names: int, source: str = "av",
        only: Optional[List[str]] = None, events_file: Optional[str] = None) -> Dict[str, Any]:
    price_syms = {os.path.basename(p).rsplit("_day", 1)[0]
                  for p in glob.glob(str(BARS_DIR / "*_day.parquet"))}
    event_map: Dict[str, List[Tuple[date, float]]] = {}
    eligible_total = 0
    priced_eligible = 0
    if source in ("sec", "pit"):
        # --events-file lets a caller point the SAME methodology at a different event set
        # (e.g. the out-of-sample daily-index crawl) without touching the in-sample cache.
        if events_file:
            event_map = load_event_file(Path(events_file))
        else:
            event_map = load_sec_buy_events() if source == "sec" else load_pit_buy_events()
        # Denominator that matters: tickers with a >min_value window cluster (i.e. that
        # could actually signal), NOT every ticker with any buy.
        cluster_eligible = {s for s, evs in event_map.items()
                            if _has_cluster(evs, min_value, window_days)}
        eligible_total = len(cluster_eligible)
        priced_eligible = len(cluster_eligible & price_syms)
        universe = sorted(set(event_map) & price_syms)
    else:
        insider_syms = {os.path.basename(p).split("_")[0]
                        for p in glob.glob(str(INSIDER_DIR / "*.json"))}
        universe = sorted(insider_syms & price_syms)
    if only:
        keep = {s.upper() for s in only}
        universe = [s for s in universe if s in keep]
    spy = load_bars("SPY")

    all_trades: List[Dict[str, Any]] = []
    per_symbol: Dict[str, int] = {}
    covered: List[str] = []
    for sym in universe:
        if sym == "SPY":
            continue
        buys = event_map.get(sym, []) if source in ("sec", "pit") else load_open_market_buys(sym)
        bars = load_bars(sym)
        if not buys or bars is None or bars.empty:
            continue
        trades = simulate_symbol(sym, buys, bars, spy, min_value, window_days,
                                 filing_lag, horizons)
        if trades:
            covered.append(sym)
            # count unique signals (per shortest horizon to avoid double count)
            per_symbol[sym] = len({t["signal_date"] for t in trades})
            all_trades.extend(trades)

    agg = aggregate(all_trades, horizons, copy_dollars, max_names)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "params": {
            "min_value": min_value, "window_days": window_days, "filing_lag_days": filing_lag,
            "horizons": horizons, "copy_dollars": copy_dollars, "max_names": max_names,
            "source": source,
        },
        "universe_size": len(universe),
        "eligible_universe": eligible_total,
        "priced_coverage": (round(priced_eligible / eligible_total, 3) if eligible_total else None),
        "symbols_with_signals_count": len(covered),
        "symbols_with_signals": covered if len(covered) <= 60 else covered[:60] + ["…"],
        "unique_signals_per_symbol": per_symbol,
        "total_unique_signals": sum(per_symbol.values()),
        "results_by_horizon": agg,
        "trades": all_trades,
    }


def _markdown(result: Dict[str, Any]) -> str:
    p = result["params"]
    lines = [
        "# Insider Trade Copier — Backtest", "",
        f"_Generated {result['generated_at']}_", "",
        "## Parameters",
        f"- Source: **{p.get('source','av')}** ({ {'sec':'SEC Form 4 cache — recent snapshot','pit':'point-in-time SEC Form 345 data sets — survivorship-clean, delisted names included','av':'AV INSIDER_TRANSACTIONS cache'}.get(p.get('source','av'),'') })",
        f"- Min open-market buy value: **${p['min_value']:,.0f}** (trailing {p['window_days']}-day window)",
        f"- Filing lag before entry: **{p['filing_lag_days']} day(s)**, entry at next-day OPEN, exit at horizon CLOSE",
        f"- Horizons (trading days): {p['horizons']}",
        f"- Copier economics: ${p['copy_dollars']:,.0f}/name, max {p['max_names']} names/cycle", "",
        f"## Coverage",
        f"- Eligible tickers in universe: **{result.get('eligible_universe') or result['universe_size']}**"
        + (f" — priced coverage **{result['priced_coverage']:.0%}** "
           f"(residual survivorship = unpriceable {(1-result['priced_coverage'])*100:.0f}%)"
           if result.get('priced_coverage') else ""),
        f"- Symbols with both insider + price data: **{result['universe_size']}**",
        f"- Symbols that produced signals: **{result.get('symbols_with_signals_count', len(result['symbols_with_signals']))}**",
        f"- Total unique copy signals: **{result['total_unique_signals']}**", "",
        "## Forward-return performance by holding horizon", "",
        "| Horizon (d) | Trades | Win% | Mean | Median | Stdev | t-stat | Mean vs SPY | $/trade |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    cd = int(p["copy_dollars"])
    for hd in p["horizons"]:
        r = result["results_by_horizon"].get(str(hd), {})
        raw = r.get("raw", {})
        exc = r.get("market_excess", {})
        if not raw.get("n"):
            lines.append(f"| {hd} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {hd} | {raw['n']} | {raw['win_rate']*100:.1f}% | {raw['mean_ret']*100:+.2f}% | "
            f"{raw['median_ret']*100:+.2f}% | {raw['stdev']*100:.2f}% | {raw['t_stat']:+.2f} | "
            f"{(exc.get('mean_ret') or 0)*100:+.2f}% | ${r.get('mean_dollar_per_trade@%d' % cd, 0):+,.2f} |")
    lines += [
        "", "## Read",
        "- **Mean vs SPY** is the edge that matters — raw return that just tracks the market is beta, not insider alpha.",
        "- **t-stat** |·|>2 ≈ statistically distinguishable from zero at this sample size; small samples warrant caution.",
        "- `$/trade` applies the live copier's fixed notional; positive-but-tiny edge × small size ≈ limited contribution",
        "  to the $400–$500/day paper target. Treat as a *conviction modifier* candidate, not a standalone engine.", "",
        "_No-lookahead: entry is delayed by the Form 4 filing lag and executed at the next-day open; a held name cannot re-signal._",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the run.py insider trade copier.")
    ap.add_argument("--min-value", type=float, default=1_000_000.0)
    ap.add_argument("--window-days", type=int, default=2)
    ap.add_argument("--filing-lag", type=int, default=2)
    ap.add_argument("--horizons", type=str, default="5,10,21,63")
    ap.add_argument("--copy-dollars", type=float, default=1500.0)
    ap.add_argument("--max-names", type=int, default=5)
    ap.add_argument("--source", choices=["av", "sec", "pit"], default="av",
                    help="av = AV INSIDER_TRANSACTIONS cache (mega-cap); "
                         "sec = SEC Form 4 cache (recent snapshot); "
                         "pit = point-in-time SEC Form 345 data sets (survivorship-clean)")
    ap.add_argument("--symbols", type=str, default="",
                    help="comma-separated universe filter (e.g. small/mid-cap names)")
    ap.add_argument("--events-file", type=str, default="",
                    help="override event-file path (same schema as pit_insider_events.json); "
                         "use to run the identical methodology on an out-of-sample event set")
    ap.add_argument("--tag", type=str, default="", help="output filename suffix")
    args = ap.parse_args()

    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    only = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    result = run(args.min_value, args.window_days, args.filing_lag, horizons,
                 args.copy_dollars, args.max_names, source=args.source, only=only,
                 events_file=(args.events_file or None))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f"_{args.source}" if args.source != "av" else "")
    (OUT_DIR / f"insider_copier_backtest{tag}.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    md = _markdown(result)
    (OUT_DIR / f"insider_copier_backtest{tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
