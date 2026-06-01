#!/usr/bin/env python3
"""
rl_alpaca_backfill.py — Phase 1: offline historical backfill/replay for RL datasets.
=====================================================================================

Pulls Alpaca HISTORICAL bars (read-only market data) and REPLAYS the real Alpha setup
detector and a Buffett daily-technical/forward-return model over that history to generate
a large LABELLED RL event log (decision → realized_R / forward_return). The output is
written as research_events.jsonl into data/rl_backfill/<tag>/, which the RL dataset
builder picks up alongside live Research-Lab events — fixing the small-sample gate.

SAFETY: fully offline. Uses ONLY alpaca.data.historical (read-only bars). It NEVER imports
AlpacaGateway / alpaca.trading, never submits an order, and never touches live state. The
replay functions are pure (operate on DataFrames) so they are unit-testable without network.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import rl_common as rc

logger = logging.getLogger(__name__)

DEFAULT_ALPHA_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
                         "TSLA", "AMD", "AVGO", "NFLX"]
DEFAULT_BUFFETT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "BRK.B", "JNJ", "KO", "PG",
                           "JPM", "V", "MA", "UNH", "HD", "PEP", "COST", "WMT", "XOM", "CVX"]


def _creds():
    try:
        from dotenv import load_dotenv
        load_dotenv(rc.PROJECT_ROOT / ".env")
    except Exception:
        pass
    key = (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
           or os.environ.get("ALPACA_API_KEY_ID") or "")
    sec = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
           or os.environ.get("ALPACA_API_SECRET_KEY") or "")
    return key, sec


def fetch_bars(symbols: List[str], start, end, timeframe: str = "day") -> Dict[str, Any]:
    """Read-only historical bars → {symbol: DataFrame[open,high,low,close,volume]}. Guarded."""
    key, sec = _creds()
    if not key or not sec:
        logger.warning("Alpaca credentials missing — no historical bars fetched")
        return {}
    try:
        import socket
        socket.setdefaulttimeout(60)
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        tf = TimeFrame.Day if timeframe == "day" else TimeFrame.Minute
        feed = DataFeed.SIP if os.getenv("ALPACA_DATA_FEED", "iex").lower() == "sip" else DataFeed.IEX
        client = StockHistoricalDataClient(key, sec)
        if timeframe.endswith("min"):
            from alpaca.data.timeframe import TimeFrameUnit
            tf = TimeFrame(int(timeframe.replace("min", "") or 5), TimeFrameUnit.Minute)
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=tf,
                               start=start, end=end, feed=feed)
        df = client.get_stock_bars(req).df
        out: Dict[str, Any] = {}
        if df is None or df.empty:
            return out
        df = df.reset_index()
        for sym, grp in df.groupby("symbol"):
            g = grp.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
            out[str(sym)] = g
        return out
    except Exception as exc:
        logger.warning("Alpaca historical fetch failed (%s)", exc)
        return {}


# ── ALPHA replay (real detect_setup + bracket simulation) ───────────────────────────
def replay_alpha(symbol: str, intraday_df, replay_min: float = 45.0,
                 risk_pct: float = 0.005, stride: int = 6, min_bars: int = 8) -> List[dict]:
    """Replay the real Alpha setup detector over intraday bars → entry/exit event pairs."""
    events: List[dict] = []
    try:
        from alpha_call_engine import detect_setup, AlphaCallConfig
        import pandas as pd
    except Exception as exc:
        logger.debug("alpha replay unavailable: %s", exc)
        return events
    if intraday_df is None or len(intraday_df) < min_bars + 2:
        return events
    cfg = AlphaCallConfig()
    mc = {"overall_market_bias": "bullish", "trade_permission": "normal"}
    try:
        idx = intraday_df.index
        et = idx.tz_convert("America/New_York") if idx.tz is not None else idx.tz_localize("UTC").tz_convert("America/New_York")
        days = sorted(set(d.date() for d in et))
    except Exception:
        return events
    for day in days:
        mask = [d.date() == day for d in et]
        day_bars = intraday_df[mask]
        if len(day_bars) < min_bars + 2:
            continue
        for i in range(min_bars, len(day_bars) - 1, max(1, stride)):
            window = day_bars.iloc[:i + 1]
            try:
                ev = detect_setup(symbol, window, cfg, mc)
            except Exception:
                continue
            if ev.reject_reason or ev.setup_score < replay_min:
                continue
            entry = float(window["close"].iloc[-1])
            ts = window.index[-1]
            risk = max(entry * risk_pct, 1e-6)
            stop, target = entry - risk, entry + 3.0 * risk
            fwd = day_bars.iloc[i + 1:]
            realized_R: Optional[float] = None
            peak_R = 0.0
            for _, bar in fwd.iterrows():
                peak_R = max(peak_R, (float(bar["high"]) - entry) / risk)
                if float(bar["low"]) <= stop:
                    realized_R = -1.0; break
                if float(bar["high"]) >= target:
                    realized_R = 3.0; break
            if realized_R is None:
                realized_R = round((float(day_bars["close"].iloc[-1]) - entry) / risk, 4)
            peak_R = round(max(peak_R, realized_R), 4)
            pop = max(0.0, min(1.0, ev.setup_score / 100.0))
            tag = str(getattr(ts, "date", lambda: day)()).replace("-", "")
            # Multi-timeframe trend over the intraday window (Roy 2024): short windows on
            # 5-min bars approximate the fast/mid/slow timeframes the thesis's bots read.
            mtf = rc.mtf_trend_features([float(c) for c in window["close"]], windows=(5, 20, 50))
            # Catalyst strength (§3.9): relative volume of the latest bars vs the session
            # mean — the thesis defines catalyst stocks by high relative volume.
            catalyst = 0.0
            try:
                vols = [float(v) for v in day_bars["volume"]]
                mean_v = (sum(vols) / len(vols)) if vols else 0.0
                recent_v = sum(float(v) for v in window["volume"].iloc[-3:]) / min(3, len(window))
                if mean_v > 0:
                    catalyst = max(0.0, min(1.0, (recent_v / mean_v - 1.0) / 2.0))   # rvol 1→0, 3→1
            except Exception:
                catalyst = 0.0
            events.append({"event_type": "alpha_entry", "symbol": symbol, "date_tag": tag,
                           "timestamp": str(ts), "setup_type": ev.setup_type,
                           "setup_score": round(float(ev.setup_score), 2), "grade": ev.grade,
                           "probability_of_profit": round(pop, 4), "expected_R": round(pop * 3 - (1 - pop), 4),
                           "vol_regime": "normal", "premarket_gap_pct": 0.0,
                           "catalyst_strength": round(catalyst, 4), **mtf, "source": "alpaca_replay"})
            events.append({"event_type": "alpha_exit", "symbol": symbol, "date_tag": tag,
                           "timestamp": str(ts), "r_multiple": round(realized_R, 4),
                           "realized_R": round(realized_R, 4), "peak_R": peak_R,
                           "probability_band": "60-69", "holding_period_days": 0,
                           **mtf, "source": "alpaca_replay"})
            break   # one trade per symbol per session
    return events


# ── BUFFETT replay (daily technicals + real forward returns) ────────────────────────
def replay_buffett(symbol: str, daily_df, step: int = 3, lookback: int = 200,
                   horizons=(5, 20, 60)) -> List[dict]:
    """Replay Buffett daily technicals + REAL forward returns over daily bars → DCA events."""
    events: List[dict] = []
    try:
        import streaming_indicator_engine as sie
    except Exception:
        sie = None
    try:
        closes = [float(x) for x in daily_df["close"]]
        highs = [float(x) for x in daily_df["high"]]
        lows = [float(x) for x in daily_df["low"]]
    except Exception:
        return events
    n = len(closes)
    max_h = max(horizons)
    if n < lookback + max_h + 1:
        return events
    for t in range(lookback, n - max_h, max(1, step)):
        wc, wh, wl = closes[:t + 1], highs[:t + 1], lows[:t + 1]
        price = closes[t]
        if price <= 0:
            continue
        rsi = (sie._rsi(wc, 14) if sie else None) or 50.0
        sma50 = (sie._sma(wc, 50) if sie else None) or price
        sma200 = (sie._sma(wc, 200) if sie else None) or price
        atr = (sie._atr(wh, wl, wc, 14) if sie else None) or 0.0
        atr_base = (sie._atr(wh[:-20], wl[:-20], wc[:-20], 14) if (sie and t > lookback + 20) else atr) or atr or 1.0
        dist200 = (price - sma200) / sma200 if sma200 else 0.0
        discount = max(0.0, min(0.5, -dist200))                 # below 200DMA = discount proxy
        p2iv = round(max(0.3, 1.0 - discount), 4)
        rsi_comp = 30.0 if rsi < 30 else (15.0 if rsi < 40 else 0.0)
        composite = round(min(100.0, discount * 100 * 0.6 + rsi_comp + (10 if dist200 < 0 else 0)), 2)
        conviction = round(min(100.0, 50 + discount * 80), 2)
        bal_score = round(min(100.0, 60 + discount * 60), 2)
        fwd = {h: round(closes[t + h] / price - 1.0, 6) for h in horizons}
        # Multi-timeframe trend (Roy 2024): 10/50/200-day SMAs on the daily series span the
        # fast/mid/slow (≈ weekly→long-term) timeframes the thesis combines for swing bots.
        mtf = rc.mtf_trend_features(wc, windows=(10, 50, 200))
        events.append({
            "event_type": "bal_buy", "symbol": symbol,
            "date_tag": str(daily_df.index[t].date()).replace("-", "") if hasattr(daily_df.index[t], "date") else str(t),
            "timestamp": str(daily_df.index[t]), "bal_score": bal_score, "conviction_score": conviction,
            "composite_dca_score": composite, "price_to_intrinsic_value": p2iv,
            "optimized_target_weight": 0.08, "current_weight": 0.0,
            "RSI": round(rsi, 2), "ATR_expansion": round((atr / atr_base) if atr_base else 1.0, 3),
            "premarket_gap_pct": 0.0, "dca_action": "normal", "tier": "starter",
            "fwd_return_5d": fwd.get(5), "fwd_return_20d": fwd.get(20), "fwd_return_60d": fwd.get(60),
            **mtf, "source": "alpaca_replay"})
    return events


# ── orchestration ───────────────────────────────────────────────────────────────────
def build_backfill(alpha_symbols: List[str] = None, buffett_symbols: List[str] = None,
                   intraday_days: int = 25, daily_days: int = 720,
                   intraday_tf: str = "5min", out_dir: Path = None) -> Dict[str, Any]:
    """Fetch history, replay Alpha + Buffett, write research_events.jsonl. OFFLINE; no orders."""
    alpha_symbols = alpha_symbols or DEFAULT_ALPHA_SYMBOLS
    buffett_symbols = buffett_symbols or DEFAULT_BUFFETT_SYMBOLS
    out_dir = Path(out_dir) if out_dir else (rc.RL_BACKFILL_DIR / rc.data_tag())
    out_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    events: List[dict] = []

    intraday = fetch_bars(alpha_symbols, end - timedelta(days=intraday_days), end, timeframe=intraday_tf)
    for sym, df in intraday.items():
        events.extend(replay_alpha(sym, df))
    daily = fetch_bars(buffett_symbols, end - timedelta(days=daily_days), end, timeframe="day")
    for sym, df in daily.items():
        events.extend(replay_buffett(sym, df))

    path = out_dir / "research_events.jsonl"
    try:
        with open(path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, default=str) + "\n")
    except Exception as exc:
        logger.warning("backfill write failed: %s", exc)
    n_entry = sum(1 for e in events if e["event_type"] == "alpha_entry")
    n_exit = sum(1 for e in events if e["event_type"] == "alpha_exit")
    n_bal = sum(1 for e in events if e["event_type"] == "bal_buy")
    summary = {"generated_at_utc": rc.now_iso(), "date_tag": rc.data_tag(),
               "output": str(path), "events": len(events),
               "alpha_entry": n_entry, "alpha_exit": n_exit, "buffett": n_bal,
               "alpha_symbols_fetched": len(intraday), "buffett_symbols_fetched": len(daily),
               "offline_only": True, "orders_submitted": 0}
    rc.write_json(out_dir / "backfill_summary.json", summary)
    return summary


# ── Source 3: repo historical dataset ingestion (offline; quality-gated) ────────────
# Reuses cached repo data — NO network, NO orders. Each row is quality-gated; bad rows
# are skipped (never fabricated). Output is research_events.jsonl that the dataset builder
# already consumes, so these labelled samples grow the RL datasets.

_VALID_OUTCOMES = {"win", "loss", "agent_exit", "timeout", "eod"}


def _scaled_pct(x: float) -> float:
    """Normalise an excursion field to a fraction (handles 'percent' vs 'fraction' inputs)."""
    v = abs(x)
    return v / 100.0 if v > 1.0 else v


def ingest_trade_replay_csv(path, max_rows: int = 20000) -> List[dict]:
    """
    Map a trade-replay CSV (entry/stop/target/mfe_pct/mae_pct/outcome) → alpha entry/exit
    events with REAL realized_R / peak_R. Quality-gated: rows with invalid entry/stop or an
    unrecognised outcome are skipped. Returns [] on any failure.
    """
    import csv
    events: List[dict] = []
    p = Path(path)
    if not p.exists():
        return events
    kept = skipped = 0
    try:
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if kept >= max_rows:
                    break
                try:
                    e = float(row.get("entry_price") or 0)
                    s = float(row.get("stop_price") or 0)
                    t = float(row.get("target_price") or 0)
                    outcome = str(row.get("outcome") or "").lower()
                    if not (e > 0 and 0 < s < e) or outcome not in _VALID_OUTCOMES:
                        skipped += 1
                        continue
                    risk = e - s
                    planned_R = min(5.0, (t - e) / risk) if t > e else 3.0
                    mfe_R = (_scaled_pct(float(row.get("mfe_pct") or 0)) * e) / risk
                    mae_R = (_scaled_pct(float(row.get("mae_pct") or 0)) * e) / risk
                    if outcome == "win":
                        realized_R = planned_R
                    elif outcome == "loss":
                        realized_R = -1.0
                    else:  # agent_exit / timeout / eod → bounded estimate from real excursions
                        realized_R = max(-1.0, min(planned_R, mfe_R * 0.4 - mae_R * 0.4))
                    peak_R = round(max(mfe_R, realized_R, 0.0), 4)
                    sym = str(row.get("symbol") or "REPLAY").upper()
                    tag = str(row.get("timestamp") or "")[:10].replace("-", "") or rc.data_tag()
                    vsr = rc.to_f(row.get("volume_spike_ratio"))
                    pop = max(0.0, min(1.0, 0.5 + 0.1 * (planned_R if realized_R > 0 else -0.5)))
                    # Catalyst strength (§3.9) from the volume-spike ratio (an RVOL proxy):
                    # vsr 1→0, vsr 3→1. No bar series here, so MTF stays neutral (absent).
                    catalyst = max(0.0, min(1.0, (vsr - 1.0) / 2.0)) if vsr > 0 else 0.0
                    events.append({"event_type": "alpha_entry", "symbol": sym, "date_tag": tag,
                                   "timestamp": row.get("timestamp"), "setup_type": row.get("setup_type", ""),
                                   "setup_score": round(min(100.0, 40 + vsr * 10), 2),
                                   "probability_of_profit": round(pop, 4),
                                   "expected_R": round(planned_R, 4), "vol_regime": "normal",
                                   "catalyst_strength": round(catalyst, 4),
                                   "source": "repo_trade_replay"})
                    events.append({"event_type": "alpha_exit", "symbol": sym, "date_tag": tag,
                                   "timestamp": row.get("timestamp"), "r_multiple": round(realized_R, 4),
                                   "realized_R": round(realized_R, 4), "peak_R": peak_R,
                                   "probability_band": "60-69", "holding_period_days": 0,
                                   "source": "repo_trade_replay"})
                    kept += 1
                except Exception:
                    skipped += 1
                    continue
    except Exception as exc:
        logger.warning("ingest_trade_replay_csv(%s) failed: %s", path, exc)
    logger.info("trade-replay %s: kept=%d skipped=%d", p.name, kept, skipped)
    return events


def ingest_daily_bars_csv(path, max_symbols: int = 150, step: int = 5,
                          min_rows: int = 270) -> List[dict]:
    """
    Group a cached daily-bars CSV (symbol,timestamp,OHLCV) by symbol and run the existing
    replay_buffett() per symbol → DCA events with REAL 5/20/60d forward returns. Offline.
    """
    events: List[dict] = []
    p = Path(path)
    if not p.exists():
        return events
    try:
        import pandas as pd
        df = pd.read_csv(p)
        if "symbol" not in df.columns or "close" not in df.columns:
            return events
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        syms = [s for s, g in df.groupby("symbol") if len(g) >= min_rows]
        syms = sorted(syms)[:max_symbols]
        for sym in syms:
            g = df[df["symbol"] == sym].sort_values("timestamp").set_index("timestamp")
            events.extend(replay_buffett(str(sym), g[["open", "high", "low", "close", "volume"]], step=step))
    except Exception as exc:
        logger.warning("ingest_daily_bars_csv(%s) failed: %s", path, exc)
    return events


def _resample(df, rule: str):
    """Resample a 1-minute OHLCV frame to `rule` (e.g. '5min','15min','1h'). Empty on failure."""
    try:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        return df.resample(rule).agg(agg).dropna()
    except Exception:
        return df


def ingest_intraday_bars_1m_csv(path, max_symbols: int = 80, stride: int = 6,
                                warmup: int = 20, horizon: int = 12,
                                max_per_symbol: int = 80, risk_pct: float = 0.003) -> List[dict]:
    """
    Replay REAL 1-minute bars (symbol,timestamp,OHLCV) into ENRICHED alpha entry/exit events.
    Per symbol: resample 1m→5m/15m/1h, walk decision points on the 5m series, bracket a long
    (stop = entry−risk, target = entry+3R) over the next `horizon` 5m bars for realized_R, and
    attach the full bar-context enrichment (MTF 5m/1h trend, rel-vol, VWAP dist, ATR%, catalyst)
    via rl_alpaca_bar_enrichment.bar_features. OFFLINE (local bars; no network). Caps logged.
    """
    events: List[dict] = []
    p = Path(path)
    if not p.exists():
        return events
    try:
        import pandas as pd
        import rl_alpaca_bar_enrichment as enr
    except Exception as exc:
        logger.debug("intraday-1m ingest unavailable: %s", exc)
        return events
    try:
        df = pd.read_csv(p)
        need = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        if not need.issubset(df.columns):
            return events
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        # real tickers only (skip degenerate numeric symbols)
        syms = [s for s in df["symbol"].astype(str).unique() if s and not s.isdigit()]
        syms = sorted(syms)[:max_symbols]
        dropped = max(0, len([s for s in df["symbol"].astype(str).unique()
                              if s and not s.isdigit()]) - len(syms))
        if dropped:
            logger.info("intraday-1m %s: capping at %d symbols (dropped %d)", p.name, max_symbols, dropped)
        for sym in syms:
            g = df[df["symbol"].astype(str) == sym].sort_values("timestamp").set_index("timestamp")
            g = g[["open", "high", "low", "close", "volume"]]
            b5 = _resample(g, "5min")
            b15 = _resample(g, "15min")
            b1h = _resample(g, "1h")
            if len(b5) < warmup + 2:
                continue
            kept = 0
            for i in range(warmup, len(b5) - 1, max(1, stride)):
                if kept >= max_per_symbol:
                    break
                ts = b5.index[i]
                entry = float(b5["close"].iloc[i])
                if entry <= 0:
                    continue
                w5 = b5.iloc[:i + 1]
                risk = max(entry * risk_pct, 1e-6)
                stop, target = entry - risk, entry + 3.0 * risk
                fwd = b5.iloc[i + 1:i + 1 + horizon]
                realized_R = None
                peak_R = 0.0
                for _, bar in fwd.iterrows():
                    peak_R = max(peak_R, (float(bar["high"]) - entry) / risk)
                    if float(bar["low"]) <= stop:
                        realized_R = -1.0
                        break
                    if float(bar["high"]) >= target:
                        realized_R = 3.0
                        break
                if realized_R is None:
                    last = float(fwd["close"].iloc[-1]) if len(fwd) else entry
                    realized_R = round((last - entry) / risk, 4)
                peak_R = round(max(peak_R, realized_R), 4)
                feats = enr.bar_features({"5m": w5,
                                          "15m": b15[b15.index <= ts],
                                          "1h": b1h[b1h.index <= ts]})
                pop = max(0.0, min(1.0, 0.5 + 0.1 * feats["mtf_alignment"]))
                tag = str(ts.date()).replace("-", "")
                base = {"symbol": sym, "date_tag": tag, "timestamp": str(ts), **feats}
                events.append({"event_type": "alpha_entry", "setup_type": "intraday_1m_replay",
                               "setup_score": round(min(100.0, 40 + feats["rel_volume"] * 15), 2),
                               "probability_of_profit": round(pop, 4),
                               "expected_R": round(pop * 3 - (1 - pop), 4), "vol_regime": "normal",
                               "source": "intraday_1m_replay", **base})
                events.append({"event_type": "alpha_exit", "r_multiple": round(realized_R, 4),
                               "realized_R": round(realized_R, 4), "peak_R": peak_R,
                               "probability_band": "60-69", "holding_period_days": 0,
                               "source": "intraday_1m_replay", **base})
                kept += 1
    except Exception as exc:
        logger.warning("ingest_intraday_bars_1m_csv(%s) failed: %s", path, exc)
    logger.info("intraday-1m %s: %d enriched alpha events", p.name, len(events))
    return events


def build_from_repo(out_dir: Path = None) -> Dict[str, Any]:
    """Sweep known repo datasets → labelled research_events.jsonl (offline; quality-gated)."""
    import glob
    out_dir = Path(out_dir) if out_dir else (rc.RL_BACKFILL_DIR / f"{rc.data_tag()}_repo")
    out_dir.mkdir(parents=True, exist_ok=True)
    events: List[dict] = []
    sources: Dict[str, int] = {}

    # Buffett: cached daily bars (1000-symbol universe) → real forward returns.
    if Path(rc.PROJECT_ROOT / "alpaca_daily_bars.csv").exists():
        ev = ingest_daily_bars_csv(rc.PROJECT_ROOT / "alpaca_daily_bars.csv")
        sources["alpaca_daily_bars.csv(buffett)"] = len(ev); events += ev

    # Alpha: trade-replay datasets (root + system-data dip-reclaim replays).
    replay_files = [str(rc.PROJECT_ROOT / "rl_trade_replay.csv")]
    replay_files += sorted(glob.glob(str(rc.PROJECT_ROOT / "system-data" / "*" / "*_dip_reclaim_replay_dataset.csv")))
    for rf in replay_files:
        if Path(rf).exists():
            ev = ingest_trade_replay_csv(rf)
            sources[f"{Path(rf).name}(alpha)"] = len(ev); events += ev

    # Alpha (ENRICHED): real 1-minute bar files → genuine MTF/rel-vol/VWAP/ATR/catalyst.
    bar_files = sorted(glob.glob(str(rc.PROJECT_ROOT / "system-data" / "*" / "*_intraday_bars_1m.csv")))
    max_bar_files = int(os.getenv("RL_MAX_INTRADAY_BAR_FILES", "3"))
    if len(bar_files) > max_bar_files:
        logger.info("intraday-1m: %d bar files found, using newest %d (set RL_MAX_INTRADAY_BAR_FILES to change)",
                    len(bar_files), max_bar_files)
    for bf_path in bar_files[-max_bar_files:]:
        ev = ingest_intraday_bars_1m_csv(bf_path)
        sources[f"{Path(bf_path).name}(alpha_enriched)"] = len(ev); events += ev

    # Regime enrichment (SPY + sector ETF, daily-trend slow) for events with real symbols.
    # Read-only Alpaca daily bars, cached; degrades to neutral when unavailable.
    try:
        import rl_alpaca_bar_enrichment as enr
        rcounts = enr.enrich_events(events, add_regime=os.getenv("RL_ENRICH_REGIME", "1") != "0")
        sources["regime_enrichment"] = rcounts
    except Exception as exc:
        logger.warning("regime enrichment skipped (%s)", exc)

    path = out_dir / "research_events.jsonl"
    try:
        with open(path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, default=str) + "\n")
    except Exception as exc:
        logger.warning("repo backfill write failed: %s", exc)
    summary = {"generated_at_utc": rc.now_iso(), "date_tag": rc.data_tag(), "output": str(path),
               "events": len(events),
               "alpha_entry": sum(1 for e in events if e["event_type"] == "alpha_entry"),
               "alpha_exit": sum(1 for e in events if e["event_type"] == "alpha_exit"),
               "buffett": sum(1 for e in events if e["event_type"] == "bal_buy"),
               "sources": sources, "offline_only": True, "orders_submitted": 0,
               "quality_gated": True}
    rc.write_json(out_dir / "repo_backfill_summary.json", summary)
    return summary


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if "--repo" in sys.argv:
        s = build_from_repo()
        print("Repo dataset ingestion complete (offline, quality-gated):")
        print(f"  events={s['events']} alpha_entry={s['alpha_entry']} alpha_exit={s['alpha_exit']} "
              f"buffett={s['buffett']}")
        for src, n in s["sources"].items():
            print(f"    {src}: {n} events")
        print(f"  -> {s['output']}")
    else:
        s = build_backfill()
        print("Alpaca backfill/replay complete (offline, read-only):")
        print(f"  events={s['events']} alpha_entry={s['alpha_entry']} buffett={s['buffett']} "
              f"(alpha syms={s['alpha_symbols_fetched']}, buffett syms={s['buffett_symbols_fetched']})")
        print(f"  -> {s['output']}")
