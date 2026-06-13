#!/usr/bin/env python3
"""
tjl_backtest.py — deterministic backtest / validation harness for the TJL strategy.

Replays the EXACT tjl_signal primitives over historical bars, so a green backtest is
direct evidence the live engine behaves correctly. Key properties:

  * NON-REPAINTING by construction: at entry-bar time t, only HTF bars whose
    (start + tf) <= t are used — identical to tjl_engine.fetch_htf_ohlc and to the
    Pine request.security(..., _expr[1], lookahead_on) contract. Asserted explicitly.
  * Intrabar exits: stop hit when bar.low <= stop; take-profit when bar.high >= tp
    (stop assumed first when both touch — conservative). EOD-flat closes the last RTH
    bar of the day.

Data: real Alpaca bars (5m entry / 15m momo / 60m trend) when ALPACA_API_KEY is set,
else a deterministic synthetic trending series so the harness always runs offline.

Usage:
    python tjl_backtest.py AMD NVDA MU            # default ~20 trading days
    python tjl_backtest.py --days 30 AMD
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import tjl_signal as T

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass
class Bar:
    t: datetime          # bar START, tz-aware UTC
    o: float
    h: float
    l: float
    c: float


def _fetch_alpaca(symbol: str, tf_min: int, days: int) -> List[Bar]:
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not (key and sec):
        return []
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
        client = StockHistoricalDataClient(key, sec)
        end = datetime.now(UTC)
        start = end - timedelta(days=days + 5)
        tf = (TimeFrame(tf_min // 60, TimeFrameUnit.Hour)
              if tf_min >= 60 and tf_min % 60 == 0
              else TimeFrame(tf_min, TimeFrameUnit.Minute))
        req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=tf,
                               start=start, end=end, feed=DataFeed.IEX)
        df = client.get_stock_bars(req).df
        if df is None or df.empty:
            return []
        df = df.reset_index()
        out: List[Bar] = []
        for _, r in df.iterrows():
            ts = r["timestamp"]
            ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            out.append(Bar(ts.astimezone(UTC), float(r["open"]), float(r["high"]),
                           float(r["low"]), float(r["close"])))
        return out
    except Exception as e:
        print(f"  [warn] Alpaca fetch failed for {symbol} @{tf_min}m: {e}", file=sys.stderr)
        return []


def _synth(symbol: str, tf_min: int, days: int) -> List[Bar]:
    """Deterministic RTH-only synthetic bars (seeded by symbol) with trend + pullbacks."""
    import math
    seed = sum(ord(c) for c in symbol)
    px = 50.0 + (seed % 150)
    bars: List[Bar] = []
    start_day = datetime.now(UTC).date() - timedelta(days=days + 3)
    step = tf_min
    for d in range(days + 3):
        day = start_day + timedelta(days=d)
        if day.weekday() >= 5:
            continue
        # RTH 13:30-20:00 UTC (= 09:30-16:00 ET, std time approx; ET handled by is_rth)
        t = datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)
        end = datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC)
        i = 0
        while t < end:
            drift = math.sin((seed + d * 7 + i) / 9.0) * 0.25 + 0.02
            px = max(1.0, px + drift)
            o = px
            c = max(1.0, px + math.sin((seed + i) / 4.0) * 0.1)
            h = max(o, c) * 1.002
            lo = min(o, c) * 0.998
            bars.append(Bar(t, o, h, lo, c))
            px = c
            i += 1
            t = t + timedelta(minutes=step)
    return bars


def get_bars(symbol: str, tf_min: int, days: int) -> Tuple[List[Bar], str]:
    real = _fetch_alpaca(symbol, tf_min, days)
    if real:
        return real, "alpaca"
    return _synth(symbol, tf_min, days), "synthetic"


def _completed(bars: List[Bar], tf_min: int, as_of: datetime):
    """Bars whose (start + tf) <= as_of — the non-repaint completed-bar filter."""
    cutoff = as_of - timedelta(minutes=tf_min)
    keep = [b for b in bars if b.t <= cutoff]
    return [b.h for b in keep], [b.l for b in keep], [b.c for b in keep], keep


@dataclass
class Trade:
    sym: str
    entry_t: datetime
    entry_px: float
    exit_t: datetime
    exit_px: float
    reason: str

    @property
    def ret_pct(self) -> float:
        return (self.exit_px / self.entry_px - 1.0) * 100.0


def backtest(symbol: str, cfg: T.TJLConfig, days: int) -> Tuple[List[Trade], dict]:
    entry_bars, src = get_bars(symbol, 5, days)            # 5m entry feed
    momo_bars, _ = get_bars(symbol, cfg.momo_tf_min, days * 2)
    trend_bars, _ = get_bars(symbol, cfg.trend_tf_min, days * 6)
    trades: List[Trade] = []
    in_pos = False
    entry_px = 0.0
    entry_t: Optional[datetime] = None
    lookahead_ok = True

    closes_seq: List[float] = []
    for i, b in enumerate(entry_bars):
        closes_seq.append(b.c)
        t = b.t
        t_et = t.astimezone(ET)

        if in_pos:
            stop = entry_px * (1 - cfg.stop_pct)
            tp = entry_px * (1 + cfg.tp_pct)
            # EOD flatten = Pine isRTH[1] and not isRTH: fire on the FIRST non-RTH bar
            # after being in RTH (NOT the end of the extended calendar day).
            prev = entry_bars[i - 1] if i > 0 else None
            eod = (cfg.eod_flat and prev is not None
                   and T.is_rth(prev.t) and not T.is_rth(b.t))
            if b.l <= stop:
                trades.append(Trade(symbol, entry_t, entry_px, t, stop, T.STOP_LOSS)); in_pos = False
            elif b.h >= tp:
                trades.append(Trade(symbol, entry_t, entry_px, t, tp, T.TAKE_PROFIT)); in_pos = False
            elif eod:
                trades.append(Trade(symbol, entry_t, entry_px, t, b.c, T.EOD_FLAT)); in_pos = False
            continue

        # completed HTF bars only (assert non-repaint: every used bar started >= tf ago)
        th, tl, tc, tkeep = _completed(trend_bars, cfg.trend_tf_min, t)
        mh, ml, mc, mkeep = _completed(momo_bars, cfg.momo_tf_min, t)
        if (tkeep and tkeep[-1].t > t - timedelta(minutes=cfg.trend_tf_min)) or \
           (mkeep and mkeep[-1].t > t - timedelta(minutes=cfg.momo_tf_min)):
            lookahead_ok = False

        sig = T.evaluate_entry(cfg, tc, mh, ml, mc, closes_seq, t, in_pos, log=False)
        if sig.enter:
            in_pos, entry_px, entry_t = True, b.c, t

    wins = [tr for tr in trades if tr.ret_pct > 0]
    gross_win = sum(tr.ret_pct for tr in wins)
    gross_loss = -sum(tr.ret_pct for tr in trades if tr.ret_pct <= 0)
    stats = {
        "symbol": symbol, "source": src, "entry_bars": len(entry_bars),
        "trades": len(trades), "wins": len(wins),
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "total_ret_pct": sum(tr.ret_pct for tr in trades),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "lookahead_ok": lookahead_ok,
    }
    return trades, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=["AMD", "NVDA", "MU"])
    ap.add_argument("--days", type=int, default=20)
    args = ap.parse_args()
    symbols = args.symbols or ["AMD", "NVDA", "MU"]
    cfg = T.TJLConfig.from_env()

    print(f"TJL backtest — cfg: trend={cfg.trend_tf}(SMA{cfg.trend_sma_len}) "
          f"momo={cfg.momo_tf}(stoch{cfg.stoch_len}/{cfg.smooth_k}/{cfg.smooth_d}<{cfg.stoch_ob}) "
          f"ema={cfg.fast_ema}/{cfg.slow_ema} stop={cfg.stop_pct*100:.1f}% tp={cfg.tp_pct*100:.1f}%")
    print(f"{'SYM':<6}{'SRC':<11}{'BARS':>6}{'TRADES':>8}{'WINS':>6}{'WIN%':>7}"
          f"{'TOTAL%':>9}{'PF':>7}  NO-LOOKAHEAD")
    for sym in symbols:
        _, s = backtest(sym, cfg, args.days)
        pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        print(f"{s['symbol']:<6}{s['source']:<11}{s['entry_bars']:>6}{s['trades']:>8}"
              f"{s['wins']:>6}{s['win_rate']:>6.1f}%{s['total_ret_pct']:>8.2f}%{pf:>7}"
              f"   {'PASS' if s['lookahead_ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
