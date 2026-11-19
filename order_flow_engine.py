#!/usr/bin/env python3
"""
order_flow_engine.py -- real-time order-flow metrics from Alpaca IEX trades and quotes.

PURPOSE
    Detect buyer/seller aggression, momentum shifts, and exhaustion to INFORM EXITS
    (see profit_exit_engine.py). This module only observes and classifies; it never
    places, modifies, or cancels an order.

HONESTY / DATA SCOPE (important -- do not misrepresent)
    * Alpaca's free tier is the IEX feed: a SINGLE exchange, not the consolidated SIP
      tape. IEX is a small, variable share of total US volume. Every metric here is
      therefore an IEX-ONLY proxy for market-wide order flow, labelled data_source="IEX".
    * Quotes are LEVEL 1 (best bid/ask + their sizes), NOT full order-book depth. Quote
      imbalance here is an L1 signal only.
    These are deliberately surfaced in the snapshot (`data_source`, `caveat`) so no
    downstream consumer treats IEX executions as consolidated activity or L1 as depth.

WHAT IT COMPUTES (per symbol)
    * Aggressive buy/sell volume via timestamp-aligned quote classification (Lee-Ready).
    * Rolling signed volume delta over 10 / 30 / 60 / 300 s.
    * Cumulative Volume Delta (CVD) and CVD-vs-price divergence.
    * Executed order-flow imbalance, buy/sell ratio.
    * Level-1 quote imbalance, spread (abs + bps) and a liquidity classification.
    * Buying/selling pressure classification and interpretable [0,1] signals
      (sell_pressure, buy_pressure, cvd_deterioration, buyer_exhaustion) for the exit engine.

ROBUSTNESS
    Stale-data detection, duplicate protection, out-of-order tolerance, bounded memory
    (time-trimmed deques). The REST/stream adapters are fail-safe: any error yields an
    INSUFFICIENT_DATA snapshot rather than raising into the trading loop.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DATA_SOURCE = "IEX"
_CAVEAT = ("IEX-only single-exchange tape (not consolidated SIP); quotes are Level 1 "
           "best bid/ask, not full depth. Treat as a proxy, not market-wide volume.")


# ======================================================================
# Config
# ======================================================================
@dataclass
class OrderFlowConfig:
    windows_s: Tuple[int, ...] = (10, 30, 60, 300)
    cvd_window_s: int = 300                 # series retained for divergence/peak tracking
    divergence_lookback_s: int = 120        # window for price-vs-CVD divergence
    max_trades: int = 20000                 # hard memory bound per symbol
    max_quotes: int = 20000
    retain_s: int = 360                     # trim events older than this (>= max window)
    stale_trade_s: float = 15.0             # no trade within this -> stale
    stale_quote_s: float = 15.0
    min_trades_for_signal: int = 8          # below this -> INSUFFICIENT_DATA
    wide_spread_bps: float = 25.0           # above this -> illiquid/caution
    out_of_order_tolerance_s: float = 2.0   # accept slightly-late events up to this lag


# ======================================================================
# Snapshot
# ======================================================================
@dataclass
class OrderFlowSnapshot:
    symbol: str
    ts: str                                  # ISO time of snapshot
    data_source: str = DATA_SOURCE
    caveat: str = _CAVEAT
    ok: bool = False                         # enough data to be meaningful
    status: str = "INSUFFICIENT_DATA"        # OK | STALE | INSUFFICIENT_DATA
    is_stale: bool = True
    last_trade_age_s: Optional[float] = None
    last_quote_age_s: Optional[float] = None
    n_trades_60s: int = 0

    # raw microstructure
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread: Optional[float] = None
    spread_bps: Optional[float] = None
    liquidity: str = "unknown"               # tight | normal | wide | unknown

    buy_vol_60s: float = 0.0
    sell_vol_60s: float = 0.0
    delta_10s: float = 0.0
    delta_30s: float = 0.0
    delta_60s: float = 0.0
    delta_300s: float = 0.0
    cvd: float = 0.0
    cvd_peak: float = 0.0
    executed_imbalance: float = 0.0          # [-1,1] (buy-sell)/(buy+sell) over 60s
    buy_sell_ratio: float = 1.0
    quote_imbalance: float = 0.0             # [-1,1] (bidsz-asksz)/(bidsz+asksz) L1
    cvd_divergence: float = 0.0              # <0 bearish (price up, CVD down)
    divergence_flag: str = "none"            # bearish | bullish | none

    pressure: str = "NEUTRAL"                # STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL
    pressure_score: float = 0.0              # [-1,1] (+ buy, - sell)

    # interpretable [0,1] signals consumed by the exit engine
    sell_pressure: float = 0.0
    buy_pressure: float = 0.0
    cvd_deterioration: float = 0.0
    buyer_exhaustion: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# ======================================================================
# Pure classification / metric functions (unit-tested, no state)
# ======================================================================
def classify_trade(price: float, bid: Optional[float], ask: Optional[float],
                   last_tick_sign: int = 0) -> int:
    """Lee-Ready trade sign: +1 aggressive buy, -1 aggressive sell, 0 unknown.

    Quote rule first (lifted offer / hit bid), then midpoint, then tick fallback.
    """
    if price <= 0:
        return 0
    if bid is not None and ask is not None and ask > 0 and bid > 0 and ask >= bid:
        if price >= ask:
            return 1
        if price <= bid:
            return -1
        mid = 0.5 * (bid + ask)
        if price > mid:
            return 1
        if price < mid:
            return -1
        return last_tick_sign if last_tick_sign in (1, -1) else 0
    # No usable quote -> fall back to tick direction.
    return last_tick_sign if last_tick_sign in (1, -1) else 0


def signed_delta(events: Sequence[Tuple[float, int, float]], now: float, window_s: float) -> float:
    """Sum signed volume (sign*size) within [now-window_s, now]. events: (ts, sign, size)."""
    lo = now - window_s
    return float(sum(sign * size for ts, sign, size in events if lo <= ts <= now))


def buy_sell_volumes(events: Sequence[Tuple[float, int, float]], now: float,
                     window_s: float) -> Tuple[float, float]:
    lo = now - window_s
    buy = sell = 0.0
    for ts, sign, size in events:
        if ts < lo or ts > now:
            continue
        if sign > 0:
            buy += size
        elif sign < 0:
            sell += size
    return buy, sell


def executed_imbalance(buy_vol: float, sell_vol: float) -> float:
    tot = buy_vol + sell_vol
    if tot <= 0:
        return 0.0
    return max(-1.0, min(1.0, (buy_vol - sell_vol) / tot))


def buy_sell_ratio(buy_vol: float, sell_vol: float) -> float:
    if sell_vol <= 0:
        return float("inf") if buy_vol > 0 else 1.0
    return buy_vol / sell_vol


def quote_imbalance(bid_size: float, ask_size: float) -> float:
    tot = bid_size + ask_size
    if tot <= 0:
        return 0.0
    return max(-1.0, min(1.0, (bid_size - ask_size) / tot))


def spread_metrics(bid: float, ask: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mid, spread, spread_bps) or (None,None,None) if invalid."""
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None, None, None
    mid = 0.5 * (bid + ask)
    spread = ask - bid
    spread_bps = (spread / mid) * 1e4 if mid > 0 else None
    return mid, spread, spread_bps


def cvd_divergence(price_series: Sequence[Tuple[float, float]],
                   cvd_series: Sequence[Tuple[float, float]], now: float,
                   lookback_s: float) -> float:
    """Signed divergence over the lookback window, in [-1, 1].

    Compares the direction of price vs the direction of CVD. Negative = bearish
    divergence (price up while CVD flat/down -> distribution / buyer exhaustion);
    positive = bullish divergence. 0 when aligned or insufficient data.
    """
    lo = now - lookback_s
    ps = [(t, v) for t, v in price_series if t >= lo]
    cs = [(t, v) for t, v in cvd_series if t >= lo]
    if len(ps) < 3 or len(cs) < 3:
        return 0.0
    p0, p1 = ps[0][1], ps[-1][1]
    c0, c1 = cs[0][1], cs[-1][1]
    if p0 <= 0:
        return 0.0
    price_chg = (p1 - p0) / p0
    cvd_range = max(abs(c1 - c0), 1.0)
    cvd_dir = (c1 - c0) / cvd_range  # normalized direction of CVD move
    # Divergence present only when the two disagree in sign.
    price_dir = 1.0 if price_chg > 0 else (-1.0 if price_chg < 0 else 0.0)
    if price_dir == 0.0:
        return 0.0
    if (price_dir > 0 and cvd_dir < 0) or (price_dir < 0 and cvd_dir > 0):
        # magnitude scaled by how strongly price moved and CVD contradicted
        mag = min(1.0, abs(price_chg) * 50.0) * min(1.0, abs(cvd_dir))
        return max(-1.0, min(1.0, -price_dir * mag))  # bearish when price up
    return 0.0


def classify_pressure(executed_imb: float, delta_60s: float, q_imb: float) -> Tuple[str, float]:
    """Blend executed imbalance, short delta sign and L1 quote imbalance into a
    pressure label + score in [-1,1] (+ buy, - sell)."""
    delta_sign = 1.0 if delta_60s > 0 else (-1.0 if delta_60s < 0 else 0.0)
    score = 0.55 * executed_imb + 0.30 * (0.5 * q_imb) + 0.15 * delta_sign
    score = max(-1.0, min(1.0, score))
    if score >= 0.5:
        return "STRONG_BUY", score
    if score >= 0.15:
        return "BUY", score
    if score <= -0.5:
        return "STRONG_SELL", score
    if score <= -0.15:
        return "SELL", score
    return "NEUTRAL", score


def norm_sell_pressure(executed_imb: float, q_imb: float, delta_60s: float, buy_vol: float,
                       sell_vol: float) -> float:
    """[0,1]: 0 = all buying, 1 = heavy selling. Blends executed + quote imbalance."""
    sell_share = 0.0
    tot = buy_vol + sell_vol
    if tot > 0:
        sell_share = sell_vol / tot
    s = 0.6 * sell_share + 0.25 * (0.5 - 0.5 * q_imb) + 0.15 * (1.0 if delta_60s < 0 else 0.0)
    return max(0.0, min(1.0, s))


def norm_cvd_deterioration(cvd: float, cvd_peak: float, delta_30s: float, delta_300s: float) -> float:
    """[0,1]: how far CVD has given back from its peak + near-term delta turning negative."""
    giveback = 0.0
    denom = max(abs(cvd_peak), 1.0)
    if cvd_peak > 0 and cvd < cvd_peak:
        giveback = min(1.0, (cvd_peak - cvd) / denom)
    turn = 0.0
    if delta_300s > 0 and delta_30s < 0:          # recent flow flipped vs the trend
        turn = min(1.0, abs(delta_30s) / max(abs(delta_300s), 1.0))
    elif delta_30s < 0:
        turn = 0.5
    return max(0.0, min(1.0, 0.65 * giveback + 0.35 * turn))


def norm_buyer_exhaustion(cvd_div: float, delta_10s: float, delta_60s: float,
                          buy_sell_r: float) -> float:
    """[0,1]: price pushing but buying is drying up. Combines bearish CVD divergence,
    short-window buy deceleration, and a falling buy/sell ratio."""
    diverge = max(0.0, -cvd_div)  # only bearish divergence contributes
    # deceleration: 10s delta much weaker than the 60s average rate
    decel = 0.0
    avg60 = delta_60s / 6.0  # per-10s average over the 60s window
    if avg60 > 0:
        decel = max(0.0, min(1.0, 1.0 - (delta_10s / avg60))) if delta_10s < avg60 else 0.0
    ratio_weak = 0.0
    if math.isfinite(buy_sell_r):
        if buy_sell_r < 1.0:
            ratio_weak = min(1.0, 1.0 - buy_sell_r)
    return max(0.0, min(1.0, 0.5 * diverge + 0.3 * decel + 0.2 * ratio_weak))


def classify_liquidity(spread_bps: Optional[float], wide_bps: float) -> str:
    if spread_bps is None:
        return "unknown"
    if spread_bps <= wide_bps * 0.4:
        return "tight"
    if spread_bps <= wide_bps:
        return "normal"
    return "wide"


# ======================================================================
# Per-symbol stateful accumulator
# ======================================================================
class SymbolOrderFlow:
    """Accumulates trades/quotes for one symbol and produces snapshots.

    Events are stored as (epoch_seconds, ...) and time-trimmed to the configured
    retention. Duplicate and badly out-of-order events are rejected.
    """

    def __init__(self, symbol: str, config: Optional[OrderFlowConfig] = None):
        self.symbol = symbol.upper()
        self.cfg = config or OrderFlowConfig()
        # signed trades: (ts, sign, size)
        self._trades: Deque[Tuple[float, int, float]] = deque(maxlen=self.cfg.max_trades)
        # cvd series: (ts, cvd_value)
        self._cvd: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.max_trades)
        # price series (last trade price): (ts, price)
        self._price: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.max_trades)
        self._cvd_running = 0.0
        self._cvd_peak = 0.0
        self._last_tick_sign = 0
        self._last_trade_ts: Optional[float] = None
        self._last_quote_ts: Optional[float] = None
        self._last_price: Optional[float] = None
        self._seen_trade_keys: Deque[Tuple[float, float, float]] = deque(maxlen=512)
        # latest quote
        self._bid: Optional[float] = None
        self._ask: Optional[float] = None
        self._bid_size: float = 0.0
        self._ask_size: float = 0.0

    # ---- ingestion ----
    def on_quote(self, ts: float, bid: float, ask: float, bid_size: float, ask_size: float) -> None:
        if self._last_quote_ts is not None and ts < self._last_quote_ts - self.cfg.out_of_order_tolerance_s:
            return  # badly out of order
        self._last_quote_ts = max(ts, self._last_quote_ts or ts)
        if bid and ask and bid > 0 and ask > 0:
            self._bid, self._ask = float(bid), float(ask)
            self._bid_size, self._ask_size = float(bid_size or 0.0), float(ask_size or 0.0)

    def on_trade(self, ts: float, price: float, size: float, trade_id: Any = None) -> None:
        if price <= 0 or size <= 0:
            return
        if self._last_trade_ts is not None and ts < self._last_trade_ts - self.cfg.out_of_order_tolerance_s:
            return  # badly out of order
        key = (round(ts, 4), round(float(price), 6), round(float(size), 6))
        if key in self._seen_trade_keys:
            return  # duplicate
        self._seen_trade_keys.append(key)

        sign = classify_trade(price, self._bid, self._ask, self._last_tick_sign)
        if sign == 0 and self._last_price is not None:
            sign = 1 if price > self._last_price else (-1 if price < self._last_price else self._last_tick_sign)
        if sign in (1, -1):
            self._last_tick_sign = sign

        self._trades.append((ts, sign, float(size)))
        self._cvd_running += sign * float(size)
        self._cvd_peak = max(self._cvd_peak, self._cvd_running)
        self._cvd.append((ts, self._cvd_running))
        self._price.append((ts, float(price)))
        self._last_trade_ts = max(ts, self._last_trade_ts or ts)
        self._last_price = float(price)
        self._trim(ts)

    def _trim(self, now: float) -> None:
        lo = now - self.cfg.retain_s
        for dq in (self._trades, self._cvd, self._price):
            while dq and dq[0][0] < lo:
                dq.popleft()

    # ---- snapshot ----
    def snapshot(self, now: Optional[float] = None) -> OrderFlowSnapshot:
        now = now if now is not None else datetime.now(timezone.utc).timestamp()
        snap = OrderFlowSnapshot(symbol=self.symbol, ts=datetime.fromtimestamp(now, timezone.utc).isoformat())
        snap.last_trade_age_s = (now - self._last_trade_ts) if self._last_trade_ts else None
        snap.last_quote_age_s = (now - self._last_quote_ts) if self._last_quote_ts else None

        # quote-derived
        if self._bid and self._ask:
            snap.bid, snap.ask = self._bid, self._ask
            snap.mid, snap.spread, snap.spread_bps = spread_metrics(self._bid, self._ask)
            snap.liquidity = classify_liquidity(snap.spread_bps, self.cfg.wide_spread_bps)
            snap.quote_imbalance = quote_imbalance(self._bid_size, self._ask_size)

        trades = list(self._trades)
        snap.n_trades_60s = sum(1 for t, _s, _sz in trades if t >= now - 60)

        # staleness / sufficiency gates
        stale = ((snap.last_trade_age_s is None or snap.last_trade_age_s > self.cfg.stale_trade_s))
        if len(trades) < self.cfg.min_trades_for_signal:
            snap.status, snap.is_stale, snap.ok = "INSUFFICIENT_DATA", True, False
            return snap
        if stale:
            snap.status, snap.is_stale, snap.ok = "STALE", True, False
            # still fill metrics below for transparency, but ok stays False

        snap.delta_10s = signed_delta(trades, now, 10)
        snap.delta_30s = signed_delta(trades, now, 30)
        snap.delta_60s = signed_delta(trades, now, 60)
        snap.delta_300s = signed_delta(trades, now, 300)
        snap.buy_vol_60s, snap.sell_vol_60s = buy_sell_volumes(trades, now, 60)
        snap.executed_imbalance = executed_imbalance(snap.buy_vol_60s, snap.sell_vol_60s)
        snap.buy_sell_ratio = buy_sell_ratio(snap.buy_vol_60s, snap.sell_vol_60s)
        snap.cvd = self._cvd_running
        snap.cvd_peak = self._cvd_peak
        snap.cvd_divergence = cvd_divergence(list(self._price), list(self._cvd), now,
                                             self.cfg.divergence_lookback_s)
        snap.divergence_flag = ("bearish" if snap.cvd_divergence < -0.05
                                else "bullish" if snap.cvd_divergence > 0.05 else "none")
        snap.pressure, snap.pressure_score = classify_pressure(
            snap.executed_imbalance, snap.delta_60s, snap.quote_imbalance)

        snap.sell_pressure = norm_sell_pressure(snap.executed_imbalance, snap.quote_imbalance,
                                                snap.delta_60s, snap.buy_vol_60s, snap.sell_vol_60s)
        snap.buy_pressure = max(0.0, min(1.0, 0.5 + 0.5 * snap.pressure_score))
        snap.cvd_deterioration = norm_cvd_deterioration(snap.cvd, snap.cvd_peak,
                                                        snap.delta_30s, snap.delta_300s)
        snap.buyer_exhaustion = norm_buyer_exhaustion(snap.cvd_divergence, snap.delta_10s,
                                                     snap.delta_60s, snap.buy_sell_ratio)
        if not stale:
            snap.status, snap.is_stale, snap.ok = "OK", False, True
        return snap


# ======================================================================
# Multi-symbol engine + Alpaca REST adapter (fail-safe)
# ======================================================================
class OrderFlowEngine:
    """Manages per-symbol order flow and (optionally) polls Alpaca IEX trades/quotes.

    The REST poller is fail-safe: any error logs and yields INSUFFICIENT_DATA snapshots
    so it can never raise into the trading loop. Pass data_client=StockHistoricalDataClient.
    """

    def __init__(self, config: Optional[OrderFlowConfig] = None, data_client: Any = None):
        self.cfg = config or OrderFlowConfig()
        self.data_client = data_client
        self._symbols: dict[str, SymbolOrderFlow] = {}

    def _sym(self, symbol: str) -> SymbolOrderFlow:
        s = symbol.upper()
        if s not in self._symbols:
            self._symbols[s] = SymbolOrderFlow(s, self.cfg)
        return self._symbols[s]

    # ---- direct ingestion (for stream handlers or tests) ----
    def ingest_trade(self, symbol: str, ts: float, price: float, size: float, trade_id: Any = None) -> None:
        self._sym(symbol).on_trade(ts, price, size, trade_id)

    def ingest_quote(self, symbol: str, ts: float, bid: float, ask: float,
                     bid_size: float, ask_size: float) -> None:
        self._sym(symbol).on_quote(ts, bid, ask, bid_size, ask_size)

    def snapshot(self, symbol: str, now: Optional[float] = None) -> OrderFlowSnapshot:
        return self._sym(symbol).snapshot(now)

    # ---- Alpaca REST poll (IEX) ----
    def poll_rest(self, symbols: Sequence[str], lookback_s: int = 300) -> dict[str, OrderFlowSnapshot]:
        """Fetch recent IEX trades + latest quotes for `symbols` and ingest them.

        Returns {symbol: snapshot}. Fail-safe: on any error a symbol gets whatever
        snapshot its accumulator currently holds (possibly INSUFFICIENT_DATA).
        """
        if self.data_client is not None:
            try:
                self._poll_trades(symbols, lookback_s)
            except Exception as exc:  # noqa: BLE001 - never raise into the loop
                logger.warning("order_flow poll_trades failed: %s", exc)
            try:
                self._poll_quotes(symbols)
            except Exception as exc:  # noqa: BLE001
                logger.warning("order_flow poll_quotes failed: %s", exc)
        return {s.upper(): self.snapshot(s) for s in symbols}

    def _poll_trades(self, symbols: Sequence[str], lookback_s: int) -> None:
        from alpaca.data.requests import StockTradesRequest
        from alpaca.data.timeframe import TimeFrame  # noqa: F401 (import parity)
        try:
            from alpaca.data.enums import DataFeed
            feed = DataFeed.IEX
        except Exception:
            feed = None
        start = datetime.now(timezone.utc) - timedelta(seconds=lookback_s)
        kwargs: dict[str, Any] = {"symbol_or_symbols": list(symbols), "start": start}
        if feed is not None:
            kwargs["feed"] = feed
        req = StockTradesRequest(**kwargs)
        resp = self.data_client.get_stock_trades(req)
        data = getattr(resp, "data", resp) or {}
        for sym in symbols:
            rows = data.get(sym.upper()) or data.get(sym) or []
            for t in rows:
                ts = _epoch(getattr(t, "timestamp", None))
                price = float(getattr(t, "price", 0.0) or 0.0)
                size = float(getattr(t, "size", 0.0) or 0.0)
                tid = getattr(t, "id", None)
                if ts is not None:
                    self._sym(sym).on_trade(ts, price, size, tid)

    def _poll_quotes(self, symbols: Sequence[str]) -> None:
        from alpaca.data.requests import StockLatestQuoteRequest
        try:
            from alpaca.data.enums import DataFeed
            feed = DataFeed.IEX
        except Exception:
            feed = None
        kwargs: dict[str, Any] = {"symbol_or_symbols": list(symbols)}
        if feed is not None:
            kwargs["feed"] = feed
        req = StockLatestQuoteRequest(**kwargs)
        quotes = self.data_client.get_stock_latest_quote(req) or {}
        for sym in symbols:
            q = quotes.get(sym.upper()) or quotes.get(sym)
            if q is None:
                continue
            ts = _epoch(getattr(q, "timestamp", None)) or datetime.now(timezone.utc).timestamp()
            self._sym(sym).on_quote(
                ts,
                float(getattr(q, "bid_price", 0.0) or 0.0),
                float(getattr(q, "ask_price", 0.0) or 0.0),
                float(getattr(q, "bid_size", 0.0) or 0.0),
                float(getattr(q, "ask_size", 0.0) or 0.0),
            )


def _epoch(ts: Any) -> Optional[float]:
    """Coerce an Alpaca timestamp (datetime / pandas Timestamp / ISO str / epoch) to seconds."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        if hasattr(ts, "timestamp"):
            return float(ts.timestamp())
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None
