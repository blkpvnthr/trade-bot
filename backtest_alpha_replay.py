#!/usr/bin/env python3
"""
backtest_alpha_replay.py — Historical replay backtester for the Alpha Call Engine.
==================================================================================

Drives the REAL `alpha_call_engine.AlphaCallEngine` over REAL historical 1-minute
underlying bars, with options priced and FILLED by the engine's own Black-Scholes
model. This is the "build a real historical replay" path: same decision code run.py
uses (evaluate_universe / check_active_positions / register_filled), minus live I/O.

Why a synthetic-but-self-consistent option chain?
    Alpaca (and most free feeds) cannot serve a *historical option chain as-of a past
    date* with greeks/IV. So at each decision point we synthesize a call chain around
    the underlying's real historical price using the engine's OWN black_scholes_call()
    pricer + an IV estimated from trailing realized volatility. Every contract is
    therefore priced on the same model the engine evaluates with, so:
      • entries always FILL (marketable limit at the modeled ask), and
      • exits are marked each bar by re-pricing the held call via the same BS model.
    This is a MODELED-FILL backtest, not a tick-accurate one (no real NBBO, no L2
    liquidity, no early-assignment). It is faithful to the engine's logic, not to
    microstructure. Stated assumptions are printed in the run summary.

Safety / fidelity:
    • $100,000 starting equity (configurable).
    • The engine's own risk logic runs (risk-per-trade sizing, delta/DTE/OI/spread
      gates, BS valuation gates, 3R/1.7R-trail/+1R-partial/EOD exits).
    • CLAUDE.md risk backstops are preserved as an OUTER gate: daily-loss limit,
      max drawdown, equity floor, max options exposure. When a backstop trips, NEW
      entries stop; exits always continue.
    • Read-only against the broker (historical bars only). No orders are ever sent.
    • Engine state is sandboxed to system-data/<RUN_DATA_DATE> so live state is never
      touched (RUN_DATA_DATE defaults to "backtest").

Outputs (under backtest_outputs/<run_tag>/):
    trades.csv          — every entry/exit fill with prices, R-multiple, PnL
    equity_curve.csv    — per-decision mark-to-market equity
    daily.csv           — per-day realized/unrealized PnL, drawdown, trade count
    summary.json        — headline metrics + assumptions

Usage:
    python backtest_alpha_replay.py \
        --symbols SPY,QQQ,NVDA,AAPL,MSFT,AMD,META,AMZN \
        --start 2026-05-12 --end 2026-06-04 --equity 100000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta, date, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# ---------------------------------------------------------------------------
# Sandbox the engine's state/output tag BEFORE importing it (engine reads
# RUN_DATA_DATE at import time to derive its state path). This guarantees the
# backtest never clobbers live system-data/<today> state.
# ---------------------------------------------------------------------------
os.environ.setdefault("RUN_DATA_DATE", "backtest")
# Engine event log is silenced in pytest; keep it quiet here too to avoid spam.
os.environ.setdefault("ALPHA_EVENTS_DISABLED", "1")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

ALPACA_KEY = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY") or ""
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET") or ""

# ---------------------------------------------------------------------------
# Controllable clock — the engine reads datetime.now(ET) for ORB windows, EOD
# close, and DTE math. We patch alpha_call_engine.datetime so "now" tracks the
# replayed bar time. FakeDatetime subclasses real datetime so strptime/combine/
# fromtimestamp keep working unchanged.
# ---------------------------------------------------------------------------
_real_datetime = datetime


class _Clock:
    current: Optional[datetime] = None  # tz-aware real datetime


_CLOCK = _Clock()


class FakeDatetime(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        c = _CLOCK.current
        if c is None:
            return _real_datetime.now(tz)
        if tz is not None:
            return c.astimezone(tz)
        return c.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        c = _CLOCK.current
        if c is None:
            return _real_datetime.utcnow()
        return c.astimezone(UTC).replace(tzinfo=None)


def set_clock(dt: datetime) -> None:
    _CLOCK.current = dt


# Import engine AFTER env + clock scaffolding is ready.
import alpha_call_engine as ace  # noqa: E402

# Patch the engine module's datetime reference with our controllable clock.
ace.datetime = FakeDatetime
# Some advisory submodules also read the clock; patch where present (best-effort).
for _modname in ("alpha_probability_model", "oops_option_utility", "oops_option_optimizer"):
    try:
        _m = sys.modules.get(_modname) or __import__(_modname)
        if hasattr(_m, "datetime"):
            _m.datetime = FakeDatetime
    except Exception:
        pass

black_scholes_call = ace.black_scholes_call


# ---------------------------------------------------------------------------
# Historical data layer
# ---------------------------------------------------------------------------
class DataLayer:
    """Fetches + caches historical 1m bars and a trailing daily-vol estimate."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._bars_1m: Dict[str, pd.DataFrame] = {}     # symbol -> full-range 1m (ET index)
        self._daily_vol: Dict[str, pd.Series] = {}      # symbol -> date -> annualized vol

    def _client_or_die(self):
        if self._client is None:
            if not (ALPACA_KEY and ALPACA_SECRET):
                raise RuntimeError("Alpaca creds (APCA_API_KEY_ID / APCA_API_SECRET_KEY) required for historical bars")
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        return self._client

    def load_1m(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Full-range 1m bars for `symbol`, ET-indexed, RTH only. Cached to parquet."""
        if symbol in self._bars_1m:
            return self._bars_1m[symbol]
        cache = self.cache_dir / f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_1m.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            client = self._client_or_die()
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.combine(start, dtime(0, 0), UTC),
                end=datetime.combine(end + timedelta(days=1), dtime(0, 0), UTC),
            )
            raw = client.get_stock_bars(req).df
            if raw is None or len(raw) == 0:
                df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            else:
                df = raw.reset_index()
                df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
                df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
            df.to_parquet(cache)
        # Localize to ET and keep RTH (09:30–16:00) so ORB/VWAP windows are correct.
        if len(df):
            if df.index.tz is None:
                df.index = df.index.tz_localize(UTC)
            df.index = df.index.tz_convert(ET)
            df = df.between_time("09:30", "16:00")
        else:
            # No data for this symbol (e.g. OTC ADR with no Alpaca feed). Keep an
            # empty frame with a DatetimeIndex so downstream `.index.date` is safe.
            df = df.iloc[0:0]
            df.index = pd.DatetimeIndex([], tz=ET)
        self._bars_1m[symbol] = df
        return df

    def daily_vol(self, symbol: str, start: date, end: date) -> pd.Series:
        """Annualized close-to-close vol (20d rolling) indexed by session date."""
        if symbol in self._daily_vol:
            return self._daily_vol[symbol]
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        client = self._client_or_die()
        lookback_start = start - timedelta(days=60)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime.combine(lookback_start, dtime(0, 0), UTC),
            end=datetime.combine(end + timedelta(days=1), dtime(0, 0), UTC),
        )
        try:
            raw = client.get_stock_bars(req).df
        except Exception:
            raw = None
        if raw is None or len(raw) == 0:
            s = pd.Series(dtype=float)
        else:
            d = raw.reset_index()
            d = d[d["symbol"] == symbol] if "symbol" in d.columns else d
            d["d"] = pd.to_datetime(d["timestamp"]).dt.tz_convert(ET).dt.date
            d = d.set_index("d")["close"].sort_index()
            rets = d.pct_change()
            s = (rets.rolling(20, min_periods=5).std() * math.sqrt(252)).dropna()
        self._daily_vol[symbol] = s
        return s

    def vol_for(self, symbol: str, on: date, fallback: float = 0.40) -> float:
        s = self._daily_vol.get(symbol)
        if s is None or len(s) == 0:
            return fallback
        prior = s[s.index < on]
        if len(prior) == 0:
            return fallback
        v = float(prior.iloc[-1])
        return min(max(v, 0.08), 1.5) if v > 0 else fallback

    def trading_days(self, ref_symbol: str, start: date, end: date) -> List[date]:
        df = self.load_1m(ref_symbol, start, end)
        if len(df) == 0:
            return []
        days = sorted({ts.date() for ts in df.index})
        return [d for d in days if start <= d <= end]


# ---------------------------------------------------------------------------
# Synthetic option-chain builder (priced by the engine's own BS model)
# ---------------------------------------------------------------------------
def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    # weekday(): Mon=0..Sun=6; Friday=4
    first_friday = 1 + (4 - d.weekday()) % 7
    return date(year, month, first_friday + 14)


def _expiries_for(today: date, min_dte: int, max_dte: int) -> List[date]:
    """Real monthly (3rd-Friday) expiries whose DTE lands in [min_dte, max_dte]."""
    out: List[date] = []
    for k in range(0, 5):
        y = today.year + (today.month - 1 + k) // 12
        m = (today.month - 1 + k) % 12 + 1
        exp = _third_friday(y, m)
        dte = (exp - today).days
        if min_dte <= dte <= max_dte:
            out.append(exp)
    # If monthlies don't land in-band (short windows), fall back to synthetic dates.
    if not out:
        for dte in (min_dte + 7, max_dte - 7):
            out.append(today + timedelta(days=dte))
    return out


# Chains depend only on (symbol, price, date, iv, dte-bounds, rate) — all constant
# across sweep configs — so memoize to avoid rebuilding identical chains per config.
_CHAIN_MEMO: Dict[tuple, List[dict]] = {}


def build_chain(symbol: str, price: float, today: date, iv: float,
                cfg: ace.AlphaCallConfig) -> List[dict]:
    """Synthesize a liquid call chain around `price`, priced by black_scholes_call.

    Strikes span ~3% ITM to ~6% OTM so the engine's delta band (min_delta..max_delta)
    has real choices. bid/ask carry a modeled 2.4% spread (under the 5% gate). OI/vol
    are set liquid so liquidity gates pass; the BS valuation layer still governs.
    """
    memo_key = (symbol, today.toordinal(), int(round(price * 100)), int(round(iv * 1000)),
                cfg.min_dte, cfg.max_dte)
    cached = _CHAIN_MEMO.get(memo_key)
    if cached is not None:
        return cached
    r = float(cfg.risk_free_rate)
    q = float(cfg.dividend_yield)
    chain: List[dict] = []
    half_spread = 0.012  # 1.2% each side -> 2.4% spread
    for exp in _expiries_for(today, cfg.min_dte, cfg.max_dte):
        dte = (exp - today).days
        T = max(dte, 1) / 365.0
        # strikes from 0.97*S to 1.06*S in ~0.5% steps, snapped to a sane increment
        step = _strike_increment(price)
        lo = _round_to(price * 0.97, step)
        hi = _round_to(price * 1.06, step)
        k = lo
        while k <= hi + 1e-9:
            bs = black_scholes_call(price, k, T, r, iv, q)
            mid = float(bs["price"])
            if mid <= 0.02:
                k += step
                continue
            bid = round(mid * (1 - half_spread), 2)
            ask = round(mid * (1 + half_spread), 2)
            occ = _occ_symbol(symbol, exp, k)
            chain.append({
                "symbol": occ,
                "type": "call",
                "expiration_date": exp.strftime("%Y-%m-%d"),
                "strike": float(k),
                "bid_price": max(bid, 0.01),
                "ask_price": max(ask, 0.02),
                "delta": abs(float(bs["delta"])),
                "implied_volatility": float(iv),
                "open_interest": 5000,
                "volume": 750,
            })
            k += step
    _CHAIN_MEMO[memo_key] = chain
    return chain


def _strike_increment(price: float) -> float:
    if price < 25:
        return 0.5
    if price < 200:
        return 1.0
    return 5.0


def _round_to(x: float, step: float) -> float:
    return round(round(x / step) * step, 2)


def _occ_symbol(underlying: str, exp: date, strike: float) -> str:
    mills = int(round(strike * 1000))
    return f"{underlying}{exp:%y%m%d}C{mills:08d}"


def reprice_call(price: float, strike: float, expiry: str, today: date,
                 iv: float, cfg: ace.AlphaCallConfig) -> float:
    """Mid value of a held call at the current underlying price (BS model)."""
    exp = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
    dte = max((exp - today).days, 0)
    if dte <= 0:
        return max(price - strike, 0.0)  # intrinsic at expiry
    T = dte / 365.0
    bs = black_scholes_call(price, strike, T, float(cfg.risk_free_rate), iv, float(cfg.dividend_yield))
    return max(float(bs["price"]), 0.0)


# ---------------------------------------------------------------------------
# Account + CLAUDE.md risk backstops
# ---------------------------------------------------------------------------
@dataclass
class Position:
    contract_symbol: str
    underlying: str
    strike: float
    expiry: str
    iv: float
    contracts: int
    entry_mid: float          # per-contract premium paid (engine 1R basis = entry_mid*100)
    entry_cost: float         # total $ paid = contracts*entry_mid*100
    opened: str
    entry_fee_pc: float = 0.0  # entry slippage+commission attributed per contract


@dataclass
class Account:
    start_equity: float
    cash: float
    realized_pnl: float = 0.0
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)  # keyed by contract_symbol
    halted_today: bool = False
    halt_reason: str = ""

    # backstop pct (mirror run.py / CLAUDE.md defaults)
    daily_loss_pct: float = float(os.getenv("DYNAMIC_DAILY_LOSS_PCT", "0.005"))
    daily_loss_min: float = float(os.getenv("DYNAMIC_DAILY_LOSS_MIN", "500"))
    max_drawdown_pct: float = float(os.getenv("DYNAMIC_MAX_DRAWDOWN_PCT", "0.10"))
    equity_floor_pct: float = float(os.getenv("DYNAMIC_EQUITY_FLOOR_PCT", "0.90"))
    max_options_exposure_pct: float = float(os.getenv("MAX_OPTIONS_EXPOSURE_PCT", "0.15"))

    def __post_init__(self):
        self.day_start_equity = self.start_equity
        self.peak_equity = self.start_equity

    def options_exposure_cost(self) -> float:
        return sum(p.entry_cost for p in self.positions.values())

    def mark_to_market(self, mark: Dict[str, float]) -> float:
        """Equity = cash + sum(open positions @ current mid). `mark` keyed by contract."""
        mv = 0.0
        for c, p in self.positions.items():
            cur = mark.get(c, p.entry_mid)
            mv += p.contracts * cur * 100.0
        return self.cash + mv

    def check_backstops(self, equity: float) -> Optional[str]:
        """Return a halt reason if a NEW-ENTRY backstop has tripped (exits still run)."""
        floor = max(self.start_equity * self.equity_floor_pct, 1000.0)
        if equity < floor:
            return f"equity_floor:{equity:.0f}<{floor:.0f}"
        dd = (self.peak_equity - equity) / max(self.peak_equity, 1.0)
        if dd > self.max_drawdown_pct:
            return f"max_drawdown:{dd*100:.1f}%>{self.max_drawdown_pct*100:.0f}%"
        day_loss = self.day_start_equity - equity
        day_limit = max(self.day_start_equity * self.daily_loss_pct, self.daily_loss_min)
        if day_loss > day_limit:
            return f"daily_loss:{day_loss:.0f}>{day_limit:.0f}"
        return None


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
class Backtest:
    def __init__(self, args, shared_data: Optional[DataLayer] = None):
        self.args = args
        self.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        self.start = datetime.strptime(args.start, "%Y-%m-%d").date()
        self.end = datetime.strptime(args.end, "%Y-%m-%d").date()
        self.data = shared_data or DataLayer(PROJECT_ROOT / "backtest_cache")
        self.cfg = ace.AlphaCallConfig()
        # ── sweepable overrides (config fields are class-level env defaults, so we
        #    set them on the instance rather than via env after import) ──
        # Absolute per-trade dollar budget takes precedence: convert to a pct of the
        # starting equity so the engine's `equity * risk_per_trade_pct` sizing yields
        # ~that many dollars at the start (drifts with mark-to-market equity).
        if getattr(args, "risk_dollars", None):
            self.cfg.risk_per_trade_pct = float(args.risk_dollars) / max(float(args.equity), 1.0)
        elif getattr(args, "risk_pct", None):
            self.cfg.risk_per_trade_pct = float(args.risk_pct)
        if getattr(args, "exposure_pct", None):
            self.cfg.max_options_exposure_pct = float(args.exposure_pct)
        self.force_overnight = bool(getattr(args, "force_overnight", False))
        self.engine = self._fresh_engine()
        self.acct = Account(start_equity=args.equity, cash=args.equity)
        self.acct.max_options_exposure_pct = self.cfg.max_options_exposure_pct
        self.trades: List[dict] = []
        self.equity_rows: List[dict] = []
        self.daily_rows: List[dict] = []
        self.entry_interval = max(1, int(args.entry_interval))
        self.exit_interval = max(1, int(args.exit_interval))
        self.iv_cache: Dict[tuple, float] = {}
        self._final_day: Optional[date] = None
        self.quiet = bool(getattr(args, "quiet", False))
        # Use REAL Alpha Vantage historical option chains (greeks+IV) instead of the
        # synthetic Black-Scholes chain for contract selection. Falls back to synthetic.
        self.av_chains = bool(getattr(args, "av_chains", False))
        # ── transaction-cost model (all zero unless --costs) ──
        self.costs = bool(getattr(args, "costs", False))
        self.commission = float(getattr(args, "commission", None) or (0.65 if self.costs else 0.0))
        self.slippage_pct = float(getattr(args, "slippage_pct", None) or (0.0075 if self.costs else 0.0))
        # entry already fills at ask (≈+1.2% vs mid); exits cross to bid → other half-spread
        self.exit_spread_haircut = 0.012 if self.costs else 0.0
        self.fees_paid = 0.0

    def _fresh_engine(self):
        # Clear any sandboxed state so each run starts flat.
        sandbox = PROJECT_ROOT / "system-data" / os.environ["RUN_DATA_DATE"]
        try:
            for f in sandbox.glob("*alpha_engine_state.json"):
                f.unlink()
        except Exception:
            pass
        return ace.AlphaCallEngine(broker=_NullBroker(), config=self.cfg)

    # ---- market context (regime gate) ----
    def _market_context(self, spy_df: pd.DataFrame, now: datetime) -> dict:
        bias = "neutral"
        try:
            day = spy_df[spy_df.index.date == now.date()]
            upto = day[day.index <= now]
            if len(upto) > 2:
                op = float(day["open"].iloc[0])
                last = float(upto["close"].iloc[-1])
                chg = (last - op) / op
                if chg > 0.001:
                    bias = "bullish"
                elif chg < -0.003:
                    bias = "cautious"
                else:
                    bias = "neutral"
        except Exception:
            pass
        return {
            "overall_market_bias": bias,
            "market_bias": bias,
            "regime": bias,
            "risk_free_rate": float(self.cfg.risk_free_rate),
            "trade_permission": "normal",
        }

    def _iv(self, symbol: str, on: date) -> float:
        key = (symbol, on)
        if key not in self.iv_cache:
            self.iv_cache[key] = self.data.vol_for(symbol, on)
        return self.iv_cache[key]

    # ---- per-decision marking of open positions ----
    def _mark_positions(self, price_at: Dict[str, float], today: date) -> Dict[str, float]:
        mark: Dict[str, float] = {}
        for c, p in self.acct.positions.items():
            S = price_at.get(p.underlying)
            if S is None:
                mark[c] = p.entry_mid
                continue
            mark[c] = reprice_call(S, p.strike, p.expiry, today, p.iv, self.cfg)
        return mark

    def _positions_for_engine(self, mark: Dict[str, float]) -> List[dict]:
        out = []
        for c, p in self.acct.positions.items():
            cur = mark.get(c, p.entry_mid)
            plpc = (cur - p.entry_mid) / p.entry_mid if p.entry_mid > 0 else 0.0
            out.append({"symbol": c, "qty": p.contracts, "unrealized_plpc": plpc})
        return out

    # ---- fills ----
    def _fill_exit(self, ex: dict, mark: Dict[str, float], now: datetime):
        c = ex["symbol"]
        p = self.acct.positions.get(c)
        if p is None:
            return
        qty = int(ex.get("qty") or p.contracts)
        qty = min(qty, p.contracts)
        if qty <= 0:
            return
        cur_mid = mark.get(c, p.entry_mid)
        # Exit fills cross to the bid (spread haircut) and take adverse slippage; a
        # flat per-contract commission is charged. All zero unless --costs is set.
        exit_px = cur_mid * (1.0 - self.exit_spread_haircut - self.slippage_pct)
        gross = qty * exit_px * 100.0
        net_proceeds = gross - self.commission * qty
        cost_basis = qty * p.entry_mid * 100.0 + qty * p.entry_fee_pc
        realized = net_proceeds - cost_basis
        self.acct.cash += net_proceeds
        self.acct.realized_pnl += realized
        self.fees_paid += (qty * cur_mid * 100.0 - gross) + self.commission * qty
        self.trades.append({
            "time": now.isoformat(), "event": "exit", "contract": c,
            "underlying": p.underlying, "qty": qty,
            "price": round(exit_px, 4), "entry_mid": round(p.entry_mid, 4),
            "proceeds": round(net_proceeds, 2), "realized_pnl": round(realized, 2),
            "r_multiple": round(ex.get("r_multiple", 0.0) or 0.0, 3),
            "reason": ex.get("source") or ex.get("reason", ""),
        })
        p.contracts -= qty
        if p.contracts <= 0:
            del self.acct.positions[c]

    def _fill_entry(self, sig, now: datetime):
        c = sig.contract_symbol
        if c in self.acct.positions:
            return False
        ask = float(getattr(sig, "ask", 0) or getattr(sig, "mid", 0) or 0)
        contracts = int(getattr(sig, "contracts", 0) or 0)
        if ask <= 0 or contracts <= 0:
            return False
        # Entry slippage + commission attributed per contract (zero unless --costs).
        fee_pc = ask * 100.0 * self.slippage_pct + self.commission
        unit = ask * 100.0 + fee_pc
        cost = contracts * unit
        if cost > self.acct.cash:
            # size down to fit cash (never go negative)
            contracts = int(self.acct.cash // unit)
            if contracts <= 0:
                return False
            cost = contracts * unit
            sig.contracts = contracts
            sig.premium_cost = contracts * ask * 100.0
        self.acct.cash -= cost
        self.fees_paid += contracts * fee_pc
        self.acct.positions[c] = Position(
            contract_symbol=c, underlying=sig.symbol, strike=float(sig.strike),
            expiry=str(sig.expiry), iv=float(getattr(sig, "implied_volatility", 0) or self._iv(sig.symbol, now.date())),
            contracts=contracts, entry_mid=ask, entry_cost=contracts * ask * 100.0,
            opened=now.isoformat(), entry_fee_pc=fee_pc,
        )
        self.trades.append({
            "time": now.isoformat(), "event": "entry", "contract": c,
            "underlying": sig.symbol, "qty": contracts, "price": round(ask, 4),
            "entry_mid": round(ask, 4), "proceeds": round(-cost, 2), "realized_pnl": 0.0,
            "r_multiple": 0.0,
            "reason": f"{getattr(sig,'setup_type','')}:{getattr(sig,'grade','')}",
        })
        return True

    # ---- main loop ----
    def run(self):
        if not self.quiet:
            print(f"[backtest] loading 1m bars for {len(self.symbols)} symbols "
                  f"{self.start}..{self.end} …", flush=True)
        for s in self.symbols:
            self.data.load_1m(s, self.start, self.end)
            self.data.daily_vol(s, self.start, self.end)
        ref = "SPY" if "SPY" in self.symbols else self.symbols[0]
        if "SPY" not in self._bars_keys():
            # ensure we have SPY for regime context even if not traded
            self.data.load_1m("SPY", self.start, self.end)
        spy_df = self.data._bars_1m.get("SPY", self.data._bars_1m.get(ref))
        days = self.data.trading_days(ref, self.start, self.end)
        if not self.quiet:
            print(f"[backtest] {len(days)} trading days: {days[0]}..{days[-1]}" if days
                  else "[backtest] NO trading days found", flush=True)
        self._final_day = days[-1] if days else None

        for d in days:
            self._run_day(d, spy_df)

        self._finalize()

    def _bars_keys(self):
        return set(self.data._bars_1m.keys())

    def _run_day(self, d: date, spy_df: pd.DataFrame):
        # new trading day → reset day-loss anchor; engine EOD logic will flush
        # non-overnight positions via check_active_positions.
        eq_open = self.acct.mark_to_market(self._mark_all_at_open(d))
        self.acct.day_start_equity = eq_open
        self.acct.halted_today = False
        self.acct.halt_reason = ""
        day_realized_start = self.acct.realized_pnl

        # decision timeline: every minute from 09:31 to 15:59 ET
        session = [datetime.combine(d, dtime(9, 31), ET) + timedelta(minutes=i)
                   for i in range(0, 6 * 60 + 28)]
        last_entry_min = -999

        for i, now in enumerate(session):
            set_clock(now)
            price_at = self._prices_at(now)
            if not price_at:
                continue
            today = d

            # ---------- EXIT PASS (every exit_interval minutes) ----------
            if i % self.exit_interval == 0 and self.acct.positions:
                mark = self._mark_positions(price_at, today)
                eng_pos = self._positions_for_engine(mark)
                mctx = self._market_context(spy_df, now)
                try:
                    exits = self.engine.check_active_positions(
                        eng_pos, None, market_context=mctx, indicator_snapshot=None, market_state=None)
                except Exception as e:
                    exits = []
                    if self.args.verbose:
                        print(f"  [exit-err] {now} {e}", flush=True)
                for ex in exits:
                    self._fill_exit(ex, mark, now)

            # ---------- ENTRY PASS (every entry_interval minutes, after ORB) ----------
            after_orb = now.time() >= dtime(9, 30 + self.cfg.orb_minutes + 1)
            entry_cutoff = now.time() <= dtime(15, 45)
            if after_orb and entry_cutoff and (i - last_entry_min) >= self.entry_interval:
                last_entry_min = i
                eq_now = self.acct.mark_to_market(self._mark_positions(price_at, today))
                self.acct.peak_equity = max(self.acct.peak_equity, eq_now)
                halt = self.acct.check_backstops(eq_now)
                if halt:
                    if not self.acct.halted_today:
                        self.acct.halted_today = True
                        self.acct.halt_reason = halt
                        if self.args.verbose:
                            print(f"  [halt] {now} {halt}", flush=True)
                else:
                    self._entry_pass(price_at, now, d, spy_df, eq_now)

            # mark equity curve sparsely (every 15 min)
            if i % 15 == 0:
                eq = self.acct.mark_to_market(self._mark_positions(price_at, today))
                self.acct.peak_equity = max(self.acct.peak_equity, eq)
                self.equity_rows.append({"time": now.isoformat(), "equity": round(eq, 2),
                                         "cash": round(self.acct.cash, 2),
                                         "open_positions": len(self.acct.positions)})

        # EOD: carry overnight positions to the next day; force-flush only on the
        # final backtest day so remaining open risk is realized for reporting.
        self._eod_flush(d, is_final=(d == self._final_day))
        eq_close = self.acct.mark_to_market(self._mark_all_at_close(d))
        self.acct.peak_equity = max(self.acct.peak_equity, eq_close)
        day_realized = self.acct.realized_pnl - day_realized_start
        dd = (self.acct.peak_equity - eq_close) / max(self.acct.peak_equity, 1.0)
        self.daily_rows.append({
            "date": d.isoformat(), "equity_close": round(eq_close, 2),
            "day_realized_pnl": round(day_realized, 2),
            "open_positions": len(self.acct.positions),
            "drawdown_pct": round(dd * 100, 3),
            "halted": self.acct.halted_today, "halt_reason": self.acct.halt_reason,
        })
        if not self.quiet:
            print(f"  {d}  realized={day_realized:+.0f}  equity={eq_close:,.0f}  "
                  f"open={len(self.acct.positions)}  dd={dd*100:.1f}%"
                  + (f"  HALT[{self.acct.halt_reason}]" if self.acct.halted_today else ""),
                  flush=True)

    def _chain_for(self, symbol: str, price: float, d: date) -> list:
        """Real AV historical chain (--av-chains) with synthetic BS fallback.

        AV historical returns one daily chain (all expiries) per symbol/date — cached,
        so at most one API call per (symbol, day). Falls back to the synthetic chain if
        AV is unavailable or returns nothing for that symbol/date.
        """
        if self.av_chains:
            try:
                import av_options_adapter as av
                chain = av.historical_chain(symbol, d)
                if chain:
                    return chain
                if self.args.verbose:
                    print(f"  [av] empty chain {symbol} {d} → synthetic fallback", flush=True)
            except Exception as e:
                if self.args.verbose:
                    print(f"  [av-err] {symbol} {d}: {e} → synthetic fallback", flush=True)
        return build_chain(symbol, price, d, self._iv(symbol, d), self.cfg)

    def _entry_pass(self, price_at: Dict[str, float], now: datetime, d: date,
                    spy_df: pd.DataFrame, equity: float):
        # build bars_by_symbol up to `now`, synthetic chains, then evaluate.
        bars_by_symbol: Dict[str, pd.DataFrame] = {}
        chains_by_symbol: Dict[str, list] = {}
        for s in self.symbols:
            df = self.data._bars_1m.get(s)
            if df is None or len(df) == 0:
                continue
            day_bars = df[(df.index.date == d) & (df.index <= now)]
            if len(day_bars) < self.cfg.orb_minutes + 2:
                continue
            bars_by_symbol[s] = day_bars
            S = price_at.get(s)
            if S:
                chains_by_symbol[s] = self._chain_for(s, S, d)
        if not bars_by_symbol:
            return
        mctx = self._market_context(spy_df, now)
        positions = {c: {"symbol": c} for c in self.acct.positions}
        try:
            signals = self.engine.evaluate_universe(
                symbols=list(bars_by_symbol.keys()),
                bars_by_symbol=bars_by_symbol,
                market_context=mctx,
                equity=equity,
                options_exposure=self.acct.options_exposure_cost(),
                waterfall_budget=0.0,
                chains_by_symbol=chains_by_symbol,
                positions=positions,
                account_state={"equity": equity, "buying_power": self.acct.cash},
            )
        except Exception as e:
            if self.args.verbose:
                import traceback
                print(f"  [entry-err] {now} {e}\n{traceback.format_exc()}", flush=True)
            return
        # exposure backstop: never exceed max_options_exposure_pct of equity
        exp_cap = equity * self.acct.max_options_exposure_pct
        for sig in signals:
            if getattr(sig, "status", "") != "ready":
                continue
            if self.acct.options_exposure_cost() >= exp_cap:
                break
            # Research lever: let winners run overnight. Keeps hard-stop / 1.7R-trail /
            # 3R-target / theta-decay exits intact; only relaxes the EOD forced close.
            if self.force_overnight:
                sig.overnight_allowed_by_probability = True
            if self._fill_entry(sig, now):
                try:
                    self.engine.register_filled(sig)
                except Exception:
                    pass

    # ---- price helpers ----
    def _prices_at(self, now: datetime) -> Dict[str, float]:
        out = {}
        for s, df in self.data._bars_1m.items():
            if len(df) == 0:
                continue
            day = df[(df.index.date == now.date()) & (df.index <= now)]
            if len(day):
                out[s] = float(day["close"].iloc[-1])
        return out

    def _mark_all_at_open(self, d: date) -> Dict[str, float]:
        mark = {}
        prices = {}
        for s, df in self.data._bars_1m.items():
            if len(df) == 0:
                continue
            day = df[df.index.date == d]
            if len(day):
                prices[s] = float(day["open"].iloc[0])
        for c, p in self.acct.positions.items():
            S = prices.get(p.underlying)
            mark[c] = reprice_call(S, p.strike, p.expiry, d, p.iv, self.cfg) if S else p.entry_mid
        return mark

    def _mark_all_at_close(self, d: date) -> Dict[str, float]:
        mark = {}
        prices = {}
        for s, df in self.data._bars_1m.items():
            if len(df) == 0:
                continue
            day = df[df.index.date == d]
            if len(day):
                prices[s] = float(day["close"].iloc[-1])
        for c, p in self.acct.positions.items():
            S = prices.get(p.underlying)
            mark[c] = reprice_call(S, p.strike, p.expiry, d, p.iv, self.cfg) if S else p.entry_mid
        return mark

    def _eod_flush(self, d: date, is_final: bool = True):
        # Non-overnight positions are closed intraday by the engine's own EOD logic.
        # Overnight positions carry across days (managed by trail/target/stop each day);
        # we only force-realize whatever remains on the LAST backtest day, for reporting.
        if not is_final or not self.acct.positions:
            return
        mark = self._mark_all_at_close(d)
        now = datetime.combine(d, dtime(15, 59), ET)
        for c in list(self.acct.positions.keys()):
            p = self.acct.positions[c]
            cur = mark.get(c, p.entry_mid)
            self._fill_exit({"symbol": c, "qty": p.contracts, "source": "backtest_eod_flush",
                             "r_multiple": (cur - p.entry_mid) / p.entry_mid if p.entry_mid else 0.0},
                            mark, now)
        # keep engine state consistent
        try:
            self.engine._active.clear()
            self.engine._save_state()
        except Exception:
            pass

    # ---- reporting ----
    def _finalize(self):
        out_dir = PROJECT_ROOT / "backtest_outputs" / self.args.run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.trades).to_csv(out_dir / "trades.csv", index=False)
        pd.DataFrame(self.equity_rows).to_csv(out_dir / "equity_curve.csv", index=False)
        pd.DataFrame(self.daily_rows).to_csv(out_dir / "daily.csv", index=False)

        exits = [t for t in self.trades if t["event"] == "exit"]
        entries = [t for t in self.trades if t["event"] == "entry"]
        wins = [t for t in exits if t["realized_pnl"] > 0]
        losses = [t for t in exits if t["realized_pnl"] <= 0]
        final_eq = self.acct.start_equity + self.acct.realized_pnl + \
            self.acct.mark_to_market({}) - self.acct.cash  # open MV (≈0 after flush)
        final_eq = self.acct.cash + sum(
            p.contracts * p.entry_mid * 100 for p in self.acct.positions.values())
        n_days = max(len(self.daily_rows), 1)
        max_dd = max((r["drawdown_pct"] for r in self.daily_rows), default=0.0)
        summary = {
            "run_tag": self.args.run_tag,
            "window": f"{self.start}..{self.end}",
            "symbols": self.symbols,
            "start_equity": self.acct.start_equity,
            "final_equity": round(final_eq, 2),
            "total_realized_pnl": round(self.acct.realized_pnl, 2),
            "return_pct": round(100 * (final_eq - self.acct.start_equity) / self.acct.start_equity, 3),
            "trading_days": n_days,
            "avg_realized_pnl_per_day": round(self.acct.realized_pnl / n_days, 2),
            "entries_filled": len(entries),
            "exits_filled": len(exits),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(100 * len(wins) / max(len(exits), 1), 1),
            "avg_win": round(sum(t["realized_pnl"] for t in wins) / max(len(wins), 1), 2),
            "avg_loss": round(sum(t["realized_pnl"] for t in losses) / max(len(losses), 1), 2),
            "max_drawdown_pct": round(max_dd, 3),
            "open_positions_remaining": len(self.acct.positions),
            "costs_enabled": self.costs,
            "fees_paid": round(self.fees_paid, 2),
            "commission_per_contract": self.commission,
            "slippage_pct": self.slippage_pct,
            "assumptions": [
                "Underlying: REAL Alpaca historical 1-minute bars (RTH).",
                "Option prices: synthesized via engine's black_scholes_call; IV = 20d trailing realized vol.",
                "Fills: entries at modeled ask (2.4% spread), exits at modeled mid — orders ALWAYS fill.",
                "No real NBBO/L2 liquidity, no commissions, no early assignment, no IV smile/skew.",
                "Engine risk + exit logic run unchanged; CLAUDE.md backstops gate new entries.",
            ],
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.summary = summary
        self.out_dir = out_dir
        if self.quiet:
            return
        print("\n" + "=" * 64)
        print("BACKTEST SUMMARY")
        print("=" * 64)
        for k in ("window", "start_equity", "final_equity", "total_realized_pnl",
                  "return_pct", "trading_days", "avg_realized_pnl_per_day",
                  "entries_filled", "exits_filled", "win_rate_pct", "avg_win",
                  "avg_loss", "max_drawdown_pct", "open_positions_remaining"):
            print(f"  {k:28s}: {summary[k]}")
        print(f"\n  outputs → {out_dir}")
        print("=" * 64)


class _NullBroker:
    """Stub broker: the harness supplies chains upfront, so this is only a fallback."""
    def get_options_chain(self, *a, **k):
        return []

    def get_account(self, *a, **k):
        return {}

    def get_all_positions(self, *a, **k):
        return []



def parse_args():
    p = argparse.ArgumentParser(description="Historical replay backtest for the Alpha Call Engine ($100k default).")
    p.add_argument("--symbols", default="VICR,MU,ASX,NOK,DELL")
    p.add_argument("--start", default="2026-05-04")
    p.add_argument("--end", default="2026-06-04")
    p.add_argument("--equity", type=float, default=6_500, help="starting equity in $")
    p.add_argument("--entry-interval", dest="entry_interval", type=int, default=5,
                   help="minutes between entry passes")
    p.add_argument("--exit-interval", dest="exit_interval", type=int, default=1,
                   help="minutes between exit passes")
    p.add_argument("--run-tag", dest="run_tag", default=None)
    # sweepable levers
    p.add_argument("--risk-pct", dest="risk_pct", type=float, default=None,
                   help="OPTIONS_RISK_PER_TRADE_PCT override (premium spend cap per trade)")
    p.add_argument("--risk-dollars", dest="risk_dollars", type=float, default=None,
                   help="absolute per-trade budget in $ (converted to pct of starting equity; takes precedence over --risk-pct)")
    p.add_argument("--exposure-pct", dest="exposure_pct", type=float, default=None,
                   help="MAX_OPTIONS_EXPOSURE_PCT override (total options exposure cap)")
    p.add_argument("--force-overnight", dest="force_overnight", action="store_true",
                   help="let winners run overnight (keeps stop/trail/target; relaxes EOD forced-close)")
    p.add_argument("--av-chains", dest="av_chains", action="store_true",
                   help="use REAL Alpha Vantage historical option chains (greeks+IV) instead of synthetic BS")
    p.add_argument("--sweep", action="store_true", help="run the sizing/overnight parameter grid")
    p.add_argument("--costs", action="store_true",
                   help="enable transaction costs (exit spread haircut + slippage + commission)")
    p.add_argument("--commission", type=float, default=None, help="commission per contract per side (default 0.65 with --costs)")
    p.add_argument("--slippage-pct", dest="slippage_pct", type=float, default=None,
                   help="adverse slippage per side as fraction of option price (default 0.0075 with --costs)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    if not a.run_tag:
        a.run_tag = f"alpha_{a.start.replace('-','')}_{a.end.replace('-','')}_{int(a.equity)}"
    return a


# ---------------------------------------------------------------------------
# Parameter sweep: sizing × overnight-hold × exposure-cap, toward $400–500/day.
# ---------------------------------------------------------------------------
import copy as _copy


def run_sweep(base_args):
    risk_grid = [0.015, 0.04, 0.08]
    exposure_grid = [0.15, 0.40]
    overnight_grid = [False, True]
    target_lo, target_hi = 400.0, 500.0

    shared = DataLayer(PROJECT_ROOT / "backtest_cache")
    rows = []
    configs = [(r, e, o) for o in overnight_grid for e in exposure_grid for r in risk_grid]
    print(f"[sweep] {len(configs)} configs over {base_args.symbols.count(',')+1} symbols "
          f"{base_args.start}..{base_args.end} @ ${int(base_args.equity):,}\n", flush=True)
    print(f"  {'risk%':>6} {'expo%':>6} {'overnight':>9} | "
          f"{'$/day':>8} {'total$':>9} {'win%':>5} {'maxDD%':>7} {'trades':>6} {'halts':>5}", flush=True)
    print("  " + "-" * 72, flush=True)

    for (risk, expo, overnight) in configs:
        a = _copy.copy(base_args)
        a.risk_pct = risk
        a.exposure_pct = expo
        a.force_overnight = overnight
        a.quiet = True
        a.run_tag = f"sweep_r{int(risk*1000)}_e{int(expo*100)}_{'on' if overnight else 'off'}"
        bt = Backtest(a, shared_data=shared)
        bt.run()
        s = bt.summary
        halts = sum(1 for r in bt.daily_rows if r["halted"])
        row = {
            "risk_pct": risk, "exposure_pct": expo, "overnight": overnight,
            "avg_per_day": s["avg_realized_pnl_per_day"], "total_pnl": s["total_realized_pnl"],
            "return_pct": s["return_pct"], "win_rate_pct": s["win_rate_pct"],
            "max_drawdown_pct": s["max_drawdown_pct"], "entries": s["entries_filled"],
            "halt_days": halts, "run_tag": a.run_tag,
        }
        rows.append(row)
        hit = target_lo <= row["avg_per_day"] <= target_hi
        flag = "  <= TARGET" if hit else ("  > target" if row["avg_per_day"] > target_hi else "")
        print(f"  {risk*100:>5.1f}% {expo*100:>5.0f}% {('YES' if overnight else 'no'):>9} | "
              f"{row['avg_per_day']:>8.0f} {row['total_pnl']:>9.0f} {row['win_rate_pct']:>5.1f} "
              f"{row['max_drawdown_pct']:>7.2f} {row['entries']:>6} {halts:>5}{flag}", flush=True)

    out_dir = PROJECT_ROOT / "backtest_outputs" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("avg_per_day", ascending=False)
    df.to_csv(out_dir / "sweep_results.csv", index=False)

    print("\n" + "=" * 74)
    print("SWEEP — ranked by $/day (risk-adjusted: watch maxDD% and halts)")
    print("=" * 74)
    best = df.iloc[0]
    for _, r in df.head(6).iterrows():
        print(f"  ${r['avg_per_day']:>5.0f}/day  total ${r['total_pnl']:>7.0f}  "
              f"win {r['win_rate_pct']:>4.1f}%  maxDD {r['max_drawdown_pct']:>5.2f}%  "
              f"halts {int(r['halt_days'])}  | risk {r['risk_pct']*100:.1f}% "
              f"expo {r['exposure_pct']*100:.0f}% overnight={'YES' if r['overnight'] else 'no'}")
    in_band = df[(df["avg_per_day"] >= target_lo) & (df["avg_per_day"] <= target_hi)]
    print("-" * 74)
    if len(in_band):
        b = in_band.sort_values("max_drawdown_pct").iloc[0]
        print(f"  IN $400–500/day BAND (lowest DD): risk {b['risk_pct']*100:.1f}%  "
              f"expo {b['exposure_pct']*100:.0f}%  overnight={'YES' if b['overnight'] else 'no'}  "
              f"→ ${b['avg_per_day']:.0f}/day, maxDD {b['max_drawdown_pct']:.2f}%")
    else:
        print(f"  No config landed inside $400–500/day. Best = ${best['avg_per_day']:.0f}/day "
              f"(risk {best['risk_pct']*100:.1f}%, expo {best['exposure_pct']*100:.0f}%, "
              f"overnight={'YES' if best['overnight'] else 'no'}).")
    print(f"\n  full grid → {out_dir / 'sweep_results.csv'}")
    print("=" * 74)
    return df


def main():
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    else:
        bt = Backtest(args)
        bt.run()


if __name__ == "__main__":
    main()
