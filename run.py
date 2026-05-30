"""
run.py — QuantumQuant Main Entry Point  (Two-Engine Architecture)

Orchestrates a two-engine trading system on top of the existing risk controls:

  1. ALPHA ENGINE  (alpha_call_engine.AlphaCallEngine)
       Primary tactical profit engine. Long calls only on A/A+ momentum, ORB,
       VWAP-reclaim, pivot-breakout, institutional-accumulation, sector-leadership
       setups. Calls are actively managed (hard stop / +1R partial / 1.7R trail /
       3R target) — never held indefinitely.

  2. BUFFETT ACCUMULATION LAYER  (buffett_accumulation_layer.BuffettAccumulationLayer)
       Strategic long-stock engine. Scores businesses 0-100 (moat, financials,
       management, valuation, macro, policy, 18-yr cycle, technical entry) and
       accumulates quality companies at optimal entry points. Held indefinitely
       unless thesis / trend / concentration / risk breaks.

Capital flow (capital_waterfall.CapitalWaterfall):
       Realized call profits → 50-70% BAL stock, 20-40% alpha buying power, 10% cash.

Data flow:
       TradingView institutional candidates  →  market_context_builder.build_market_context()
       →  macro/policy/cycle context  →  Alpha Engine (calls) + BAL (stock)
       →  risk-gated Alpaca execution  →  journal / equity logging.

SAFETY GATES (never bypassed — preserved from prior run.py):
  • Daily loss limit             DAILY_LOSS_LIMIT (env, default -$500)
  • Max portfolio drawdown       MAX_DRAWDOWN     (env, default -5%)
  • Max open positions           MAX_POSITIONS    (env, default 3)
  • Kill switch (manual file)    KILL_SWITCH_PATH (env, default .kill_switch)
  • Alpaca paper-only gate       ALPACA_PAPER_ONLY (env, default "true")
  • Equity floor                 EQUITY_FLOOR     (env, default $1000)
  • Max order size               MAX_ORDER_SIZE   (env, default 1000 shares)
  • Cash buffer                  CASH_BUFFER_PCT  (env, default 8%)
  • Max options exposure         MAX_OPTIONS_EXPOSURE_PCT (env, default 15%)
  • Max stock exposure           MAX_STOCK_EXPOSURE_PCT   (env, default 80%)
  • Max single-stock weight      MAX_SINGLE_STOCK_WEIGHT  (env, default 10%)
  • PDT awareness                (block equity day-trade buys if flagged < $25k)
  • Market hours gate            (no orders outside RTH unless allowed)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env before any module-level env reads so creds/config are populated.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Environment / config
# ---------------------------------------------------------------------------

MODE               = os.environ.get("MODE",             "paper").lower()   # paper | live | train | scalp
SYMBOLS_ENV        = os.environ.get("SYMBOLS",          "")
DAILY_LOSS_LIMIT   = float(os.environ.get("DAILY_LOSS_LIMIT",  -500.0))
MAX_DRAWDOWN       = float(os.environ.get("MAX_DRAWDOWN",       -0.05))
MAX_POSITIONS      = int(os.environ.get("MAX_POSITIONS",         3))
EQUITY_FLOOR       = float(os.environ.get("EQUITY_FLOOR",     1000.0))
MAX_ORDER_SIZE     = int(os.environ.get("MAX_ORDER_SIZE",       1000))
ALPACA_PAPER_ONLY  = os.environ.get("ALPACA_PAPER_ONLY",       "true").lower() == "true"
KILL_SWITCH_PATH   = os.environ.get("KILL_SWITCH_PATH",        ".kill_switch")
BAR_INTERVAL_SEC   = float(os.environ.get("BAR_INTERVAL_SEC",  60.0))
STARTING_EQUITY    = float(os.environ.get("STARTING_EQUITY", 100_000.0))

# Two-engine toggles & exposure controls (also consumed by the engine configs via env)
ENABLE_ALPHA_CALL_ENGINE        = os.environ.get("ENABLE_ALPHA_CALL_ENGINE", "true").lower() in {"1", "true"}
ENABLE_BUFFETT_ACCUMULATION_LAYER = os.environ.get("ENABLE_BUFFETT_ACCUMULATION_LAYER", "true").lower() in {"1", "true"}
ENABLE_BAL_PORTFOLIO_MONITOR    = os.environ.get("ENABLE_BAL_PORTFOLIO_MONITOR", "false").lower() in {"1", "true"}
ENABLE_ALPHA_OPTIONS_SCREENER   = os.environ.get("ENABLE_ALPHA_OPTIONS_SCREENER", "true").lower() in {"1", "true"}
ENABLE_BUFFETT_ACCUMULATION_SCREENER = os.environ.get("ENABLE_BUFFETT_ACCUMULATION_SCREENER", "true").lower() in {"1", "true"}
MAX_OPTIONS_EXPOSURE_PCT        = float(os.environ.get("MAX_OPTIONS_EXPOSURE_PCT", 0.15))
MAX_STOCK_EXPOSURE_PCT          = float(os.environ.get("MAX_STOCK_EXPOSURE_PCT",  0.80))
MAX_SINGLE_STOCK_WEIGHT         = float(os.environ.get("MAX_SINGLE_STOCK_WEIGHT", 0.10))
OPTIONS_RISK_PER_TRADE_PCT      = float(os.environ.get("OPTIONS_RISK_PER_TRADE_PCT", 0.015))
BAL_MIN_SCORE_TO_BUY            = float(os.environ.get("BAL_MIN_SCORE_TO_BUY", 55))
BAL_AGGRESSIVE_SCORE            = float(os.environ.get("BAL_AGGRESSIVE_SCORE", 85))
# Composite DCA framework — value decides WHETHER, RSI decides WHEN, tranches decide
# HOW MUCH, RiskManager decides whether ALLOWED. (config lives in DCAConfig)
ENABLE_BAL_COMPOSITE_DCA        = os.environ.get("ENABLE_BAL_COMPOSITE_DCA", "true").lower() in {"1", "true"}
# SEC EDGAR Form 4 insider-buy DISCOVERY (expands the Buffett review universe only —
# never generates orders, never bypasses any gate).
ENABLE_SEC_FORM4_FEED           = os.environ.get("ENABLE_SEC_FORM4_FEED", "true").lower() in {"1", "true"}
# Adaptive Capital Allocation (evidence-driven sizing PLAN; advisory, within caps).
ENABLE_ADAPTIVE_ALLOCATION      = os.environ.get("ENABLE_ADAPTIVE_ALLOCATION", "true").lower() in {"1", "true"}
OPTIONS_PROFIT_TO_BAL_PCT       = float(os.environ.get("OPTIONS_PROFIT_TO_BAL_PCT", 0.60))
OPTIONS_PROFIT_TO_ALPHA_PCT     = float(os.environ.get("OPTIONS_PROFIT_TO_ALPHA_PCT", 0.30))
OPTIONS_PROFIT_TO_CASH_PCT      = float(os.environ.get("OPTIONS_PROFIT_TO_CASH_PCT", 0.10))
CASH_BUFFER_PCT                 = float(os.environ.get("CASH_BUFFER_PCT", 0.08))
CONTEXT_REFRESH_MIN             = float(os.environ.get("CONTEXT_REFRESH_MIN", 30.0))
ALPHA_BARS_LOOKBACK_MIN         = int(os.environ.get("ALPHA_BARS_LOOKBACK_MIN", 120))

# Alpaca creds (required for live/paper modes).
# Resolve across both the alpaca-py (ALPACA_*) and classic SDK (APCA_*) env names —
# .env files commonly carry APCA_API_KEY_ID / APCA_API_SECRET_KEY.
ALPACA_KEY    = (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
                 or os.environ.get("ALPACA_API_KEY_ID") or "")
ALPACA_SECRET = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
                 or os.environ.get("ALPACA_API_SECRET_KEY") or "")
ALPACA_URL    = os.environ.get("ALPACA_BASE_URL", os.environ.get("APCA_API_BASE_URL",
                "https://paper-api.alpaca.markets" if ALPACA_PAPER_ONLY
                else "https://api.alpaca.markets"))

# Option-chain DTE window for contract retrieval (matches the Alpha Engine's gate).
ALPHA_MIN_DTE = int(os.environ.get("ALPHA_MIN_DTE", 21))
ALPHA_MAX_DTE = int(os.environ.get("ALPHA_MAX_DTE", 60))

# Daily tag shared with the engine modules (so all outputs land in the same dated buckets)
DATA_TAG       = os.getenv("RUN_DATA_DATE", datetime.now(timezone.utc).strftime("%Y%m%d"))
DAILY_OUTPUTS  = PROJECT_ROOT / "daily_system_outputs"
SKIPPED_TRADES_PATH = PROJECT_ROOT / "logs" / "execution" / DATA_TAG / f"{DATA_TAG}_skipped_trades.jsonl"
SCREENER_FUNNEL_PATH = DAILY_OUTPUTS / "screener_funnel_report.json"

# Canonical skip reasons (every rejected trade must log one of these)
SKIP_REASONS = {
    "rejected_by_market_regime",
    "rejected_by_options_liquidity",
    "rejected_by_setup_quality",
    "rejected_by_BAL_score",
    "rejected_by_valuation",
    "rejected_by_macro",
    "rejected_by_cycle",
    "rejected_by_cash_buffer",
    "rejected_by_exposure_limit",
    "rejected_by_PDT",
    "rejected_by_kill_switch",
    "rejected_by_equity_floor",
    "rejected_by_daily_loss",
    "rejected_by_drawdown",
    "rejected_by_order_size",
    "rejected_by_data_unavailable",
}


def log_skip(symbol: str, reason: str, **detail) -> None:
    """Record a skipped trade with a canonical reason. Always logged + journaled."""
    base = reason.split(":", 1)[0]
    if base not in SKIP_REASONS:
        logger.debug("log_skip: non-canonical reason '%s' (symbol=%s)", reason, symbol)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": DATA_TAG,
        "symbol": symbol,
        "reason": reason,
        "detail": detail,
    }
    logger.info("SKIP %s — %s %s", symbol, reason, detail if detail else "")
    try:
        SKIPPED_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SKIPPED_TRADES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def is_option_symbol(symbol: str) -> bool:
    """OCC option symbols are >= 15 chars and end with the strike block, e.g. AAPL250620C00150000."""
    s = (symbol or "").strip().upper()
    if len(s) < 15:
        return False
    tail = s[-15:]
    return (tail[6] in ("C", "P")) and tail[:6].isdigit() and tail[7:].isdigit()


# ---------------------------------------------------------------------------
# Risk manager  (preserved + extended for options/stock exposure and reason codes)
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Central guard. All order submissions must pass gate()/check_order().
    Positions / daily PnL / drawdown / PDT tracked here.
    """

    def __init__(self, equity: float = STARTING_EQUITY):
        self.starting_equity = equity
        self.equity          = equity
        self.buying_power    = equity
        self.daily_pnl       = 0.0
        self.open_positions: Dict[str, int] = {}
        self.options_exposure = 0.0     # $ notional in option positions
        self.stock_exposure   = 0.0     # $ notional in equity positions
        self.pdt_restricted   = False   # account flagged & < $25k & near daytrade cap
        self._killed         = False

    # ------------------------------------------------------------------

    def gate(self, symbol: str, side: str, qty: int, price: float,
             asset_class: str = "equity") -> Tuple[bool, str]:
        """
        Return (allowed, reason). reason is a canonical SKIP_REASONS code when blocked,
        or "" when allowed. asset_class ∈ {"equity", "option"}.
        """
        if self._check_kill_switch():
            return False, "rejected_by_kill_switch"

        if self.equity < EQUITY_FLOOR:
            return False, f"rejected_by_equity_floor:equity={self.equity:.0f}<{EQUITY_FLOOR:.0f}"

        if self.daily_pnl <= DAILY_LOSS_LIMIT:
            return False, f"rejected_by_daily_loss:pnl={self.daily_pnl:.0f}<={DAILY_LOSS_LIMIT:.0f}"

        dd = (self.equity - self.starting_equity) / (self.starting_equity + 1e-8)
        if dd <= MAX_DRAWDOWN:
            return False, f"rejected_by_drawdown:dd={dd*100:.1f}%<={MAX_DRAWDOWN*100:.1f}%"

        if qty > MAX_ORDER_SIZE:
            return False, f"rejected_by_order_size:qty={qty}>{MAX_ORDER_SIZE}"

        # Buy-side exposure / position controls
        if side == "buy":
            notional = qty * price * (100.0 if asset_class == "option" else 1.0)
            if asset_class == "option":
                if (self.options_exposure + notional) > self.equity * MAX_OPTIONS_EXPOSURE_PCT:
                    return False, (f"rejected_by_exposure_limit:options="
                                   f"{self.options_exposure + notional:.0f}>"
                                   f"{self.equity * MAX_OPTIONS_EXPOSURE_PCT:.0f}")
            else:
                # equity buy — PDT, position-count and stock-exposure limits
                if self.pdt_restricted:
                    return False, "rejected_by_PDT"
                n_open = len([s for s, q in self.open_positions.items() if q > 0])
                if n_open >= MAX_POSITIONS:
                    return False, f"rejected_by_exposure_limit:positions={n_open}>={MAX_POSITIONS}"
                if (self.stock_exposure + notional) > self.equity * MAX_STOCK_EXPOSURE_PCT:
                    return False, (f"rejected_by_exposure_limit:stock="
                                   f"{self.stock_exposure + notional:.0f}>"
                                   f"{self.equity * MAX_STOCK_EXPOSURE_PCT:.0f}")

        return True, ""

    def check_order(self, symbol: str, side: str, qty: int,
                    price: float, dry_run: bool = False) -> bool:
        """Backward-compatible boolean gate (used by the legacy scalping path)."""
        allowed, reason = self.gate(symbol, side, qty, price, asset_class="equity")
        if not allowed:
            logger.warning("BLOCKED %s %s %d: %s", side, symbol, qty, reason)
        return allowed

    def record_fill(self, symbol: str, side: str, qty: int, price: float,
                    asset_class: str = "equity"):
        if side == "buy":
            self.open_positions[symbol] = self.open_positions.get(symbol, 0) + qty
        else:
            prev = self.open_positions.get(symbol, 0)
            self.open_positions[symbol] = max(0, prev - qty)

    def record_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.equity    += pnl

    def sync_account(self, account: Dict):
        """Refresh equity/buying-power/PDT from a broker account snapshot."""
        if account.get("equity"):
            self.equity = float(account["equity"])
        if account.get("buying_power") is not None:
            self.buying_power = float(account["buying_power"])
        # PDT: flagged AND under the $25k pattern-day-trader floor AND near the daytrade cap
        flagged   = bool(account.get("pattern_day_trader", False))
        dt_count  = int(account.get("daytrade_count", 0) or 0)
        self.pdt_restricted = flagged and self.equity < 25_000 and dt_count >= 3

    def sync_exposure(self, options_exposure: float, stock_exposure: float):
        self.options_exposure = max(0.0, options_exposure)
        self.stock_exposure   = max(0.0, stock_exposure)

    def _check_kill_switch(self) -> bool:
        if self._killed:
            return True
        if os.path.exists(KILL_SWITCH_PATH):
            logger.critical("KILL SWITCH FILE DETECTED — halting all orders")
            self._killed = True
        return self._killed

    @property
    def is_killed(self) -> bool:
        return self._check_kill_switch()

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.open_positions = {}
        logger.info("Risk manager: daily state reset")


# ---------------------------------------------------------------------------
# Broker  (extends the prior AlpacaGateway: equity + option orders, chain, positions)
# ---------------------------------------------------------------------------

class AlpacaGateway:
    """
    Unified, risk-gated broker for both engines.

    Paper gate enforced: if ALPACA_PAPER_ONLY=true the client is forced to the paper
    endpoint. Every order passes through RiskManager.gate() first. Supports:
        • equity market/limit orders        (BAL stock buys, exits)
        • option limit orders on OCC symbols (Alpha call buys, exits, BAL hedges)
        • options-chain retrieval            (Alpha contract selection)
        • account + positions snapshots
    """

    def __init__(self, risk: RiskManager):
        self.risk       = risk
        self._client    = None       # TradingClient
        self._opt_data  = None       # OptionHistoricalDataClient
        self._mode      = "sim"
        self._available = False
        self._init_client()

    # ── init ──────────────────────────────────────────────────────────────────
    def _init_client(self):
        if not ALPACA_KEY or not ALPACA_SECRET:
            logger.warning("Alpaca creds not set — order submission disabled (simulation mode)")
            return
        try:
            from alpaca.trading.client import TradingClient
            self._client    = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=ALPACA_PAPER_ONLY)
            self._mode      = "alpaca_py"
            self._available = True
            logger.info("Alpaca-py client initialised  paper=%s  url=%s", ALPACA_PAPER_ONLY, ALPACA_URL)
        except ImportError:
            try:
                import alpaca_trade_api as tradeapi
                self._client    = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)
                self._mode      = "alpaca_trade_api"
                self._available = True
                logger.info("alpaca_trade_api client initialised  url=%s", ALPACA_URL)
            except ImportError:
                logger.warning("Neither alpaca-py nor alpaca_trade_api installed — simulation only")
        # Options data client (best-effort; only used by alpaca_py mode)
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self._opt_data = OptionHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        except Exception:
            self._opt_data = None

    @property
    def available(self) -> bool:
        return self._available

    # ── account / positions ─────────────────────────────────────────────────
    def get_account(self) -> Dict:
        if not self._available:
            return {"equity": self.risk.equity, "buying_power": self.risk.buying_power, "sim": True}
        try:
            acc = self._client.get_account()
            return {
                "equity":            float(acc.equity),
                "buying_power":      float(acc.buying_power),
                "cash":              float(getattr(acc, "cash", 0) or 0),
                "pattern_day_trader": bool(getattr(acc, "pattern_day_trader", False)),
                "daytrade_count":    int(getattr(acc, "daytrade_count", 0) or 0),
            }
        except Exception as e:
            logger.error("get_account error: %s", e)
            return {}

    def get_positions_raw(self) -> List[Any]:
        """Return raw Alpaca position objects (attributes: symbol, qty, market_value,
        avg_entry_price, current_price, unrealized_plpc, asset_class)."""
        if not self._available or self._mode != "alpaca_py":
            return []
        try:
            return list(self._client.get_all_positions())
        except Exception as e:
            logger.error("get_all_positions error: %s", e)
            return []

    @staticmethod
    def _is_option_position(p: Any) -> bool:
        ac = str(getattr(p, "asset_class", "") or "").lower()
        if "option" in ac:
            return True
        return is_option_symbol(str(getattr(p, "symbol", "")))

    def split_positions(self, positions: List[Any]) -> Tuple[List[Dict], Dict[str, Any], float, float]:
        """
        Split raw positions into:
          option_positions : list[dict]  (for AlphaCallEngine — keys: symbol, qty, unrealized_plpc)
          equity_positions : dict[symbol, position_obj]  (for BAL)
          options_exposure : float ($ market value)
          stock_exposure   : float ($ market value)
        """
        option_positions: List[Dict] = []
        equity_positions: Dict[str, Any] = {}
        opt_exp = stk_exp = 0.0
        for p in positions:
            mv = abs(float(getattr(p, "market_value", 0) or 0))
            if self._is_option_position(p):
                option_positions.append({
                    "symbol": str(getattr(p, "symbol", "")),
                    "qty": int(float(getattr(p, "qty", 0) or 0)),
                    "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0) or 0),
                    "market_value": mv,
                })
                opt_exp += mv
            else:
                equity_positions[str(getattr(p, "symbol", ""))] = p
                stk_exp += mv
        return option_positions, equity_positions, opt_exp, stk_exp

    # ── options chain ─────────────────────────────────────────────────────────
    def get_options_chain(self, symbol: str) -> List[Dict]:
        """
        Return a list of call/put contract dicts for `symbol` in the shape the
        Alpha Engine expects (keys: symbol, type, expiration_date, strike_price,
        delta, open_interest, volume, bid_price, ask_price).

        Strikes / expirations / open-interest come from the trading client's
        option-contracts endpoint; bid/ask/greeks are merged from the options
        data snapshot when available. Returns [] gracefully on any failure.
        """
        if not self._available or self._mode != "alpaca_py":
            return []
        contracts: Dict[str, Dict] = {}
        # 1. Contract metadata (strike / expiry / type / open_interest)
        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import ContractType, AssetStatus
            from datetime import timedelta
            # An UNWINDOWED request returns only the nearest (often 0-DTE) expiry, which
            # silently fails the Alpha Engine's 21-60 DTE gate. Request the DTE window so
            # the real, tradeable contracts are returned.
            _today = datetime.now(timezone.utc).date()
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status=AssetStatus.ACTIVE,
                type=ContractType.CALL,
                expiration_date_gte=_today + timedelta(days=ALPHA_MIN_DTE),
                expiration_date_lte=_today + timedelta(days=ALPHA_MAX_DTE),
                limit=1000,
            )
            resp = self._client.get_option_contracts(req)
            for c in getattr(resp, "option_contracts", []) or []:
                occ = str(getattr(c, "symbol", ""))
                if not occ:
                    continue
                contracts[occ] = {
                    "symbol": occ,
                    "type": str(getattr(c, "type", "call")).lower().replace("contracttype.", ""),
                    "expiration_date": str(getattr(c, "expiration_date", "")),
                    "strike_price": float(getattr(c, "strike_price", 0) or 0),
                    "open_interest": int(float(getattr(c, "open_interest", 0) or 0)),
                    "volume": 0,   # Alpaca contract metadata carries no daily volume; left to liquidity gate
                    "delta": 0.0,
                    "implied_volatility": 0.0,
                    "bid_price": 0.0,
                    "ask_price": 0.0,
                }
        except Exception as exc:
            logger.debug("get_option_contracts failed for %s: %s", symbol, exc)
            return []
        # 2. Merge live quotes + greeks from the options data snapshot
        if self._opt_data is not None and contracts:
            try:
                from alpaca.data.requests import OptionChainRequest
                chain = self._opt_data.get_option_chain(OptionChainRequest(underlying_symbol=symbol))
                snap_map = chain if isinstance(chain, dict) else getattr(chain, "data", {}) or {}
                for occ, snap in snap_map.items():
                    if occ not in contracts:
                        continue
                    q = getattr(snap, "latest_quote", None)
                    g = getattr(snap, "greeks", None)
                    t = getattr(snap, "latest_trade", None)
                    if q is not None:
                        contracts[occ]["bid_price"] = float(getattr(q, "bid_price", 0) or 0)
                        contracts[occ]["ask_price"] = float(getattr(q, "ask_price", 0) or 0)
                    if g is not None:
                        contracts[occ]["delta"] = float(getattr(g, "delta", 0) or 0)
                    iv = getattr(snap, "implied_volatility", None)
                    if iv is not None:
                        contracts[occ]["implied_volatility"] = float(iv or 0)
                    # Daily volume is not on the contract metadata; take it from the
                    # snapshot's day bar / trade size so the min_volume gate has real data.
                    day = getattr(snap, "daily_bar", None)
                    if day is not None and getattr(day, "volume", None) is not None:
                        contracts[occ]["volume"] = int(getattr(day, "volume", 0) or 0)
                    elif t is not None and getattr(t, "size", None) is not None:
                        contracts[occ]["volume"] = int(getattr(t, "size", 0) or 0)
            except Exception as exc:
                logger.debug("option chain snapshot merge failed for %s: %s", symbol, exc)
        return list(contracts.values())

    # ── order submission ───────────────────────────────────────────────────────
    def _resolve_reference_price(self, symbol: str) -> float:
        """
        Resolve a real per-share/contract price for exposure/notional. Order: latest
        trade → latest quote mid → most recent daily close. Returns 0.0 only when no
        market data is available. Used so RiskManager never gates on qty × 0.0.
        """
        if not self._available or self._mode != "alpaca_py":
            return 0.0
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import (StockLatestTradeRequest,
                                              StockLatestQuoteRequest, StockBarsRequest)
            from alpaca.data.timeframe import TimeFrame
            from datetime import timedelta
            dc = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
            # 1. latest trade
            try:
                t = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
                px = float(getattr(t.get(symbol), "price", 0) or 0)
                if px > 0:
                    return px
            except Exception:
                pass
            # 2. latest quote mid
            try:
                q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol)).get(symbol)
                bid = float(getattr(q, "bid_price", 0) or 0)
                ask = float(getattr(q, "ask_price", 0) or 0)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
                if ask > 0:
                    return ask
            except Exception:
                pass
            # 3. most recent daily close
            try:
                end = datetime.now(timezone.utc)
                req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                       start=end - timedelta(days=7), end=end)
                df = dc.get_stock_bars(req).df
                if df is not None and not df.empty:
                    return float(df["close"].iloc[-1])
            except Exception:
                pass
        except Exception as e:
            logger.debug("_resolve_reference_price(%s) failed: %s", symbol, e)
        return 0.0

    def submit_market_order(self, symbol: str, side: str, qty: int,
                            reason: str = "", limit_price: float = 0.0,
                            order_type: str = "market", reference_price: float = 0.0) -> Optional[Dict]:
        """
        Risk-gated order submission for equities AND options (OCC symbols).
        Options are always routed as DAY limit orders (market options orders are
        commonly rejected); equities use market unless a limit_price is supplied.

        `reference_price` (or limit_price, or a resolved market price) is used ONLY for
        the RiskManager exposure/notional check — it never changes the order TYPE. A BUY
        is rejected when no real reference price can be found, so exposure is never
        computed on qty × 0.0.
        Returns the broker response (or {"sim": True}) on success, None if blocked.
        """
        qty = int(qty)
        if qty <= 0:
            return None
        opt = is_option_symbol(symbol)
        asset_class = "option" if opt else "equity"

        # Notional reference price: explicit limit → caller-supplied reference → live
        # market lookup. Must be > 0 for BUYs (exits never depend on notional).
        ref_price = float(limit_price or reference_price or 0.0)
        if ref_price <= 0:
            ref_price = self._resolve_reference_price(symbol)
        if ref_price <= 0 and side == "buy":
            log_skip(symbol, "rejected_by_data_unavailable:no_reference_price",
                     side=side, qty=qty, source=reason or "broker")
            return None

        allowed, gate_reason = self.risk.gate(symbol, side, qty, ref_price, asset_class=asset_class)
        if not allowed:
            log_skip(symbol, gate_reason, side=side, qty=qty, source=reason or "broker")
            return None

        if not self._available:
            logger.info("[SIM] %s %s x%d  reason=%s", side.upper(), symbol, qty, reason)
            self.risk.record_fill(symbol, side, qty, ref_price, asset_class=asset_class)
            return {"sim": True, "symbol": symbol, "side": side, "qty": qty}

        try:
            if self._mode == "alpaca_py":
                resp = self._submit_alpaca_py(symbol, side, qty, opt, limit_price, order_type)
            else:
                resp = self._client.submit_order(
                    symbol=symbol, qty=qty, side=side,
                    type=("limit" if (opt or limit_price) else "market"),
                    time_in_force="day",
                    limit_price=(limit_price or None),
                )
            logger.info("ORDER %s %s x%d  reason=%s  asset=%s", side, symbol, qty, reason, asset_class)
            self.risk.record_fill(symbol, side, qty, ref_price, asset_class=asset_class)
            return resp
        except Exception as e:
            logger.error("Alpaca order error (%s %s x%d): %s", side, symbol, qty, e)
            return None

    def _submit_alpaca_py(self, symbol, side, qty, opt, limit_price, order_type):
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        os_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        if opt or limit_price or order_type == "limit":
            px = limit_price
            if px <= 0:
                # Best-effort marketable limit for exits without an explicit price
                px = 0.01
            req = LimitOrderRequest(symbol=symbol, qty=qty, side=os_side,
                                    time_in_force=TimeInForce.DAY, limit_price=round(px, 2))
        else:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=os_side,
                                     time_in_force=TimeInForce.DAY)
        return self._client.submit_order(req)

    # Legacy alias (scalping path called submit_order(symbol, side, qty, type, limit))
    def submit_order(self, symbol: str, side: str, qty: int,
                     order_type: str = "market", limit_price: float = 0.0):
        return self.submit_market_order(symbol, side, qty, reason="scalping",
                                        limit_price=limit_price, order_type=order_type)


# ---------------------------------------------------------------------------
# Market hours check
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    now = datetime.now(timezone.utc)
    # NYSE RTH: Mon-Fri 13:30-20:00 UTC
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    mins = h * 60 + m
    return 13 * 60 + 30 <= mins <= 20 * 60


# ---------------------------------------------------------------------------
# Universe / data helpers
# ---------------------------------------------------------------------------

def load_candidate_universe() -> List[str]:
    """
    Load the TradingView institutional-accumulation candidates (the locked
    trading universe). Order of preference:
        1. SYMBOLS env override
        2. {DATA_TAG}_locked_trading_universe.txt
        3. locked_trading_universe.txt
    """
    if SYMBOLS_ENV.strip():
        return [s.strip().upper() for s in SYMBOLS_ENV.split(",") if s.strip()]

    for candidate in (PROJECT_ROOT / f"{DATA_TAG}_locked_trading_universe.txt",
                      PROJECT_ROOT / "locked_trading_universe.txt"):
        if candidate.exists():
            try:
                syms = [ln.strip().upper() for ln in candidate.read_text().splitlines() if ln.strip()]
                if syms:
                    logger.info("Loaded %d candidates from %s", len(syms), candidate.name)
                    return syms
            except Exception as e:
                logger.warning("Failed to read %s: %s", candidate, e)
    logger.warning("No candidate universe found — using default watchlist")
    return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def fetch_1m_bars(gateway: AlpacaGateway, symbols: List[str]) -> Dict[str, "Any"]:
    """
    Fetch recent 1-minute OHLCV bars per symbol as a pandas DataFrame
    (columns: open, high, low, close, volume; tz-aware index).
    Falls back to a deterministic simulation when Alpaca data is unavailable.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed — cannot build bars")
        return {}

    out: Dict[str, Any] = {}
    if gateway.available and gateway._mode == "alpaca_py":
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import DataFeed
            from datetime import timedelta
            data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=ALPHA_BARS_LOOKBACK_MIN + 5)
            feed = DataFeed.SIP if os.getenv("ALPACA_DATA_FEED", "iex").lower() == "sip" else DataFeed.IEX
            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                                   start=start, end=end, feed=feed)
            df = data_client.get_stock_bars(req).df
            if df is not None and not df.empty:
                df = df.reset_index()
                for sym, grp in df.groupby("symbol"):
                    g = grp.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
                    out[sym] = g
                return out
        except Exception as e:
            logger.warning("Alpaca 1m bar fetch failed (%s) — using simulation", e)

    # Simulation fallback (deterministic per-symbol so the loop stays runnable offline)
    import pandas as pd
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    for sym in symbols:
        seed = sum(ord(c) for c in sym)
        base = 20.0 + (seed % 200)
        idx = [now - timedelta(minutes=ALPHA_BARS_LOOKBACK_MIN - i) for i in range(ALPHA_BARS_LOOKBACK_MIN)]
        rows = []
        px = base
        for i in range(ALPHA_BARS_LOOKBACK_MIN):
            drift = math.sin((seed + i) / 17.0) * 0.15
            px = max(1.0, px + drift)
            rows.append({"open": px, "high": px * 1.003, "low": px * 0.997,
                         "close": px, "volume": 50_000 + (seed * 7 + i) % 40_000})
        out[sym] = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="timestamp"))
    return out


def build_bars_ctx(bars_by_symbol: Dict[str, "Any"]) -> Dict[str, Dict]:
    """Derive per-symbol VWAP and relative-volume context for BAL technical scoring."""
    ctx: Dict[str, Dict] = {}
    for sym, df in bars_by_symbol.items():
        try:
            if df is None or len(df) == 0:
                continue
            close = df["close"]
            vol = df["volume"]
            typical = (df["high"] + df["low"] + df["close"]) / 3.0
            vwap = float((typical * vol).sum() / max(1e-9, vol.sum()))
            recent_v = float(vol.tail(20).mean())
            avg_v = float(vol.mean())
            rvol = recent_v / max(1e-9, avg_v)
            ctx[sym] = {
                "vwap": vwap,
                "rel_volume": round(rvol, 3),
                "rvol": round(rvol, 3),
                "close": float(close.iloc[-1]),
            }
        except Exception:
            continue
    return ctx


# ---------------------------------------------------------------------------
# Required output writers
# ---------------------------------------------------------------------------

def write_capital_waterfall(waterfall) -> None:
    """Persist the capital waterfall summary (engine only writes on allocate)."""
    try:
        from capital_waterfall import WATERFALL_PATH
        payload = dict(waterfall.summary)
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["config"] = {
            "to_bal_pct": OPTIONS_PROFIT_TO_BAL_PCT,
            "to_alpha_pct": OPTIONS_PROFIT_TO_ALPHA_PCT,
            "to_cash_pct": OPTIONS_PROFIT_TO_CASH_PCT,
        }
        WATERFALL_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug("write_capital_waterfall failed: %s", e)


def ensure_outputs_exist() -> None:
    """Guarantee the four required daily outputs exist (empty scaffolds if engines haven't run)."""
    DAILY_OUTPUTS.mkdir(parents=True, exist_ok=True)
    scaffolds = {
        "options_alpha_signals.json":      {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "signals": []},
        "buffett_accumulation_scores.json": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "scores": []},
        "buffett_dca_signals.json":        {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "signals": []},
        "valuation_diagnostic.json":       {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "valuations": []},
        "conviction_score_report.json":    {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "convictions": []},
        "alpha_volatility_diagnostic.json": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "count": 0, "diagnostics": []},
        "sec_form4_insider_buys.json":     {"generated_at": datetime.now(timezone.utc).isoformat(),
                                            "lookback_days": 0, "min_transaction_value": 0,
                                            "candidates": [], "rejected": [], "errors": [],
                                            "source": "SEC_EDGAR_FORM4"},
        "insider_buy_candidates.json":     {"generated_at": datetime.now(timezone.utc).isoformat(),
                                            "source": "SEC_EDGAR_FORM4", "count": 0, "candidates": {}},
        "ownership_universe.json":         {},
        "capital_waterfall.json":          {"date_tag": DATA_TAG, "realized_today": 0.0,
                                            "bal_reserve": 0.0, "alpha_reserve": 0.0, "cash_reserve": 0.0,
                                            "allocation_today": {"bal": 0.0, "alpha": 0.0, "cash": 0.0}},
        "alpha_options_universe.json":     {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "scanned": 0, "candidate_count": 0,
                                            "candidates": [], "all_scanned": []},
        "buffett_accumulation_universe.json": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                            "date_tag": DATA_TAG, "scanned": 0, "candidate_count": 0,
                                            "candidates": [], "all_scanned": []},
        "screener_funnel_report.json":     {"date_tag": DATA_TAG,
                                            "alpha_options_screener": {}, "buffett_accumulation_screener": {}},
    }
    for fname, default in scaffolds.items():
        p = DAILY_OUTPUTS / fname
        if not p.exists():
            try:
                p.write_text(json.dumps(default, indent=2), encoding="utf-8")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Engine cycle helpers
# ---------------------------------------------------------------------------

def regime_allows_new_buys(market_context: dict) -> Tuple[bool, str]:
    """Top-level regime gate shared by both engines for NEW buys (exits always allowed)."""
    perm = str(market_context.get("trade_permission") or "normal").lower()
    if perm == "halt_new_buys":
        return False, "rejected_by_market_regime:halt_new_buys"
    bias = str(market_context.get("overall_market_bias") or "").lower()
    if any(b in bias for b in ("crisis", "crash", "systemic_risk")):
        return False, f"rejected_by_macro:{bias}"
    return True, ""


def _alpha_waterfall_cb(waterfall):
    def cb(profit, source="alpha_call"):
        try:
            waterfall.allocate(float(profit), source=source)
        except Exception as e:
            logger.debug("waterfall.allocate failed: %s", e)
    return cb


# ── Research layer (Evidence System) — fully fail-safe, never affects trading ─────
def _research(method: str, *args, **kwargs) -> None:
    """Call a portfolio_research_lab recorder/report fn, swallowing any error."""
    try:
        import portfolio_research_lab as prl
        getattr(prl, method)(*args, **kwargs)
    except Exception as e:
        logger.debug("research layer (%s) skipped: %s", method, e)


# ── Alpha: EXIT pass (runs every cycle, independent of regime/entries/screeners) ──
def run_alpha_exits(engine, gateway: AlpacaGateway, waterfall,
                    option_positions: List[Dict], market_context: dict) -> int:
    """Manage open calls via the engine's public exit pass. ALWAYS runs."""
    try:
        exits = engine.check_active_positions(option_positions, _alpha_waterfall_cb(waterfall),
                                              market_context=market_context)
    except Exception as e:
        logger.error("alpha check_active_positions failed: %s", e)
        exits = []
    for ex in exits:
        gateway.submit_market_order(ex["symbol"], ex["side"], ex["qty"],
                                    reason=ex.get("source", "alpha_exit"))
        _research("record_alpha_exit_safe", ex, market_context)   # evidence: realized result
    return len(exits)


# ── Alpha: ENTRY pass (engine owns all screening / contract / BS decisions) ───────
def run_alpha_entries(engine, gateway: AlpacaGateway, waterfall, risk: RiskManager,
                      symbols: List[str], bars_by_symbol: Dict[str, Any],
                      chains_by_symbol: Dict[str, list], market_context: dict,
                      positions: Optional[dict] = None) -> int:
    """
    Hand PRE-FETCHED data to the engine; it writes alpha_options_universe.json and
    returns order intents. run.py only routes intents through Risk/Gateway.
    waterfall_budget=0 so reserves never cap tactical sizing (primary engine).
    """
    try:
        signals = engine.evaluate_universe(
            symbols=symbols, bars_by_symbol=bars_by_symbol, market_context=market_context,
            equity=risk.equity, options_exposure=risk.options_exposure, waterfall_budget=0.0,
            chains_by_symbol=chains_by_symbol, positions=positions,
            account_state={"equity": risk.equity, "buying_power": risk.buying_power},
        )
    except Exception as e:
        logger.error("alpha evaluate_universe failed: %s", e)
        return 0
    placed = 0
    for sig in signals:
        if sig.status != "ready":
            if sig.reject_reason:
                log_skip(sig.symbol, sig.reject_reason, engine="alpha",
                         grade=sig.grade, score=sig.setup_score)
                _research("record_alpha_signal_safe", getattr(sig, "__dict__", {}), market_context, False)
            continue
        limit_px = sig.ask or sig.mid
        resp = gateway.submit_market_order(sig.contract_symbol, "buy", sig.contracts,
                                           reason=f"alpha_call:{sig.setup_type}:{sig.grade}",
                                           limit_price=limit_px, order_type="limit")
        if resp is not None:
            placed += 1
            _research("record_alpha_signal_safe", getattr(sig, "__dict__", {}), market_context, True)
            try:
                engine.register_filled(sig)
                waterfall.consume_alpha(sig.premium_cost)
            except Exception as e:
                logger.debug("register_filled/consume_alpha failed: %s", e)
    return placed


# ── Alpha: screener orchestration (I/O lives in run.py; intelligence in engine) ───
def run_alpha_screener(engine, gateway: AlpacaGateway, market_context: dict,
                       dynamic_symbols: List[str]) -> dict:
    """
    run.py fetches data; the ENGINE decides the universe. Returns the base universe,
    bars, pre-fetched chains (watch list only) and the screen funnel. No alpha logic
    is performed here — only fetching + delegation to engine.screen_alpha_options_universe.
    """
    from alpha_call_engine import build_alpha_universe
    base = build_alpha_universe(dynamic_symbols)
    bars = fetch_1m_bars(gateway, base)
    if ENABLE_ALPHA_OPTIONS_SCREENER:
        screen = engine.screen_alpha_options_universe(base, bars, market_context)
        watch, funnel = screen["watch"], screen["funnel"]
    else:
        watch = base
        funnel = {"scanned": len(base), "passed_momentum": len(base), "note": "screener_disabled"}
    # Fetch option chains ONLY for the watch list (fetching, not filtering).
    chains: Dict[str, list] = {}
    for sym in watch:
        try:
            chains[sym] = gateway.get_options_chain(sym) or []
        except Exception as e:
            logger.debug("chain fetch failed for %s: %s", sym, e)
            chains[sym] = []
    return {"base": base, "bars": bars, "chains": chains, "watch": watch, "funnel": funnel}


# ── BAL: EXIT pass (runs every cycle) ─────────────────────────────────────────────
def run_bal_exits(bal, gateway: AlpacaGateway, equity_positions: Dict[str, Any],
                  market_context: dict) -> int:
    n = 0
    for sym, pos in list(equity_positions.items()):
        try:
            should_exit, reason = bal.should_exit(sym, pos, market_context)
        except Exception as e:
            logger.debug("BAL should_exit failed for %s: %s", sym, e)
            continue
        if should_exit:
            qty = int(abs(float(getattr(pos, "qty", 0) or 0)))
            if qty > 0:
                gateway.submit_market_order(sym, "sell", qty, reason=f"bal_exit:{reason}")
                n += 1
    return n


# ── BAL: ENTRY pass (long-term accumulation) ──────────────────────────────────────
def run_bal_entries(bal, gateway: AlpacaGateway, waterfall, risk: RiskManager,
                    symbols: List[str], market_context: dict,
                    bars_ctx_by_symbol: Dict[str, Dict], equity_positions: Dict[str, Any],
                    insider_data_by_symbol: Optional[Dict[str, Dict]] = None) -> int:
    if not symbols:
        return 0
    stock_room = max(0.0, risk.equity * MAX_STOCK_EXPOSURE_PCT - risk.stock_exposure)
    spendable_bp = max(0.0, risk.buying_power * (1.0 - CASH_BUFFER_PCT))
    available_capital = min(spendable_bp, stock_room)
    if available_capital < 200:
        return 0
    # Composite DCA framework is the accumulation mechanism when enabled (value decides
    # WHETHER, RSI decides WHEN, tranches decide HOW MUCH); legacy score-based scan is
    # the fallback. SEC Form 4 insider data is a CONVICTION MODIFIER only — it can never
    # generate an order; every intent still routes through RiskManager/AlpacaGateway.
    use_dca = ENABLE_BAL_COMPOSITE_DCA and hasattr(bal, "evaluate_dca")
    try:
        if use_dca:
            orders = bal.evaluate_dca(
                symbols=symbols, market_context=market_context,
                bars_ctx_by_symbol=bars_ctx_by_symbol, equity=risk.equity,
                positions=equity_positions, available_capital=available_capital,
                insider_data_by_symbol=insider_data_by_symbol or {},
            )
        else:
            orders = bal.scan(
                symbols=symbols, market_context=market_context,
                bars_ctx_by_symbol=bars_ctx_by_symbol, equity=risk.equity,
                positions=equity_positions, available_capital=available_capital,
            )
    except Exception as e:
        logger.error("BAL %s failed: %s", "DCA" if use_dca else "scan", e)
        return 0
    placed = 0
    for o in orders:
        # Pass the BAL-evaluated price as the notional reference so RiskManager gates on
        # real exposure (never qty × 0.0). It does NOT make this a limit order.
        resp = gateway.submit_market_order(o["symbol"], "buy", int(o["qty"]),
                                           reason=o.get("reason", "buffett_accumulation"),
                                           reference_price=float(o.get("price", 0.0) or 0.0))
        if resp is not None:
            placed += 1
            _research("record_bal_order_safe", o, market_context)   # evidence: tranche/conviction
            try:
                waterfall.consume_bal(float(o.get("order_dollars", 0.0)))
            except Exception:
                pass
    # Evidence: record BAL rejections from the just-written DCA signals file.
    try:
        from buffett_accumulation_layer import BUFFETT_DCA_SIGNALS_PATH
        sigs = json.loads(BUFFETT_DCA_SIGNALS_PATH.read_text()).get("signals", [])
        for s in sigs:
            if not s.get("risk_limits_passed", False):
                _research("record_bal_rejection_safe", s, market_context)
    except Exception:
        pass
    return placed


# ── Buffett screener orchestration (fully isolated from Alpha) ────────────────────
def run_buffett_screener(market_context: dict) -> dict:
    """Guarded: any failure returns empty candidates so Alpha exits/entries are never affected."""
    if not ENABLE_BUFFETT_ACCUMULATION_SCREENER:
        return {"candidates": [], "funnel": {"note": "screener_disabled"}, "detail": []}
    try:
        import buffett_accumulation_screener as bscr
        return bscr.screen(market_context)
    except Exception as e:
        logger.error("Buffett screener failed (Alpha unaffected): %s", e)
        return {"candidates": [], "funnel": {"error": str(e)[:160]}, "detail": []}


def write_screener_funnel_report(alpha_screen_funnel: dict, buffett_res: dict,
                                 market_context: dict) -> None:
    """Combined funnel diagnostics for both screeners."""
    alpha_candidates = 0
    alpha_liquidity_rejects = 0
    try:
        from alpha_call_engine import ALPHA_OPTIONS_UNIVERSE_PATH
        if ALPHA_OPTIONS_UNIVERSE_PATH.exists():
            au = json.loads(ALPHA_OPTIONS_UNIVERSE_PATH.read_text())
            alpha_candidates = au.get("candidate_count", 0)
            alpha_liquidity_rejects = sum(
                1 for r in au.get("all_scanned", [])
                if str(r.get("reject_reason") or "").startswith("rejected_by_options_liquidity"))
    except Exception:
        pass
    bf = buffett_res.get("funnel", {})
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": DATA_TAG,
        "market_regime": str(market_context.get("overall_market_bias")),
        "alpha_options_screener": {
            "total_scanned": alpha_screen_funnel.get("scanned", 0),
            "passed_alpha_options_screener": alpha_screen_funnel.get("passed_momentum", 0),
            "rejected_by_regime": alpha_screen_funnel.get("rejected_by_regime", 0),
            "rejected_by_momentum": alpha_screen_funnel.get("rejected_by_momentum", 0),
            "rejected_by_option_liquidity": alpha_liquidity_rejects,
            "final_candidates_sent_to_alpha_engine": alpha_candidates,
        },
        "buffett_accumulation_screener": {
            "total_scanned": bf.get("scanned", 0),
            "passed_buffett_screener": bf.get("passed", 0),
            "rejected_by_fundamentals": bf.get("rejected_by_fundamentals", 0),
            "rejected_by_valuation": bf.get("rejected_by_valuation", 0),
            "rejected_by_macro_cycle": bf.get("rejected_by_macro_cycle", 0),
            "final_candidates_sent_to_bal": len(buffett_res.get("candidates", [])),
        },
        "universes_independent": True,
    }
    try:
        SCREENER_FUNNEL_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug("write_screener_funnel_report failed: %s", e)


def run_portfolio_monitor(monitor, gateway: AlpacaGateway, risk: RiskManager,
                          equity_positions: Dict[str, Any], market_context: dict) -> dict:
    """Optional BAL portfolio monitor / hedging (opt-in via ENABLE_BAL_PORTFOLIO_MONITOR)."""
    try:
        return monitor.evaluate(
            positions=equity_positions,
            equity=risk.equity,
            market_context=market_context,
            broker=gateway,
        )
    except Exception as e:
        logger.error("BAL portfolio monitor failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Two-engine trading loop
# ---------------------------------------------------------------------------

def build_engines(gateway: AlpacaGateway):
    """Instantiate engines + waterfall. Returns (alpha, bal, monitor, waterfall)."""
    alpha = bal = monitor = waterfall = None
    try:
        from capital_waterfall import CapitalWaterfall, WaterfallConfig
        waterfall = CapitalWaterfall(WaterfallConfig())
    except Exception as e:
        logger.error("CapitalWaterfall init failed: %s", e)
    if ENABLE_ALPHA_CALL_ENGINE:
        try:
            from alpha_call_engine import AlphaCallEngine, AlphaCallConfig
            alpha = AlphaCallEngine(broker=gateway, config=AlphaCallConfig())
            logger.info("Alpha Call Engine ENABLED")
        except Exception as e:
            logger.error("AlphaCallEngine init failed: %s", e)
    if ENABLE_BUFFETT_ACCUMULATION_LAYER:
        try:
            from buffett_accumulation_layer import BuffettAccumulationLayer, BALConfig, DCAConfig
            bal = BuffettAccumulationLayer(config=BALConfig(), dca_config=DCAConfig())
            logger.info("Buffett Accumulation Layer ENABLED (composite_dca=%s)", ENABLE_BAL_COMPOSITE_DCA)
        except Exception as e:
            logger.error("BuffettAccumulationLayer init failed: %s", e)
    if ENABLE_BAL_PORTFOLIO_MONITOR:
        try:
            from buffett_accumulation_layer import BALPortfolioMonitor, HedgeConfig, BALConfig
            monitor = BALPortfolioMonitor(hedge_config=HedgeConfig(), bal_config=BALConfig())
            logger.info("BAL Portfolio Monitor ENABLED")
        except Exception as e:
            logger.error("BALPortfolioMonitor init failed: %s", e)
    return alpha, bal, monitor, waterfall


def run_two_engine_cycle(risk: RiskManager, gateway: AlpacaGateway, waterfall,
                         alpha, bal, monitor, market_context: dict) -> dict:
    """
    One orchestration iteration. ORDER (hard requirement):
        sync  →  ALPHA EXITS  →  BAL EXITS  →  screeners  →  ALPHA ENTRIES  →  BAL ENTRIES
    Exits ALWAYS run (independent of regime / entries-enabled / screener success).
    Buffett work is fully wrapped so its failure never affects Alpha exits or entries.
    run.py performs ONLY fetching + execution; all Alpha intelligence is in the engine.
    """
    # 1. Sync account / positions / exposure
    risk.sync_account(gateway.get_account())
    raw_positions = gateway.get_positions_raw()
    option_positions, equity_positions, opt_exp, stk_exp = gateway.split_positions(raw_positions)
    risk.sync_exposure(opt_exp, stk_exp)
    new_buys_allowed, regime_reason = regime_allows_new_buys(market_context)

    summary = {"alpha_exits": 0, "bal_exits": 0, "alpha_entries": 0, "bal_entries": 0,
               "new_buys_allowed": new_buys_allowed}

    # 2. EXIT PASS — unconditional (regime/entries/screeners cannot block exits) ─────
    if alpha is not None:
        summary["alpha_exits"] = run_alpha_exits(alpha, gateway, waterfall, option_positions, market_context)
    if bal is not None:
        try:
            summary["bal_exits"] = run_bal_exits(bal, gateway, equity_positions, market_context)
        except Exception as e:
            logger.error("BAL exits failed (Alpha unaffected): %s", e)

    # 3. SCREENERS — separate, independent universes. Buffett fully isolated. ────────
    dynamic = load_candidate_universe()          # TradingView institutional dynamic discovery
    alpha_scr = {"base": [], "bars": {}, "chains": {}, "funnel": {}}
    if alpha is not None:
        try:
            alpha_scr = run_alpha_screener(alpha, gateway, market_context, dynamic)
        except Exception as e:
            logger.error("Alpha screener failed: %s", e)
    # SEC Form 4 insider-buy discovery: DISCOVERY ONLY. Expands the Buffett review
    # universe; never generates orders, never bypasses any gate. Fully fail-safe —
    # a failure here must not stop the Buffett screener or the Alpha engine.
    insider_data: Dict[str, Dict] = {}
    if bal is not None and ENABLE_SEC_FORM4_FEED:
        try:
            import sec_form4_feed
            sec_res = sec_form4_feed.run_sec_form4_discovery()
            insider_data = sec_form4_feed.insider_data_for_layer()
            for _cand in sec_res.get("candidates", []):
                _research("record_insider_discovery_safe", _cand)   # evidence: discoveries
            logger.info("SEC Form 4 discovery: %d candidates, %d errors",
                        len(sec_res.get("candidates", [])), len(sec_res.get("errors", [])))
        except Exception as e:
            logger.error("SEC Form 4 discovery failed (Buffett/Alpha unaffected): %s", e)
    buffett_res = run_buffett_screener(market_context) if bal is not None else \
        {"candidates": [], "funnel": {}, "detail": []}

    # 4. ENTRY PASS — after all exits. Alpha first, then BAL. ────────────────────────
    if not new_buys_allowed:
        log_skip("UNIVERSE", regime_reason, phase="entry_gate")
    # Alpha entries ALWAYS call evaluate_universe so the universe file is written; the
    # ENGINE rejects entries internally when the regime is hostile (engine setup gate).
    if alpha is not None:
        summary["alpha_entries"] = run_alpha_entries(
            alpha, gateway, waterfall, risk,
            alpha_scr["base"], alpha_scr["bars"], alpha_scr["chains"],
            market_context, positions=equity_positions)
    # BAL buys are hard-gated on regime permission (halt_new_buys / crisis).
    if bal is not None and new_buys_allowed:
        try:
            bcands = buffett_res.get("candidates", [])
            bal_bars = fetch_1m_bars(gateway, bcands) if bcands else {}
            bal_ctx = build_bars_ctx(bal_bars)
            summary["bal_entries"] = run_bal_entries(bal, gateway, waterfall, risk,
                                                     bcands, market_context, bal_ctx, equity_positions,
                                                     insider_data_by_symbol=insider_data)
        except Exception as e:
            logger.error("BAL entries failed (Alpha unaffected): %s", e)

    # 5. Optional monitor + outputs
    if monitor is not None:
        run_portfolio_monitor(monitor, gateway, risk, equity_positions, market_context)
    if waterfall is not None:
        write_capital_waterfall(waterfall)
    write_screener_funnel_report(alpha_scr.get("funnel", {}), buffett_res, market_context)
    # 6. Evidence System — regenerate research analytics from the recorded events.
    bal_value = sum(float(getattr(p, "market_value", 0) or 0) for p in equity_positions.values())
    _research("generate_all_reports_safe", market_context=market_context,
              start_equity=STARTING_EQUITY, bal_current_value=bal_value)
    # 7. Adaptive Capital Allocation — turn the evidence into an allocation PLAN.
    # Advisory only: it never submits orders, never overrides RiskManager, and all
    # recommended sizes are clamped to the configured caps. Fully fail-safe.
    if ENABLE_ADAPTIVE_ALLOCATION:
        try:
            import adaptive_capital_allocator as aca
            aca.build_allocation_plan_safe(market_context=market_context)
        except Exception as e:
            logger.debug("adaptive allocator skipped: %s", e)
    return summary


def run_two_engine_trading():
    logger.info("=== TWO-ENGINE TRADING  mode=%s ===", MODE)
    _log_config()

    risk    = RiskManager(equity=STARTING_EQUITY)
    gateway = AlpacaGateway(risk)
    ensure_outputs_exist()
    alpha, bal, monitor, waterfall = build_engines(gateway)

    if alpha is None and bal is None:
        logger.critical("No engines enabled — nothing to do. Exiting.")
        return

    market_context: dict = {}
    last_context_build = 0.0

    logger.info("Entering two-engine loop (Ctrl-C to stop) …")
    try:
        while True:
            if risk.is_killed:
                logger.critical("Kill switch active — exiting loop")
                break
            if not is_market_open():
                logger.debug("Market closed — sleeping 60s")
                time.sleep(60)
                continue
            if (time.time() - last_context_build) > CONTEXT_REFRESH_MIN * 60 or not market_context:
                market_context = _build_context_safe()
                last_context_build = time.time()

            run_two_engine_cycle(risk, gateway, waterfall, alpha, bal, monitor, market_context)
            time.sleep(BAR_INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("Interrupted by user — finalising outputs")
    finally:
        if waterfall is not None:
            write_capital_waterfall(waterfall)
        ensure_outputs_exist()
        logger.info("Two-engine session complete. Risk: equity=%.2f daily_pnl=%.2f opt_exp=%.0f stk_exp=%.0f",
                    risk.equity, risk.daily_pnl, risk.options_exposure, risk.stock_exposure)


def _build_context_safe() -> dict:
    try:
        from market_context_builder import build_market_context
        ctx = build_market_context()
        logger.info("Market context: bias=%s permission=%s risk=%s",
                    ctx.get("overall_market_bias"), ctx.get("trade_permission"), ctx.get("risk_level"))
        return ctx or {}
    except Exception as e:
        logger.error("build_market_context failed (%s) — defaulting to cautious context", e)
        return {"overall_market_bias": "mixed", "trade_permission": "cautious", "risk_level": "high"}


def _log_config() -> None:
    logger.info("Risk limits: daily_loss=%.0f max_dd=%.1f%% max_pos=%d equity_floor=%.0f max_order=%d",
                DAILY_LOSS_LIMIT, MAX_DRAWDOWN * 100, MAX_POSITIONS, EQUITY_FLOOR, MAX_ORDER_SIZE)
    logger.info("Engines: alpha=%s bal=%s monitor=%s | paper_only=%s",
                ENABLE_ALPHA_CALL_ENGINE, ENABLE_BUFFETT_ACCUMULATION_LAYER,
                ENABLE_BAL_PORTFOLIO_MONITOR, ALPACA_PAPER_ONLY)
    logger.info("Exposure: options<=%.0f%% stock<=%.0f%% single_stock<=%.0f%% opt_risk/trade=%.1f%% cash_buffer=%.0f%%",
                MAX_OPTIONS_EXPOSURE_PCT * 100, MAX_STOCK_EXPOSURE_PCT * 100,
                MAX_SINGLE_STOCK_WEIGHT * 100, OPTIONS_RISK_PER_TRADE_PCT * 100, CASH_BUFFER_PCT * 100)
    logger.info("BAL thresholds: min_score=%.0f aggressive=%.0f | Waterfall: bal=%.0f%% alpha=%.0f%% cash=%.0f%%",
                BAL_MIN_SCORE_TO_BUY, BAL_AGGRESSIVE_SCORE,
                OPTIONS_PROFIT_TO_BAL_PCT * 100, OPTIONS_PROFIT_TO_ALPHA_PCT * 100, OPTIONS_PROFIT_TO_CASH_PCT * 100)
    logger.info("BAL composite DCA: enabled=%s (value=WHETHER, RSI=WHEN, tranches=HOW MUCH, RiskManager=ALLOWED)",
                ENABLE_BAL_COMPOSITE_DCA)


# ---------------------------------------------------------------------------
# Training orchestrator  (preserved)
# ---------------------------------------------------------------------------

def run_training():
    logger.info("=== TRAINING MODE ===")
    from rl_intraday_structure_trainer import train as train_intraday
    from tune_rl_intraday             import run_tuning as tune_intraday
    from sr_rl_agent                  import train as train_sr
    from tune_sr_rl_agent             import run_tuning as tune_sr

    logger.info("--- Step 1: Train Intraday RL ---");  train_intraday()
    logger.info("--- Step 2: Tune Intraday RL ---");   tune_intraday()
    logger.info("--- Step 3: Train SR RL ---");        train_sr()
    logger.info("--- Step 4: Tune SR RL ---");         tune_sr()
    logger.info("=== TRAINING COMPLETE ===")


# ---------------------------------------------------------------------------
# Legacy scalping loop  (preserved — run with MODE=scalp)
# ---------------------------------------------------------------------------

def run_scalping():
    logger.info("=== SCALPING MODE  symbols=%s ===", SYMBOLS_ENV or "default")
    symbols = load_candidate_universe()

    risk    = RiskManager(equity=STARTING_EQUITY)
    gateway = AlpacaGateway(risk)

    def order_callback(symbol, side, qty, order_type, limit_price):
        gateway.submit_order(symbol, side, qty, order_type, limit_price)

    from trading_day_monitor_agent import TradingDayMonitorAgent

    monitor = TradingDayMonitorAgent(
        symbols          = symbols,
        starting_equity  = STARTING_EQUITY,
        session_csv      = "logs/session_summary.csv",
    )
    for sym in symbols:
        if sym in getattr(monitor, "_scalpers", {}):
            monitor._scalpers[sym].order_cb = lambda **kw: order_callback(**kw)

    logger.info("Entering scalping loop (Ctrl-C to stop) …")
    try:
        while True:
            if risk.is_killed:
                logger.critical("Kill switch active — exiting loop")
                break
            if not is_market_open():
                time.sleep(60)
                continue
            acc = gateway.get_account()
            if acc.get("equity"):
                risk.equity = float(acc["equity"])
            bars = _fetch_latest_bars(symbols)
            for symbol, bar in bars.items():
                monitor.update(symbol, bar)
            time.sleep(BAR_INTERVAL_SEC)
    except KeyboardInterrupt:
        logger.info("Interrupted by user — flushing session")
    finally:
        monitor.flush_session()
        s = monitor.session_summary()
        logger.info("Session: %s", json.dumps(s, default=str))


def _fetch_latest_bars(symbols: List[str]) -> Dict[str, Dict]:
    """Stub feed for the legacy scalping path (deterministic-ish single bars)."""
    import random
    bars = {}
    for sym in symbols:
        p = {"CRM": 190.0, "TE": 10.0, "RKLB": 140.0, "PLUG": 3.9}.get(sym, 50.0)
        p += random.gauss(0, 0.2)
        bars[sym] = dict(
            timestamp   = datetime.now(timezone.utc).isoformat(),
            open=p, high=p + random.uniform(0.05, 0.3), low=p - random.uniform(0.05, 0.4),
            close=p + random.gauss(0, 0.1), volume=random.uniform(3e4, 3e5),
            volume_ma20=1e5, vwap=p * 0.999, ema9=p * 0.998, ema20=p * 0.996,
        )
        bars[sym]["close"] = max(bars[sym]["close"], bars[sym]["low"])
    return bars


# ===========================================================================
# Legacy compatibility surface
# ===========================================================================
#
# The two-engine architecture above is the live system. The symbols below are a
# thin compatibility layer for the intraday-bracket / premarket-recovery test
# suite (tests/test_runtime_recovery.py, tests/test_first_candle_rule.py,
# tests/test_live_experience_logging.py). They form an INDEPENDENT execution path
# (its own BrokerPosition + bracket broker abstraction) that is NOT wired into
# run_two_engine_cycle — so the Alpha/Buffett separation, RiskManager, and
# AlpacaGateway are untouched. The heavy classes live in production_executor.py;
# only the small patchable hooks live here so `patch.object(run, ...)` resolves.

from dataclasses import dataclass as _dataclass

LOG_DIR = PROJECT_ROOT / "logs" / "execution" / DATA_TAG
LIVE_EXPERIENCE_JSONL_PATH   = LOG_DIR / f"{DATA_TAG}_live_experience_replay.jsonl"
LIVE_EXPERIENCE_CSV_PATH     = LOG_DIR / f"{DATA_TAG}_live_experience_replay.csv"
LIVE_EXPERIENCE_PARQUET_PATH = LOG_DIR / f"{DATA_TAG}_live_experience_replay.parquet"

# Whether bracket (OCO) order submission is available. Real systems flip this
# based on broker capability; tests patch it. Default on.
_BRACKET_ORDERS_AVAILABLE = True

# Numeric fields coerced to float in the live-experience replay store (empty
# string / None → NaN) so the parquet schema stays numeric for offline training.
_LIVE_NUMERIC_FIELDS = (
    "account_equity_before", "account_equity_after", "filled_avg_price",
    "filled_qty", "order_qty", "price", "stop_price", "target_price",
    "capped_dollars", "slippage_pct",
)


@_dataclass
class BrokerPosition:
    """Minimal broker position snapshot used by the legacy bracket execution path."""
    symbol: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    market_value: float = 0.0
    current_price: float = 0.0
    unrealized_plpc: float = 0.0


def write_event(event: str, payload: Optional[dict] = None) -> None:
    """Best-effort execution-event journal line. Patched in tests."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{DATA_TAG}_execution_events.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": event, "payload": payload or {},
            }, default=str) + "\n")
    except Exception:
        pass


def save_df(df: "Any", path: "Any") -> None:
    """Best-effort DataFrame → CSV writer. Patched in tests."""
    try:
        if df is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        pass


def log_executed_order(**kwargs) -> None:
    """Best-effort executed-order log. Patched in tests."""
    try:
        write_event("executed_order", kwargs)
    except Exception:
        pass


def row_has_complete_bracket_plan(row: "Any") -> bool:
    """
    True when an order row carries a complete long bracket plan: a buy with a
    positive entry price, a stop below it, and a target above it. Accepts a dict
    or a pandas Series (both expose .get).
    """
    def _num(v) -> float:
        try:
            f = float(v)
            return 0.0 if math.isnan(f) else f
        except (TypeError, ValueError):
            return 0.0

    get = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
    side = str(get("side") or "").lower()
    if side and side != "buy":
        return False
    price = _num(get("price")) or _num(get("entry_price"))
    stop = _num(get("stop_price"))
    target = _num(get("target_price"))
    if price <= 0 or stop <= 0 or target <= 0:
        return False
    return stop < price < target


def append_live_experience(row: dict, write_parquet: bool = True) -> None:
    """
    Append a live decision/execution sample to the replay store (jsonl + csv +
    optional parquet) for offline ML training. Empty strings become NaN for the
    declared numeric fields so the parquet schema stays numeric. Does NOT train
    here. Reads the LIVE_EXPERIENCE_* paths from this module so tests can patch
    them; parquet accumulates across calls.
    """
    import pandas as pd
    rec: dict = {}
    for key, value in dict(row).items():
        key = str(key)
        if key in _LIVE_NUMERIC_FIELDS:
            if value == "" or value is None:
                rec[key] = float("nan")
            else:
                try:
                    rec[key] = float(value)
                except (TypeError, ValueError):
                    rec[key] = float("nan")
        else:
            rec[key] = None if value == "" else value

    try:
        LIVE_EXPERIENCE_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LIVE_EXPERIENCE_JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass

    df = pd.DataFrame([rec])
    try:
        write_header = not LIVE_EXPERIENCE_CSV_PATH.exists()
        df.to_csv(LIVE_EXPERIENCE_CSV_PATH, mode="a", header=write_header, index=False)
    except Exception:
        pass

    if write_parquet:
        try:
            if LIVE_EXPERIENCE_PARQUET_PATH.exists():
                old = pd.read_parquet(LIVE_EXPERIENCE_PARQUET_PATH)
                pd.concat([old, df], ignore_index=True).to_parquet(
                    LIVE_EXPERIENCE_PARQUET_PATH, index=False)
            else:
                df.to_parquet(LIVE_EXPERIENCE_PARQUET_PATH, index=False)
        except Exception as exc:
            # Parquet is optional (pyarrow/fastparquet may be absent).
            write_event("live_experience_parquet_failed", {"error": str(exc)[:200]})


# Re-export the heavy legacy execution classes so `run.X` resolves for the tests.
# Imported at module end (after the hooks above are defined) to avoid a circular
# import — production_executor reaches the hooks lazily via `import run`.
try:
    from production_executor import (  # noqa: E402
        ScannerConfig, ExecutorConfig, build_default_config,
        MomentumSetupScanner, ProductionOrderExecutor,
    )
except Exception as _exc:   # pragma: no cover - keeps run.py importable if module missing
    logger.debug("production_executor compat import failed: %s", _exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    os.makedirs("logs",    exist_ok=True)
    os.makedirs("models",  exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("data",    exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("logs/run.log", mode="a")],
    )

    logger.info("QuantumQuant  mode=%s  paper_only=%s  data_tag=%s", MODE, ALPACA_PAPER_ONLY, DATA_TAG)

    if MODE == "train":
        run_training()
    elif MODE == "scalp":
        run_scalping()
    elif MODE in ("paper", "live"):
        if MODE == "live" and ALPACA_PAPER_ONLY:
            logger.critical("Live mode requested but ALPACA_PAPER_ONLY=true — set to false explicitly")
            sys.exit(1)
        run_two_engine_trading()
    else:
        logger.error("Unknown MODE=%s  (use: train | paper | live | scalp)", MODE)
        sys.exit(1)


if __name__ == "__main__":
    main()
