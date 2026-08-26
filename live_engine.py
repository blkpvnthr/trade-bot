#!/usr/bin/env python3
"""
V2.9.8 live execution engine -- the ONE authoritative orchestration path.
=========================================================================

``live_trader.py`` (CLI) and ``run_engine.py`` (no-arg runner) both call into
this module; there is no alternative order-submission path anywhere (#20).

Pipeline (after the strategy decision is computed by ``strat_decision``):

  target weights (LOCKED at the completed close)
    -> target dollars from LIVE equity + EXECUTABLE prices (#10)
    -> per-symbol delta vs ACTUAL positions (trade only the delta) (#9)
    -> fractional-per-asset sizing; non-fractionable -> integer qty ->
       recompute realized weights -> re-check strategy constraints -> REJECT if
       the rounded book violates a hard constraint (#7)
    -> THREE gate layers, all must pass, REJECT (never clip) (#8):
         1. V2.9.8 strategy constraints (on the realized rounded book)
         2. operational safety limits (LIVE_MAX_*)
         3. broker/account constraints (blocked, tradable, buying power)
    -> kill switches: file = hard halt; daily-loss / drawdown = block only
       RISK-INCREASING moves (classified by projected gross exposure), allow
       de-risking (#16); never auto-flatten (#17)
    -> sell-before-buy, refreshing account/prices between phases (#9,#10)
    -> write-ahead + deterministic client-order-ids + open-order/partial-fill
       reconciliation (idempotent, restart-safe) (#12,#13,#14)
    -> poll fills, persist, report realized drift

Fail-closed: any error blocks new orders, marks the run FAILED, releases the
lock, and returns nonzero. Real money is NEVER submitted here during
implementation testing; live requires the CLI's three-gate protection.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd

from strat_allocator import AllocatorConfig
from strat import VOL_BREACH_TRIGGER
from strat_decision import (
    compute_target_portfolio, make_client_order_id, PortfolioDecision,
    DEFAULT_START, COLD_START_SESSIONS)
from execution_store import ExecutionStore, RebalanceLockError, is_terminal
from bond_regime import BondOverlayConfig

log = logging.getLogger("v298.engine")


class Mode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"


KILL_SWITCH_PATH = os.environ.get("KILL_SWITCH_PATH", ".kill_switch")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v is not None and v != "" else default
    except ValueError:
        return default


@dataclass
class ExecutionConfig:
    """Operational circuit breakers -- SEPARATE from AllocatorConfig (#8)."""
    live_enabled: bool = False
    max_order_notional: float = 5_000.0
    max_total_notional: float = 25_000.0
    max_gross_exposure: float = 1.50          # operational cap on target gross
    max_single_asset_weight: float = 1.25
    max_rebalance_turnover: float = 1.00      # 0.5 * sum|w_target - w_current|
    max_daily_loss_pct: float = 0.05          # of prior-close equity
    max_account_drawdown_pct: float = 0.25    # of high-water mark
    minimum_trade_notional: float = 1.0
    residual_share_threshold: float = 1e-4
    expected_account_id: str | None = None
    fill_poll_attempts: int = 20
    fill_poll_sleep_sec: float = 2.0

    @classmethod
    def from_env(cls) -> "ExecutionConfig":
        live = os.environ.get("ENABLE_LIVE_TRADING", "").strip().lower() in ("1", "true", "yes", "on")
        # Repo convention: ALLOW_LIVE_TRADING is also honored.
        live = live or os.environ.get("ALLOW_LIVE_TRADING", "").strip().lower() in ("1", "true", "yes", "on")
        return cls(
            live_enabled=live,
            max_order_notional=_env_float("LIVE_MAX_ORDER_NOTIONAL", 5_000.0),
            max_total_notional=_env_float("LIVE_MAX_TOTAL_NOTIONAL", 25_000.0),
            max_gross_exposure=_env_float("LIVE_MAX_GROSS_EXPOSURE", 1.50),
            max_single_asset_weight=_env_float("LIVE_MAX_SINGLE_ASSET_WEIGHT", 1.25),
            max_rebalance_turnover=_env_float("LIVE_MAX_REBALANCE_TURNOVER", 1.00),
            max_daily_loss_pct=_env_float("LIVE_MAX_DAILY_LOSS_PCT", 0.05),
            max_account_drawdown_pct=_env_float("LIVE_MAX_ACCOUNT_DRAWDOWN_PCT", 0.25),
            minimum_trade_notional=_env_float("LIVE_MINIMUM_TRADE_NOTIONAL", 1.0),
            expected_account_id=(os.environ.get("EXPECTED_ALPACA_ACCOUNT_ID") or None),
            fill_poll_attempts=int(_env_float("LIVE_FILL_POLL_ATTEMPTS", 20)),
            fill_poll_sleep_sec=_env_float("LIVE_FILL_POLL_SLEEP_SEC", 2.0),
        )


class GateRejection(RuntimeError):
    """A hard gate rejected the rebalance. No orders are placed. Fail closed."""


@dataclass
class OrderPlan:
    symbol: str
    side: str                 # buy / sell
    order_type: str           # market
    target_weight: float
    current_weight: float
    target_notional: float
    current_notional: float
    delta_notional: float
    exec_price: float
    fractionable: bool
    qty: float | None         # for sells and non-fractionable buys
    notional: float | None    # for fractionable buys
    realized_weight: float
    client_order_id: str
    # filled at execution time
    submitted: bool = False
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    status: str = "planned"


@dataclass
class RebalanceReport:
    mode: str
    as_of_close: str | None = None
    target_session: str | None = None
    forecast_source: str | None = None
    recovery_state: str | None = None
    severe_bear: bool = False
    vol_budget: float | None = None
    ex_ante_vol: float | None = None
    leverage_cap: float | None = None
    growth_ceiling: float | None = None
    current_gross: float | None = None
    target_gross: float | None = None
    equity: float | None = None
    decision_hash: str | None = None
    bond_regime: dict | None = None
    rebalanced: bool = False
    rebalance_reason: str = "scheduled"
    loss_replacement: list = field(default_factory=list)
    exhaustion: list = field(default_factory=list)
    adaptive: list = field(default_factory=list)
    executed: bool = False
    dry_run: bool = False
    kill_switch: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    rejected_reason: str | None = None
    skipped_reason: str | None = None
    final_drift: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["plans"] = [asdict(p) if isinstance(p, OrderPlan) else p for p in self.plans]
        return d


class LiveEngine:
    """Dependency-injected so tests can pass fake broker/data/store."""

    def __init__(self, mode: Mode, ecfg: ExecutionConfig, acfg: AllocatorConfig,
                 store: ExecutionStore, broker=None, data=None,
                 forecast_hook=None, start: str = DEFAULT_START,
                 strategy_name: str = "v298",
                 bond_overlay: BondOverlayConfig | None = None):
        self.mode = mode
        self.ecfg = ecfg
        self.acfg = acfg
        self.store = store
        self.broker = broker           # AlpacaBroker or fake; None only for pure compute
        self.data = data               # AlpacaData or fake
        self.forecast_hook = forecast_hook
        self.start = start
        self.name = strategy_name
        self.bond_overlay = bond_overlay   # None/disabled -> byte-identical decisions

    # ================= decision (READ-ONLY) =================
    def compute_decision(self, prices: pd.DataFrame | None = None,
                         as_of_close=None, target_session=None,
                         positions: dict | None = None) -> PortfolioDecision:
        """
        Compute (and persist) the V2.9.8 decision for the last completed session.

        READ-ONLY with respect to the broker: it never submits an order. Used by
        the CLI `signal` command and internally by `run_rebalance`.

        ``positions`` (broker positions incl. avg_entry_price) is used ONLY to feed
        the loss-replacement overlay (when enabled); it never triggers a trade here.
        """
        if prices is None:
            prices = self._fetch_decision_prices()
        decision_date = pd.Timestamp(as_of_close) if as_of_close is not None else prices.index[-1]

        # Exceptional-exit overlay inputs (only when an overlay is enabled AND
        # positions are known). avg_entry_price is the ACTUAL broker cost basis;
        # cooldowns come from the durable ledger. Absent positions -> no-op preview.
        position_avg_entry = None
        cooldown_state = None
        position_entry_dates = None
        if (self.acfg.loss_replacement_enabled or self.acfg.momentum_exhaustion_enabled
                or self.acfg.adaptive_rotation_enabled) and positions:
            position_avg_entry = {
                s: float(v.get("avg_entry_price", 0.0) or 0.0)
                for s, v in positions.items()
                if float(v.get("avg_entry_price", 0.0) or 0.0) > 0}
            cooldown_state = self.store.get_cooldowns()
            if self.acfg.adaptive_rotation_enabled:
                # days_held from the persisted first-entry ledger; a broker position
                # with no ledger row (cold ledger / manual trade) backfills to the
                # decision date -> treated as freshly entered (safe max hold-penalty).
                entries = self.store.get_entries()
                today = decision_date.strftime("%Y-%m-%d")
                position_entry_dates = {}
                for s in position_avg_entry:
                    if s not in entries:
                        self.store.set_entry(s, today)
                    position_entry_dates[s] = entries.get(s, today)

        st = self.store.get_strategy_state(self.name)
        if st is None:
            prev = np.zeros(len(self.acfg.risk_assets))
            ssr = COLD_START_SESSIONS
            last_rebal = None
        else:
            prev = np.asarray(st["prev_target"], float)
            last_rebal = st["last_rebalance_date"]
            ssr = self._sessions_since_rebalance(last_rebal, decision_date, prices.index)

        if target_session is None and self.data is not None:
            try:
                target_session = self.data.get_next_trading_session(after=decision_date)
            except Exception:
                target_session = None

        bond_prices = None
        if self.bond_overlay is not None and self.bond_overlay.enabled and self.data is not None:
            try:
                bond_prices = self.data.get_completed_daily_bars(
                    list(self.bond_overlay.bond_assets),
                    max(self.acfg.cov_lookback, 260) + 40)
            except Exception as e:
                log.warning("bond overlay enabled but bond data fetch failed (%s); "
                            "proceeding with overlay NEUTRAL (mult=1.0)", e)
                bond_prices = None

        dec = compute_target_portfolio(
            prices, prev_weights=prev, sessions_since_rebalance=ssr,
            cfg=self.acfg, as_of_close=decision_date, target_session=target_session,
            forecast_hook=self.forecast_hook, start=self.start,
            bond_overlay=self.bond_overlay, bond_prices=bond_prices,
            position_avg_entry=position_avg_entry, cooldown_state=cooldown_state,
            position_entry_dates=position_entry_dates)

        self.store.record_decision({
            "decision_hash": dec.decision_hash,
            "as_of_close": dec.as_of_close.strftime("%Y-%m-%d"),
            "target_session": (dec.target_session.strftime("%Y-%m-%d")
                               if dec.target_session is not None else None),
            "forecast_source": dec.forecast_source,
            "recovery_state": dec.recovery_state, "severe_bear": dec.severe_bear,
            "leverage_cap": dec.leverage_cap, "growth_ceiling": dec.growth_ceiling,
            "vol_budget": dec.vol_budget, "ex_ante_vol": dec.ex_ante_vol,
            "gross": dec.gross_exposure, "growth": dec.growth_exposure,
            "defensive": dec.defensive_exposure,
            "target_weights": dec.target_weights, "mode": self.mode.value,
            "rebalance_reason": getattr(dec, "rebalance_reason", "scheduled"),
            "rotation_events": dec.rotation_events()})
        return dec

    def _sessions_since_rebalance(self, last_rebalance_date, decision_date, index) -> int:
        if last_rebalance_date is None:
            return COLD_START_SESSIONS
        lr = pd.Timestamp(last_rebalance_date)
        idx = pd.DatetimeIndex(index)
        return int(((idx > lr) & (idx <= pd.Timestamp(decision_date))).sum())

    def _fetch_decision_prices(self) -> pd.DataFrame:
        if self.data is None:
            raise GateRejection("no market-data adapter configured")
        from alpaca_data import validate_market_data
        symbols = list(self.acfg.risk_assets)
        need = max(self.acfg.cov_lookback, 260) + 40
        close = self.data.get_completed_daily_bars(symbols, need)
        validate_market_data(close, symbols)
        return close

    # ================= rebalance (THE execution path) =================
    def run_rebalance(self, dry_run: bool = False) -> RebalanceReport:
        rep = RebalanceReport(mode=self.mode.value, dry_run=dry_run)
        try:
            # 0) restart recovery FIRST -- complete any interrupted rebalance with
            #    the sell-before-buy fail-closed discipline. On "failed" (a sell did
            #    not free capacity) we STOP and submit nothing. On "completed" we
            #    proceed to compute today's decision: strategy_state was persisted, so
            #    if the recovered run WAS this session the decision is "holding"
            #    (idempotent, no duplicate orders); if it was a STALE prior session,
            #    today's own rebalance is still evaluated rather than silently skipped.
            if not dry_run and self.broker is not None:
                rec = self._recover_incomplete_run(rep)
                if rec == "failed":
                    return rep                       # fail-closed; rejected_reason set
                if rec == "completed":
                    rep.executed = True
                    rep.notes.append("recovery complete; evaluating current session")

            # 1) prices for the last completed session (LOCKED weights)
            prices = self._fetch_decision_prices() if self.data is not None else None

            # compute-only (no broker): read-only decision, no positions, no trade
            if self.broker is None:
                dec = self.compute_decision(prices=prices)
                self._fill_decision_report(rep, dec)
                rep.skipped_reason = "no broker configured (compute-only)"
                return rep

            # 2) account + positions. Positions (with broker avg_entry_price) feed
            #    the loss-replacement overlay; the decision stays READ-ONLY here.
            acct = self.broker.account_snapshot()
            rep.equity = acct.equity
            self._account_guard(acct)               # broker-layer hard blocks
            positions = self.broker.positions()

            # 3) decision from the last completed session (overlay consumes cost basis)
            dec = self.compute_decision(prices=prices, positions=positions)
            self._fill_decision_report(rep, dec)
            current_gross = (acct.long_market_value / acct.equity) if acct.equity > 0 else 0.0
            rep.current_gross = current_gross
            rep.target_gross = dec.gross_exposure

            self.store.record_equity(self.mode.value, acct.equity, acct.cash,
                                     acct.buying_power, acct.portfolio_value,
                                     acct.long_market_value)
            hwm = self.store.update_high_water_mark(acct.equity)

            # 3) kill switches
            kill = self._evaluate_kill_switches(acct, hwm)
            rep.kill_switch = [k for k, _ in kill]
            if any(k == "file" for k, _ in kill):
                rep.skipped_reason = "KILL_SWITCH_ACTIVE (file) -- all new orders halted"
                log.critical("KILL_SWITCH_ACTIVE file present; no orders")
                return rep

            # 4) if the strategy is not rebalancing this session, hold (no trade)
            if not dec.rebalanced:
                rep.skipped_reason = ("strategy holding (not a rebalance session: "
                                      "not due / deadband)")
                rep.final_drift = self._drift(dec, positions, acct.equity)
                self._maybe_circuit_break(rep, dec, positions, acct.equity)
                return rep

            # 5) build order plans + realized book
            exec_prices = self._executable_prices(list(self.acfg.risk_assets), positions)
            asset_caps = self._asset_caps(list(self.acfg.risk_assets))
            plans, realized = self._build_plans(dec, acct.equity, positions, exec_prices, asset_caps)
            rep.plans = plans

            # 6) THREE gate layers (REJECT, never clip)
            self._strategy_gate(realized, dec)                     # layer 1
            self._operational_gate(plans, dec, current_gross)      # layer 2
            self._broker_gate(plans, acct, asset_caps)             # layer 3

            # 7) kill-switch: block RISK-INCREASING moves, allow de-risking (#16)
            realized_gross = float(sum(realized.values()))
            risk_increasing = realized_gross > current_gross + 1e-6
            if kill and risk_increasing:
                rep.skipped_reason = (f"KILL_SWITCH_ACTIVE ({[k for k,_ in kill]}): "
                                      f"risk-increasing rebalance blocked "
                                      f"(target gross {realized_gross:.2f} > "
                                      f"current {current_gross:.2f})")
                log.critical(rep.skipped_reason)
                return rep

            if dry_run:
                rep.notes.append("DRY RUN: 0 orders submitted")
                return rep

            # 8) execute (sell-before-buy, write-ahead, idempotent, reconcile)
            self._execute(dec, plans, rep, acct)
            rep.executed = True
            return rep

        except RebalanceLockError as e:
            rep.rejected_reason = f"another rebalance is active: {e}"
            log.error(rep.rejected_reason)
            return rep
        except GateRejection as e:
            rep.rejected_reason = str(e)
            log.error("REBALANCE REJECTED: %s", e)
            return rep
        except Exception as e:                        # fail closed (#17): NO auto-flatten
            rep.rejected_reason = f"engine error (fail-closed, no orders): {e}"
            log.exception("engine error -- failing closed, no orders")
            return rep

    # ---- restart recovery ----
    def _recover_incomplete_run(self, rep: RebalanceReport) -> str:
        """
        On restart, COMPLETE any interrupted rebalance (#14) using the SAME
        sell-before-buy fail-closed discipline as _execute. Every leg was
        write-ahead recorded as an intent: already-submitted legs reconcile against
        the broker; never-submitted legs are submitted now. Returns:

          "none"      -- no active run; caller proceeds normally.
          "completed" -- the interrupted run was finished (or was an orphan with no
                         decision); caller proceeds to compute today's decision,
                         which is "holding" when it was this session (strategy_state
                         is now persisted) or a fresh rebalance for a stale session.
          "failed"    -- a SELL leg did not free its capacity; caller FAILS CLOSED
                         and submits nothing further this run.

        The critical invariant: a rejected / short / price-missing SELL during
        recovery must STOP the dependent replacement BUY, exactly like _execute.
        """
        from types import SimpleNamespace
        active = self.store.active_run()
        if active is None:
            return "none"
        run_id = int(active["run_id"])
        dec_row = self.store.get_decision(active["decision_hash"])
        if dec_row is None:
            self.store.finish_run(run_id, "FAILED", "no decision for active run")
            rep.notes.append(f"cleared orphan active run {run_id}")
            return "completed"
        rep.notes.append(f"resuming incomplete run {run_id} (state={active['state']})")
        self.store.set_run_state(run_id, "RECONCILING", "restart recovery")

        dec_ctx = SimpleNamespace(as_of_close=pd.Timestamp(dec_row["as_of_close"]),
                                  decision_hash=dec_row["decision_hash"])
        legs = self.store.orders_for_run(run_id)
        sells = [o for o in legs if o["side"] == "sell"]
        buys = [o for o in legs if o["side"] == "buy"]

        def _drive(o):
            price = self._executable_prices([o["symbol"]], self.broker.positions()).get(o["symbol"])
            if price is None or price <= 0:
                return None
            plan = OrderPlan(
                symbol=o["symbol"], side=o["side"], order_type="market",
                target_weight=0.0, current_weight=0.0, target_notional=0.0,
                current_notional=0.0, delta_notional=0.0, exec_price=price,
                fractionable=(o.get("intent_notional") is not None),
                qty=o.get("intent_qty"), notional=o.get("intent_notional"),
                realized_weight=0.0, client_order_id=o["client_order_id"])
            self._drive_leg(plan, dec_ctx)
            return plan

        # SELLS first -- confirm every one freed its capacity BEFORE any buy.
        incomplete = []
        for o in sells:
            plan = _drive(o)
            if plan is None:
                incomplete.append(o["symbol"])          # no price -> cannot confirm
                continue
            desired = plan.qty if plan.qty is not None else 0.0
            if (not is_terminal(plan.status)) or (desired - plan.filled_qty
                                                  > self.ecfg.residual_share_threshold):
                incomplete.append(o["symbol"])
        if incomplete:
            self.store.finish_run(run_id, "FAILED", f"recovery sells incomplete {incomplete}")
            rep.rejected_reason = (f"restart recovery: sell legs did not complete "
                                   f"{incomplete}; failing closed before dependent buys")
            log.error(rep.rejected_reason)
            return "failed"

        # sells reconciled -> drive dependent buys
        for o in buys:
            _drive(o)

        # Persist the recovered book + cooldowns BEFORE finish_run releases the lock
        # (mirror _execute; do NOT swallow). If a persist fails we must NOT declare
        # the run COMPLETE: leaving the lock held lets the next recovery retry, and
        # avoids a COMPLETE-but-stale-strategy_state window in which the fall-through
        # would recompute a divergent decision that reverses the recovered rotation.
        # Cooldowns come from the persisted EXCEPTIONAL-EXIT events (not every sell
        # leg), so a crash-recovery reproduces the clean-completion cooldown set.
        tw = json.loads(dec_row["target_weights"])
        realized_vec = [float(tw.get(a, 0.0)) for a in self.acfg.risk_assets]
        self.store.set_strategy_state(realized_vec, dec_row["as_of_close"],
                                      dec_row["decision_hash"], self.name)
        for sym in self._rotation_exits(dec_row):
            self.store.set_cooldown(sym, dec_row["as_of_close"])
        if self.acfg.adaptive_rotation_enabled:
            eps = self.acfg.loss_replacement_hold_eps
            for a in self.acfg.risk_assets:
                if float(tw.get(a, 0.0)) > eps:
                    self.store.set_entry(a, dec_row["as_of_close"])
                else:
                    self.store.clear_entry(a)
        self.store.finish_run(run_id, "COMPLETE", "recovered on restart")
        return "completed"

    @staticmethod
    def _rotation_exits(dec_row: dict) -> list:
        """Exited assets from a persisted decision's exceptional-exit events."""
        try:
            evs = json.loads(dec_row.get("rotation_events") or "[]")
        except (TypeError, ValueError):
            return []
        return [e["exited_asset"] for e in evs if e.get("exited_asset")]

    # ---- plan building ----
    def _executable_prices(self, symbols, positions) -> dict:
        prices = {}
        try:
            prices = dict(self.data.get_latest_prices(symbols)) if self.data else {}
        except Exception as e:
            log.warning("latest-price fetch failed (%s)", e)
        need = [s for s in symbols if s not in prices or prices[s] <= 0]
        # Fallback 1: most recent RAW daily close (works without real-time entitlement).
        if need and self.data is not None:
            try:
                raw = self.data.get_recent_raw_closes(need)
                for s, px in raw.items():
                    if px > 0:
                        prices[s] = px
            except Exception as e:
                log.warning("raw-close fallback failed (%s)", e)
        # Fallback 2: current position mark.
        for s in symbols:
            if s not in prices or prices[s] <= 0:
                cp = positions.get(s, {}).get("current_price", 0.0)
                if cp > 0:
                    prices[s] = cp
        return prices

    def _asset_caps(self, symbols) -> dict:
        caps = {}
        for s in symbols:
            try:
                caps[s] = self.broker.asset_caps(s)
            except Exception as e:
                raise GateRejection(f"could not fetch asset metadata for {s}: {e}")
        return caps

    def _build_plans(self, dec: PortfolioDecision, equity: float, positions: dict,
                     exec_prices: dict, asset_caps: dict):
        if equity <= 0:
            raise GateRejection(f"non-positive account equity {equity}")
        assets = list(self.acfg.risk_assets)
        plans, realized = [], {}
        for sym in assets:
            w = float(dec.target_weights[sym])
            price = exec_prices.get(sym)
            if price is None or price <= 0:
                raise GateRejection(f"no executable price for {sym}")
            caps = asset_caps[sym]
            if not caps["tradable"]:
                raise GateRejection(f"{sym} is not tradable")

            cur_qty = float(positions.get(sym, {}).get("qty", 0.0))
            cur_notional = cur_qty * price
            target_notional = w * equity
            delta_notional = target_notional - cur_notional

            if abs(delta_notional) < self.ecfg.minimum_trade_notional:
                realized[sym] = cur_notional / equity
                continue

            side = "buy" if delta_notional > 0 else "sell"
            delta_qty = delta_notional / price
            fractionable = bool(caps["fractionable"])

            if fractionable:
                trade_qty = abs(delta_qty)
            else:
                trade_qty = math.floor(abs(delta_qty))  # never overshoot on rounding
                if trade_qty <= 0:
                    realized[sym] = cur_notional / equity
                    continue

            if side == "sell":
                trade_qty = min(trade_qty, cur_qty)     # cannot sell more than held
                realized_qty = cur_qty - trade_qty
                qty, notional = trade_qty, None
            else:
                realized_qty = cur_qty + trade_qty
                if fractionable:
                    qty, notional = None, round(trade_qty * price, 2)
                else:
                    qty, notional = trade_qty, None

            realized[sym] = (realized_qty * price) / equity
            coid = make_client_order_id(dec.as_of_close, sym, side, dec.decision_hash)
            plans.append(OrderPlan(
                symbol=sym, side=side, order_type="market",
                target_weight=w, current_weight=cur_notional / equity,
                target_notional=target_notional, current_notional=cur_notional,
                delta_notional=delta_notional, exec_price=price,
                fractionable=fractionable, qty=qty, notional=notional,
                realized_weight=realized[sym], client_order_id=coid))
        return plans, realized

    # ---- gate layer 1: strategy constraints on the REALIZED rounded book ----
    def _strategy_gate(self, realized: dict, dec: PortfolioDecision) -> None:
        cfg = self.acfg
        w = np.array([realized.get(a, 0.0) for a in cfg.risk_assets], float)
        if not np.all(np.isfinite(w)):
            raise GateRejection("realized weights contain NaN/inf")
        if np.any(w < -1e-9):
            raise GateRejection("realized weights contain a negative (short) weight")
        caps = cfg.caps
        over = [(a, w[i], caps[i]) for i, a in enumerate(cfg.risk_assets)
                if w[i] > caps[i] + 1e-6]
        if over:
            raise GateRejection(f"per-asset cap violated on realized book: {over}")
        gsum = float(w[cfg.growth_idx].sum())
        if gsum > dec.growth_ceiling + 1e-6:
            raise GateRejection(f"growth {gsum:.4f} > ceiling {dec.growth_ceiling:.4f}")
        dsum = float(w[cfg.defensive_idx].sum())
        if dsum > cfg.defensive_combined_max + 1e-6:
            raise GateRejection(f"defensive {dsum:.4f} > {cfg.defensive_combined_max:.4f}")
        gross = float(w.sum())
        if gross > dec.leverage_cap + 1e-6:
            raise GateRejection(f"gross {gross:.4f} > strategy leverage cap "
                                f"{dec.leverage_cap:.4f}")
        cov = dec.diagnostics.get("cov_adj")
        if cov is not None:
            ex_ante = math.sqrt(max(float(w @ np.asarray(cov) @ w), 0.0))
            if ex_ante > dec.vol_budget * VOL_BREACH_TRIGGER + 1e-6:
                raise GateRejection(
                    f"realized ex-ante vol {ex_ante:.4f} > "
                    f"{dec.vol_budget * VOL_BREACH_TRIGGER:.4f}")

    # ---- gate layer 2: operational limits (REJECT, never clip) ----
    def _operational_gate(self, plans, dec, current_gross) -> None:
        e = self.ecfg
        total = 0.0
        for p in plans:
            notional = abs(p.notional if p.notional is not None else p.qty * p.exec_price)
            total += notional
            if notional > e.max_order_notional + 1e-6:
                raise GateRejection(f"{p.symbol} order ${notional:,.0f} > "
                                    f"LIVE_MAX_ORDER_NOTIONAL ${e.max_order_notional:,.0f}")
            if p.target_weight > e.max_single_asset_weight + 1e-9:
                raise GateRejection(f"{p.symbol} target weight {p.target_weight:.3f} > "
                                    f"LIVE_MAX_SINGLE_ASSET_WEIGHT {e.max_single_asset_weight}")
        if total > e.max_total_notional + 1e-6:
            raise GateRejection(f"total order ${total:,.0f} > LIVE_MAX_TOTAL_NOTIONAL "
                                f"${e.max_total_notional:,.0f}")
        if dec.gross_exposure > e.max_gross_exposure + 1e-9:
            raise GateRejection(f"target gross {dec.gross_exposure:.3f} > "
                                f"LIVE_MAX_GROSS_EXPOSURE {e.max_gross_exposure} "
                                f"(operational cap; NOT clipped)")
        turnover = 0.5 * float(sum(abs(p.realized_weight - p.current_weight) for p in plans))
        if turnover > e.max_rebalance_turnover + 1e-9:
            raise GateRejection(f"turnover {turnover:.3f} > LIVE_MAX_REBALANCE_TURNOVER "
                                f"{e.max_rebalance_turnover}")

    # ---- gate layer 3: broker / buying power ----
    def _account_guard(self, acct) -> None:
        reason = acct.blocked_reason()
        if reason:
            raise GateRejection(f"account not tradable: {reason}")
        if self.mode == Mode.LIVE:
            self.broker.assert_expected_account(acct, self.ecfg.expected_account_id)

    def _broker_gate(self, plans, acct, asset_caps) -> None:
        buy_notional = sum(
            (p.notional if p.notional is not None else p.qty * p.exec_price)
            for p in plans if p.side == "buy")
        # Buying power is checked AFTER sells in _execute; this is a pre-flight guard
        # that broker BP is never treated as permission to exceed the strategy cap.
        if buy_notional > acct.buying_power + acct.long_market_value + 1e-6:
            raise GateRejection(
                f"buy notional ${buy_notional:,.0f} exceeds account capacity "
                f"(bp ${acct.buying_power:,.0f} + lmv ${acct.long_market_value:,.0f})")

    # ---- kill switches ----
    def _evaluate_kill_switches(self, acct, hwm) -> list:
        active = []
        if os.path.exists(KILL_SWITCH_PATH):
            active.append(("file", f"{KILL_SWITCH_PATH} present"))
            self.store.record_kill_switch("file", True, KILL_SWITCH_PATH)
        if acct.last_equity > 0:
            daily = (acct.equity - acct.last_equity) / acct.last_equity
            if daily <= -abs(self.ecfg.max_daily_loss_pct):
                active.append(("daily_loss", f"daily P&L {daily:.2%}"))
                self.store.record_kill_switch("daily_loss", True, f"{daily:.4f}")
        if hwm and hwm > 0:
            dd = (acct.equity - hwm) / hwm
            if dd <= -abs(self.ecfg.max_account_drawdown_pct):
                active.append(("drawdown", f"drawdown {dd:.2%} from HWM"))
                self.store.record_kill_switch("drawdown", True, f"{dd:.4f}")
        return active

    # ---- execution: sell-before-buy, idempotent, reconcile ----
    def _execute(self, dec, plans, rep, acct) -> None:
        run_id, state, resumed = self.store.begin_or_resume_run(dec.decision_hash, self.mode.value)
        if state == "COMPLETE":
            rep.notes.append("decision already executed this session (idempotent no-op)")
            rep.executed = True
            rep.final_drift = self._drift(dec, self.broker.positions(), acct.equity)
            return
        for p in plans:                                   # tag plans to this run
            self.store.record_order_intent(p.client_order_id, run_id, dec.decision_hash,
                                           p.symbol, p.side, p.order_type,
                                           p.qty, p.notional)
        try:
            sells = [p for p in plans if p.side == "sell"]
            buys = [p for p in plans if p.side == "buy"]

            self.store.set_run_state(run_id, "SELLING")
            for p in sells:
                self._drive_leg(p, dec)
            # A rejected/short/UNCONFIRMED SELL must STOP dependent execution:
            # proceeding to buy without having freed the intended capacity would
            # over-lever the book. A sell that is NOT terminal after poll exhaustion
            # (still new/accepted/partially_filled) has NOT freed capacity either, so
            # it also blocks. Fail closed before any buy.
            short_sells = [p.symbol for p in sells
                           if (not is_terminal(p.status))
                           or (p.qty or 0.0) - p.filled_qty > self.ecfg.residual_share_threshold]
            if short_sells:
                raise GateRejection(f"sell legs did not complete {short_sells}; "
                                    f"failing closed before buys")
            self.store.set_run_state(run_id, "SELLS_RECONCILED")

            # refresh account/prices; recompute buy sizes from CURRENT state (#9,#10)
            acct2 = self.broker.account_snapshot()
            positions2 = self.broker.positions()
            exec_prices2 = self._executable_prices(list(self.acfg.risk_assets), positions2)
            buys = self._resize_buys(dec, buys, acct2, positions2, exec_prices2, run_id)
            buy_notional = sum((p.notional if p.notional is not None else p.qty * p.exec_price)
                               for p in buys)
            if buy_notional > acct2.buying_power + 1e-6:
                raise GateRejection(
                    f"insufficient buying power after sells: need ${buy_notional:,.0f}, "
                    f"have ${acct2.buying_power:,.0f}")

            self.store.set_run_state(run_id, "BUYING")
            for p in buys:
                self._drive_leg(p, dec)

            self.store.set_run_state(run_id, "RECONCILING")
            rep.plans = sells + buys
            rep.final_drift = self._drift(dec, self.broker.positions(), acct2.equity)

            # Persist the strategy book + cooldowns BEFORE finish_run releases the
            # active-rebalance lock. If the process dies in the gap between "book is
            # durable" and "lock released", the lock is still held so recovery
            # re-drives (idempotent no-ops) and re-persists; if it dies after the
            # lock is released, strategy_state already reflects this session, so the
            # next decision is "holding" (ssr=0) instead of recomputing a divergent,
            # non-short-circuited decision that reverses the rotation (#3).
            realized_vec = [dec.target_weights[a] for a in self.acfg.risk_assets]
            self.store.set_strategy_state(
                realized_vec, dec.as_of_close.strftime("%Y-%m-%d"),
                dec.decision_hash, self.name)
            # churn protection: start the cooldown clock for each asset this run
            # rotated OUT (loss / exhaustion / adaptive), barred as a replacement.
            aoc = dec.as_of_close.strftime("%Y-%m-%d")
            for ev in dec.rotation_events():
                if ev.get("exited_asset"):
                    self.store.set_cooldown(ev["exited_asset"], aoc)
            # first-entry ledger (days_held): stamp new holds, clear full exits.
            if self.acfg.adaptive_rotation_enabled:
                eps = self.acfg.loss_replacement_hold_eps
                for a in self.acfg.risk_assets:
                    if dec.target_weights[a] > eps:
                        self.store.set_entry(a, aoc)      # DO NOTHING preserves first entry
                    else:
                        self.store.clear_entry(a)
            self.store.finish_run(run_id, "COMPLETE")
        except Exception:
            self.store.finish_run(run_id, "FAILED", "exception during execution")
            raise

    def _resize_buys(self, dec, buys, acct, positions, exec_prices, run_id):
        """Recompute buy quantities from refreshed equity/prices/positions (target LOCKED)."""
        resized = []
        for p in buys:
            sym = p.symbol
            price = exec_prices.get(sym, p.exec_price)
            cur_qty = float(positions.get(sym, {}).get("qty", 0.0))
            target_notional = dec.target_weights[sym] * acct.equity
            delta_notional = target_notional - cur_qty * price
            if delta_notional < self.ecfg.minimum_trade_notional:
                continue
            if p.fractionable:
                q, notional = None, round(delta_notional, 2)
            else:
                shares = math.floor(delta_notional / price)
                if shares <= 0:
                    continue
                q, notional = shares, None
            np_ = OrderPlan(symbol=sym, side="buy", order_type="market",
                            target_weight=p.target_weight, current_weight=cur_qty * price / acct.equity,
                            target_notional=target_notional, current_notional=cur_qty * price,
                            delta_notional=delta_notional, exec_price=price,
                            fractionable=p.fractionable, qty=q, notional=notional,
                            realized_weight=p.realized_weight,
                            client_order_id=p.client_order_id)
            self.store.record_order_intent(np_.client_order_id, run_id, dec.decision_hash,
                                           sym, "buy", "market", q, notional)
            resized.append(np_)
        return resized

    def _drive_leg(self, p: OrderPlan, dec: PortfolioDecision) -> None:
        """
        Drive one leg to completion idempotently. Reconciles any existing order
        under the coid (crash/restart), submits only if needed, then submits ONE
        residual slice if a terminal fill fell short (#12).
        """
        desired_qty = (p.qty if p.qty is not None else (p.notional / p.exec_price))
        existing = self.broker.get_order_by_client_id(p.client_order_id)
        if existing is not None:
            if not is_terminal(existing["status"]):
                existing = self.broker.poll_fill(existing["broker_order_id"])
            self.store.update_order_fill(p.client_order_id, existing["status"],
                                         existing["filled_qty"], existing["filled_avg_price"],
                                         existing["broker_order_id"])
            final = existing
        else:
            submitted = self.broker.submit_market_order(
                p.symbol, p.side, qty=p.qty, notional=p.notional,
                client_order_id=p.client_order_id)
            self.store.mark_order_submitted(p.client_order_id, submitted["broker_order_id"],
                                            p.qty, p.notional, submitted["status"])
            final = self.broker.poll_fill(submitted["broker_order_id"],
                                          attempts=self.ecfg.fill_poll_attempts,
                                          sleep_sec=self.ecfg.fill_poll_sleep_sec)
            self.store.update_order_fill(p.client_order_id, final["status"],
                                         final["filled_qty"], final["filled_avg_price"],
                                         final["broker_order_id"])
        p.submitted = True
        p.filled_qty = final["filled_qty"]
        p.avg_fill_price = final["filled_avg_price"]
        p.status = final["status"]

        remaining = desired_qty - final["filled_qty"]
        if is_terminal(final["status"]) and remaining > self.ecfg.residual_share_threshold:
            slice_coid = make_client_order_id(dec.as_of_close, p.symbol, p.side,
                                              dec.decision_hash, slice_n=1)
            if self.broker.get_order_by_client_id(slice_coid) is None:
                self.store.record_order_intent(slice_coid, None, dec.decision_hash,
                                               p.symbol, p.side, "market", remaining, None)
                sub2 = self.broker.submit_market_order(
                    p.symbol, p.side, qty=remaining, notional=None, client_order_id=slice_coid)
                self.store.mark_order_submitted(slice_coid, sub2["broker_order_id"],
                                                remaining, None, sub2["status"])
                f2 = self.broker.poll_fill(sub2["broker_order_id"])
                self.store.update_order_fill(slice_coid, f2["status"], f2["filled_qty"],
                                             f2["filled_avg_price"], f2["broker_order_id"])
                p.filled_qty += f2["filled_qty"]

    # ---- drift + circuit breaker ----
    def _drift(self, dec, positions, equity) -> dict:
        out = {}
        for a in self.acfg.risk_assets:
            cur_mv = float(positions.get(a, {}).get("market_value", 0.0))
            cur_w = cur_mv / equity if equity > 0 else 0.0
            out[a] = round(cur_w - dec.target_weights[a], 6)
        return out

    def _maybe_circuit_break(self, rep, dec, positions, equity) -> None:
        """Execution-failure breaker: if actual persistently far from target, alert."""
        drift = rep.final_drift or self._drift(dec, positions, equity)
        max_abs = max(abs(v) for v in drift.values()) if drift else 0.0
        if max_abs > 0.10:
            rep.notes.append(f"WARNING: actual book drifts {max_abs:.1%} from strategy "
                             f"target -- possible persistent execution failure")

    def _fill_decision_report(self, rep, dec) -> None:
        rep.as_of_close = dec.as_of_close.strftime("%Y-%m-%d")
        rep.target_session = (dec.target_session.strftime("%Y-%m-%d")
                              if dec.target_session is not None else None)
        rep.forecast_source = dec.forecast_source
        rep.recovery_state = dec.recovery_state
        rep.severe_bear = dec.severe_bear
        rep.vol_budget = dec.vol_budget
        rep.ex_ante_vol = dec.ex_ante_vol
        rep.leverage_cap = dec.leverage_cap
        rep.growth_ceiling = dec.growth_ceiling
        rep.decision_hash = dec.decision_hash
        rep.bond_regime = dec.bond_regime
        rep.rebalanced = dec.rebalanced
        rep.rebalance_reason = getattr(dec, "rebalance_reason", "scheduled")
        rep.loss_replacement = list(getattr(dec, "loss_replacement_events", []) or [])
        rep.exhaustion = list(getattr(dec, "exhaustion_events", []) or [])
        rep.adaptive = list(getattr(dec, "adaptive_events", []) or [])

    # ================= status (read-only) =================
    def status(self) -> dict:
        out = {"mode": self.mode.value}
        st = self.store.get_strategy_state(self.name)
        out["strategy_state"] = st
        out["active_run"] = self.store.active_run()
        out["high_water_mark"] = self.store.get_high_water_mark()
        out["kill_switch_file"] = os.path.exists(KILL_SWITCH_PATH)
        if self.broker is not None:
            try:
                acct = self.broker.account_snapshot()
                out["account"] = {"suffix": acct.account_suffix, "equity": acct.equity,
                                  "cash": acct.cash, "buying_power": acct.buying_power,
                                  "long_market_value": acct.long_market_value,
                                  "status": acct.status,
                                  "pattern_day_trader": acct.pattern_day_trader,
                                  "daytrade_count": acct.daytrade_count}
                out["positions"] = self.broker.positions()
                out["open_orders"] = self.broker.open_orders()
            except Exception as e:
                out["account_error"] = str(e)
        return out
