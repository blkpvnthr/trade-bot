"""
production_executor.py — legacy execution compatibility layer.

This module is NOT part of the live two-engine architecture (Alpha Call Engine +
Buffett Accumulation Layer). It restores the intraday-bracket / premarket-recovery
execution surface that an earlier `run.py` exposed and that the test suite still
exercises:

    tests/test_runtime_recovery.py    (premarket plan → bracket rows, RL timing gate,
                                        bracket-vs-market submission)
    tests/test_first_candle_rule.py   (9:35 first-candle-rule long brackets)
    tests/test_live_experience_logging.py (order execution + live-experience logging)

Design constraints honoured:
  * It is a SEPARATE, self-contained execution path with its own broker abstraction
    (run.BrokerPosition + a bracket-capable broker passed in by the caller/tests).
    It does NOT touch the two-engine RiskManager or AlpacaGateway, so nothing routes
    around those gates and the Alpha/Buffett separation is preserved.
  * Patchable hooks (write_event, save_df, log_executed_order, _BRACKET_ORDERS_AVAILABLE,
    row_has_complete_bracket_plan) live in run.py and are reached lazily via `import run`
    so `patch.object(run, ...)` in the tests takes effect.

The numeric contracts (one-tick stops, fixed-R targets, capped target multiples)
match docs/first_candle_rule_930.md and the committed premarket trade-plan schema.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(timezone.utc).strftime("%Y%m%d"))


def _num(value) -> float:
    """Coerce to float, mapping None/blank/NaN/non-numeric to 0.0."""
    try:
        f = float(value)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


# ── configuration ──────────────────────────────────────────────────────────────
@dataclass
class ScannerConfig:
    """Scanner-side config (subset used by the compat execution path)."""
    # First-candle-rule (9:35) parameters
    first_candle_rule_enabled: bool = False
    first_candle_rule_require_volume_confirmation: bool = False
    fcr_window_minutes: int = 5
    fcr_target_r: float = 3.0
    fcr_tick_size: float = 0.01
    fcr_require_bullish: bool = True
    fcr_min_range_pct: float = 0.001
    fcr_max_range_pct: float = 0.08
    fcr_min_stop_distance_pct: float = 0.0
    fcr_max_stop_distance_pct: float = 0.50
    fcr_risk_pct: float = 0.01
    # RL timing gate
    use_rl_timing_gate: bool = False
    rl_shadow_mode: bool = True
    # Direction-confirmation / streaming flags
    require_direction_confirmation_before_buy: bool = True
    stream_confirmed_append_direction_rejects: bool = False
    stream_confirmed_submit_trailing_stop_after_fill: bool = False


@dataclass
class ExecutorConfig:
    """Executor-side config (subset used by the compat execution path)."""
    max_order_dollars: float = 1500.0
    max_single_order_qty: int = 100_000
    min_notional_to_trade: float = 1.0
    risk_per_trade_pct: float = 0.01
    bracket_stop_tick: float = 0.01
    max_target_r_multiple: float = 5.0
    time_in_force: str = "day"
    scanner: ScannerConfig = field(default_factory=ScannerConfig)


def build_default_config() -> ExecutorConfig:
    """Return a default ExecutorConfig (with a nested ScannerConfig)."""
    return ExecutorConfig(scanner=ScannerConfig())


# ── momentum scanner (first-candle-rule) ─────────────────────────────────────────
class MomentumSetupScanner:
    """
    Intraday momentum scanner. The compat surface implements the 9:35 first-candle
    rule: long-only bracket at the close of the 09:30-09:34 ET candle when it is
    bullish and tradable. Entry = first-candle close, stop = one tick below the
    first-candle low, target = entry + target_r * risk (default 3R).
    """

    def __init__(self, config: ScannerConfig | None = None, executor_config=None):
        self.config = config or ScannerConfig()
        self.executor_config = executor_config
        self._intraday_bar_cache: dict = {}

    def _download_intraday_bars(self, symbols, minutes: int) -> pd.DataFrame:
        """Default no-op fetch (tests inject a real one)."""
        return pd.DataFrame()

    def _first_candle_rule_signal_for_symbol(self, symbol: str, bars: pd.DataFrame) -> dict:
        cfg = self.config

        def blocked(reason: str) -> dict:
            return {"symbol": symbol, "available": False,
                    "blocked_by": "first_candle_rule", "reason": reason}

        b = bars
        if b is None or len(b) == 0:
            return blocked("incomplete_first_candle")
        cols = list(getattr(b, "columns", []))
        if "symbol" in cols:
            b = b[b["symbol"] == symbol]
        if "timestamp" in cols:
            b = b.sort_values("timestamp")
        if len(b) < 5:
            return blocked("incomplete_first_candle")

        first5 = b.head(5)
        open_ = float(first5["open"].iloc[0])
        high = float(first5["high"].max())
        low = float(first5["low"].min())
        close = float(first5["close"].iloc[-1])

        if high <= low:
            return blocked("first_candle_invalid_range")
        mid = (high + low) / 2.0
        range_pct = (high - low) / mid if mid > 0 else 0.0
        if range_pct < float(cfg.fcr_min_range_pct):
            return blocked("first_candle_range_too_tight")
        if range_pct > float(cfg.fcr_max_range_pct):
            return blocked("first_candle_range_too_wide")
        if cfg.fcr_require_bullish and close <= open_:
            return blocked("first_candle_not_bullish")

        stop = low - float(cfg.fcr_tick_size)
        risk = close - stop
        if risk <= 0:
            return blocked("stop_distance_too_small")
        stop_dist_pct = risk / close if close > 0 else 0.0
        if stop_dist_pct < float(cfg.fcr_min_stop_distance_pct):
            return blocked("stop_distance_too_small")
        if stop_dist_pct > float(cfg.fcr_max_stop_distance_pct):
            return blocked("stop_distance_too_wide")

        target_r = float(cfg.fcr_target_r)
        target = close + target_r * risk
        return {
            "symbol": symbol, "available": True, "blocked_by": None, "reason": "ok",
            "entry": close, "stop": stop, "target": target,
            "risk_reward_ratio": target_r,
            "first_candle_high": high, "first_candle_low": low,
        }

    def _build_first_candle_rule_orders(self, symbols, features, equity: float) -> pd.DataFrame:
        cfg = self.config
        if not cfg.first_candle_rule_enabled:
            return pd.DataFrame()
        try:
            bars = self._download_intraday_bars(symbols, int(cfg.fcr_window_minutes))
        except Exception:
            bars = pd.DataFrame()
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        exec_cfg = getattr(self, "executor_config", None)
        max_w = float(getattr(exec_cfg, "max_position_weight", 0.10) or 0.10)
        min_notional = float(getattr(exec_cfg, "min_notional_to_trade", 1.0) or 1.0)

        rows = []
        for symbol in symbols:
            sig = self._first_candle_rule_signal_for_symbol(symbol, bars)
            if not sig.get("available"):
                continue
            entry = float(sig["entry"])
            stop = float(sig["stop"])
            target = float(sig["target"])
            risk = entry - stop
            if risk <= 0 or entry <= 0:
                continue
            qty_by_risk = int((equity * float(cfg.fcr_risk_pct)) / risk) if risk > 0 else 0
            qty_by_notional = int((max_w * equity) / entry) if entry > 0 else 0
            qty = max(0, min(qty_by_risk, qty_by_notional))
            if qty <= 0 or (qty * entry) < min_notional:
                continue
            rows.append({
                "symbol": symbol, "side": "buy",
                "source": "first_candle_rule_935", "entry_type": "first_5m_close",
                "price": round(entry, 4), "stop_price": round(stop, 4),
                "target_price": round(target, 4),
                "risk_reward_ratio": float(sig.get("risk_reward_ratio", cfg.fcr_target_r)),
                "order_qty": int(qty), "reason": "first_candle_rule_935",
            })
        return pd.DataFrame(rows)


# ── production order executor ────────────────────────────────────────────────────
class ProductionOrderExecutor:
    """
    Bracket-first order executor for the legacy intraday path. Converts premarket
    trade plans to executable bracket rows, applies an RL timing gate that never
    removes a complete bracket buy, and executes plans through a bracket-capable
    broker. All patchable journal hooks are reached via the `run` module.
    """

    def __init__(self, config: ExecutorConfig | None = None, broker=None, scanner=None):
        self.config = config or build_default_config()
        self.broker = broker
        self.scanner = scanner

    # ── premarket plan → executable bracket rows ────────────────────────────────
    def _premarket_plan_to_order_rows(self, payload: dict, equity: float) -> list[dict]:
        cfg = self.config
        max_order_dollars = float(getattr(cfg, "max_order_dollars", 0) or 0)
        max_qty = int(getattr(cfg, "max_single_order_qty", 0) or 0)
        min_notional = float(getattr(cfg, "min_notional_to_trade", 0) or 0)
        risk_pct = float(getattr(cfg, "risk_per_trade_pct", 0.01) or 0.01)
        tick = float(getattr(cfg, "bracket_stop_tick", 0.01) or 0.01)
        max_r = float(getattr(cfg, "max_target_r_multiple", 5.0) or 5.0)

        plans = (payload or {}).get("raw_premarket_pattern_plans") or []
        rows: list[dict] = []
        seen: set[str] = set()
        for p in plans:
            if not isinstance(p, dict) or not p.get("valid"):
                continue
            symbol = str(p.get("symbol") or "").upper()
            if not symbol or symbol in seen:
                continue
            entry = _num(p.get("entry_price"))
            raw_stop = _num(p.get("stop_loss"))
            # long-only bracket: entry above the structural stop
            if entry <= 0 or raw_stop <= 0 or raw_stop >= entry:
                continue
            stop = round(raw_stop - tick, 4)        # one tick below the plan stop
            risk = entry - stop
            if risk <= 0:
                continue
            raw_rr = _num(p.get("risk_reward_ratio"))
            target_r = min(raw_rr, max_r) if raw_rr > 0 else max_r   # cap target R
            target = round(entry + target_r * risk, 4)

            qty_by_risk = int((equity * risk_pct) / risk) if risk > 0 else 0
            qty_by_notional = int(max_order_dollars / entry) if (max_order_dollars > 0 and entry > 0) else qty_by_risk
            qty = min(qty_by_risk, qty_by_notional) if qty_by_notional > 0 else qty_by_risk
            if max_qty > 0:
                qty = min(qty, max_qty)
            qty = max(0, qty)
            if qty <= 0 or (qty * entry) < min_notional:
                continue

            seen.add(symbol)
            rows.append({
                "symbol": symbol, "side": "buy", "order_qty": int(qty),
                "price": round(entry, 4), "stop_price": stop, "target_price": target,
                "target_r_multiple": round(target_r, 4),
                "client_order_id": f"qq_{DATA_TAG}_premarket_{symbol}_buy",
                "source": "premarket_trade_plan", "reason": p.get("reason", ""),
            })
        return rows

    # ── RL timing gate (never removes a complete bracket buy) ───────────────────
    def _apply_rl_timing_gate(self, rebalance: pd.DataFrame, exits: pd.DataFrame):
        import run
        if rebalance is None or len(rebalance) == 0:
            return rebalance, exits
        scanner_cfg = getattr(self.config, "scanner", None)
        if not getattr(scanner_cfg, "use_rl_timing_gate", False):
            return rebalance, exits
        shadow = bool(getattr(scanner_cfg, "rl_shadow_mode", False))

        buy_syms = [str(r.get("symbol")) for _, r in rebalance.iterrows()
                    if str(r.get("side", "")).lower() == "buy"]
        try:
            bars = self._download_latest_bars_for_rl(buy_syms)
        except Exception:
            bars = pd.DataFrame()
        try:
            structure = self._load_rl_structure_map()
        except Exception:
            structure = {}
        positions: dict = {}
        broker = getattr(self, "broker", None)
        try:
            if broker is not None and hasattr(broker, "get_all_open_positions"):
                positions = {getattr(p, "symbol", None): p
                             for p in (broker.get_all_open_positions() or [])}
        except Exception:
            positions = {}

        kept: list[dict] = []
        for _, r in rebalance.iterrows():
            row = r.to_dict()
            if str(row.get("side", "")).lower() != "buy":
                kept.append(row)
                continue
            try:
                sig = self._predict_rl_signal(row.get("symbol"), bars, structure, positions) or {}
            except Exception:
                sig = {}
            action = str(sig.get("action", "")).upper()
            if action == "BUY":
                kept.append(row)
                continue
            # RL did not confirm a BUY this cycle.
            if run.row_has_complete_bracket_plan(row):
                # Never remove a complete bracket buy — keep it, attach a warning.
                row["rl_warning"] = "rl_timing_not_confirmed"
                kept.append(row)
            elif shadow:
                kept.append(row)
            # else: drop the unconfirmed, non-bracketed buy

        out = pd.DataFrame(kept)
        try:
            run.write_event("rl_timing_gate", {"buys": len(buy_syms), "kept": int(len(out))})
            run.save_df(out, run.LOG_DIR / f"{run.DATA_TAG}_rl_timing_gate.csv")
        except Exception:
            pass
        return out, exits

    # ── live-experience logging (failure-tolerant) ──────────────────────────────
    def _safe_log_live_experience(self, **kwargs) -> None:
        """Call _log_live_experience but never let a logging failure abort an order."""
        try:
            log = getattr(self, "_log_live_experience", None)
            if callable(log):
                log(**kwargs)
        except Exception:
            pass

    # ── execute an order plan (bracket-first, never naked market by default) ─────
    def _execute_order_plan(self, plan: pd.DataFrame, cycle_id: str) -> pd.DataFrame:
        import run
        if plan is None or len(plan) == 0:
            return pd.DataFrame()

        min_notional = float(getattr(self.config, "min_notional_to_trade", 0.0) or 0.0)
        tif = str(getattr(self.config, "time_in_force", "day") or "day")
        broker = self.broker
        results: list[dict] = []

        for _, prow in plan.iterrows():
            row = prow.to_dict()
            symbol = str(row.get("symbol") or "")
            side = str(row.get("side", "buy")).lower()
            qty = int(_num(row.get("order_qty")))
            price = _num(row.get("price"))
            stop_price = _num(row.get("stop_price"))
            target_price = _num(row.get("target_price"))
            client_order_id = row.get("client_order_id") or f"qq_{DATA_TAG}_{symbol}_{side}"
            reason = row.get("reason", "")

            # decision-stage sample — must never break execution
            self._safe_log_live_experience(
                stage="decision", cycle_id=cycle_id, symbol=symbol, side=side,
                order_qty=qty, price=price, stop_price=stop_price,
                target_price=target_price, reason=reason)

            if broker is not None and hasattr(broker, "has_open_order_for_symbol") \
                    and broker.has_open_order_for_symbol(symbol):
                continue
            if qty <= 0 or price <= 0 or (qty * price) < min_notional:
                continue

            has_bracket = bool(run._BRACKET_ORDERS_AVAILABLE) and run.row_has_complete_bracket_plan(row)
            if has_bracket:
                resp = broker.submit_bracket_buy(symbol, qty, stop_price, target_price, client_order_id, tif)
            elif side == "sell":
                if not hasattr(broker, "submit_market_sell"):
                    run.write_event("executor_sell_not_supported", {"symbol": symbol, "qty": qty})
                    continue
                resp = broker.submit_market_sell(symbol, qty, client_order_id, tif)
            else:
                resp = broker.submit_market_buy(symbol, qty, client_order_id, tif)
            bracket_attached = bool(has_bracket)

            order_id = getattr(resp, "id", None)
            fill = self._poll_fill(order_id, price) or {}
            status = str(fill.get("fill_status", getattr(resp, "status", "submitted")))
            filled_qty = fill.get("filled_qty", 0)
            filled_avg_price = fill.get("filled_avg_price", 0.0)

            # submitted-stage sample — must never break execution
            self._safe_log_live_experience(
                stage="submitted", cycle_id=cycle_id, symbol=symbol, side=side,
                order_qty=qty, price=price, filled_qty=filled_qty,
                filled_avg_price=filled_avg_price, status=status,
                bracket_attached=bracket_attached)

            run.log_executed_order(
                cycle_id=cycle_id, symbol=symbol, side=side, qty=qty, price=price,
                status=status, bracket_attached=bracket_attached, client_order_id=client_order_id)
            run.write_event("order_executed", {
                "symbol": symbol, "status": status, "bracket_attached": bracket_attached})

            results.append({
                "symbol": symbol, "side": side, "order_qty": qty, "price": price,
                "stop_price": stop_price, "target_price": target_price,
                "status": status, "filled_qty": filled_qty,
                "filled_avg_price": filled_avg_price, "bracket_attached": bracket_attached,
                "client_order_id": client_order_id, "reason": reason,
            })

        return pd.DataFrame(results)
