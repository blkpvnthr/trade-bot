#!/usr/bin/env python3
"""Profit-First Daily Engine v2 entrypoint.

``run.py`` remains the canonical safety substrate.  This entrypoint adds a
buy-side opportunity router and keeps exits exit-first/unconditional.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import run
from daily_profit_engine import DailyOpportunityRouter, RouterConfig, TradeOpportunity

logger = logging.getLogger("profit_engine_v2")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def engine_enabled() -> bool:
    return _env_bool("DAILY_PROFIT_ENGINE_ENABLED", True)


def paper_mode_allowed() -> bool:
    if not _env_bool("DAILY_PROFIT_ENGINE_PAPER_ONLY", True):
        return True
    mode = os.getenv("MODE", getattr(run, "MODE", "paper")).strip().lower()
    paper_flag = bool(getattr(run, "ALPACA_PAPER_ONLY", True))
    return mode != "live" and paper_flag


def live_orders_allowed() -> bool:
    mode = os.getenv("MODE", getattr(run, "MODE", "paper")).strip().lower()
    return mode == "live" and not _env_bool("DAILY_PROFIT_ENGINE_PAPER_ONLY", True)


def _date_tag() -> str:
    return str(getattr(run, "DATA_TAG", datetime.now().strftime("%Y%m%d")))


def funnel_path() -> Path:
    base = os.getenv("DAILY_PROFIT_ENGINE_LOG_DIR")
    if base:
        return Path(base) / f"{_date_tag()}_daily_profit_funnel.json"
    return Path(getattr(run, "LOG_DIR", Path("logs/execution") / _date_tag())) / f"{_date_tag()}_daily_profit_funnel.json"


def demo_opportunities() -> list[TradeOpportunity]:
    return [
        TradeOpportunity(
            symbol="SPY",
            asset_class="equity",
            strategy_id="alpaca_daily_income",
            setup_type="VWAP_RECLAIM",
            score=84,
            probability=0.76,
            expected_r=2.2,
            liquidity_score=0.95,
            regime_alignment=0.90,
            risk_score=0.15,
            freshness_score=1.0,
            qty=1,
            reference_price=500.0,
            reason="demo:vwap_reclaim",
            overnight_candidate=False,
            metadata={"tactical": True, "trend_intact": True},
        ),
        TradeOpportunity(
            symbol="QQQ260117C00450000",
            asset_class="option",
            strategy_id="alpha_call",
            setup_type="ORB",
            score=92,
            probability=0.83,
            expected_r=2.7,
            liquidity_score=0.85,
            regime_alignment=0.85,
            risk_score=0.25,
            freshness_score=1.0,
            qty=1,
            reference_price=2.50,
            limit_price=2.50,
            reason="demo:alpha_call",
            overnight_candidate=True,
            metadata={
                "tactical": True,
                "dte": 30,
                "iv_crush_risk": 15,
                "protected_or_partialed": True,
                "active_exit_ladder": True,
            },
        ),
    ]


def _append_skip_logs(result) -> None:
    for rejected in result.rejected:
        try:
            run.log_skip(
                rejected.symbol,
                rejected.blocked_by,
                source="daily_profit_engine_v2",
                note=rejected.reason,
                setup_type=rejected.opportunity.setup_type,
            )
        except Exception:
            pass


def execute_approved(gateway: Any, opportunities: Iterable[TradeOpportunity]) -> int:
    placed = 0
    for opp in opportunities:
        qty = 1 if opp.asset_class == "option" else max(1, int(opp.qty or 1))
        kwargs = {
            "reason": f"daily_profit_v2:{opp.strategy_id}:{opp.setup_type}:{opp.reason}",
            "reference_price": float(opp.reference_price or opp.limit_price or 0.0),
        }
        if opp.asset_class == "option":
            kwargs.update({"limit_price": float(opp.limit_price or opp.reference_price), "order_type": "limit"})
        resp = gateway.submit_market_order(opp.symbol, "buy", qty, **kwargs)
        if resp is not None:
            placed += 1
    return placed


def run_v2_cycle(
    alpha: Any,
    bal: Any,
    monitor: Any,
    gateway: Any,
    risk: Any,
    waterfall: Any,
    market_context: dict[str, Any],
) -> dict[str, Any]:
    if not engine_enabled():
        return run.run_two_engine_cycle(risk, gateway, waterfall, alpha, bal, monitor, market_context)
    if not paper_mode_allowed():
        raise SystemExit("DAILY_PROFIT_ENGINE_PAPER_ONLY=true refuses live mode by default")

    # Keep the canonical sync and EXIT-FIRST behavior in run.py.
    risk.sync_account(gateway.get_account())
    raw_positions = gateway.get_positions_raw()
    option_positions, equity_positions, opt_exp, stk_exp = gateway.split_positions(raw_positions)
    risk.sync_exposure(opt_exp, stk_exp)

    summary = {
        "alpha_exits": 0,
        "bal_exits": 0,
        "regime_exits": 0,
        "ema_exits": 0,
        "insider_copy_exits": 0,
        "option_safety_exits": 0,
        "v2_entries": 0,
    }
    if alpha is not None:
        summary["alpha_exits"] = run.run_alpha_exits(alpha, gateway, waterfall, option_positions, market_context, risk=risk)
    if bal is not None:
        summary["bal_exits"] = run.run_bal_exits(bal, gateway, equity_positions, market_context)
    if getattr(run, "ENABLE_MARKET_REGIME", False):
        summary["regime_exits"] = run.run_regime_exit(gateway, equity_positions)
    if getattr(run, "ENABLE_EMA_EXIT", False):
        summary["ema_exits"] = run.run_ema_exits(gateway, equity_positions)
    if getattr(run, "ENABLE_INSIDER_COPIER", False):
        summary["insider_copy_exits"] = run.run_insider_copy_exits(gateway, risk, option_positions, equity_positions)
    summary["option_safety_exits"] = run.run_option_safety_exits(gateway, risk, option_positions, alpha)

    if os.getenv("DAILY_PROFIT_ENGINE_DEMO") == "1":
        opportunities = demo_opportunities()
        stale_state = feed_timeout = missing_data = False
    else:
        collected = run.collect_daily_profit_opportunities(
            alpha=alpha,
            bal=bal,
            gateway=gateway,
            risk=risk,
            market_context=market_context,
            equity_positions=equity_positions,
            option_positions=option_positions,
        )
        opportunities = collected.get("opportunities", [])
        stale_state = bool(collected.get("stale_state"))
        feed_timeout = bool(collected.get("feed_timeout"))
        missing_data = bool(collected.get("missing_data"))
        # Optional intraday-reversal candidate source (isolated; gated OFF by default via
        # INTRADAY_REVERSAL_ENABLED, only after 1:30 PM ET, reads its own universe feed).
        # Extends the candidate pool; the router/risk/exec gates still apply unchanged.
        try:
            import intraday_reversal_engine as _ire
            _rev = _ire.collect_reversal_opportunities(market_context=market_context)
            if _rev:
                opportunities = list(opportunities) + list(_rev)
                logger.info("v2: +%d intraday-reversal candidates", len(_rev))
        except Exception as exc:
            logger.debug("intraday reversal candidate source skipped: %s", exc)

    router = DailyOpportunityRouter(RouterConfig())
    result = router.route(
        opportunities,
        market_context=market_context,
        stale_state=stale_state,
        feed_timeout=feed_timeout,
        missing_data=missing_data,
        open_intraday_tactical_positions=len(equity_positions) + len(option_positions),
    )
    _append_skip_logs(result)
    summary["v2_entries"] = execute_approved(gateway, result.approved)
    path = router.write_funnel(
        funnel_path(),
        result,
        actual_trades=summary["v2_entries"],
        realized_next_day_outcome={"checked": False, "asof_utc": datetime.now(timezone.utc).isoformat()},
    )
    summary["daily_profit_funnel_path"] = str(path)
    return summary


def _build_context_safe() -> dict[str, Any]:
    try:
        return run._build_context_safe()
    except Exception as exc:
        logger.warning("v2 market context unavailable; safe-mode context used: %s", exc)
        return {
            "overall_market_bias": "bearish",
            "trade_permission": "halt_new_buys",
            "risk_level": "high",
            "decision_reasons": ["v2_context_unavailable_safe_mode"],
        }


def run_v2_trading() -> None:
    risk = run.RiskManager(equity=run.STARTING_EQUITY)
    gateway = run.AlpacaGateway(risk)

    logger.info("PROFIT_ENGINE_V2_ACTIVE=true")
    logger.info("=== PROFIT-FIRST DAILY ENGINE V2  mode=%s ===", run.MODE)
    run.seed_risk_from_broker(risk, gateway)
    run.log_account_stats(gateway, risk)
    run._log_config(risk)
    run.ensure_outputs_exist()
    alpha, bal, monitor, waterfall = run.build_engines(gateway)
    try:
        run._start_tv_screen_refresher()
    except Exception as exc:
        logger.debug("TV refresher skipped in v2: %s", exc)

    market_context: dict[str, Any] = {}
    last_context_build = 0.0
    last_account_log = time.time()
    cycle_count = 0
    logger.info("Entering v2 loop (Ctrl-C to stop) ...")
    try:
        import threading as _threading
        run._LOOP_HEARTBEAT[0] = time.time()
        run._start_loop_watchdog(_threading.get_ident())
    except Exception as exc:
        logger.debug("v2 watchdog skipped: %s", exc)

    while True:
        try:
            cycle_count += 1
            run._LOOP_HEARTBEAT[0] = time.time()
            if risk.is_killed:
                logger.critical("Kill switch active — exiting v2 loop")
                break
            now = time.time()
            if now - last_account_log >= run.ACCOUNT_STATS_LOG_SEC:
                run.log_account_stats(gateway, risk, label=f"V2 Account cycle={cycle_count}")
                last_account_log = now
            if not run.is_market_open():
                if run.is_premarket():
                    if not market_context:
                        market_context = _build_context_safe()
                    try:
                        run.run_premarket_phase(gateway, bal, market_context)
                    except Exception as exc:
                        logger.debug("v2 premarket phase skipped: %s", exc)
                    time.sleep(run.PREMARKET_REFRESH_SEC)
                    continue
                logger.debug("Market closed — v2 sleeping 60s")
                time.sleep(60)
                continue
            try:
                run._stop_premarket_feed()
            except Exception:
                pass
            if (time.time() - last_context_build) > run.CONTEXT_REFRESH_MIN * 60 or not market_context:
                market_context = _build_context_safe()
                last_context_build = time.time()
            run_v2_cycle(alpha, bal, monitor, gateway, risk, waterfall, market_context)
            time.sleep(run.BAR_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Interrupted by user — finalising v2 outputs")
            break
        except Exception:
            logger.exception("V2 loop error — continuing")
            try:
                run.log_account_stats(gateway, risk, label=f"V2 Account after_error cycle={cycle_count}")
            except Exception:
                pass
            time.sleep(max(5.0, run.BAR_INTERVAL_SEC))

    try:
        run._stop_premarket_feed()
    except Exception:
        pass
    if waterfall is not None:
        run.write_capital_waterfall(waterfall)
    run.ensure_outputs_exist()
    logger.info("V2 session complete. Risk: equity=%.2f daily_pnl=%.2f opt_exp=%.0f stk_exp=%.0f",
                risk.equity, risk.daily_pnl, risk.options_exposure, risk.stock_exposure)


def run_demo_cycle() -> Path:
    if not paper_mode_allowed():
        raise SystemExit("DAILY_PROFIT_ENGINE_PAPER_ONLY=true refuses live mode by default")
    router = DailyOpportunityRouter(RouterConfig())
    result = router.route(demo_opportunities(), market_context={"overall_market_bias": "bullish", "trade_permission": "normal"})
    path = router.write_funnel(
        funnel_path(),
        result,
        actual_trades=0,
        realized_next_day_outcome={"checked": False, "demo": True},
    )
    print(f"FUNNEL_JSON={path}")
    print(json.dumps(router.funnel_payload(result, actual_trades=0, realized_next_day_outcome={"checked": False, "demo": True}), indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profit-First Daily Engine v2")
    parser.add_argument("--demo-cycle", action="store_true", help="run a local paper-safe demo cycle and write funnel JSON")
    args = parser.parse_args(argv)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("logs/run.log", mode="a")],
    )
    if not paper_mode_allowed():
        print("DAILY_PROFIT_ENGINE_PAPER_ONLY=true refuses live mode by default", file=sys.stderr)
        return 2
    if args.demo_cycle:
        run_demo_cycle()
        return 0
    if not engine_enabled():
        return run.main() if hasattr(run, "main") else 0
    run_v2_trading()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
