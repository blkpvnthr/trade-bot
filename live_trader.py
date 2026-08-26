#!/usr/bin/env python3
"""
V2.9.8 live trader CLI.
=======================

Commands
--------
    python live_trader.py status
    python live_trader.py signal                 # READ-ONLY: compute + persist, 0 orders
    python live_trader.py rebalance --dry-run
    python live_trader.py rebalance --paper
    python live_trader.py rebalance --live --confirm-live LIVE
    python live_trader.py cancel-open-orders --paper
    python live_trader.py cancel-open-orders --live --confirm-live LIVE
    python live_trader.py flatten --paper --confirm-live FLATTEN
    python live_trader.py flatten --live  --confirm-live FLATTEN

`signal` NEVER submits an order. Only `rebalance` executes. LIVE requires ALL of
``--live`` + ``--confirm-live LIVE`` + ``ENABLE_LIVE_TRADING=true`` (or the repo's
``ALLOW_LIVE_TRADING``); missing any prints ``LIVE TRADING BLOCKED`` and exits
nonzero. Paper and live environments never cross (#19).

This module also exposes the shared orchestration helpers (`build_engine`,
`execute_rebalance`, `print_report`) that ``run_engine.py`` reuses so there is
exactly one execution path (#20).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_env() -> str:
    """Load exactly one env file: ENV_FILE=<path> (override) or PROJECT_ROOT/.env."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return "(python-dotenv not installed)"
    ep = os.environ.get("ENV_FILE", "").strip()
    if ep:
        p = Path(ep)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        load_dotenv(p, override=True)
        return str(p)
    load_dotenv(PROJECT_ROOT / ".env")
    return str(PROJECT_ROOT / ".env")


# BLAS single-threading for reproducible SLSQP (must precede numpy import).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ENV_FILE = load_env()

from strat_allocator import AllocatorConfig
from execution_store import ExecutionStore
from live_engine import LiveEngine, ExecutionConfig, Mode
from bond_regime import BondOverlayConfig

log = logging.getLogger("v298.cli")


class LiveBlocked(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Mode resolution + hard live gate
# ---------------------------------------------------------------------------
def resolve_mode(args, ecfg: ExecutionConfig) -> Mode:
    """
    Map CLI flags to a Mode with the three-gate live protection. Never infers or
    auto-falls back between paper and live.
    """
    picks = [("dry_run", getattr(args, "dry_run", False)),
             ("paper", getattr(args, "paper", False)),
             ("live", getattr(args, "live", False))]
    chosen = [name for name, on in picks if on]
    if len(chosen) != 1:
        raise LiveBlocked("choose exactly one of --dry-run / --paper / --live")
    which = chosen[0]

    if which == "dry_run":
        return Mode.DRY_RUN
    if which == "paper":
        return Mode.PAPER

    # LIVE: require all three gates.
    reasons = []
    if getattr(args, "confirm_live", None) != "LIVE":
        reasons.append("--confirm-live LIVE missing")
    if not ecfg.live_enabled:
        reasons.append("ENABLE_LIVE_TRADING=true (or ALLOW_LIVE_TRADING) not set")
    paper_only = os.environ.get("ALPACA_PAPER_ONLY", "true").strip().lower() not in ("0", "false", "no", "off")
    if paper_only:
        reasons.append("ALPACA_PAPER_ONLY must be explicitly false")
    if os.environ.get("MODE", "paper").strip().lower() != "live":
        reasons.append("MODE=live not set")
    if reasons:
        raise LiveBlocked("LIVE TRADING BLOCKED: " + "; ".join(reasons))
    return Mode.LIVE


def build_engine(mode: Mode, *, need_broker: bool = True,
                 acfg: AllocatorConfig | None = None,
                 ecfg: ExecutionConfig | None = None,
                 store: ExecutionStore | None = None) -> LiveEngine:
    """Construct the engine with real adapters. paper flag is explicit per mode."""
    acfg = acfg or AllocatorConfig()
    ecfg = ecfg or ExecutionConfig.from_env()
    store = store or ExecutionStore()

    broker = data = None
    if need_broker:
        try:
            from alpaca_data import AlpacaData
            from alpaca_broker import AlpacaBroker
            data = AlpacaData()
            paper = (mode != Mode.LIVE)      # explicit; LIVE => paper=False only
            broker = AlpacaBroker(paper=paper)
        except Exception as e:
            if mode == Mode.DRY_RUN:
                log.warning("no broker/data (%s); dry-run will compute-only where possible", e)
                broker = data = None
            else:
                raise
    bond = BondOverlayConfig.from_env()
    if bond.enabled:
        log.warning("BOND_REGIME_OVERLAY is ON -- this changes the frozen strategy "
                    "and is NOT falsification-validated (see LIVE_TRADING.md).")
    return LiveEngine(mode, ecfg, acfg, store, broker=broker, data=data, bond_overlay=bond)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def print_report(rep) -> None:
    d = rep.to_dict() if hasattr(rep, "to_dict") else rep
    line = "=" * 84
    print(line)
    print("V2.9.8 LIVE ENGINE")
    print(line)
    print(f"Mode             : {d['mode'].upper()}{'  (DRY RUN)' if d.get('dry_run') else ''}")
    print(f"As of close      : {d.get('as_of_close')}")
    print(f"Target session   : {d.get('target_session')}")
    print(f"Forecast         : {d.get('forecast_source')}")
    print(f"State            : {d.get('recovery_state')}   severe_bear={d.get('severe_bear')}")
    print(f"Vol budget       : {_pct(d.get('vol_budget'))}   target ex-ante {_pct(d.get('ex_ante_vol'))}")
    print(f"Leverage cap     : {_x(d.get('leverage_cap'))}   growth ceiling {_x(d.get('growth_ceiling'))}")
    print(f"Current gross    : {_x(d.get('current_gross'))}   target gross {_x(d.get('target_gross'))}")
    print(f"Equity           : {_usd(d.get('equity'))}")
    print(f"Decision hash    : {d.get('decision_hash')}")
    print(f"Rebalanced       : {d.get('rebalanced')}   reason={d.get('rebalance_reason', 'scheduled')}"
          f"   executed={d.get('executed')}")
    for ev in d.get("loss_replacement") or []:
        print(f"Loss-replacement : {ev['exited_asset']} (pr {ev['position_return']:+.2%} <= "
              f"{ev['threshold']:.2%}) -> {ev['replacement_asset'] or 'CASH'}  [{ev['reason']}]")
    for ev in d.get("exhaustion") or []:
        print(f"Exhaustion exit  : {ev['exited_asset']} (score {ev['exhaustion_score']:.2f} "
              f"{ev['state']}) -> {ev['replacement_asset'] or 'CASH'}  [{ev['reason']}]")
    for ev in d.get("adaptive") or []:
        print(f"Adaptive rotate  : {ev['exited_asset']} (dmg {ev['damage']} regime {ev['regime']}) "
              f"-> {ev['replacement_asset'] or 'CASH'}  net_edge {ev['net_rotation_edge']:+.3f}")
    if d.get("bond_regime"):
        b = d["bond_regime"]
        print(f"Bond overlay     : risk_off_z={b.get('risk_off_z', 0):+.2f}  "
              f"vol x{b.get('vol_budget_mult', 1):.3f}  growth x{b.get('growth_ceiling_mult', 1):.3f}  "
              f"[ftq {b.get('z_ftq', 0):+.1f} credit {b.get('z_credit', 0):+.1f} "
              f"bondvol {b.get('z_bondvol', 0):+.1f}]")
    if d.get("kill_switch"):
        print(f"KILL SWITCH      : {d['kill_switch']}")
    plans = d.get("plans") or []
    if plans:
        print("-" * 84)
        print(f"{'Symbol':<7}{'CurWt':>8}{'TgtWt':>8}{'Current$':>12}{'Target$':>12}"
              f"{'Delta$':>12}  Action")
        for p in plans:
            action = f"{p['side'].upper()} " + (f"${p['notional']:,.0f}" if p.get('notional')
                                                else f"{p['qty']:.4f}sh")
            print(f"{p['symbol']:<7}{p['current_weight']:>8.3f}{p['target_weight']:>8.3f}"
                  f"{p['current_notional']:>12,.0f}{p['target_notional']:>12,.0f}"
                  f"{p['delta_notional']:>12,.0f}  {action}")
    if d.get("rejected_reason"):
        print("-" * 84)
        print(f"REJECTED: {d['rejected_reason']}")
    if d.get("skipped_reason"):
        print("-" * 84)
        print(f"SKIPPED: {d['skipped_reason']}")
    if d.get("final_drift"):
        worst = max(abs(v) for v in d["final_drift"].values()) if d["final_drift"] else 0.0
        print(f"Max realized drift vs target: {worst:.2%}")
    for n in d.get("notes", []):
        print(f"note: {n}")
    print(line)


def _pct(x):
    return f"{x:.2%}" if isinstance(x, (int, float)) else "n/a"


def _x(x):
    return f"{x:.2f}x" if isinstance(x, (int, float)) else "n/a"


def _usd(x):
    return f"${x:,.2f}" if isinstance(x, (int, float)) else "n/a"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_status(args) -> int:
    ecfg = ExecutionConfig.from_env()
    mode = Mode.LIVE if args.live else Mode.PAPER
    eng = build_engine(mode, need_broker=True)
    st = eng.status()
    print(json.dumps(st, indent=2, default=str))
    return 0


def cmd_signal(args) -> int:
    """READ-ONLY: compute + persist the decision, submit ZERO orders."""
    ecfg = ExecutionConfig.from_env()
    mode = Mode.LIVE if args.live else Mode.PAPER
    eng = build_engine(mode, need_broker=True)
    # READ-ONLY: read positions so the overlay preview reflects real cost basis.
    positions = None
    if eng.broker is not None:
        try:
            positions = eng.broker.positions()
        except Exception as e:
            log.warning("could not read positions for signal preview (%s)", e)
    dec = eng.compute_decision(positions=positions)
    print("=" * 84)
    print("V2.9.8 SIGNAL (read-only -- NO orders submitted)")
    print("=" * 84)
    print(f"As of close      : {dec.as_of_close.date()}")
    print(f"Target session   : {dec.target_session.date() if dec.target_session is not None else 'n/a'}")
    print(f"Forecast         : {dec.forecast_source}")
    print(f"State            : {dec.recovery_state}   severe_bear={dec.severe_bear}")
    print(f"Vol budget       : {_pct(dec.vol_budget)}   target ex-ante {_pct(dec.ex_ante_vol)}")
    print(f"Leverage cap     : {_x(dec.leverage_cap)}   growth ceiling {_x(dec.growth_ceiling)}")
    print(f"Target gross     : {_x(dec.gross_exposure)}   rebalanced={dec.rebalanced}")
    print(f"Rebalance reason : {getattr(dec, 'rebalance_reason', 'scheduled')}")
    for ev in getattr(dec, "loss_replacement_events", []) or []:
        print(f"  loss-replace   : {ev['exited_asset']} (pr {ev['position_return']:+.2%}) "
              f"-> {ev['replacement_asset'] or 'CASH'} "
              f"({ev['reason']})")
    print(f"Decision hash    : {dec.decision_hash}")
    print("-" * 84)
    print(f"{'Symbol':<8}{'TargetWt':>10}")
    for a, w in dec.target_weights.items():
        print(f"{a:<8}{w:>10.4f}")
    print("=" * 84)
    return 0


def cmd_rebalance(args) -> int:
    ecfg = ExecutionConfig.from_env()
    try:
        mode = resolve_mode(args, ecfg)
    except LiveBlocked as e:
        print(str(e), file=sys.stderr)
        return 2
    return execute_rebalance(mode, dry_run=(mode == Mode.DRY_RUN), ecfg=ecfg)


def execute_rebalance(mode: Mode, *, dry_run: bool, ecfg: ExecutionConfig | None = None,
                      acfg: AllocatorConfig | None = None) -> int:
    """Shared execution path used by the CLI AND run_engine.py (#20)."""
    ecfg = ecfg or ExecutionConfig.from_env()
    eng = build_engine(mode, need_broker=True, ecfg=ecfg, acfg=acfg)
    rep = eng.run_rebalance(dry_run=dry_run)
    print_report(rep)
    if rep.rejected_reason:
        return 3
    return 0


def cmd_cancel(args) -> int:
    ecfg = ExecutionConfig.from_env()
    if args.live and (args.confirm_live != "LIVE" or not ecfg.live_enabled):
        print("LIVE TRADING BLOCKED: cancel-open-orders --live needs --confirm-live LIVE "
              "and ENABLE_LIVE_TRADING=true", file=sys.stderr)
        return 2
    mode = Mode.LIVE if args.live else Mode.PAPER
    eng = build_engine(mode, need_broker=True)
    n = eng.broker.cancel_all_open_orders()
    print(f"cancelled {n} open order(s) in {mode.value}")
    return 0


def cmd_flatten(args) -> int:
    """Explicit operator action ONLY. Never invoked automatically."""
    ecfg = ExecutionConfig.from_env()
    if args.confirm_live != "FLATTEN":
        print("FLATTEN BLOCKED: requires --confirm-live FLATTEN", file=sys.stderr)
        return 2
    if args.live and not ecfg.live_enabled:
        print("LIVE TRADING BLOCKED: flatten --live needs ENABLE_LIVE_TRADING=true",
              file=sys.stderr)
        return 2
    mode = Mode.LIVE if args.live else Mode.PAPER
    eng = build_engine(mode, need_broker=True)
    eng.broker.cancel_all_open_orders()
    closed = eng.broker.close_all_positions(cancel_orders=True)
    print(f"FLATTEN {mode.value}: submitted {len(closed)} close order(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="V2.9.8 live trader")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="account + positions + target + drift (read-only)")
    s.add_argument("--live", action="store_true")
    s.set_defaults(func=cmd_status)

    g = sub.add_parser("signal", help="compute + persist the decision, 0 orders (read-only)")
    g.add_argument("--live", action="store_true")
    g.set_defaults(func=cmd_signal)

    r = sub.add_parser("rebalance", help="reconcile actual positions to target weights")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--paper", action="store_true")
    r.add_argument("--live", action="store_true")
    r.add_argument("--confirm-live", default=None)
    r.set_defaults(func=cmd_rebalance)

    c = sub.add_parser("cancel-open-orders", help="cancel all open orders")
    c.add_argument("--paper", action="store_true")
    c.add_argument("--live", action="store_true")
    c.add_argument("--confirm-live", default=None)
    c.set_defaults(func=cmd_cancel)

    f = sub.add_parser("flatten", help="EMERGENCY: cancel orders and close all positions")
    f.add_argument("--paper", action="store_true")
    f.add_argument("--live", action="store_true")
    f.add_argument("--confirm-live", default=None)
    f.set_defaults(func=cmd_flatten)
    return ap


def main(argv=None) -> int:
    logging.basicConfig(level=os.environ.get("V298_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
