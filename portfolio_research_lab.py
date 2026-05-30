#!/usr/bin/env python3
"""
portfolio_research_lab.py — Evidence & Performance Analytics Layer.

The system already makes decisions. This module measures which decisions actually
create returns, so optimisation can be driven by measured expectancy / returns /
capital efficiency / risk-adjusted performance instead of opinion.

Every Alpha trade, Alpha rejection, Alpha exit, Buffett accumulation, Buffett
rejection, DCA tranche, and insider-discovered candidate becomes a research event
appended to:

    daily_system_outputs/<YYYYMMDD>/research_events.jsonl

Analytics + rankings + a $100k walk-forward simulation are written to:

    daily_system_outputs/<YYYYMMDD>/research_lab.json
    daily_system_outputs/<YYYYMMDD>/alpha_expectancy_report.json
    daily_system_outputs/<YYYYMMDD>/buffett_performance_report.json
    daily_system_outputs/<YYYYMMDD>/capital_waterfall_performance.json
    daily_system_outputs/<YYYYMMDD>/walkforward_report.json

Design: the analytics are PURE functions over a list of event dicts (offline-testable).
Recording and report generation are wrapped so they can NEVER break the trading loop.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"


def research_dir(date_tag: str | None = None) -> Path:
    d = DAILY_OUTPUTS / (date_tag or DATA_TAG)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _path(name: str, date_tag: str | None = None) -> Path:
    return research_dir(date_tag) / name


RESEARCH_EVENTS = "research_events.jsonl"
RESEARCH_LAB = "research_lab.json"
ALPHA_EXPECTANCY_REPORT = "alpha_expectancy_report.json"
BUFFETT_PERFORMANCE_REPORT = "buffett_performance_report.json"
CAPITAL_WATERFALL_PERFORMANCE = "capital_waterfall_performance.json"
WALKFORWARD_REPORT = "walkforward_report.json"

EVENT_TYPES = {
    "alpha_entry", "alpha_exit", "alpha_rejection",
    "bal_buy", "bal_dca", "bal_rejection",
    "insider_discovery", "conviction_update",
}


# ── event recording ──────────────────────────────────────────────────────────────
def record_event(event_type: str, symbol: str, *, date_tag: str | None = None, **fields) -> dict:
    """Append one research event to research_events.jsonl. Returns the event dict."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date_tag": date_tag or DATA_TAG,
        "symbol": str(symbol or "").upper(),
        "event_type": event_type,
        "regime": fields.pop("regime", ""),
        "sector": fields.pop("sector", ""),
    }
    event.update(fields)
    try:
        with open(_path(RESEARCH_EVENTS, date_tag), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as exc:
        logger.debug("record_event failed: %s", exc)
    return event


def read_events(date_tag: str | None = None) -> list[dict]:
    """Read all recorded research events for a day (empty list on any failure)."""
    p = _path(RESEARCH_EVENTS, date_tag)
    if not p.exists():
        return []
    out = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _regime(mc: dict) -> str:
    return str((mc or {}).get("overall_market_bias") or (mc or {}).get("regime") or "").lower()


def _vol_regime(mc: dict) -> str:
    return str((mc or {}).get("volatility_regime") or (mc or {}).get("vix_regime") or "").lower()


# Convenience recorders (accept the existing engine dict/dataclass shapes) ──────────
def record_alpha_signal(sig: dict, market_context: dict | None = None, placed: bool = False,
                        date_tag: str | None = None) -> dict:
    sig = dict(sig or {})
    et = "alpha_entry" if placed else "alpha_rejection"
    return record_event(et, sig.get("symbol", ""), date_tag=date_tag,
                        regime=_regime(market_context), sector=sig.get("sector", ""),
                        setup_type=sig.get("setup_type", ""), grade=sig.get("grade", ""),
                        setup_score=sig.get("setup_score"), vol_regime=_vol_regime(market_context),
                        iv_rank=sig.get("iv_rank"), iv_crush_risk=sig.get("iv_crush_risk"),
                        gamma_theta_efficiency=sig.get("gamma_theta_efficiency"),
                        black_scholes_score=sig.get("black_scholes_score"),
                        contracts=sig.get("contracts"), premium_cost=sig.get("premium_cost"),
                        reject_reason=sig.get("reject_reason", ""))


def record_alpha_exit(exit_order: dict, market_context: dict | None = None,
                      date_tag: str | None = None) -> dict:
    eo = dict(exit_order or {})
    return record_event("alpha_exit", eo.get("underlying") or eo.get("symbol", ""), date_tag=date_tag,
                        regime=_regime(market_context), sector=eo.get("sector", ""),
                        setup_type=eo.get("setup_type", ""), vol_regime=_vol_regime(market_context),
                        source=eo.get("source", ""), reason=eo.get("reason", ""),
                        pnl=eo.get("realized_pnl"), r_multiple=eo.get("r_multiple"),
                        return_pct=eo.get("return_pct"),
                        holding_period_days=eo.get("holding_period_days"),
                        win=(None if eo.get("realized_pnl") is None else eo["realized_pnl"] > 0))


def record_bal_order(order: dict, market_context: dict | None = None,
                     date_tag: str | None = None) -> dict:
    o = dict(order or {})
    tier = str(o.get("tier", ""))
    et = "bal_buy" if tier in ("starter", "") else "bal_dca"
    return record_event(et, o.get("symbol", ""), date_tag=date_tag,
                        regime=_regime(market_context), sector=o.get("sector", ""),
                        tier=tier, dca_action=o.get("action", ""),
                        price=o.get("price"), order_dollars=o.get("order_dollars"),
                        conviction_score=o.get("conviction_score"),
                        composite_dca_score=o.get("composite_dca_score"),
                        bal_score=o.get("bal_score"),
                        target_allocation_pct=o.get("target_allocation_pct"),
                        journal_reason=o.get("journal_reason", ""),
                        return_pct=o.get("return_pct"), pnl=o.get("pnl"))


def record_bal_rejection(sig: dict, market_context: dict | None = None,
                         date_tag: str | None = None) -> dict:
    s = dict(sig or {})
    return record_event("bal_rejection", s.get("symbol", ""), date_tag=date_tag,
                        regime=_regime(market_context),
                        bal_score=s.get("BAL_score"), conviction_score=s.get("conviction_score"),
                        price_to_intrinsic_value=s.get("price_to_intrinsic_value"),
                        composite_dca_score=s.get("composite_dca_score"),
                        insider_buy_score=s.get("insider_buy_score"),
                        journal_reason=s.get("journal_reason", ""),
                        reject_reason=s.get("reject_reason", ""))


def record_insider_discovery(candidate: dict, date_tag: str | None = None) -> dict:
    c = dict(candidate or {})
    return record_event("insider_discovery", c.get("symbol", ""), date_tag=date_tag,
                        company=c.get("company", ""),
                        insider_buy_score=c.get("insider_buy_score"),
                        insider_buy_amount=c.get("total_open_market_purchase_value",
                                                 c.get("insider_buy_amount")),
                        insider_count=c.get("insider_count"),
                        insider_roles=c.get("insider_roles", []),
                        latest_insider_buy_date=c.get("latest_transaction_date",
                                                      c.get("latest_insider_buy_date", "")))


# ── metric primitives (pure) ─────────────────────────────────────────────────────
def _has_result(e: dict) -> bool:
    return any(e.get(k) is not None for k in ("pnl", "r_multiple", "return_pct"))


def _value_field(events: list[dict]) -> str:
    if any(e.get("pnl") is not None for e in events):
        return "pnl"
    if any(e.get("return_pct") is not None for e in events):
        return "return_pct"
    return "r_multiple"


def _max_drawdown(curve: list[float]) -> float:
    peak, mdd = (curve[0] if curve else 0.0), 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, (v - peak) / peak)
        elif peak < 0:
            mdd = min(mdd, (v - peak) / abs(peak))
    return round(mdd, 4)


def _sharpe(returns: list[float], periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns)
    if sd <= 1e-12:
        return 0.0
    return round((mu / sd) * math.sqrt(periods_per_year), 3)


def trade_metrics(events: list[dict]) -> dict:
    """Expectancy / profit-factor / win-rate / R / Sharpe / drawdown for a trade group."""
    closed = [e for e in events if _has_result(e)]
    n = len(closed)
    base = {"trades": n, "wins": 0, "losses": 0, "win_rate": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "avg_r_multiple": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "total_pnl": 0.0}
    if n == 0:
        return base
    vf = _value_field(closed)
    vals = [float(e.get(vf) or 0.0) for e in closed]
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v <= 0]
    win_rate = len(wins) / n
    loss_rate = len(losses) / n
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = abs(statistics.mean(losses)) if losses else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    r_vals = [float(e["r_multiple"]) for e in closed if e.get("r_multiple") is not None]
    cum, curve = 0.0, [0.0]
    for v in vals:
        cum += v
        curve.append(cum)
    return {
        "trades": n, "wins": len(wins), "losses": len(losses),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4),
        "avg_r_multiple": round(statistics.mean(r_vals), 3) if r_vals else 0.0,
        "expectancy": round(win_rate * avg_win - loss_rate * avg_loss, 4),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 1e-12 else (
            float("inf") if gross_win > 0 else 0.0),
        "sharpe": _sharpe([v for v in vals]),
        "max_drawdown": _max_drawdown(curve),
        "total_pnl": round(sum(vals), 4),
        "value_field": vf,
    }


def group_metrics(events: list[dict], key_fn) -> dict:
    groups: dict[str, list] = {}
    for e in events:
        groups.setdefault(key_fn(e), []).append(e)
    return {k: trade_metrics(v) for k, v in sorted(groups.items())}


# ── bucketers ────────────────────────────────────────────────────────────────────
def _val_bucket(p2iv) -> str:
    try:
        p = float(p2iv)
    except (TypeError, ValueError):
        return "unknown"
    if p <= 0:
        return "unknown"
    if p < 0.65:
        return "<65%"
    if p < 0.75:
        return "65-75%"
    if p < 0.85:
        return "75-85%"
    if p < 0.95:
        return "85-95%"
    if p <= 1.00:
        return "95-100%"
    return ">100%"


def _conv_bucket(score) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if s >= 90:
        return "90-100"
    if s >= 75:
        return "75-89"
    if s >= 60:
        return "60-74"
    return "<60"


def _insider_bucket(score) -> str:
    try:
        s = float(score or 0)
    except (TypeError, ValueError):
        return "none"
    if s <= 0:
        return "none"
    if s < 40:
        return "low(1-39)"
    if s < 70:
        return "medium(40-69)"
    return "high(70-100)"


# ── Alpha analytics ──────────────────────────────────────────────────────────────
def alpha_analytics(events: list[dict]) -> dict:
    exits = [e for e in events if e.get("event_type") == "alpha_exit"]
    entries = [e for e in events if e.get("event_type") == "alpha_entry"]
    rejections = [e for e in events if e.get("event_type") == "alpha_rejection"]
    return {
        "totals": trade_metrics(exits),
        "by_setup_type": group_metrics(exits, lambda e: e.get("setup_type") or "unknown"),
        "by_symbol": group_metrics(exits, lambda e: e.get("symbol") or "unknown"),
        "by_sector": group_metrics(exits, lambda e: e.get("sector") or "unknown"),
        "by_market_regime": group_metrics(exits, lambda e: e.get("regime") or "unknown"),
        "by_volatility_regime": group_metrics(exits, lambda e: e.get("vol_regime") or "unknown"),
        "funnel": {"entries": len(entries), "exits": len(exits), "rejections": len(rejections)},
    }


# ── Buffett analytics ────────────────────────────────────────────────────────────
def buffett_analytics(events: list[dict]) -> dict:
    bal = [e for e in events if e.get("event_type") in ("bal_buy", "bal_dca")]
    rejections = [e for e in events if e.get("event_type") == "bal_rejection"]

    # DCA tranche effectiveness (tier label or dca sequence).
    def _tranche_key(e):
        t = e.get("tier")
        if t:
            return str(t)
        return f"tranche_{int(e.get('dca_count') or 1)}"

    tranche_groups: dict[str, list] = {}
    for e in bal:
        tranche_groups.setdefault(_tranche_key(e), []).append(e)
    overall_pnl = sum(float(e.get("pnl") or 0.0) for e in bal) or 0.0
    by_tranche = {}
    for k, grp in sorted(tranche_groups.items()):
        m = trade_metrics(grp)
        rets = [float(e["return_pct"]) for e in grp if e.get("return_pct") is not None]
        holds = [float(e["holding_period_days"]) for e in grp if e.get("holding_period_days") is not None]
        grp_pnl = sum(float(e.get("pnl") or 0.0) for e in grp)
        m.update({
            "count": len(grp),
            "average_return": round(statistics.mean(rets), 4) if rets else 0.0,
            "median_return": round(statistics.median(rets), 4) if rets else 0.0,
            "holding_period_days": round(statistics.mean(holds), 1) if holds else 0.0,
            "contribution_to_total_return": round(grp_pnl / overall_pnl, 4) if abs(overall_pnl) > 1e-9 else 0.0,
        })
        by_tranche[k] = m

    return {
        "totals": trade_metrics(bal),
        "by_tranche": by_tranche,
        "by_valuation_bucket": group_metrics(bal, lambda e: _val_bucket(e.get("price_to_intrinsic_value"))),
        "by_conviction_bucket": group_metrics(bal, lambda e: _conv_bucket(e.get("conviction_score"))),
        "by_insider_bucket": group_metrics(bal, lambda e: _insider_bucket(e.get("insider_buy_score"))),
        "funnel": {"accumulations": len(bal), "rejections": len(rejections),
                   "insider_discoveries": len([e for e in events if e.get("event_type") == "insider_discovery"])},
    }


# ── Capital waterfall analytics ──────────────────────────────────────────────────
def waterfall_analytics(events: list[dict], waterfall_json: dict | None = None,
                        bal_current_value: float | None = None) -> dict:
    wf = waterfall_json or {}
    bal_events = [e for e in events if e.get("event_type") in ("bal_buy", "bal_dca")]
    bal_invested = sum(float(e.get("order_dollars") or 0.0) for e in bal_events)
    alloc_today = wf.get("allocation_today", {}) if isinstance(wf.get("allocation_today"), dict) else {}
    alpha_generated = float(wf.get("realized_today", wf.get("realized_options_profit_today", 0.0)) or 0.0)
    alpha_transferred = float(alloc_today.get("bal", 0.0) or 0.0) + float(wf.get("bal_reserve", 0.0) or 0.0)
    current_value = float(bal_current_value) if bal_current_value is not None else bal_invested
    waterfall_return = round((current_value - bal_invested) / bal_invested, 4) if bal_invested > 1e-9 else 0.0
    return {
        "alpha_generated": round(alpha_generated, 2),
        "alpha_transferred": round(alpha_transferred, 2),
        "bal_invested": round(bal_invested, 2),
        "bal_current_value": round(current_value, 2),
        "bal_unrealized_gain": round(current_value - bal_invested, 2),
        "waterfall_return": waterfall_return,
        "transfer_efficiency": round(alpha_transferred / alpha_generated, 4) if alpha_generated > 1e-9 else 0.0,
    }


# ── Self-improvement rankings ────────────────────────────────────────────────────
def _top(d: dict, metric: str, n: int = 5, reverse: bool = True, min_trades: int = 1) -> list[dict]:
    rows = [{"key": k, **v} for k, v in d.items() if v.get("trades", 0) >= min_trades]
    # Treat +inf profit_factor as very large but finite for ranking stability.
    def _m(r):
        x = r.get(metric, 0.0)
        return 1e12 if x == float("inf") else (x or 0.0)
    rows.sort(key=_m, reverse=reverse)
    return rows[:n]


def rankings(alpha_an: dict, buffett_an: dict) -> dict:
    setups = alpha_an.get("by_setup_type", {})
    return {
        "top_alpha_setups": {
            "by_expectancy": _top(setups, "expectancy"),
            "by_sharpe": _top(setups, "sharpe"),
            "by_profit_factor": _top(setups, "profit_factor"),
        },
        "worst_alpha_setups": {
            "by_expectancy": _top(setups, "expectancy", reverse=False),
            "by_max_drawdown": _top(setups, "max_drawdown", reverse=False),
        },
        "top_buffett_characteristics": {
            "valuation_buckets": _top(buffett_an.get("by_valuation_bucket", {}), "expectancy"),
            "conviction_buckets": _top(buffett_an.get("by_conviction_bucket", {}), "expectancy"),
            "insider_buckets": _top(buffett_an.get("by_insider_bucket", {}), "expectancy"),
        },
    }


# ── Walk-forward simulation ──────────────────────────────────────────────────────
def walk_forward(events: list[dict], start_equity: float = 100_000.0,
                 risk_fraction: float = 0.02, position_fraction: float = 0.10) -> dict:
    """
    Replay closed (result-bearing) events in time order over a starting equity and
    produce a risk-adjusted summary. Dollar pnl is applied directly; an r_multiple is
    sized at risk_fraction of equity; a bare return_pct is sized at position_fraction.
    """
    closed = sorted([e for e in events if _has_result(e)], key=lambda e: str(e.get("timestamp", "")))
    equity = float(start_equity)
    curve = [equity]
    rets: list[float] = []
    wins = 0
    gross_win = gross_loss = 0.0
    for e in closed:
        prev = equity
        if e.get("pnl") is not None:
            pnl = float(e["pnl"])
        elif e.get("r_multiple") is not None:
            pnl = prev * risk_fraction * float(e["r_multiple"])
        else:
            pnl = prev * position_fraction * float(e.get("return_pct") or 0.0)
        equity += pnl
        curve.append(equity)
        rets.append(pnl / prev if prev else 0.0)
        if pnl > 0:
            wins += 1
            gross_win += pnl
        else:
            gross_loss += abs(pnl)
    n = len(closed)
    # Span in days from timestamps (fallback to ~1 trade/day).
    days = 1.0
    if n >= 2:
        try:
            t0 = datetime.fromisoformat(str(closed[0]["timestamp"]).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(closed[-1]["timestamp"]).replace("Z", "+00:00"))
            days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
        except Exception:
            days = float(max(n, 1))
    total_return = (equity / start_equity) - 1.0 if start_equity else 0.0
    cagr = ((equity / start_equity) ** (365.0 / days) - 1.0) if (start_equity > 0 and equity > 0) else 0.0
    return {
        "start_equity": round(start_equity, 2),
        "end_equity": round(equity, 2),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "trades": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 1e-12 else (
            float("inf") if gross_win > 0 else 0.0),
        "sharpe": _sharpe(rets),
        "max_drawdown": _max_drawdown(curve),
        "span_days": round(days, 2),
        "equity_curve": [round(c, 2) for c in curve],
    }


# ── orchestration + file I/O ─────────────────────────────────────────────────────
def _write(name: str, payload: dict, date_tag: str | None = None) -> None:
    try:
        _path(name, date_tag).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("write %s failed: %s", name, exc)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def generate_all_reports(events: list[dict] | None = None, market_context: dict | None = None,
                         start_equity: float = 100_000.0, date_tag: str | None = None,
                         bal_current_value: float | None = None) -> dict:
    """Compute every analytic from the event stream and write all report files."""
    if events is None:
        events = read_events(date_tag)
    wf_json = _load_json(DAILY_OUTPUTS / "capital_waterfall.json")

    alpha_an = alpha_analytics(events)
    buffett_an = buffett_analytics(events)
    waterfall_an = waterfall_analytics(events, wf_json, bal_current_value)
    rank = rankings(alpha_an, buffett_an)
    wf_sim = walk_forward(events, start_equity=start_equity)

    now = datetime.now(timezone.utc).isoformat()
    meta = {"generated_at": now, "date_tag": date_tag or DATA_TAG, "event_count": len(events),
            "market_regime": _regime(market_context)}

    _write(ALPHA_EXPECTANCY_REPORT, {**meta, "alpha": alpha_an}, date_tag)
    _write(BUFFETT_PERFORMANCE_REPORT, {**meta, "buffett": buffett_an}, date_tag)
    _write(CAPITAL_WATERFALL_PERFORMANCE, {**meta, "capital_waterfall": waterfall_an}, date_tag)
    _write(WALKFORWARD_REPORT, {**meta, "walkforward": wf_sim,
                                "alpha_expectancy": alpha_an["totals"].get("expectancy", 0.0),
                                "buffett_expectancy": buffett_an["totals"].get("expectancy", 0.0),
                                "capital_waterfall": waterfall_an}, date_tag)
    research_lab = {**meta, "alpha": alpha_an, "buffett": buffett_an,
                    "capital_waterfall": waterfall_an, "rankings": rank, "walkforward": wf_sim}
    _write(RESEARCH_LAB, research_lab, date_tag)
    return research_lab


# ── fail-safe wrappers for run.py (never raise) ──────────────────────────────────
def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("research lab call %s skipped: %s", getattr(fn, "__name__", fn), exc)
        return None


def record_alpha_signal_safe(*a, **k): return _safe(record_alpha_signal, *a, **k)
def record_alpha_exit_safe(*a, **k): return _safe(record_alpha_exit, *a, **k)
def record_bal_order_safe(*a, **k): return _safe(record_bal_order, *a, **k)
def record_bal_rejection_safe(*a, **k): return _safe(record_bal_rejection, *a, **k)
def record_insider_discovery_safe(*a, **k): return _safe(record_insider_discovery, *a, **k)
def generate_all_reports_safe(*a, **k): return _safe(generate_all_reports, *a, **k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rl = generate_all_reports()
    print(json.dumps({"events": rl.get("event_count"),
                      "alpha_expectancy": rl["alpha"]["totals"].get("expectancy"),
                      "buffett_expectancy": rl["buffett"]["totals"].get("expectancy"),
                      "walkforward": {k: rl["walkforward"][k] for k in
                                      ("end_equity", "cagr", "sharpe", "max_drawdown", "profit_factor")}},
                     indent=2, default=str))
