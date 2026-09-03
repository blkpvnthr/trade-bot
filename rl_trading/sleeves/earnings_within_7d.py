"""earnings_within_7d sleeve engine — daily forward-earnings pre-print drift. Paper/scanner only.

Long-only, equal-weight names with a SCHEDULED earnings report within the next 7 calendar days;
exit at the close of the trading day BEFORE the print. NEVER holds through an announcement. Only
FORWARD scheduled dates feed the signal (no estimate/overview snapshots).

Layering:
  - `plan_day(...)`         pure, deterministic planner over injected data (fully unit-testable).
  - `load_calendar / spy_200sma_regime / liquidity_view`  thin causal adapters over cached data.
  - `append_shadow_log`     durable append-only CSV of intended decisions (SCANNER_ONLY).
  - `main()`                CLI; SCANNER_ONLY by default, paper route gated behind the sleeve gate.

Validation: rl_trading.sleeves.promote.from_extended_validation('earnings_within_7d') -> ACCEPT
(net Sharpe 1.11@5bps, breakeven 56bps, turnover 0.24/day, boot+placebo p~0). Live is impossible.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set

from .initial import EARNINGS_UNIVERSE, earnings_within_7d_sleeve
from .paper import Order, PaperExecutor

_REPO_ROOT = Path(__file__).resolve().parents[2]
CALENDAR_DIR = _REPO_ROOT / "backtest_cache" / "earnings_calendar"
SHADOW_LOG_DIR = _REPO_ROOT / "logs" / "earnings_within_7d"
STATUS_PATH = _REPO_ROOT / "system-data" / "earnings_within_7d_status.json"
INSIDER_HELD_PATH = _REPO_ROOT / "system-data" / "insider_holdings.json"
# Reconciliation summary that the >=4wk shadow stage must produce before paper can be enabled.
RECON_SUMMARY_PATH = _REPO_ROOT / "system-data" / "earnings_within_7d_reconciliation.json"

# Pre-registered shadow-stage acceptance (Task 3). Encoded so the gate and report agree.
SHADOW_ACCEPTANCE = {
    "min_weeks": 4,
    "zero_holds_through_actual_report": True,        # never held into a true print
    "document_calendar_vs_actual_mismatch_rate": True,
    "no_material_degradation_vs_reportedDate_proxy": True,
    "every_reschedule_failclosed_or_exited_early": True,
    "reconcile_forward_calendar_vs_actual_reportedDate": True,
}

# calendar_reconciliation table columns (Task 5): one row per flagged event, written during shadow.
RECONCILIATION_FIELDS = (
    "symbol", "entry_date", "scheduled_earnings_date_at_entry", "scheduled_earnings_date_latest",
    "actual_reported_date", "exit_date", "held_through_actual_report", "date_changed",
    "days_changed", "fail_closed_reason", "return_to_exit", "return_to_actual_minus_1",
)

# Shadow-log column order (Task 6). Append-only; header written once.
SHADOW_FIELDS = (
    "date", "symbol", "scheduled_earnings_date", "days_to_earnings", "signal_flag",
    "rank_score", "target_weight", "target_notional", "spy_regime", "spy_above_200sma",
    "market_exposure_multiplier", "calendar_age_days", "coverage",
    "action", "reason", "realized_drift",
)


# ── config ───────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EarningsConfig:
    lookahead_days: int = 7            # entry window: earnings within next N calendar days
    exit_days_before: int = 1         # exit when days_to_earnings <= this (close day before print)
    min_price: float = 5.0
    min_adv_usd: float = 5_000_000.0  # 20d dollar ADV floor (preferred 10M, see pref_adv_usd)
    pref_adv_usd: float = 10_000_000.0
    max_gross_pct: float = 0.50       # gross long exposure cap as fraction of equity
    max_concurrent: int = 10
    max_position_pct: float = 0.10
    cash_buffer_pct: float = 0.08
    regime_policy: str = "pause"      # 'pause' | 'size_down' | 'off'  (default conservative)
    regime_size_down_factor: float = 0.33
    max_calendar_age_days: int = 3
    min_coverage: float = 0.95


# ── results ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Calendar:
    scheduled: Mapping[str, Optional[str]]  # symbol -> next forward scheduled earnings date (or None)
    coverage: float                          # fraction of universe with calendar info
    age_days: Optional[float]                # days since the calendar was fetched
    feed_ok: bool
    source: str = ""
    fetched_at: Optional[str] = None


@dataclass
class DayPlan:
    asof: str
    fail_closed: bool
    fail_reason: str
    spy_regime: str               # 'above_200sma' | 'below_200sma' | 'unknown'
    regime_action: str            # 'normal' | 'pause' | 'size_down'
    market_exposure_multiplier: float  # 1.0 normal | size_down factor | 0.0 paused (new entries)
    variant: str                  # 'A' | 'B' | 'C'
    calendar_age_days: Optional[float]
    coverage: Optional[float]
    rows: List[dict] = field(default_factory=list)

    @property
    def entries(self) -> List[str]:
        return [r["symbol"] for r in self.rows if r["action"] == "ENTER"]

    @property
    def exits(self) -> List[str]:
        return [r["symbol"] for r in self.rows if r["action"] == "EXIT"]

    @property
    def holds(self) -> List[str]:
        return [r["symbol"] for r in self.rows if r["action"] == "HOLD"]

    @property
    def flagged(self) -> List[str]:
        return [r["symbol"] for r in self.rows if r["signal_flag"]]


# ── helpers ──────────────────────────────────────────────────────────────────────
def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


# ── pure planner ─────────────────────────────────────────────────────────────────
def plan_day(
    asof,
    universe: Sequence[str],
    scheduled: Mapping[str, Optional[str]],
    liquidity: Mapping[str, Mapping[str, Optional[float]]],
    currently_held: Mapping[str, float],
    equity: float,
    cfg: EarningsConfig,
    *,
    spy_above: Optional[bool],
    calendar_age_days: Optional[float],
    coverage: Optional[float],
    feed_ok: bool,
    insider_held: Set[str] = frozenset(),
    scores: Optional[Mapping[str, float]] = None,
    variant: str = "A",
) -> DayPlan:
    """Deterministic daily plan. Pure function of its inputs (no clock, no I/O).

    Fail-closed (stale/low-coverage/feed-down) blocks all NEW entries AND flattens existing holds
    (we cannot verify we are not about to hold through a print). Exits are ALWAYS permitted.
    """
    asof_d = _as_date(asof)

    # ---- fail-closed gate ----
    if not feed_ok:
        fc_reason = "feed_unavailable"
    elif calendar_age_days is None or calendar_age_days > cfg.max_calendar_age_days:
        fc_reason = f"calendar_stale(age={calendar_age_days},max={cfg.max_calendar_age_days})"
    elif coverage is None or coverage < cfg.min_coverage:
        fc_reason = f"coverage_low({coverage}<{cfg.min_coverage})"
    else:
        fc_reason = ""
    fail_closed = bool(fc_reason)

    # ---- regime ----
    if spy_above is True:
        spy_regime = "above_200sma"
    elif spy_above is False:
        spy_regime = "below_200sma"
    else:
        spy_regime = "unknown"
    regime_action = (
        cfg.regime_policy if (spy_above is False and cfg.regime_policy != "off") else "normal"
    )
    # market_exposure_multiplier: feature column + the actual gross scaler applied below.
    if regime_action == "pause":
        market_mult = 0.0
    elif regime_action == "size_down":
        market_mult = cfg.regime_size_down_factor
    else:
        market_mult = 1.0
    scores = scores or {}

    # ---- per-symbol signal ----
    info: Dict[str, tuple] = {}
    for sym in universe:
        sched = scheduled.get(sym)
        days_to = (_as_date(sched) - asof_d).days if sched else None
        flag = bool(days_to is not None and 1 <= days_to <= cfg.lookahead_days)
        info[sym] = (sched, days_to, flag)

    held = {s for s, n in currently_held.items() if abs(float(n)) > 1e-9}

    def liquid(sym: str) -> bool:
        lq = liquidity.get(sym) or {}
        p, adv = lq.get("price"), lq.get("adv20")
        return p is not None and p >= cfg.min_price and adv is not None and adv >= cfg.min_adv_usd

    # ---- target end-of-day book ----
    target: Set[str] = set()
    if not fail_closed:
        for sym, (sched, days_to, flag) in info.items():
            if (
                flag
                and days_to > cfg.exit_days_before  # far enough from print to hold
                and liquid(sym)
                and sym not in insider_held
            ):
                target.add(sym)

    keep = held & target                       # held + still wanted -> HOLD
    # admission order: variant B/C rank by injected score desc (drift/fundamentals); A = arbitrary
    # alphabetical. Ties + missing scores fall back to symbol for determinism.
    new_candidates = sorted(target - held, key=lambda s: (-float(scores.get(s, 0.0)), s))
    slots = max(cfg.max_concurrent - len(keep), 0)
    admit: Set[str] = set() if regime_action == "pause" else set(new_candidates[:slots])
    book = keep | admit                          # names held at close

    # ---- equal-weight sizing ----
    gross = min(cfg.max_gross_pct * equity, (1.0 - cfg.cash_buffer_pct) * equity)
    if regime_action == "size_down":
        gross *= cfg.regime_size_down_factor
    n = len(book)
    per_name = 0.0 if n == 0 else min(gross / n, cfg.max_position_pct * equity)

    # ---- emit one row per universe symbol ----
    rows: List[dict] = []
    for sym in universe:
        sched, days_to, flag = info[sym]
        reason = ""
        if fail_closed:
            if sym in held:
                action, reason = "EXIT", f"fail_closed:{fc_reason}"
            else:
                action, reason = "FAIL_CLOSED", fc_reason
            tw = tn = 0.0
        elif sym in book:
            action = "HOLD" if sym in held else "ENTER"
            tn = per_name
            tw = per_name / equity if equity else 0.0
        elif sym in held:
            action = "EXIT"
            if days_to is not None and days_to <= cfg.exit_days_before:
                reason = "before_print"
            elif not flag:
                reason = "left_window"
            elif sym in insider_held:
                reason = "insider_overlap"
            elif not liquid(sym):
                reason = "illiquid"
            else:
                reason = "capacity_or_regime"
            tw = tn = 0.0
        else:
            action = "FLAT"
            if flag and sym in insider_held:
                reason = "insider_overlap"
            elif flag and not liquid(sym):
                reason = "illiquid"
            elif flag and regime_action == "pause":
                reason = "regime_pause"
            elif flag and days_to is not None and days_to <= cfg.exit_days_before:
                reason = "too_close_to_print"
            elif flag:
                reason = "capacity"
            tw = tn = 0.0
        rows.append({
            "date": asof_d.isoformat(),
            "symbol": sym,
            "scheduled_earnings_date": sched,
            "days_to_earnings": days_to,
            "signal_flag": flag,
            "rank_score": round(float(scores.get(sym, 0.0)), 4),
            "target_weight": round(tw, 6),
            "target_notional": round(tn, 2),
            "spy_regime": spy_regime,
            "spy_above_200sma": spy_above,
            "market_exposure_multiplier": market_mult,
            "calendar_age_days": calendar_age_days,
            "coverage": coverage,
            "action": action,
            "reason": reason,
            "realized_drift": None,
        })

    return DayPlan(
        asof=asof_d.isoformat(),
        fail_closed=fail_closed,
        fail_reason=fc_reason,
        spy_regime=spy_regime,
        regime_action=regime_action,
        market_exposure_multiplier=market_mult,
        variant=variant,
        calendar_age_days=calendar_age_days,
        coverage=coverage,
        rows=rows,
    )


# ── adapters (causal, best-effort, fail-closed) ───────────────────────────────────
def load_calendar(
    universe: Sequence[str], asof, cal_dir: Path = CALENDAR_DIR, now: Optional[datetime] = None,
) -> Calendar:
    """Load the freshest forward earnings calendar. Forward-only: keep the earliest scheduled date
    on/after `asof` per symbol. coverage = fraction of universe present in the calendar."""
    asof_d = _as_date(asof)
    now = now or datetime.now(timezone.utc)
    files = sorted(Path(cal_dir).glob("cal_*.json")) if Path(cal_dir).exists() else []
    if not files:
        return Calendar(scheduled={}, coverage=0.0, age_days=None, feed_ok=False, source="missing")

    best = None
    for f in files:
        try:
            data = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        fetched = data.get("fetched_at")
        if best is None or (fetched or "") > (best[1].get("fetched_at") or ""):
            best = (f, data)
    if best is None:
        return Calendar(scheduled={}, coverage=0.0, age_days=None, feed_ok=False, source="unreadable")

    _f, data = best
    events = data.get("events", {})
    scheduled: Dict[str, Optional[str]] = {}
    covered = 0
    for sym in universe:
        if sym not in events:
            continue
        covered += 1
        forward = sorted(d for d in events[sym] if _as_date(d) >= asof_d)
        scheduled[sym] = forward[0] if forward else None
    coverage = covered / len(universe) if universe else 0.0

    age_days = None
    fetched_at = data.get("fetched_at")
    if fetched_at:
        try:
            ts = datetime.fromisoformat(fetched_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).total_seconds() / 86400.0
        except ValueError:
            age_days = None
    return Calendar(
        scheduled=scheduled, coverage=coverage, age_days=age_days, feed_ok=True,
        source=str(data.get("source", "")), fetched_at=fetched_at,
    )


def spy_200sma_regime(asof) -> Optional[bool]:
    """True if SPY close > 200d SMA as of the last completed daily bar on/before asof, else False;
    None if fewer than 200 bars. Uses the adjusted Alpaca daily cache."""
    try:
        from .. import data as rl_data
    except Exception:  # pragma: no cover - import guard
        return None
    df = rl_data.load_daily("SPY", source="adj")
    if df is None or df.empty:
        return None
    asof_ts = _as_date(asof).isoformat()
    closes = df.loc[df.index <= asof_ts + "T23:59:59+00:00", "close"]
    if len(closes) < 200:
        return None
    sma = float(closes.iloc[-200:].mean())
    return bool(float(closes.iloc[-1]) > sma)


def liquidity_view(universe: Sequence[str], asof) -> Dict[str, Dict[str, Optional[float]]]:
    """{symbol: {price, adv20}} from the adjusted daily cache, causal as of `asof`."""
    try:
        from .. import data as rl_data
    except Exception:  # pragma: no cover - import guard
        return {s: {"price": None, "adv20": None} for s in universe}
    asof_ts = _as_date(asof).isoformat() + "T23:59:59+00:00"
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for sym in universe:
        df = rl_data.load_daily(sym, source="adj")
        if df is None or df.empty:
            out[sym] = {"price": None, "adv20": None}
            continue
        d = df.loc[df.index <= asof_ts]
        if d.empty:
            out[sym] = {"price": None, "adv20": None}
            continue
        price = float(d["close"].iloc[-1])
        tail = d.tail(20)
        adv20 = float((tail["close"] * tail["volume"]).mean())
        out[sym] = {"price": price, "adv20": adv20}
    return out


def load_insider_held(path: Path = INSIDER_HELD_PATH) -> Set[str]:
    """Symbols currently held by the insider sleeve (excluded to avoid double exposure).
    Optional: a JSON list of tickers at `path`; absent -> empty set."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return set()
    if isinstance(data, dict):
        data = data.get("symbols", [])
    return {str(s).upper() for s in data}


# ── shadow logger (durable, append-only) ──────────────────────────────────────────
def append_shadow_log(rows: Sequence[dict], log_dir: Path = SHADOW_LOG_DIR) -> Path:
    """Append decision rows to a durable monthly CSV. Header written once per file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    month = (rows[0]["date"][:7] if rows else datetime.now(timezone.utc).strftime("%Y-%m"))
    path = log_dir / f"shadow_{month}.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHADOW_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ── calendar reconciliation (forward-scheduled vs actual reportedDate) ─────────────
def reconciliation_row(symbol: str, entry_date, scheduled_at_entry, scheduled_latest,
                       actual_reported=None, exit_date=None, fail_closed_reason: str = "",
                       return_to_exit: Optional[float] = None,
                       return_to_actual_minus_1: Optional[float] = None) -> dict:
    """One row reconciling the forward-scheduled date (what we acted on) against the actual
    reportedDate (what really happened). `held_through_actual_report` MUST be False for every event
    in an accepted shadow run; any True is a hard violation."""
    sa = _as_date(scheduled_at_entry) if scheduled_at_entry else None
    sl = _as_date(scheduled_latest) if scheduled_latest else None
    ar = _as_date(actual_reported) if actual_reported else None
    ex = _as_date(exit_date) if exit_date else None
    held_through = bool(ar and ex and ex >= ar)               # exited on/after the print = violation
    date_changed = bool(sa and sl and sa != sl)
    days_changed = (sl - sa).days if (sa and sl) else None
    return {
        "symbol": symbol,
        "entry_date": _as_date(entry_date).isoformat() if entry_date else None,
        "scheduled_earnings_date_at_entry": sa.isoformat() if sa else None,
        "scheduled_earnings_date_latest": sl.isoformat() if sl else None,
        "actual_reported_date": ar.isoformat() if ar else None,
        "exit_date": ex.isoformat() if ex else None,
        "held_through_actual_report": held_through,
        "date_changed": date_changed,
        "days_changed": days_changed,
        "fail_closed_reason": fail_closed_reason,
        "return_to_exit": return_to_exit,
        "return_to_actual_minus_1": return_to_actual_minus_1,
    }


def append_reconciliation_log(rows: Sequence[dict], log_dir: Path = SHADOW_LOG_DIR) -> Path:
    """Append reconciliation rows to a durable monthly CSV (append-only, header once)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    month = rows[0]["entry_date"][:7] if rows and rows[0].get("entry_date") else \
        datetime.now(timezone.utc).strftime("%Y-%m")
    path = log_dir / f"calendar_reconciliation_{month}.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RECONCILIATION_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def reconciliation_gate_passed(path: Path = RECON_SUMMARY_PATH) -> tuple[bool, str]:
    """Promotion gate (Task 2): paper stays blocked until the >=4wk shadow reconciliation passes.

    Reads a summary JSON the shadow stage produces. Fail-closed: missing/incomplete -> blocked.
    """
    path = Path(path)
    if not path.exists():
        return False, "reconciliation_summary_missing(>=4wk shadow not completed)"
    try:
        s = json.loads(path.read_text())
    except (ValueError, OSError):
        return False, "reconciliation_summary_unreadable"
    if not s.get("passed"):
        return False, "reconciliation_not_passed"
    if float(s.get("weeks", 0)) < SHADOW_ACCEPTANCE["min_weeks"]:
        return False, f"reconciliation_weeks<{SHADOW_ACCEPTANCE['min_weeks']}"
    if int(s.get("holds_through_actual_report", 1)) != 0:
        return False, "held_through_actual_report_occurred"
    return True, "reconciliation_passed"


# ── dashboard surface ─────────────────────────────────────────────────────────────
def dashboard_payload(plan: DayPlan, calendar: Calendar) -> dict:
    """Operator/alert surface (Task 8). `ok=False` whenever anything is wrong."""
    from .promote import CANONICAL_STATUS, PROVENANCE_CAVEAT
    calendar_ok = bool(calendar.feed_ok and not plan.fail_closed)
    recon_ok, recon_reason = reconciliation_gate_passed()
    return {
        "asof": plan.asof,
        "ok": calendar_ok,
        "canonical_status": CANONICAL_STATUS,
        "provenance_caveat": PROVENANCE_CAVEAT,
        "reconciliation_gate_passed": recon_ok,
        "reconciliation_gate_reason": recon_reason,
        "calendar_ok": calendar_ok,
        "calendar_stale": plan.fail_closed and plan.fail_reason.startswith("calendar_stale"),
        "coverage": plan.coverage,
        "age_days": plan.calendar_age_days,
        "feed_ok": calendar.feed_ok,
        "spy_regime": plan.spy_regime,
        "regime_action": plan.regime_action,
        "market_exposure_multiplier": plan.market_exposure_multiplier,
        "variant": plan.variant,
        "fail_closed": plan.fail_closed,
        "fail_closed_reason": plan.fail_reason,
        "flagged_symbols": plan.flagged,
        "intended_entries": plan.entries,
        "intended_exits": plan.exits,
        "holds": plan.holds,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────
def resolve_paper_authorization(readiness: Mapping[str, object], override: bool):
    """Decide paper authorization from the readiness payload + an explicit override flag.

    Returns (ready, override_active, reason). The override bypasses ONLY the live reconciliation
    gate; it CANNOT bypass methodology validation. It does not touch fail-closed, the sleeve gate,
    risk limits, the kill switch, or the permanent live block (those are enforced elsewhere).
    """
    if readiness.get("recommendation") == "READY_FOR_PAPER":
        return True, False, "ready"
    if override and readiness.get("methodology_validation_passed"):
        return True, True, "override_active:" + str(readiness.get("reconciliation_reason", ""))
    if override:
        return False, False, "override_refused:methodology_not_passed"
    return False, False, "blocked:" + str(readiness.get("recommendation", ""))


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _load_equity(default: float = 100_000.0) -> float:
    snap = _REPO_ROOT / "system-data" / "dashboard_snapshot.json"
    if snap.exists():
        try:
            data = json.loads(snap.read_text())
            eq = data.get("equity") or data.get("account", {}).get("equity")
            if eq:
                return float(eq)
        except (ValueError, OSError, TypeError):
            pass
    return default


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="earnings_within_7d sleeve runner (paper/scanner only)")
    ap.add_argument("--asof", default=date.today().isoformat(), help="planning date YYYY-MM-DD")
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--json", action="store_true", help="print the dashboard payload as JSON")
    ap.add_argument("--reconcile", action="store_true",
                    help="reconcile shadow logs vs actual reportedDate; write CSV+summary; no orders")
    args = ap.parse_args(argv)

    if args.reconcile:
        from . import earnings_reconciliation as RC
        summary = RC.run_reconcile()
        readiness = RC.paper_readiness(write=True)
        print(json.dumps({"reconciliation": summary, "readiness": readiness}, indent=2, default=str)
              if args.json else
              f"reconcile: events={summary['events_reconciled']} weeks={summary['weeks_observed']} "
              f"held_through={summary['held_through_actual_count']} gate_passed={summary['gate_passed']} "
              f"-> {readiness['recommendation']} ({'; '.join(readiness['blocking_reasons']) or 'ok'})")
        return 0

    # SCANNER_ONLY default 1; paper requires BOTH SCANNER_ONLY=0 AND EARNINGS_SLEEVE_PAPER_ENABLED=1.
    scanner_only = _env_flag("SCANNER_ONLY", "1")
    paper_enabled = _env_flag("EARNINGS_SLEEVE_PAPER_ENABLED", "0") and not scanner_only

    cfg = EarningsConfig()
    universe = list(EARNINGS_UNIVERSE)
    equity = args.equity if args.equity is not None else _load_equity()

    # Variant: default = validated winner (A). C requires explicit opt-in and is non-PIT.
    from . import earnings_features as EF
    variant = os.environ.get("EARNINGS_VARIANT", EF.DEFAULT_VARIANT).upper()
    if variant == "C" and not _env_flag("EARNINGS_ALLOW_FUNDAMENTALS", "0"):
        print(f"WARNING: variant C disabled by default ({EF.FUNDAMENTALS_WARNING}). Using A.")
        variant = "A"
    if variant not in EF.VARIANTS:
        variant = "A"

    calendar = load_calendar(universe, args.asof)
    asof_d = _as_date(args.asof)
    feats = EF.drift_features(universe, args.asof)
    scores: Dict[str, float] = {}
    for sym in universe:
        sched = calendar.scheduled.get(sym)
        days_to = (_as_date(sched) - asof_d).days if sched else None
        fund = EF.fundamentals_features(sym) if variant == "C" else None
        scores[sym] = EF.rank_score(sym, feats.get(sym, {}), days_to, variant, fund)

    plan = plan_day(
        args.asof, universe,
        scheduled=calendar.scheduled,
        liquidity=liquidity_view(universe, args.asof),
        currently_held={},  # scanner has no book; paper book is tracked by the executor's audit log
        equity=equity,
        cfg=cfg,
        spy_above=spy_200sma_regime(args.asof),
        calendar_age_days=calendar.age_days,
        coverage=calendar.coverage,
        feed_ok=calendar.feed_ok,
        insider_held=load_insider_held(),
        scores=scores,
        variant=variant,
    )

    log_path = append_shadow_log(plan.rows)
    payload = dashboard_payload(plan, calendar)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, default=str))

    # Paper route — GATED. The sleeve's gate must pass AND paper must be explicitly enabled; the
    # PaperExecutor independently hard-blocks anything that is not tradeable (and live forever).
    sleeve = earnings_within_7d_sleeve(paper_enabled=paper_enabled)
    submitted = 0
    # Paper route requires ALL of (Task 8 + provenance gates): canonical record loads (sleeve gate
    # passes), paper explicitly enabled, fresh/covered calendar with no fail-closed this run, the
    # >=4wk forward-calendar-vs-actual reconciliation passed, AND the historical methodology
    # validation passed -> i.e. the readiness recommendation is READY_FOR_PAPER.
    from . import earnings_reconciliation as RC
    readiness = RC.paper_readiness()
    override = _env_flag("EARNINGS_PAPER_OVERRIDE", "0")
    ready, override_active, auth_msg = resolve_paper_authorization(readiness, override)
    if override_active:
        print("WARNING: EARNINGS_PAPER_OVERRIDE active - bypassing the live reconciliation gate "
              f"({readiness['reconciliation_reason']}). Methodology validated; fail-closed + risk "
              "limits + no-live remain enforced.")
    elif override and not ready:
        print("OVERRIDE REFUSED: methodology validation has not passed; paper stays blocked.")
    payload["paper_override_active"] = override_active
    payload["readiness_recommendation"] = readiness["recommendation"]
    STATUS_PATH.write_text(json.dumps(payload, indent=2, default=str))  # re-write with override audit
    paper_ready = (not scanner_only and paper_enabled and sleeve.can_trade()
                   and not plan.fail_closed and ready)
    if not scanner_only and paper_enabled and not ready:
        print(f"PAPER BLOCKED: {readiness['recommendation']} "
              f"({'; '.join(readiness['blocking_reasons'])})")
    # ponytail: stateless per-run executor submits ENTER notionals only; cross-session book +
    # exit reconciliation is deferred until paper is enabled after the >=4wk shadow stage.
    broker_kind = os.environ.get("EARNINGS_PAPER_BROKER", "sim").strip().lower()
    broker = None
    placed = 0
    if paper_ready:
        audit = SHADOW_LOG_DIR / "paper_audit.jsonl"
        ex = PaperExecutor(equity=equity, risk_limits=sleeve.risk_limits, audit_path=audit)
        if broker_kind == "alpaca":
            from .alpaca_paper_broker import AlpacaPaperBroker
            broker = AlpacaPaperBroker()
            if not broker.ready:
                print(f"ALPACA PAPER BROKER NOT READY: {broker.reason} -> simulation only, "
                      "no broker orders placed")
        for r in plan.rows:
            if r["action"] == "ENTER" and r["target_notional"] > 0:
                # PaperExecutor is the risk-gate authority (limits + sleeve gate + kill switch).
                dec = ex.submit(sleeve, Order(
                    sleeve=sleeve.name, symbol=r["symbol"], side="buy",
                    notional=r["target_notional"], ts=plan.asof,
                ))
                submitted += int(dec.accepted)
                # Only a risk-accepted order is forwarded to the real paper broker.
                if dec.accepted and broker is not None and broker.ready:
                    res = broker.submit_paper_buy(
                        r["symbol"], r["target_notional"],
                        client_order_id=f"earn-{plan.asof}-{r['symbol']}")
                    placed += int(res.get("placed", False))

    mode = "SCANNER_ONLY" if scanner_only or not paper_enabled else "PAPER"
    broker_status = ("alpaca_paper" if (broker is not None and broker.ready)
                     else f"sim({broker.reason})" if broker is not None else "sim")
    if args.json:
        print(json.dumps({**payload, "mode": mode, "paper_submitted": submitted,
                          "broker": broker_status, "broker_orders_placed": placed,
                          "shadow_log": str(log_path)}, indent=2, default=str))
    else:
        print(f"earnings_within_7d [{mode}] asof={plan.asof} ok={payload['ok']} "
              f"regime={plan.spy_regime}/{plan.regime_action} "
              f"flagged={len(plan.flagged)} entries={len(plan.entries)} "
              f"exits={len(plan.exits)} fail_closed={plan.fail_reason or 'no'} "
              f"-> {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
