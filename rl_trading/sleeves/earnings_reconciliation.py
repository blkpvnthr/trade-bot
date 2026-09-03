"""earnings_reconciliation — forward-calendar vs actual-reportedDate reconciliation + paper readiness.

The earnings_within_7d backtest used `reportedDate` (ex-post) as a near-PIT proxy; the LIVE sleeve
acts on forward-scheduled dates. This module closes that gap operationally:

  Phase 1  reconcile live shadow logs (scheduled-date-at-entry we ACTED on) against the actual
           reportedDate (what really happened), per held event, idempotently.
  Phase 3  combine the live reconciliation gate with the historical methodology validation into a
           single READY_FOR_PAPER / NOT_READY_FOR_PAPER recommendation.

Never places orders. Never enables live. Reconciliation is a read/measure/write step only.

LIMITATION (load-bearing, honest): there is NO historical archive of forward-scheduled dates (the
earnings cache stores only reportedDate), so the scheduled-vs-actual MISMATCH can only be measured
GOING FORWARD from live shadow logs. The retrospective study (run_reconciliation_validation.py) is a
date-perturbation proxy for it, not a direct observation.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from . import earnings_within_7d as EW
from .earnings_within_7d import (
    RECONCILIATION_FIELDS,
    RECON_SUMMARY_PATH,
    SHADOW_ACCEPTANCE,
    SHADOW_LOG_DIR,
    _as_date,
    reconciliation_row,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
RECON_CSV_PATH = SHADOW_LOG_DIR / "calendar_reconciliation.csv"
AV_EARNINGS_CACHE = _REPO_ROOT / "data" / "cache" / "av_earnings"
# Historical methodology-validation report (Phase 2 output). out/ is gitignored runtime artifact.
RECON_VALIDATION_PATH = (
    _REPO_ROOT / "rl_strategy_meta_agent" / "experiments" / "orthogonal_features"
    / "out" / "reconciliation_validation_report.json"
)
READINESS_PATH = _REPO_ROOT / "system-data" / "earnings_within_7d_paper_readiness.json"

MIN_RECON_EVENTS = 10        # need enough reconciled events for the gate to mean anything
MAX_DEGRADATION_FRAC = 0.50  # live forward-calendar must capture >= 50% of the proxy drift


# ── data adapters ─────────────────────────────────────────────────────────────────
def load_actual_reported_dates(
    universe: Sequence[str], cache_dir: Path = AV_EARNINGS_CACHE
) -> Dict[str, List[date]]:
    """{symbol: sorted actual reportedDate list} from the local earnings cache. reportedDate is the
    ACTUAL ex-post report date (the only date the cache stores)."""
    out: Dict[str, List[date]] = {}
    for sym in universe:
        p = Path(cache_dir) / f"{sym}.json"
        dates: List[date] = []
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except (ValueError, OSError):
                data = []
            recs = data if isinstance(data, list) else data.get("quarterlyEarnings", [])
            for r in recs:
                rd = r.get("reportedDate")
                if rd:
                    try:
                        dates.append(date.fromisoformat(str(rd)[:10]))
                    except ValueError:
                        pass
        out[sym] = sorted(dates)
    return out


def make_close_lookup() -> Callable[[str, str], Optional[float]]:
    """Causal close lookup: close on/at-or-before a date from cached adjusted Alpaca bars."""
    try:
        from .. import data as rl_data
    except Exception:  # pragma: no cover
        return lambda sym, iso: None
    cache: Dict[str, object] = {}

    def lookup(sym: str, iso: str) -> Optional[float]:
        if sym not in cache:
            cache[sym] = rl_data.load_daily(sym, source="adj")
        df = cache[sym]
        if df is None or getattr(df, "empty", True):
            return None
        d = df.loc[df.index <= iso + "T23:59:59+00:00"]
        return float(d["close"].iloc[-1]) if len(d) else None

    return lookup


# ── shadow-log reading + episode detection ────────────────────────────────────────
def _read_shadow_rows(log_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for p in sorted(Path(log_dir).glob("shadow_*.csv")):
        with p.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _episodes_for_symbol(rows: List[dict]) -> List[dict]:
    """Segment a symbol's chronological rows into holding episodes: ENTER -> HOLD* -> EXIT.

    Each closed episode (has an EXIT) is one reconcilable event. Open episodes (no EXIT yet) are
    skipped until the exit appears."""
    rows = sorted(rows, key=lambda r: r["date"])
    episodes, cur = [], None
    for r in rows:
        act = r.get("action")
        if act == "ENTER":
            if cur:
                episodes.append(cur)          # defensively close an unterminated prior episode
            cur = {"entry": r, "last": r, "exit": None}
        elif act == "HOLD" and cur:
            cur["last"] = r
        elif act == "EXIT" and cur:
            cur["exit"] = r
            cur["last"] = r          # the exit row carries the latest scheduled date
            episodes.append(cur)
            cur = None
    if cur:
        episodes.append(cur)                  # still open -> incomplete (no exit)
    return episodes


def _next_reported_on_or_after(dates: Sequence[date], d: date) -> Optional[date]:
    for rd in dates:
        if rd >= d:
            return rd
    return None


def reconcile_from_shadow_logs(
    log_dir: Path = SHADOW_LOG_DIR,
    earnings_dates: Optional[Mapping[str, Sequence[date]]] = None,
    close_lookup: Optional[Callable[[str, str], Optional[float]]] = None,
    now: Optional[date] = None,
) -> List[dict]:
    """Build reconciliation rows from live shadow logs. Deterministic; one row per CLOSED holding
    episode whose actual reportedDate is already known (<= now). Pure given its injected inputs."""
    rows = _read_shadow_rows(log_dir)
    if not rows:
        return []
    now = now or datetime.now(timezone.utc).date()
    universe = sorted({r["symbol"] for r in rows})
    earnings_dates = earnings_dates or load_actual_reported_dates(universe)
    close_lookup = close_lookup or make_close_lookup()

    by_sym: Dict[str, List[dict]] = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], []).append(r)

    out: List[dict] = []
    for sym, srows in by_sym.items():
        for ep in _episodes_for_symbol(srows):
            if ep["exit"] is None:
                continue                                  # not yet exited -> incomplete
            entry, last, exit_ = ep["entry"], ep["last"], ep["exit"]
            entry_d = _as_date(entry["date"])
            exit_d = _as_date(exit_["date"])
            actual = _next_reported_on_or_after(earnings_dates.get(sym, []), entry_d)
            if actual is None or actual > now:
                continue                                  # actual report not yet known -> skip
            ce, cx = close_lookup(sym, entry_d.isoformat()), close_lookup(sym, exit_d.isoformat())
            # day before the actual print: proxy would capture drift to the actual-1 trading day
            actual_m1 = date.fromordinal(actual.toordinal() - 1)
            ca = close_lookup(sym, actual_m1.isoformat())
            ret_exit = (cx / ce - 1.0) if (ce and cx) else None
            ret_actual_m1 = (ca / ce - 1.0) if (ce and ca) else None
            out.append(reconciliation_row(
                symbol=sym, entry_date=entry_d,
                scheduled_at_entry=entry.get("scheduled_earnings_date"),
                scheduled_latest=last.get("scheduled_earnings_date"),
                actual_reported=actual, exit_date=exit_d,
                fail_closed_reason=exit_.get("reason", ""),
                return_to_exit=ret_exit, return_to_actual_minus_1=ret_actual_m1,
            ))
    out.sort(key=lambda r: (r["entry_date"] or "", r["symbol"]))
    return out


def write_reconciliation_csv(rows: Sequence[dict], path: Path = RECON_CSV_PATH) -> Path:
    """Idempotent: rebuilds the consolidated reconciliation CSV from `rows` (full overwrite)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RECONCILIATION_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ── summary + gate ─────────────────────────────────────────────────────────────────
def _weeks_observed(log_dir: Path) -> float:
    rows = _read_shadow_rows(log_dir)
    if not rows:
        return 0.0
    ds = [_as_date(r["date"]) for r in rows]
    return round((max(ds) - min(ds)).days / 7.0, 2)


def build_reconciliation_summary(rows: Sequence[dict], weeks_observed: float) -> dict:
    n = len(rows)
    held_through = sum(1 for r in rows if r["held_through_actual_report"])
    reschedules = sum(1 for r in rows if r["date_changed"])
    days_changed = [r["days_changed"] for r in rows if r["days_changed"] is not None]
    shadow_rets = [r["return_to_exit"] for r in rows if r["return_to_exit"] is not None]
    proxy_rets = [r["return_to_actual_minus_1"] for r in rows if r["return_to_actual_minus_1"] is not None]
    shadow_mean = sum(shadow_rets) / len(shadow_rets) if shadow_rets else 0.0
    proxy_mean = sum(proxy_rets) / len(proxy_rets) if proxy_rets else 0.0
    degradation = proxy_mean - shadow_mean

    why: List[str] = []
    if weeks_observed < SHADOW_ACCEPTANCE["min_weeks"]:
        why.append(f"weeks_observed {weeks_observed}<{SHADOW_ACCEPTANCE['min_weeks']}")
    if n < MIN_RECON_EVENTS:
        why.append(f"events_reconciled {n}<{MIN_RECON_EVENTS}")
    if held_through != 0:
        why.append(f"held_through_actual_report={held_through} (must be 0)")
    if proxy_mean > 0 and degradation > MAX_DEGRADATION_FRAC * proxy_mean:
        why.append(f"degradation_vs_proxy {degradation:.4f} > {MAX_DEGRADATION_FRAC}*proxy")
    gate_passed = len(why) == 0

    return {
        "weeks_observed": weeks_observed,
        "events_reconciled": n,
        "held_through_actual_count": held_through,
        "mismatch_rate": round(reschedules / n, 4) if n else 0.0,
        "reschedule_count": reschedules,
        "avg_days_changed": round(sum(days_changed) / len(days_changed), 3) if days_changed else 0.0,
        "forward_shadow_return_mean": round(shadow_mean, 6),
        "reportedDate_proxy_return_mean": round(proxy_mean, 6),
        "degradation_vs_proxy": round(degradation, 6),
        "gate_passed": gate_passed,
        "why_not_passed": why,
        # compatibility keys read by earnings_within_7d.reconciliation_gate_passed():
        "passed": gate_passed,
        "weeks": weeks_observed,
        "holds_through_actual_report": held_through,
    }


def run_reconcile(log_dir: Path = SHADOW_LOG_DIR, now: Optional[date] = None,
                  summary_path: Path = RECON_SUMMARY_PATH) -> dict:
    """Phase-1 driver: rebuild reconciliation CSV + summary JSON from shadow logs. Never orders."""
    rows = reconcile_from_shadow_logs(log_dir=log_dir, now=now)
    write_reconciliation_csv(rows, RECON_CSV_PATH)   # global resolved at call time (monkeypatchable)
    summary = build_reconciliation_summary(rows, _weeks_observed(log_dir))
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ── Phase 3: paper readiness ───────────────────────────────────────────────────────
def methodology_validation_passed(path: Path = RECON_VALIDATION_PATH) -> tuple[bool, str]:
    """Reads the Phase-2 historical methodology report. Fail-closed if missing/failing."""
    path = Path(path)
    if not path.exists():
        return False, "methodology_report_missing(run_reconciliation_validation.py not run)"
    try:
        r = json.loads(path.read_text())
    except (ValueError, OSError):
        return False, "methodology_report_unreadable"
    if not r.get("methodology_validation_passed"):
        return False, "methodology_validation_failed:" + ",".join(r.get("why_not_passed", []) or ["unknown"])
    return True, "methodology_validated"


def paper_readiness(write: bool = False) -> dict:
    """Combine the live reconciliation gate (operational) with the historical methodology gate.
    READY_FOR_PAPER only if BOTH pass. Paper still defaults OFF; live remains impossible."""
    recon_ok, recon_reason = EW.reconciliation_gate_passed()
    meth_ok, meth_reason = methodology_validation_passed()
    reasons = []
    if not recon_ok:
        reasons.append(f"reconciliation: {recon_reason}")
    if not meth_ok:
        reasons.append(f"methodology: {meth_reason}")
    ready = recon_ok and meth_ok
    payload = {
        "recommendation": "READY_FOR_PAPER" if ready else "NOT_READY_FOR_PAPER",
        "reconciliation_gate_passed": recon_ok,
        "reconciliation_reason": recon_reason,
        "methodology_validation_passed": meth_ok,
        "methodology_reason": meth_reason,
        "blocking_reasons": reasons,
        "paper_default": "OFF",
        "live": "PERMANENTLY_DISABLED",
        "canonical_status": "ACCEPT_WITH_PROVENANCE_CAVEAT",
    }
    if write:
        READINESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        READINESS_PATH.write_text(json.dumps(payload, indent=2, default=str))
    return payload
