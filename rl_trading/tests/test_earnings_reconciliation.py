"""earnings_reconciliation guarantees: episode reconciliation, summary/gate logic, historical
methodology report, paper readiness composition, and paper-only enforcement (no orders, no live).
Hermetic: the driver takes injected earnings dates + close lookup (no network, no cache)."""
from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from rl_trading.sleeves import earnings_reconciliation as RC
from rl_trading.sleeves import earnings_within_7d as EW

SHADOW_HEADER = EW.SHADOW_FIELDS


def _write_shadow(tmp_path, rows):
    p = tmp_path / "shadow_2026-06.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHADOW_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return tmp_path


def _row(d, sym, action, sched, reason=""):
    return {"date": d, "symbol": sym, "scheduled_earnings_date": sched,
            "days_to_earnings": 5, "signal_flag": True, "action": action, "reason": reason}


def _closes(mapping):
    # mapping: {(sym, iso): close}; lookup returns close on/at-or-before requested date by exact key
    def lookup(sym, iso):
        return mapping.get((sym, iso))
    return lookup


# ── reconciliation builder ─────────────────────────────────────────────────────────
def test_reconcile_clean_exit_before_print(tmp_path):
    ld = _write_shadow(tmp_path, [
        _row("2026-06-20", "AAPL", "ENTER", "2026-06-29"),
        _row("2026-06-22", "AAPL", "HOLD", "2026-06-29"),
        _row("2026-06-26", "AAPL", "EXIT", "2026-06-29", reason="before_print"),
    ])
    closes = _closes({("AAPL", "2026-06-20"): 100.0, ("AAPL", "2026-06-26"): 105.0,
                      ("AAPL", "2026-06-28"): 106.0})
    rows = RC.reconcile_from_shadow_logs(
        log_dir=ld, earnings_dates={"AAPL": [date(2026, 6, 29)]},
        close_lookup=closes, now=date(2026, 7, 1))
    assert len(rows) == 1
    r = rows[0]
    assert r["held_through_actual_report"] is False
    assert r["date_changed"] is False and r["days_changed"] == 0
    assert r["return_to_exit"] == pytest.approx(0.05)
    assert r["return_to_actual_minus_1"] == pytest.approx(0.06)


def test_reconcile_flags_held_through_and_reschedule(tmp_path):
    ld = _write_shadow(tmp_path, [
        _row("2026-06-20", "MSFT", "ENTER", "2026-06-30"),
        _row("2026-06-24", "MSFT", "EXIT", "2026-06-24", reason="before_print"),  # rescheduled earlier
    ])
    closes = _closes({("MSFT", "2026-06-20"): 50.0, ("MSFT", "2026-06-24"): 49.0,
                      ("MSFT", "2026-06-23"): 49.5})
    rows = RC.reconcile_from_shadow_logs(
        log_dir=ld, earnings_dates={"MSFT": [date(2026, 6, 24)]},
        close_lookup=closes, now=date(2026, 7, 1))
    r = rows[0]
    assert r["held_through_actual_report"] is True        # exit on the actual report date
    assert r["date_changed"] is True and r["days_changed"] == -6


def test_reconcile_skips_when_actual_not_yet_known(tmp_path):
    ld = _write_shadow(tmp_path, [
        _row("2026-06-20", "AAPL", "ENTER", "2026-06-29"),
        _row("2026-06-26", "AAPL", "EXIT", "2026-06-29"),
    ])
    rows = RC.reconcile_from_shadow_logs(
        log_dir=ld, earnings_dates={"AAPL": [date(2026, 6, 29)]},
        close_lookup=_closes({}), now=date(2026, 6, 27))   # now < actual report
    assert rows == []


def test_reconcile_skips_open_episode(tmp_path):
    ld = _write_shadow(tmp_path, [
        _row("2026-06-20", "AAPL", "ENTER", "2026-06-29"),
        _row("2026-06-22", "AAPL", "HOLD", "2026-06-29"),   # no EXIT yet
    ])
    rows = RC.reconcile_from_shadow_logs(
        log_dir=ld, earnings_dates={"AAPL": [date(2026, 6, 29)]},
        close_lookup=_closes({}), now=date(2026, 7, 1))
    assert rows == []


def test_reconcile_is_idempotent(tmp_path):
    ld = _write_shadow(tmp_path, [
        _row("2026-06-20", "AAPL", "ENTER", "2026-06-29"),
        _row("2026-06-26", "AAPL", "EXIT", "2026-06-29"),
    ])
    ed, cl, now = {"AAPL": [date(2026, 6, 29)]}, _closes({}), date(2026, 7, 1)
    r1 = RC.reconcile_from_shadow_logs(log_dir=ld, earnings_dates=ed, close_lookup=cl, now=now)
    r2 = RC.reconcile_from_shadow_logs(log_dir=ld, earnings_dates=ed, close_lookup=cl, now=now)
    assert r1 == r2                                          # deterministic
    out = tmp_path / "recon.csv"
    RC.write_reconciliation_csv(r1, out)
    RC.write_reconciliation_csv(r1, out)                    # rewrite -> same content (no dupes)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_date") and len(lines) == 1 + len(r1)


# ── summary + gate ──────────────────────────────────────────────────────────────────
def _recon_row(held=False, changed=False, days=0, exit_ret=0.01, proxy_ret=0.012):
    return EW.reconciliation_row(
        "X", "2026-06-20", "2026-06-29", "2026-06-29" if not changed else "2026-06-23",
        actual_reported="2026-06-29", exit_date="2026-06-29" if held else "2026-06-26",
        return_to_exit=exit_ret, return_to_actual_minus_1=proxy_ret)


def test_summary_gate_blocks_until_enough_weeks_and_events():
    rows = [_recon_row() for _ in range(12)]
    s = build = RC.build_reconciliation_summary(rows, weeks_observed=2.0)  # < 4 weeks
    assert s["gate_passed"] is False and any("weeks_observed" in w for w in s["why_not_passed"])
    s2 = RC.build_reconciliation_summary(rows, weeks_observed=5.0)         # 12 events, 5 weeks
    assert s2["gate_passed"] is True and s2["events_reconciled"] == 12
    # compatibility keys the gate reader uses
    assert s2["passed"] is True and s2["weeks"] == 5.0 and s2["holds_through_actual_report"] == 0


def test_summary_held_through_is_hard_fail():
    rows = [_recon_row() for _ in range(11)] + [_recon_row(held=True)]
    s = RC.build_reconciliation_summary(rows, weeks_observed=6.0)
    assert s["gate_passed"] is False
    assert s["held_through_actual_count"] == 1
    assert any("held_through" in w for w in s["why_not_passed"])


def test_summary_reschedule_and_degradation_fields():
    rows = [_recon_row(changed=True, days=-6, exit_ret=0.002, proxy_ret=0.02) for _ in range(12)]
    s = RC.build_reconciliation_summary(rows, weeks_observed=6.0)
    assert s["reschedule_count"] == 12 and s["mismatch_rate"] == 1.0
    assert s["degradation_vs_proxy"] > 0          # proxy captured more than the early exit
    assert s["gate_passed"] is False              # degradation too large


# ── methodology report + readiness ──────────────────────────────────────────────────
def test_methodology_gate_failclosed(tmp_path):
    p = tmp_path / "rep.json"
    assert RC.methodology_validation_passed(p)[0] is False           # missing
    p.write_text(json.dumps({"methodology_validation_passed": True}))
    assert RC.methodology_validation_passed(p)[0] is True
    p.write_text(json.dumps({"methodology_validation_passed": False, "why_not_passed": ["x"]}))
    assert RC.methodology_validation_passed(p)[0] is False


def test_historical_validation_report_present_and_valid():
    if not RC.RECON_VALIDATION_PATH.exists():
        pytest.skip("run experiments/orthogonal_features/run_reconciliation_validation.py first")
    r = json.loads(RC.RECON_VALIDATION_PATH.read_text())
    for k in ("reportedDate_proxy", "perturbation_scenarios", "date_shuffle_placebo_frac_ge_real",
              "methodology_validation_passed", "limitation"):
        assert k in r
    assert "perturbation proxy" in r["limitation"].lower() or "perturbation" in r["limitation"].lower()


def test_paper_readiness_ready_only_when_both_gates_pass(monkeypatch):
    monkeypatch.setattr(RC.EW, "reconciliation_gate_passed", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(RC, "methodology_validation_passed", lambda *a, **k: (True, "ok"))
    assert RC.paper_readiness()["recommendation"] == "READY_FOR_PAPER"
    # either gate failing -> NOT_READY
    monkeypatch.setattr(RC.EW, "reconciliation_gate_passed", lambda *a, **k: (False, "no_recon"))
    out = RC.paper_readiness()
    assert out["recommendation"] == "NOT_READY_FOR_PAPER"
    assert any("reconciliation" in r for r in out["blocking_reasons"])
    assert out["live"] == "PERMANENTLY_DISABLED" and out["paper_default"] == "OFF"


def test_run_reconcile_writes_files_and_never_orders(tmp_path, monkeypatch):
    ld = _write_shadow(tmp_path, [
        _row("2026-06-20", "AAPL", "ENTER", "2026-06-29"),
        _row("2026-06-26", "AAPL", "EXIT", "2026-06-29"),
    ])
    monkeypatch.setattr(RC, "RECON_CSV_PATH", tmp_path / "recon.csv")
    summary = RC.run_reconcile(log_dir=ld, now=date(2026, 7, 1),
                               summary_path=tmp_path / "summary.json")
    assert (tmp_path / "recon.csv").exists() and (tmp_path / "summary.json").exists()
    assert "gate_passed" in summary and summary["gate_passed"] in (True, False)
    # reconciliation module imports no broker/live path
    src = open(RC.__file__).read()
    for forbidden in ("TradingClient", "submit_order", "live=True", "alpaca.trading"):
        assert forbidden not in src


# ── explicit paper override (operator decision) ───────────────────────────────────
def test_override_bypasses_only_reconciliation_not_methodology():
    from rl_trading.sleeves.earnings_within_7d import resolve_paper_authorization
    # methodology passed but reconciliation not -> override authorizes paper
    r = {"recommendation": "NOT_READY_FOR_PAPER", "methodology_validation_passed": True,
         "reconciliation_reason": "reconciliation_not_passed"}
    ready, active, _ = resolve_paper_authorization(r, override=True)
    assert ready is True and active is True
    # without the override -> blocked
    assert resolve_paper_authorization(r, override=False) == (False, False, "blocked:NOT_READY_FOR_PAPER")
    # override CANNOT bypass a failed methodology validation
    bad = {"recommendation": "NOT_READY_FOR_PAPER", "methodology_validation_passed": False,
           "reconciliation_reason": "x"}
    ready2, active2, reason2 = resolve_paper_authorization(bad, override=True)
    assert ready2 is False and active2 is False and "methodology" in reason2
    # genuinely READY needs no override
    ok = {"recommendation": "READY_FOR_PAPER", "methodology_validation_passed": True}
    assert resolve_paper_authorization(ok, override=False)[:2] == (True, False)


# ── real Alpaca PAPER broker: hard refuses live keys, fail-closed ─────────────────
def test_alpaca_paper_broker_refuses_live_and_missing_keys(monkeypatch):
    from rl_trading.sleeves.alpaca_paper_broker import AlpacaPaperBroker
    # no paper keys -> not ready, no orders
    monkeypatch.delenv("EARNINGS_ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("EARNINGS_ALPACA_PAPER_SECRET_KEY", raising=False)
    b = AlpacaPaperBroker()
    assert b.ready is False and "no_paper_keys" in b.reason
    assert b.submit_paper_buy("AAPL", 1000.0)["placed"] is False
    # a LIVE 'AK' key is hard-refused even if provided as the paper key
    monkeypatch.setenv("EARNINGS_ALPACA_PAPER_KEY_ID", "AKLIVEXYZ")
    monkeypatch.setenv("EARNINGS_ALPACA_PAPER_SECRET_KEY", "secret")
    b2 = AlpacaPaperBroker()
    assert b2.ready is False and "non-paper" in b2.reason
    # a key identical to the live APCA_API_KEY_ID is refused
    monkeypatch.setenv("APCA_API_KEY_ID", "PKSAME")
    monkeypatch.setenv("EARNINGS_ALPACA_PAPER_KEY_ID", "PKSAME")
    b3 = AlpacaPaperBroker()
    assert b3.ready is False and "identical to the live" in b3.reason
    # never advertises live capability
    assert AlpacaPaperBroker.LIVE_SUPPORTED is False


def test_alpaca_paper_broker_module_forces_paper_endpoint():
    from rl_trading.sleeves import alpaca_paper_broker as B
    src = open(B.__file__).read()
    assert "paper=True" in src and "paper=False" not in src          # paper endpoint forced
    assert "//api.alpaca.markets" not in src                         # no hardcoded live endpoint
    assert "APCA_API_KEY_ID" in src  # referenced ONLY to refuse reuse, never to authenticate
