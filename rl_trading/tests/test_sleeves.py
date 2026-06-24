"""Sleeve framework guarantees: gate enforcement, paper-only, risk caps, kill switch, audit,
scanner-always-works, determinism. Live is never tradeable.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rl_trading.sleeves import (
    Order,
    PaperExecutor,
    RiskLimits,
    Sleeve,
    SleeveMode,
    ValidationRecord,
    status_report,
    validation_gate,
)
from rl_trading.sleeves.initial import build_default_registry, trend_ranker_sleeve

LIMITS = RiskLimits(
    max_notional=10_000.0, per_order_notional=2_000.0, max_position_pct=0.10,
    max_gross_exposure=1.0, max_concurrent=3, max_daily_loss=500.0,
)

PASS_RECORD = ValidationRecord(
    source="hypothetical_passing_source",
    ledger_id="hypo_pass_001",
    bootstrap_ci=(0.004, 0.012),
    placebo_passed=True, oos_passed=True, ood_passed=True,
    attribution_passed=True, concentration_passed=True,
    value=0.008,
)


def _sleeve(record, paper_enabled, **kw):
    base = dict(
        name="t", signal_source="s", native_execution_type="event_hold",
        eligible_symbols=("AAA",), entry_rules={"native_horizon_days": 20}, exit_rules={},
        sizing_rules={}, risk_limits=LIMITS, required_validation=record, paper_enabled=paper_enabled,
    )
    base.update(kw)
    return Sleeve(**base)


def _validated_sleeve(tmp_path):
    return _sleeve(PASS_RECORD, True)


def _exec(tmp_path, equity=50_000.0):
    return PaperExecutor(equity=equity, risk_limits=LIMITS, audit_path=tmp_path / "audit.jsonl")


# ---- gate ------------------------------------------------------------------
def test_gate_passes_only_when_complete():
    assert validation_gate(PASS_RECORD).passed
    assert not validation_gate(None).passed
    assert "no_validation_record" in validation_gate(None).failures


@pytest.mark.parametrize("field_,val,key", [
    ("bootstrap_ci", None, "bootstrap_ci_missing"),
    ("bootstrap_ci", (-0.01, 0.01), "bootstrap_ci_not_positive"),
    ("placebo_passed", False, "placebo_not_passed"),
    ("oos_passed", None, "oos_not_passed"),
    ("ood_passed", False, "ood_not_passed"),
    ("attribution_passed", None, "attribution_not_passed"),
    ("concentration_passed", False, "concentration_not_passed"),
])
def test_each_missing_criterion_blocks(field_, val, key):
    rec = replace(PASS_RECORD, **{field_: val})
    g = validation_gate(rec)
    assert not g.passed
    assert key in g.failures


# ---- sleeve modes ----------------------------------------------------------
def test_unvalidated_sleeve_cannot_trade():
    s = _sleeve(None, paper_enabled=True)
    assert not s.can_trade()
    assert s.mode() is SleeveMode.DISABLED


def test_missing_ledger_pass_blocks_orders(tmp_path):
    # record present but incomplete (no placebo/ood) -> scanner only, orders blocked
    partial = replace(PASS_RECORD, placebo_passed=None, ood_passed=None)
    s = _sleeve(partial, paper_enabled=True)
    assert s.mode() is SleeveMode.SCANNER_ONLY
    ex = _exec(tmp_path)
    d = ex.submit(s, Order(s.name, "AAA", "buy", 1000.0, "2024-01-02"))
    assert not d.accepted
    assert any(r.startswith("sleeve_not_tradeable") for r in d.reasons)


def test_validated_but_paper_disabled_is_scanner_only():
    s = _sleeve(PASS_RECORD, paper_enabled=False)
    assert s.gate().passed
    assert not s.can_trade()
    assert s.mode() is SleeveMode.SCANNER_ONLY
    assert s.why_not_tradeable() == "paper_not_enabled"


def test_validated_and_enabled_can_trade():
    assert _sleeve(PASS_RECORD, paper_enabled=True).can_trade()


# ---- live always blocked ---------------------------------------------------
def test_live_mode_always_blocked(tmp_path):
    s = _validated_sleeve(tmp_path)
    ex = _exec(tmp_path)
    d = ex.submit(s, Order(s.name, "AAA", "buy", 1000.0, "2024-01-02", live=True))
    assert not d.accepted
    assert d.reasons == ["live_blocked"]
    assert PaperExecutor.LIVE_SUPPORTED is False
    assert s.live_enabled is False


def test_live_enabled_field_cannot_be_set_via_init():
    with pytest.raises(TypeError):
        Sleeve(
            name="x", signal_source="s", native_execution_type="t", eligible_symbols=(),
            entry_rules={}, exit_rules={}, sizing_rules={}, risk_limits=LIMITS,
            live_enabled=True,  # not an init arg
        )


# ---- paper-only guard ------------------------------------------------------
def test_paper_only_guard_accepts_valid_paper_order(tmp_path):
    s = _validated_sleeve(tmp_path)
    ex = _exec(tmp_path)
    d = ex.submit(s, Order(s.name, "AAA", "buy", 1000.0, "2024-01-02"))
    assert d.accepted and not d.reasons
    assert ex.audit_records[-1]["mode"] == "paper"


# ---- kill switch -----------------------------------------------------------
def test_kill_switch_blocks_all_orders(tmp_path):
    s = _validated_sleeve(tmp_path)
    ex = _exec(tmp_path)
    ex.trip_kill_switch("manual")
    d = ex.submit(s, Order(s.name, "AAA", "buy", 100.0, "2024-01-02"))
    assert not d.accepted
    assert any(r.startswith("kill_switch") for r in d.reasons)


def test_daily_loss_trips_kill_switch(tmp_path):
    s = _validated_sleeve(tmp_path)
    ex = _exec(tmp_path)
    ex.record_pnl(-600.0)  # exceeds max_daily_loss 500
    assert ex.killed
    d = ex.submit(s, Order(s.name, "AAA", "buy", 100.0, "2024-01-02"))
    assert not d.accepted
    assert "daily_loss_limit" in d.reasons or any(r.startswith("kill_switch") for r in d.reasons)


# ---- risk caps -------------------------------------------------------------
def test_per_order_notional_cap(tmp_path):
    s = _validated_sleeve(tmp_path)
    ex = _exec(tmp_path)
    d = ex.submit(s, Order(s.name, "AAA", "buy", 5000.0, "2024-01-02"))  # > 2000
    assert not d.accepted and "per_order_notional_exceeded" in d.reasons


def test_max_position_pct_cap(tmp_path):
    s = _validated_sleeve(tmp_path)
    ex = _exec(tmp_path, equity=10_000.0)  # 10% = 1000 cap
    # two 600 buys on same symbol -> 1200 projected > 1000
    ex.submit(s, Order(s.name, "AAA", "buy", 600.0, "2024-01-02"))
    d = ex.submit(s, Order(s.name, "AAA", "buy", 600.0, "2024-01-03"))
    assert not d.accepted and "max_position_pct_exceeded" in d.reasons


def test_max_concurrent_cap(tmp_path):
    s = _validated_sleeve(tmp_path)
    ex = _exec(tmp_path)  # max_concurrent 3
    for i, sym in enumerate(["AAA", "BBB", "CCC"]):
        assert ex.submit(s, Order(s.name, sym, "buy", 500.0, f"d{i}")).accepted
    d = ex.submit(s, Order(s.name, "DDD", "buy", 500.0, "d4"))
    assert not d.accepted and "max_concurrent_exceeded" in d.reasons


def test_max_gross_and_notional_caps(tmp_path):
    s = _validated_sleeve(tmp_path)
    # equity 5000, gross cap = 1.0*5000 = 5000; also max_notional 10000 -> gross binds first
    ex = _exec(tmp_path, equity=5_000.0)
    lim = replace(LIMITS, max_position_pct=1.0, max_concurrent=100)
    ex.limits = lim
    ok = 0
    for i in range(20):
        if ex.submit(s, Order(s.name, f"S{i}", "buy", 1000.0, f"d{i}")).accepted:
            ok += 1
    assert ok == 5  # 5 * 1000 = 5000 = gross cap; 6th blocked
    d = ex.submit(s, Order(s.name, "Z", "buy", 1000.0, "dz"))
    assert "max_gross_exposure_exceeded" in d.reasons


# ---- scanner works even when trading blocked -------------------------------
def test_scanner_works_when_trading_blocked():
    s = trend_ranker_sleeve()  # scanner-only (does not beat SPY)
    assert not s.can_trade()
    rows = s.scan([{"symbol": "AAPL", "ts": "2024-01-02", "strength": 0.9}])
    assert len(rows) == 1
    r = rows[0]
    assert r["tradeable"] is False
    assert r["signal_strength"] == 0.9
    assert r["pit_timestamp"] == "2024-01-02"
    assert r["proposed_native_horizon_days"] == 20
    assert r["reason_not_tradeable"].startswith("gate_failed")
    assert r["required_validation_next_step"] != "none"


def test_scanner_works_for_disabled_sleeve():
    from rl_trading.sleeves.initial import event_hold_sleeve
    rows = event_hold_sleeve().scan([{"symbol": "X", "ts": "t", "strength": 1.0}])
    assert rows[0]["required_validation_next_step"] == "no_validation_record"


# ---- audit log -------------------------------------------------------------
def test_audit_log_written(tmp_path):
    s = _validated_sleeve(tmp_path)
    ex = _exec(tmp_path)
    ex.submit(s, Order(s.name, "AAA", "buy", 1000.0, "2024-01-02"))
    ex.submit(s, Order(s.name, "BBB", "buy", 9999.0, "2024-01-03"))  # blocked (per-order cap)
    path = tmp_path / "audit.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(x) for x in lines]
    assert recs[0]["accepted"] is True
    assert recs[1]["accepted"] is False
    assert all("reasons" in r and "ts" in r and r["sleeve"] == s.name for r in recs)


# ---- determinism -----------------------------------------------------------
def test_deterministic_behavior(tmp_path):
    s = _validated_sleeve(tmp_path)
    orders = [
        Order(s.name, "AAA", "buy", 800.0, "d1"),
        Order(s.name, "BBB", "buy", 3000.0, "d2"),
        Order(s.name, "AAA", "buy", 800.0, "d3"),
        Order(s.name, "CCC", "buy", 500.0, "d4"),
    ]

    def run(p):
        ex = PaperExecutor(equity=50_000.0, risk_limits=LIMITS, audit_path=p)
        return [(d.accepted, tuple(d.reasons)) for d in (ex.submit(s, o) for o in orders)]

    r1 = run(tmp_path / "a1.jsonl")
    r2 = run(tmp_path / "a2.jsonl")
    assert r1 == r2
    assert (tmp_path / "a1.jsonl").read_text() == (tmp_path / "a2.jsonl").read_text()


# ---- registry / report -----------------------------------------------------
def test_registry_status_and_disclaimer():
    reg = build_default_registry()
    rep = status_report(reg)
    assert "NO VALIDATED LIVE EDGE" in rep["disclaimer"]
    assert "1%/day" in rep["disclaimer"]
    assert rep["live_status"].startswith("DISABLED")
    assert rep["counts"]["validated"] == 0       # honest: nothing tradeable yet (paper not enabled)
    assert rep["counts"]["total"] == 8           # 4 scaffolded + 3 legacy + earnings_within_7d
    assert rep["champion"] == "trend_ranker_sleeve"
    # nothing is validated/tradeable because no sleeve has paper enabled (earnings clears the gate
    # but defaults paper_enabled=False -> scanner-only)
    names_validated = [s["name"] for s in rep["validated_sleeves"]]
    assert names_validated == []
    earn = reg.get("earnings_within_7d_sleeve")
    assert earn.gate().passed                     # FIRST sleeve to clear the gate
    assert earn.mode().value == "scanner_only"    # ...but still scanner-only (paper disabled)
    assert not earn.can_trade()


def test_no_sleeve_is_live_in_report():
    reg = build_default_registry()
    rep = status_report(reg)
    assert all(s["live_enabled"] is False for s in rep["all_sleeves"])


# ---- promotion bridge (event_study -> ValidationRecord) --------------------
from rl_trading.sleeves.promote import from_event_study  # noqa: E402

_GOOD_REAL = {"n": 300, "mean": 0.012, "median": 0.006, "hit_rate": 0.58,
              "boot_lo": 0.004, "boot_hi": 0.020, "block_lo": 0.003, "block_hi": 0.021}
_WEAK_PLACEBO = {"mean": 0.001, "block_hi": 0.004, "boot_hi": 0.004}
_GOOD_DOMAINS = {y: {"n": 30, "mean": 0.01, "hit": 0.6} for y in range(2015, 2024)}
_GOOD_DROPTOP = {"block_lo": 0.002, "boot_lo": 0.002}


def test_promote_passes_with_clean_stats():
    rec = from_event_study("hypo", "led1", _GOOD_REAL, _WEAK_PLACEBO, _GOOD_DOMAINS, _GOOD_DROPTOP)
    assert validation_gate(rec).passed


def test_promote_fails_when_placebo_beats_signal():
    # Replays the real PEAD market-neutral falsification: placebo mean >= signal mean.
    strong_placebo = {"mean": 0.013, "block_hi": 0.014, "boot_hi": 0.014}
    rec = from_event_study("pead", "led2", _GOOD_REAL, strong_placebo, _GOOD_DOMAINS, _GOOD_DROPTOP)
    g = validation_gate(rec)
    assert not g.passed and "placebo_not_passed" in g.failures


def test_promote_fails_when_concentrated():
    rec = from_event_study("conc", "led3", _GOOD_REAL, _WEAK_PLACEBO, _GOOD_DOMAINS,
                           {"block_lo": -0.003, "boot_lo": -0.003})
    g = validation_gate(rec)
    assert not g.passed and "concentration_not_passed" in g.failures


def test_promote_fails_on_negative_ci():
    bad = dict(_GOOD_REAL, block_lo=-0.002)
    rec = from_event_study("neg", "led4", bad, _WEAK_PLACEBO, _GOOD_DOMAINS, _GOOD_DROPTOP)
    g = validation_gate(rec)
    assert not g.passed and "bootstrap_ci_not_positive" in g.failures


def test_promote_fails_when_domains_flip():
    flip = {y: {"n": 30, "mean": (0.01 if y % 2 else -0.01), "hit": 0.5} for y in range(2015, 2024)}
    rec = from_event_study("flip", "led5", _GOOD_REAL, _WEAK_PLACEBO, flip, _GOOD_DROPTOP)
    g = validation_gate(rec)
    assert not g.passed and "ood_not_passed" in g.failures


def test_promoted_source_becomes_tradeable_paper_sleeve():
    """End-to-end: a clean source mints a passing record -> sleeve can paper-trade (never live)."""
    rec = from_event_study("hypo", "led1", _GOOD_REAL, _WEAK_PLACEBO, _GOOD_DOMAINS, _GOOD_DROPTOP)
    s = _sleeve(rec, paper_enabled=True)
    assert s.can_trade() and s.mode() is SleeveMode.PAPER
    assert s.live_enabled is False
