"""earnings_within_7d sleeve guarantees: canonical-record loading, forward-only calendar,
fail-closed, long-only equal weights, liquidity, SPY 200SMA gate, exit-before-print, and the
paper gate. Hermetic: the planner takes injected data (no network, no pandas)."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from rl_trading.sleeves import validation_gate
from rl_trading.sleeves.core import SleeveMode
from rl_trading.sleeves.paper import Order, PaperExecutor
from rl_trading.sleeves.promote import (
    CANONICAL_EXTENDED_VALIDATION,
    from_extended_validation,
)
from rl_trading.sleeves.initial import EARNINGS_UNIVERSE, earnings_within_7d_sleeve
from rl_trading.sleeves import earnings_within_7d as E
from rl_trading.sleeves import earnings_features as EF

ASOF = date(2026, 6, 24)
EQUITY = 100_000.0
CFG = E.EarningsConfig()
VALIDATION_JSON = (
    CANONICAL_EXTENDED_VALIDATION.parent / "earnings_variant_validation.json"
)


def _liq(syms, price=50.0, adv=20_000_000.0):
    return {s: {"price": price, "adv20": adv} for s in syms}


def _plan(scheduled, liquidity, held=None, *, spy_above=True, age=0.0, cov=1.0, feed=True,
          insider=frozenset(), universe=None, cfg=CFG, scores=None, variant="A"):
    universe = universe or list(scheduled)
    return E.plan_day(
        ASOF, universe, scheduled, liquidity, held or {}, EQUITY, cfg,
        spy_above=spy_above, calendar_age_days=age, coverage=cov, feed_ok=feed,
        insider_held=insider, scores=scores, variant=variant,
    )


def _row(plan, sym):
    return next(r for r in plan.rows if r["symbol"] == sym)


# ── 1 / 2: canonical record loaded; stale artifact rejected ───────────────────────
def test_canonical_validation_record_loads_and_clears_gate():
    rec = from_extended_validation("earnings_within_7d")
    assert validation_gate(rec).passed
    assert rec.ledger_id == "extended_validation::earnings_within_7d"
    assert rec.value == pytest.approx(1.11)
    assert rec.bootstrap_ci[0] > 0
    # canonical headline numbers are pinned in the record notes
    for token in ("1.11", "55.9", "0.242", "Supersedes intraday lead_validation.json"):
        assert token in rec.notes


def test_stale_lead_validation_artifact_is_rejected(tmp_path):
    stale = tmp_path / "lead_validation.json"
    stale.write_text(json.dumps({"leads": {"earnings_within_7d": {}}}))
    with pytest.raises(ValueError, match="superseded"):
        from_extended_validation("earnings_within_7d", path=stale)


def test_non_accept_signal_is_not_promotable():
    # idio_reversal is REJECT in the canonical artifact -> must refuse to mint a record
    with pytest.raises(ValueError, match="not ACCEPT"):
        from_extended_validation("idio_reversal")


def test_sleeve_is_scanner_only_by_default():
    s = earnings_within_7d_sleeve()
    assert s.gate().passed and not s.paper_enabled
    assert s.mode() is SleeveMode.SCANNER_ONLY and not s.can_trade()


# ── 3: forward calendar only ──────────────────────────────────────────────────────
def test_calendar_keeps_only_forward_scheduled_dates(tmp_path):
    cal = tmp_path / "cal_x.json"
    cal.write_text(json.dumps({
        "events": {"AAPL": ["2026-05-01", "2026-06-30"], "MSFT": ["2026-01-01"]},
        "fetched_at": "2026-06-24T08:00:00+00:00", "source": "test",
    }))
    c = E.load_calendar(["AAPL", "MSFT"], ASOF, cal_dir=tmp_path,
                        now=datetime(2026, 6, 24, 12, tzinfo=timezone.utc))
    assert c.scheduled["AAPL"] == "2026-06-30"   # past 2026-05-01 dropped, forward kept
    assert c.scheduled["MSFT"] is None           # only past date -> no forward event
    assert c.coverage == 1.0 and c.feed_ok


# ── 4: no estimate/overview snapshot endpoints feed the signal ────────────────────
def test_signal_path_uses_no_estimate_or_overview_snapshots():
    text = open(E.__file__).read()
    for forbidden in ("OVERVIEW", "EARNINGS_ESTIMATES", "EARNINGS_CALENDAR_ESTIMATE", "ESTIMATES"):
        assert forbidden not in text
    # signal comes from forward scheduled cal_*.json only
    assert "earnings_calendar" in text and "cal_*.json" in text


# ── 5 / 6: fail-closed ────────────────────────────────────────────────────────────
def test_fail_closed_on_stale_calendar_blocks_entries_and_flattens():
    sched = {"AAPL": "2026-06-29"}
    p = _plan(sched, _liq(["AAPL"]), held={"AAPL": 5_000.0}, age=10.0)  # age 10 > 3
    assert p.fail_closed and p.fail_reason.startswith("calendar_stale")
    assert _row(p, "AAPL")["action"] == "EXIT"   # held name flattened for safety
    assert p.entries == []


def test_fail_closed_on_low_coverage():
    p = _plan({"AAPL": "2026-06-29"}, _liq(["AAPL"]), cov=0.50)  # 0.50 < 0.95
    assert p.fail_closed and "coverage_low" in p.fail_reason
    assert _row(p, "AAPL")["action"] == "FAIL_CLOSED"
    assert p.entries == []


def test_fail_closed_on_feed_unavailable():
    p = _plan({"AAPL": "2026-06-29"}, _liq(["AAPL"]), feed=False)
    assert p.fail_closed and p.fail_reason == "feed_unavailable"


# ── 7: long-only equal weights ────────────────────────────────────────────────────
def test_long_only_equal_weight_targets():
    syms = ["AAPL", "MSFT", "NVDA"]
    sched = {s: "2026-06-29" for s in syms}  # all 5 days out -> flagged, far enough to hold
    p = _plan(sched, _liq(syms))
    entered = [r for r in p.rows if r["action"] == "ENTER"]
    assert {r["symbol"] for r in entered} == set(syms)
    notionals = {r["target_notional"] for r in entered}
    assert len(notionals) == 1 and notionals.pop() > 0      # equal weight
    assert all(r["target_weight"] > 0 for r in entered)     # long-only (no shorts/negatives)
    gross = sum(r["target_notional"] for r in entered)
    assert gross <= CFG.max_gross_pct * EQUITY + 1e-6        # gross cap respected
    assert all(r["target_notional"] <= CFG.max_position_pct * EQUITY + 1e-6 for r in entered)


def test_max_concurrent_and_cash_buffer():
    syms = [f"S{i}" for i in range(15)]
    sched = {s: "2026-06-29" for s in syms}
    p = _plan(sched, _liq(syms))
    assert len(p.entries) <= CFG.max_concurrent
    gross = sum(r["target_notional"] for r in p.rows if r["action"] == "ENTER")
    assert gross <= (1 - CFG.cash_buffer_pct) * EQUITY + 1e-6


# ── 8: liquidity filter ───────────────────────────────────────────────────────────
def test_liquidity_filter_excludes_cheap_and_thin():
    sched = {"GOOD": "2026-06-29", "CHEAP": "2026-06-29", "THIN": "2026-06-29"}
    liq = {
        "GOOD": {"price": 50.0, "adv20": 20_000_000.0},
        "CHEAP": {"price": 4.0, "adv20": 20_000_000.0},     # price < $5
        "THIN": {"price": 50.0, "adv20": 1_000_000.0},      # ADV < $5M
    }
    p = _plan(sched, liq)
    assert _row(p, "GOOD")["action"] == "ENTER"
    assert _row(p, "CHEAP")["action"] == "FLAT" and _row(p, "CHEAP")["reason"] == "illiquid"
    assert _row(p, "THIN")["action"] == "FLAT" and _row(p, "THIN")["reason"] == "illiquid"


def test_insider_overlap_excluded():
    sched = {"AAPL": "2026-06-29", "MSFT": "2026-06-29"}
    p = _plan(sched, _liq(["AAPL", "MSFT"]), insider={"MSFT"})
    assert _row(p, "AAPL")["action"] == "ENTER"
    assert _row(p, "MSFT")["action"] == "FLAT" and _row(p, "MSFT")["reason"] == "insider_overlap"


# ── 9: SPY 200SMA gate ────────────────────────────────────────────────────────────
def test_spy_below_200sma_pauses_new_entries():
    sched = {"AAPL": "2026-06-29"}
    p = _plan(sched, _liq(["AAPL"]), spy_above=False)  # default policy = pause
    assert p.regime_action == "pause" and p.spy_regime == "below_200sma"
    assert _row(p, "AAPL")["action"] == "FLAT" and _row(p, "AAPL")["reason"] == "regime_pause"
    assert p.entries == []


def test_spy_below_200sma_size_down_keeps_entries_smaller():
    # 6 names so equal-weight (gross/n) binds before the 10%/name cap, exposing the size-down cut
    syms = [f"S{i}" for i in range(6)]
    sched = {s: "2026-06-29" for s in syms}
    cfg = E.EarningsConfig(regime_policy="size_down")
    full = _plan(sched, _liq(syms), spy_above=True)
    down = _plan(sched, _liq(syms), spy_above=False, cfg=cfg)
    assert set(down.entries) == set(syms)  # entries kept, just smaller
    full_gross = sum(r["target_notional"] for r in full.rows if r["action"] == "ENTER")
    down_gross = sum(r["target_notional"] for r in down.rows if r["action"] == "ENTER")
    assert down_gross < full_gross
    assert down_gross == pytest.approx(full_gross * cfg.regime_size_down_factor, rel=1e-6)


def test_spy_below_200sma_does_not_force_exit_existing():
    sched = {"AAPL": "2026-06-29"}  # 5 days out, still a valid hold
    p = _plan(sched, _liq(["AAPL"]), held={"AAPL": 5_000.0}, spy_above=False)
    assert _row(p, "AAPL")["action"] == "HOLD"  # pause blocks NEW entries, not existing holds


# ── 10 / 11: exit before print; never hold through ────────────────────────────────
def test_exit_at_close_day_before_print():
    sched = {"AAPL": "2026-06-25"}  # earnings tomorrow -> days_to == 1 -> exit today
    p = _plan(sched, _liq(["AAPL"]), held={"AAPL": 5_000.0})
    r = _row(p, "AAPL")
    assert r["action"] == "EXIT" and r["reason"] == "before_print"


def test_revised_earlier_earnings_forces_earlier_exit():
    # held name that would normally HOLD (5d out) but calendar revised earnings to tomorrow
    far = _plan({"AAPL": "2026-06-29"}, _liq(["AAPL"]), held={"AAPL": 5_000.0})
    assert _row(far, "AAPL")["action"] == "HOLD"
    revised = _plan({"AAPL": "2026-06-25"}, _liq(["AAPL"]), held={"AAPL": 5_000.0})
    assert _row(revised, "AAPL")["action"] == "EXIT"


def test_never_hold_through_print():
    for d in ("2026-06-24", "2026-06-23"):  # print today / already passed
        p = _plan({"AAPL": d}, _liq(["AAPL"]), held={"AAPL": 5_000.0})
        assert _row(p, "AAPL")["action"] == "EXIT"
        assert _row(p, "AAPL")["signal_flag"] is False


def test_no_entry_inside_exit_window():
    # earnings tomorrow, not held -> must not ENTER (would have to exit immediately)
    p = _plan({"AAPL": "2026-06-25"}, _liq(["AAPL"]))
    assert _row(p, "AAPL")["action"] == "FLAT"


# ── 12 / 13: paper gate ───────────────────────────────────────────────────────────
def test_paper_orders_blocked_when_paper_disabled(tmp_path):
    s = earnings_within_7d_sleeve(paper_enabled=False)
    ex = PaperExecutor(EQUITY, s.risk_limits, tmp_path / "a.jsonl")
    dec = ex.submit(s, Order(s.name, "AAPL", "buy", 1_000.0, "2026-06-24"))
    assert not dec.accepted
    assert any("sleeve_not_tradeable" in r for r in dec.reasons)


def test_paper_orders_allowed_only_when_gate_passes_and_paper_enabled(tmp_path):
    s = earnings_within_7d_sleeve(paper_enabled=True)
    assert s.gate().passed and s.can_trade()
    ex = PaperExecutor(EQUITY, s.risk_limits, tmp_path / "a.jsonl")
    dec = ex.submit(s, Order(s.name, "AAPL", "buy", 1_000.0, "2026-06-24"))
    assert dec.accepted


def test_live_order_always_blocked_even_when_paper_enabled(tmp_path):
    s = earnings_within_7d_sleeve(paper_enabled=True)
    ex = PaperExecutor(EQUITY, s.risk_limits, tmp_path / "a.jsonl")
    dec = ex.submit(s, Order(s.name, "AAPL", "buy", 1_000.0, "2026-06-24", live=True))
    assert not dec.accepted and "live_blocked" in dec.reasons


# ── shadow logger ─────────────────────────────────────────────────────────────────
def test_shadow_log_is_append_only_with_header(tmp_path):
    p = _plan({"AAPL": "2026-06-29"}, _liq(["AAPL"]))
    path1 = E.append_shadow_log(p.rows, log_dir=tmp_path)
    path2 = E.append_shadow_log(p.rows, log_dir=tmp_path)
    assert path1 == path2
    lines = path1.read_text().splitlines()
    assert lines[0].startswith("date,symbol,scheduled_earnings_date")  # one header
    assert len([ln for ln in lines if ln.startswith("date,symbol")]) == 1
    assert len(lines) == 1 + 2 * len(p.rows)  # header + two appends


def test_dashboard_payload_flags_not_ok_on_fail_closed():
    p = _plan({"AAPL": "2026-06-29"}, _liq(["AAPL"]), feed=False)
    cal = E.Calendar(scheduled={}, coverage=0.0, age_days=None, feed_ok=False)
    pay = E.dashboard_payload(p, cal)
    assert pay["ok"] is False and pay["fail_closed"] is True


# ── integration: PIT firewall (fundamentals cannot create a signal) ───────────────
def test_fundamentals_cannot_create_signal():
    # an UNflagged name with an absurdly high rank score must NEVER be entered: the binary
    # earnings_within_7d signal comes only from the calendar; ranking only orders flagged names.
    p = _plan({"AAPL": None}, _liq(["AAPL"]), scores={"AAPL": 9e9}, variant="C")
    r = _row(p, "AAPL")
    assert r["signal_flag"] is False and r["action"] == "FLAT"
    # the fundamentals path itself refuses to be a signal source
    with pytest.raises(RuntimeError, match="cannot create"):
        EF.assert_not_signal_source()
    f = EF.fundamentals_features("AAPL")
    assert f["_pit_safe"] is False and "snapshot" in f["_warning"].lower()


def test_variant_C_is_non_pit_and_disabled_by_default():
    assert EF.DEFAULT_VARIANT == "A"          # validated winner, not C
    assert EF.LIVE_ONLY_NON_PIT_BACKTESTABLE is True
    # rank_score for C uses fundamentals but only on an already-flagged name (caller gates by flag)
    sA = EF.rank_score("X", {"price": 50, "ret_5d": 0.1, "avg_dollar_vol_20d": 2e7}, 3, "A")
    sB = EF.rank_score("X", {"price": 50, "ret_5d": 0.1, "avg_dollar_vol_20d": 2e7}, 3, "B")
    assert sA == 0.0 and sB > 0.0            # A is equal-weight; B uses PIT drift only


def test_variant_B_selects_top_drift_among_flagged():
    syms = [f"S{i}" for i in range(12)]      # 12 flagged, cap = 10 -> 2 dropped
    sched = {s: "2026-06-29" for s in syms}
    scores = {f"S{i}": float(i) for i in range(12)}   # S11 best, S0/S1 worst
    p = _plan(sched, _liq(syms), scores=scores, variant="B")
    entered = set(p.entries)
    assert len(entered) == CFG.max_concurrent
    assert "S0" not in entered and "S1" not in entered      # lowest drift dropped
    assert "S11" in entered and "S10" in entered            # highest drift kept


def test_drift_score_is_price_only_and_safe_on_missing():
    assert EF.drift_score({}, None) == 0.0
    assert EF.drift_score({"price": None}, 3) == 0.0
    s = EF.drift_score({"price": 50, "ret_5d": 0.08, "ret_20d": 0.05,
                        "vol_20d": 0.02, "avg_dollar_vol_20d": 5e7}, 3)
    assert s > 0.0


def test_new_feature_columns_present_in_rows_and_log():
    p = _plan({"AAPL": "2026-06-29"}, _liq(["AAPL"]), scores={"AAPL": 1.0}, variant="B")
    r = _row(p, "AAPL")
    for col in ("rank_score", "spy_above_200sma", "market_exposure_multiplier"):
        assert col in r
    for col in ("rank_score", "spy_above_200sma", "market_exposure_multiplier"):
        assert col in E.SHADOW_FIELDS


def test_av_calendar_fetch_requires_key(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        EF.fetch_av_earnings_calendar(["AAPL"], ASOF)


# ── retraining/validation artifact (A vs B, PIT-safe splits, placebo) ──────────────
def _load_validation():
    if not VALIDATION_JSON.exists():
        pytest.skip("run experiments/orthogonal_features/run_earnings_variants.py first")
    return json.loads(VALIDATION_JSON.read_text())


def test_validation_has_canonical_parity():
    v = _load_validation()
    assert v["parity_check"]["within_0.05_of_pin"] is True
    assert abs(v["parity_check"]["canonical_net_sharpe_5bps"] - 1.11) <= 0.05


def test_retraining_uses_pit_safe_splits():
    v = _load_validation()
    for r in v["variants"]:
        assert r["pit_safe"] is True
        assert "wf_by_year" in r and len(r["wf_by_year"]) >= 5      # walk-forward by year
        assert "holdout_net_sharpe" in r                            # symbol holdout
    # winner is A or B only (C never enters validation)
    assert v["winner"]["variant"] in ("variant_A", "variant_B")
    assert "NON_PIT" in v["variant_C_status"]


def test_placebo_shuffled_dates_lose_edge():
    v = _load_validation()
    pl = v["date_shuffle_placebo"]
    assert pl["frac_placebo_ge_real"] < 0.05                        # real edge not replicated
    assert pl["placebo_mean_net_sharpe_10bps"] < pl["real_net_sharpe_10bps"]


# ── scanner logs but does not order ───────────────────────────────────────────────
def test_scanner_mode_logs_but_does_not_order(tmp_path):
    p = _plan({"AAPL": "2026-06-29"}, _liq(["AAPL"]), scores={"AAPL": 1.0}, variant="A")
    path = E.append_shadow_log(p.rows, log_dir=tmp_path)
    assert path.exists() and path.read_text().count("AAPL") >= 1     # logged
    # a scanner-config sleeve (paper disabled) cannot place an order
    s = earnings_within_7d_sleeve(paper_enabled=False)
    ex = PaperExecutor(EQUITY, s.risk_limits, tmp_path / "a.jsonl")
    dec = ex.submit(s, Order(s.name, "AAPL", "buy", 1_000.0, ASOF.isoformat()))
    assert not dec.accepted


# ── provenance caveat + reconciliation gate (blocking disclosure) ─────────────────
def test_canonical_status_is_accept_with_provenance_caveat():
    from rl_trading.sleeves import promote as P
    assert P.CANONICAL_STATUS == "ACCEPT_WITH_PROVENANCE_CAVEAT"
    assert "reportedDate" in P.PROVENANCE_CAVEAT and "forward scheduled" in P.PROVENANCE_CAVEAT
    rec = from_extended_validation("earnings_within_7d")
    assert "ACCEPT_WITH_PROVENANCE_CAVEAT" in rec.notes      # status pinned in the record
    assert "reportedDate" in rec.notes


def test_reconciliation_row_flags_held_through_and_reschedule():
    # clean exit before the print, date unchanged
    ok = E.reconciliation_row("AAPL", "2026-06-20", "2026-06-29", "2026-06-29",
                              actual_reported="2026-06-29", exit_date="2026-06-26")
    assert ok["held_through_actual_report"] is False and ok["date_changed"] is False
    assert ok["days_changed"] == 0
    # rescheduled earlier AND exit lands on/after the actual print -> violation
    bad = E.reconciliation_row("MSFT", "2026-06-20", "2026-06-30", "2026-06-24",
                               actual_reported="2026-06-24", exit_date="2026-06-24")
    assert bad["held_through_actual_report"] is True        # hard fail
    assert bad["date_changed"] is True and bad["days_changed"] == -6


def test_reconciliation_log_append_only(tmp_path):
    rows = [E.reconciliation_row("AAPL", "2026-06-20", "2026-06-29", "2026-06-29",
                                 actual_reported="2026-06-29", exit_date="2026-06-26")]
    p1 = E.append_reconciliation_log(rows, log_dir=tmp_path)
    p2 = E.append_reconciliation_log(rows, log_dir=tmp_path)
    assert p1 == p2
    lines = p1.read_text().splitlines()
    assert lines[0].startswith("symbol,entry_date,scheduled_earnings_date_at_entry")
    assert len([ln for ln in lines if ln.startswith("symbol,entry_date")]) == 1


def test_reconciliation_gate_blocks_paper_until_passed(tmp_path):
    missing = tmp_path / "nope.json"
    ok, reason = E.reconciliation_gate_passed(missing)
    assert ok is False and "missing" in reason
    # incomplete: <4 weeks
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"passed": True, "weeks": 2, "holds_through_actual_report": 0}))
    assert E.reconciliation_gate_passed(short)[0] is False
    # held through a real print -> blocked
    held = tmp_path / "held.json"
    held.write_text(json.dumps({"passed": True, "weeks": 6, "holds_through_actual_report": 1}))
    assert E.reconciliation_gate_passed(held)[0] is False
    # fully passing
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"passed": True, "weeks": 6, "holds_through_actual_report": 0}))
    assert E.reconciliation_gate_passed(good) == (True, "reconciliation_passed")


def test_dashboard_surfaces_status_and_recon_gate():
    p = _plan({"AAPL": "2026-06-29"}, _liq(["AAPL"]))
    cal = E.Calendar(scheduled={"AAPL": "2026-06-29"}, coverage=1.0, age_days=0.0, feed_ok=True)
    pay = E.dashboard_payload(p, cal)
    assert pay["canonical_status"] == "ACCEPT_WITH_PROVENANCE_CAVEAT"
    assert "reconciliation_gate_passed" in pay and pay["reconciliation_gate_passed"] in (True, False)
    assert "provenance_caveat" in pay
