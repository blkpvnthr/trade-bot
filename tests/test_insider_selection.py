"""Tests for the insider-selection buy algorithm (trade.py).

The strategy itself is QUARANTINE per the OOS study (selection edge real, timing edge
not validated); these tests pin the mechanics so it is safe to enable for a paper study:
default-OFF, correct eligibility/horizon logic, state tracking, and clean dispatch.
"""
import trade
from datetime import date, datetime

import pandas as pd


def _cfg(**over):
    return trade.ScannerConfig(**over)


def _ecfg(**over):
    return trade.ExecutorConfig(**over)


# --------------------------------------------------------------------------- config
def test_insider_config_defaults_off():
    c = trade.ScannerConfig()
    assert c.insider_strategy_enabled is False  # ships OFF
    assert c.insider_min_cluster_value == 1_000_000.0
    assert c.insider_max_filing_age_days == 3
    assert c.insider_hold_days == 10
    assert c.insider_per_position_weight == 0.08
    assert c.insider_max_positions == 5
    assert c.insider_stop_loss_pct == -0.15


# ------------------------------------------------------------------- horizon helpers
def test_hold_elapsed_trading_days_counts_weekdays_only():
    # Counts trading days from entry up to and including `today` (the exit day).
    # Mon 2026-11-16 -> Fri 2026-11-20 : Tue/Wed/Thu/Fri = 4
    assert trade.insider_hold_elapsed_trading_days(date(2026, 11, 16), date(2026, 11, 20)) == 4
    # Fri 2026-11-20 -> Mon 2026-11-23 : weekend skipped, Mon counted = 1
    assert trade.insider_hold_elapsed_trading_days(date(2026, 11, 20), date(2026, 11, 23)) == 1
    # same day / past entry -> 0
    assert trade.insider_hold_elapsed_trading_days(date(2026, 11, 16), date(2026, 11, 16)) == 0
    assert trade.insider_hold_elapsed_trading_days(date(2026, 11, 16), date(2026, 11, 10)) == 0


def test_exit_decision_stop_horizon_hold():
    # catastrophe stop fires regardless of horizon
    assert trade.insider_exit_decision(date(2026, 11, 2), date(2026, 11, 3), 80.0, 100.0, 10, -0.15) == (True, "insider_stop_loss")
    # horizon reached (hold_days=3): Mon 11-02 -> Fri 11-06 = 4 trading days >= 3
    assert trade.insider_exit_decision(date(2026, 11, 2), date(2026, 11, 6), 105.0, 100.0, 3, -0.15) == (True, "insider_hold_horizon")
    # within horizon, no stop -> hold
    assert trade.insider_exit_decision(date(2026, 11, 2), date(2026, 11, 4), 98.0, 100.0, 3, -0.15) == (False, "")
    # stop takes precedence over a reached horizon
    assert trade.insider_exit_decision(date(2026, 11, 2), date(2026, 11, 6), 80.0, 100.0, 3, -0.15)[1] == "insider_stop_loss"


# ----------------------------------------------------------------------- entry rows
def test_entry_rows_selects_eligible_unheld():
    price_map = {"AAA": 10.0, "BBB": 20.0, "CCC": 0.0}
    rows = trade.build_insider_entry_rows(
        ["AAA", "BBB", "CCC"], price_map, held_symbols={"BBB"}, holdings={},
        equity=13000.0, config=_cfg(), executor_config=_ecfg(),
    )
    assert {r["symbol"] for r in rows} == {"AAA"}  # BBB held, CCC priceless
    assert rows[0]["side"] == "buy"
    assert rows[0]["reason"] == "insider_selection_entry"
    assert rows[0]["delta_shares"] > 0


def test_entry_rows_skips_existing_holdings_and_caps_positions():
    price_map = {"AAA": 10.0, "BBB": 10.0, "CCC": 10.0}
    # already a tracked holding -> skip
    rows = trade.build_insider_entry_rows(
        ["AAA", "BBB"], price_map, held_symbols=set(), holdings={"AAA": "2026-11-02"},
        equity=13000.0, config=_cfg(), executor_config=_ecfg(),
    )
    assert "AAA" not in {r["symbol"] for r in rows}
    # cap at max_positions (0 held, cap 1 -> 1 row)
    rows2 = trade.build_insider_entry_rows(
        ["AAA", "BBB", "CCC"], price_map, held_symbols=set(), holdings={},
        equity=13000.0, config=_cfg(insider_max_positions=1), executor_config=_ecfg(),
    )
    assert len(rows2) == 1


def test_entry_rows_excludes_dust_below_min_notional():
    # equity 10 * 0.08 = 0.80 < dynamic min_notional floor -> excluded
    rows = trade.build_insider_entry_rows(
        ["AAA"], {"AAA": 5.0}, held_symbols=set(), holdings={},
        equity=10.0, config=_cfg(), executor_config=_ecfg(),
    )
    assert rows == []


def test_entry_rows_fractional_high_priced_name():
    rows = trade.build_insider_entry_rows(
        ["BRKA"], {"BRKA": 600000.0}, set(), {}, 13000.0, _cfg(), _ecfg(),
    )
    assert len(rows) == 1 and 0 < rows[0]["delta_shares"] < 1
    off = trade.build_insider_entry_rows(
        ["BRKA"], {"BRKA": 600000.0}, set(), {}, 13000.0,
        _cfg(enable_fractional_orders=False), _ecfg(),
    )
    assert off == []  # integer shares -> 0 -> excluded


# ------------------------------------------------------------------------ exit rows
def test_exit_rows_horizon_stop_and_untracked():
    pos_horizon = trade.BrokerPosition(symbol="AAA", qty=10, market_value=1050.0, avg_entry_price=100.0, current_price=105.0)
    pos_stop = trade.BrokerPosition(symbol="BBB", qty=10, market_value=700.0, avg_entry_price=100.0, current_price=70.0)
    pos_hold = trade.BrokerPosition(symbol="CCC", qty=10, market_value=980.0, avg_entry_price=100.0, current_price=98.0)
    pos_untracked = trade.BrokerPosition(symbol="DDD", qty=10, market_value=1000.0, avg_entry_price=100.0, current_price=100.0)
    holdings = {"AAA": "2026-11-02", "BBB": "2026-11-05", "CCC": "2026-11-05"}  # DDD absent
    rows = trade.build_insider_exit_rows(
        [pos_horizon, pos_stop, pos_hold, pos_untracked], holdings,
        today=date(2026, 11, 6), config=_cfg(insider_hold_days=3), equity=13000.0,
    )
    by = {r["symbol"]: r["reason"] for r in rows}
    assert by["AAA"] == "insider_hold_horizon"
    assert by["BBB"] == "insider_stop_loss"
    assert "CCC" not in by      # within horizon, no stop
    assert "DDD" not in by      # not an insider-tracked holding
    for r in rows:
        assert r["side"] == "sell" and r["delta_shares"] == 10


# --------------------------------------------------------------------- state helpers
def test_holdings_state_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "insider_holdings.json"
    monkeypatch.setattr(trade, "INSIDER_HOLDINGS_STATE_PATH", p)
    assert trade.load_insider_holdings() == {}
    trade.mark_insider_entry("aapl", "2026-11-02")
    trade.mark_insider_entry("MSFT", "2026-11-03")
    trade.mark_insider_entry("AAPL", "2026-11-09")  # idempotent: first entry date wins
    h = trade.load_insider_holdings()
    assert h == {"AAPL": "2026-11-02", "MSFT": "2026-11-03"}
    trade.clear_insider_holding("AAPL")
    assert trade.load_insider_holdings() == {"MSFT": "2026-11-03"}


# ------------------------------------------------------------------- eligibility scan
def test_eligible_candidates_filters_by_value_and_freshness(monkeypatch):
    import sec_form4_feed as sf
    today = datetime.now(trade.ET).date()
    fresh = today.isoformat()
    stale = date(today.year - 1, 1, 1).isoformat()
    monkeypatch.setattr(sf, "run_sec_form4_discovery", lambda *a, **k: None)
    monkeypatch.setattr(sf, "load_insider_candidates", lambda: {
        "BIG": {"insider_buy_amount": 5_000_000, "insider_buy_score": 0.9, "latest_filing_date": fresh},
        "SMALL": {"insider_buy_amount": 100_000, "insider_buy_score": 0.8, "latest_filing_date": fresh},  # below cluster
        "OLD": {"insider_buy_amount": 9_000_000, "insider_buy_score": 0.95, "latest_filing_date": stale},  # stale filing
        "MID": {"insider_buy_amount": 2_000_000, "insider_buy_score": 0.99, "latest_filing_date": fresh},
    })
    s = object.__new__(trade.MomentumSetupScanner)
    s.config = trade.ScannerConfig(insider_strategy_enabled=True)
    out = s._insider_eligible_candidates()
    assert out == ["MID", "BIG"]  # both eligible, sorted by score desc; SMALL/OLD filtered out


# -------------------------------------------------------------------------- dispatch
class _FakeBroker:
    def __init__(self, equity, positions):
        self._equity = equity
        self._positions = positions

    def get_account_snapshot(self):
        return trade.AccountSnapshot(buying_power=self._equity, equity=self._equity, cash=self._equity, portfolio_value=self._equity)

    def get_all_open_positions(self):
        return self._positions


def test_build_orders_branches_to_insider(tmp_path, monkeypatch):
    monkeypatch.setattr(trade, "DEFAULT_REBALANCE_ORDERS", tmp_path / "rb.csv")
    monkeypatch.setattr(trade, "DEFAULT_EXIT_ORDERS", tmp_path / "ex.csv")
    monkeypatch.setattr(trade, "INSIDER_HOLDINGS_STATE_PATH", tmp_path / "hold.json")

    bars = pd.DataFrame([
        {"symbol": "AAA", "timestamp": pd.Timestamp("2026-11-17 20:00", tz="UTC"), "open": 9.0, "close": 10.0},
    ])
    s = object.__new__(trade.MomentumSetupScanner)
    s.config = trade.ScannerConfig(insider_strategy_enabled=True, daily_green_strategy_enabled=False)
    s.executor_config = trade.ExecutorConfig()
    s.broker = _FakeBroker(13000.0, positions=[])
    s._download_bars = lambda syms: bars
    s._insider_eligible_candidates = lambda: ["AAA"]

    rebalance, exits, scores = s.build_orders()
    assert list(rebalance["symbol"]) == ["AAA"]
    assert rebalance.iloc[0]["reason"] == "insider_selection_entry"
    assert exits.empty and scores.empty
    # entry recorded in holdings state
    assert "AAA" in trade.load_insider_holdings()


def test_build_default_config_env_toggle(monkeypatch):
    monkeypatch.setenv("INSIDER_STRATEGY", "1")
    monkeypatch.setenv("INSIDER_HOLD_DAYS", "15")
    monkeypatch.setenv("INSIDER_MAX_POSITIONS", "7")
    cfg = trade.build_default_config().scanner
    assert cfg.insider_strategy_enabled is True
    assert cfg.daily_green_strategy_enabled is False  # mutually exclusive
    assert cfg.insider_hold_days == 15
    assert cfg.insider_max_positions == 7


def test_build_default_config_insider_off_by_default(monkeypatch):
    monkeypatch.delenv("INSIDER_STRATEGY", raising=False)
    cfg = trade.build_default_config().scanner
    assert cfg.insider_strategy_enabled is False
    assert cfg.daily_green_strategy_enabled is True
