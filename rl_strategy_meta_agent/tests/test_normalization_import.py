import json
from pathlib import Path

from rl_meta.tv_import import (parse_tv_csv, normalize_trades, parse_canonical,
                              REFERENCE_QTY)
from rl_meta.performance import PerformanceDB, compute_metrics
from rl_meta.types import TradeRecord


def tr(qty, pnl, pct, ed="2025-03-04", xd="2025-03-06"):
    return TradeRecord(ed, xd, 100.0, 110.0, pnl, pct, 2.0, raw_qty=qty,
                       normalized_qty=qty, raw_pnl_usd=pnl)


def write_tv_csv(path: Path, rows):
    """rows: (edt, xdt, entry_px, exit_px, qty, pnl, pct)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Trade number,Type,Date and time,Price USD,Size (qty),Net PnL USD,Net PnL %"]
    for i, (edt, xdt, ep, xp, qty, pnl, pct) in enumerate(rows, 1):
        lines.append(f"{i},Entry long,{edt},{ep},{qty},,")
        lines.append(f"{i},Exit long,{xdt},{xp},{qty},{pnl},{pct}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- normalization rules ---------------------------------------------------
def test_scale_factor_rules():
    assert REFERENCE_QTY == 10.0
    trades = [tr(1000, 1000, 5), tr(100, 100, 5), tr(50, 50, 5),
              tr(10, 10, 5), tr(5, 5, 5)]
    normalize_trades(trades, REFERENCE_QTY, enabled=True)
    assert [round(t.scale_factor, 4) for t in trades] == [0.01, 0.10, 0.20, 1.0, 1.0]
    assert [round(t.normalized_qty, 1) for t in trades] == [10, 10, 10, 10, 5]
    # pnl scaled by each trade's own factor; raw preserved
    assert trades[0].pnl == 10.0 and trades[0].raw_pnl_usd == 1000.0
    assert trades[4].pnl == 5.0           # <=10 qty unchanged
    assert all(t.normalized for t in trades)


def test_no_normalize_keeps_raw():
    trades = [tr(1000, 1000, 5)]
    normalize_trades(trades, REFERENCE_QTY, enabled=False)
    assert trades[0].scale_factor == 1.0 and trades[0].pnl == 1000.0
    assert trades[0].normalized is False


def test_normalization_preserves_pct_and_direction():
    trades = [tr(1000, 1000, 5.0), tr(1000, -800, -4.0)]
    normalize_trades(trades, REFERENCE_QTY)
    assert trades[0].return_pct == 5.0 and trades[1].return_pct == -4.0   # unchanged
    assert trades[0].pnl > 0 and trades[1].pnl < 0                        # signs kept


def test_parse_captures_qty(tmp_path):
    p = tmp_path / "x.csv"
    write_tv_csv(p, [("2025-03-04 17:15", "2025-03-06 17:45", 100, 110, 1000, 500.0, 5.0)])
    t = parse_tv_csv(p)[0]
    assert t.raw_qty == 1000.0 and t.raw_pnl_usd == 500.0


def test_normalization_changes_pf_when_sizes_differ():
    # One oversized winner + one small loser: raw PF inflated by size; normalized
    # PF reflects equal risk.
    raw = [tr(2000, 2000, 10), tr(50, -100, -5)]
    norm = [tr(2000, 2000, 10), tr(50, -100, -5)]
    pf_raw = compute_metrics(raw).profit_factor
    normalize_trades(norm, REFERENCE_QTY)
    pf_norm = compute_metrics(norm).profit_factor
    assert pf_raw != pf_norm                       # size distortion removed
    # winner 2000sh -> x0.005 -> +10 ; loser 50sh -> x0.2 -> -20
    assert abs(norm[0].pnl - 10.0) < 1e-6 and norm[1].pnl == -20.0


# -- canonical path resolution --------------------------------------------
def test_parse_canonical_variants():
    assert parse_canonical("linear_regression__AAPL__15Min__20250306_20260620.csv") \
        == ("linear_regression", "AAPL", "15Min")          # 4-part name
    assert parse_canonical("sample__NVDA__1Day.csv") == ("sample", "NVDA", "1Day")
    assert parse_canonical("a/b/martingale/IONQ/15Min/martingale__IONQ__15Min__x.csv") \
        == ("martingale", "IONQ", "15Min")                 # nested + name
    assert parse_canonical(Path("mean_reversion/NVDA/15Min/raw_export.csv")) \
        == ("mean_reversion", "NVDA", "15Min")             # nested dirs only
    assert parse_canonical("not_canonical.csv") is None


# -- DB round trip with normalization + manifest plumbing ------------------
def test_db_roundtrip_stores_normalization(tmp_path):
    db = PerformanceDB(tmp_path / "p.db")
    trades = [tr(1000, 1000, 5.0)]
    normalize_trades(trades, REFERENCE_QTY)
    db.record_import("martingale", "IONQ", "15Min", "tv_abc", "f.csv", "h1",
                     trades, "2026-01-01")
    got = db.imported_trades("martingale", "15Min", "tv_abc")["IONQ"][0]
    assert got.normalized and got.scale_factor == 0.01
    assert got.pnl == 10.0 and got.raw_pnl_usd == 1000.0
    assert db.imported_strategies("15Min") == [("martingale", "tv_abc")]


def test_manifest_ingestion_round_trip(tmp_path):
    # Build a canonical export + a manifest, then walk it like the importer does.
    csv = tmp_path / "exports" / "martingale" / "IONQ" / "15Min" / \
        "martingale__IONQ__15Min__20250304_20260205.csv"
    write_tv_csv(csv, [("2025-03-04 10:00", "2025-03-06 11:00", 100, 110, 1000, 1000.0, 5.0),
                       ("2025-04-04 10:00", "2025-04-06 11:00", 110, 100, 1000, -900.0, -4.0)])
    manifest = {"records": [{
        "strategy_id": "martingale", "symbol": "IONQ", "timeframe": "15Min",
        "dest_path": str(csv), "file_hash": "deadbeef"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    db = PerformanceDB(tmp_path / "p.db")
    recs = json.loads((tmp_path / "manifest.json").read_text())["records"]
    for r in recs:
        trades = parse_tv_csv(Path(r["dest_path"]))
        normalize_trades(trades, REFERENCE_QTY)
        db.record_import(r["strategy_id"], r["symbol"], r["timeframe"], "tv_x",
                         Path(r["dest_path"]).name, r["file_hash"], trades, "2026-01-01")
    out = db.imported_trades("martingale", "15Min", "tv_x")
    assert "IONQ" in out and len(out["IONQ"]) == 2
    # outcome from PnL, normalized (x0.01): winner -> +10, loser -> -9
    assert out["IONQ"][0].pnl == 10.0 and out["IONQ"][1].pnl == -9.0


# -- equity curve + drawdown ----------------------------------------------
def _trade(ed, xd, pnl):
    return TradeRecord(ed, xd, 100.0, 100.0, pnl, pnl / 1000.0, 1.0)


def test_equity_curve_starts_from_initial_equity():
    from rl_meta.performance import INITIAL_EQUITY
    m = compute_metrics([_trade("2023-01-01", "2023-01-02", 500.0)])
    assert m.initial_equity == INITIAL_EQUITY
    assert m.ending_equity == INITIAL_EQUITY + 500.0


def test_drawdown_uses_running_peak_not_cumulative_pnl():
    # up to +2000 (peak 12000), then -3000 (equity 9000). DD = 3000/12000 = 0.25.
    # A naive cumulative-PnL drawdown would give 3000/2000 = 1.5 -> wrong.
    m = compute_metrics([_trade("2023-01-01", "2023-01-02", 2000.0),
                         _trade("2023-02-01", "2023-02-02", -3000.0)])
    assert m.max_drawdown == 0.25 and not m.blown_up


def test_drawdown_is_chronological_not_input_order():
    # Same trades, reversed input order -> same (date-ordered) drawdown.
    a = [_trade("2023-01-01", "2023-01-02", 2000.0),
         _trade("2023-02-01", "2023-02-02", -3000.0)]
    assert compute_metrics(a).max_drawdown == compute_metrics(list(reversed(a))).max_drawdown


def test_blown_up_when_equity_hits_zero():
    # initial 10000, a single -10000 loss -> equity 0 -> blown up, DD capped 100%.
    m = compute_metrics([_trade("2023-01-01", "2023-01-02", -10000.0)])
    assert m.blown_up and m.max_drawdown == 1.0 and m.min_equity <= 0


def test_not_blown_up_normal_curve():
    m = compute_metrics([_trade("2023-01-01", "2023-01-02", 100.0),
                         _trade("2023-02-01", "2023-02-02", -50.0)])
    assert not m.blown_up and m.max_drawdown < 1.0


# -- hard RL risk gates ----------------------------------------------------
def gate(**kw):
    from rl_meta.ranking import rl_risk_gate
    from rl_meta.types import Metrics
    base = dict(trade_count=600, profit_factor=2.0, max_drawdown=0.1, blown_up=False)
    base.update({k: v for k, v in kw.items() if k in
                 ("trade_count", "profit_factor", "max_drawdown", "blown_up")})
    full = Metrics(**base)
    oos = Metrics(trade_count=kw.get("oos", 300))
    return rl_risk_gate(full, oos, kw.get("sid", "strat"), kw.get("family", "trend"))


def test_gate_blocks_blown_up():
    assert not gate(blown_up=True)[0]


def test_gate_blocks_high_drawdown():
    assert not gate(max_drawdown=0.5)[0]
    assert gate(max_drawdown=0.49)[0]


def test_gate_blocks_low_profit_factor():
    assert not gate(profit_factor=1.24)[0]
    assert gate(profit_factor=1.25)[0]


def test_gate_blocks_low_trade_count():
    assert not gate(trade_count=499)[0]


def test_gate_blocks_low_oos_trade_count():
    assert not gate(oos=249)[0]


def test_gate_blocks_martingale_by_id_and_family():
    assert not gate(sid="my_martingale_v2")[0]      # id contains martingale
    assert not gate(family="martingale")[0]
    # even with otherwise-perfect metrics
    assert not gate(sid="martingale", profit_factor=5.0, max_drawdown=0.01)[0]


def test_gate_passes_clean_strategy():
    ok, reason = gate(trade_count=600, profit_factor=2.0, max_drawdown=0.2, oos=300)
    assert ok and reason == "passes risk gates"
