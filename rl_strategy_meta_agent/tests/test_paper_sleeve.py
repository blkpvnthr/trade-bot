"""Tests for the earnings_within_7d paper sleeve promotion. Network-free."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1] / "experiments" / "orthogonal_features"
sys.path.insert(0, str(PKG))

import paper_sleeve as PS


def test_no_live_broker_path_imported():
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "place_order", "run_v2", "live_trade", ".orders", "alpaca.broker")
    for f in ("paper_sleeve.py", "run_paper_sleeve.py"):
        mods = []
        for n in ast.walk(ast.parse((PKG / f).read_text())):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            assert not any(b in m.lower() for b in banned), f"{f} imports broker/order path {m}"


def test_allocation_cap_enforced():
    # weights that exceed gross + per-name caps must be clamped, never scaled up, no leverage
    raw = {"A": 0.5, "B": -0.5, "C": 0.3}                 # gross 1.3, names over 0.04
    w = PS.enforce_caps(raw)
    assert max(abs(v) for v in w.values()) <= PS.SLEEVE["safety"]["max_name_weight"] + 1e-9
    assert sum(abs(v) for v in w.values()) <= PS.SLEEVE["safety"]["max_allocation"] + 1e-9
    # a tiny book is NOT scaled up to the cap (no leverage manufactured)
    small = PS.enforce_caps({"A": 0.01, "B": -0.01})
    assert sum(abs(v) for v in small.values()) <= 0.02 + 1e-9


def test_stale_earnings_blocks_signals():
    sig = {f"S{i}": (1.0 if i < 3 else 0.0) for i in range(10)}
    w_fresh, d_fresh = PS.target_weights(sig, fetched_at="2026-06-17", as_of="2026-06-18")
    assert not d_fresh["blocked"] and w_fresh                          # fresh -> trades
    w_stale, d_stale = PS.target_weights(sig, fetched_at="2026-06-01", as_of="2026-06-18")
    assert d_stale["blocked"] and w_stale == {}                        # stale -> flat
    w_missing, d_missing = PS.target_weights(sig, fetched_at=None, as_of="2026-06-18")
    assert d_missing["blocked"] and "missing" in d_missing["reason"]


def test_signal_metadata_preserved():
    m = PS.SLEEVE["validated_metrics"]
    assert m["net_sharpe_5bps"] == 1.11 and m["breakeven_bps"] == 56.0
    assert m["bootstrap_p"] == 0.0 and m["placebo_p"] == 0.0 and m["fdr_significant"] is True
    assert m["walk_forward_positive_years"] == "8/11" and m["holdout_net_sharpe"] == 0.32
    assert m["vol_corr"] == 0.03
    assert PS.SLEEVE["safety"]["max_allocation"] == 0.10 and PS.SLEEVE["safety"]["options_allowed"] is False


def test_registry_export_includes_earnings_within_7d(tmp_path):
    path = tmp_path / "reg.jsonl"
    PS.export_registry(path, as_of="2026-06-18")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    assert any(r["alpha_id"] == "earnings_within_7d" and r["decision"] == "ACCEPT"
               and r["status"] == "PAPER_PROMOTED" and r["promotion_requires_shadow_period"] is True
               for r in rows)


def test_monitor_computes_realized_metrics():
    L = pd.DataFrame({"gross_pnl": np.full(50, 0.001), "turnover": 0.03, "n_wins": 6,
                      "n_positions": 10, "slippage_bps": 5.0, "earnings_age_days": 0,
                      "post_earnings_exits": 2, "post_earnings_eligible": 2})
    m = PS.monitor(L)
    assert m["hit_rate"] == 0.6 and m["post_earnings_exit_rate"] == 1.0 and m["n_days"] == 50
    assert "realized_net_sharpe" in m


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except TypeError:
                fn(Path("/tmp"))
            print("ok", name)
