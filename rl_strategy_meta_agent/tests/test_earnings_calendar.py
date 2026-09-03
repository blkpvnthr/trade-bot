"""Tests for the forward earnings-calendar adapter + source-agnostic fail-closed wiring.
Network-free (synthetic)."""
import ast
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "experiments" / "orthogonal_features"
sys.path.insert(0, str(PKG))

import earnings_calendar as EC
import paper_sleeve as PS

NOW = "2026-06-18T00:00:00+00:00"
EV = {"AAA": ["2026-06-20"], "BBB": ["2026-07-30"], "CCC": ["2026-06-22"]}
U = ["AAA", "BBB", "CCC"]


def test_assess_fails_closed():
    assert EC.assess(EV, "2026-06-17T00:00:00+00:00", NOW, U)["ok"]                 # fresh + full coverage
    assert not EC.assess(EV, "2026-06-10T00:00:00+00:00", NOW, U)["ok"]             # age > 3 days
    assert not EC.assess(EV, "2026-06-19T00:00:00+00:00", NOW, U)["ok"]             # fetched in future
    assert not EC.assess({"AAA": ["2026-06-20"]}, "2026-06-17T00:00:00+00:00", NOW, U + ["DDD"])["ok"]  # coverage 25%
    assert not EC.assess({}, None, NOW, U)["ok"]                                    # source unavailable


def test_within_7d_signal():
    r = EC.CalendarResult(EV, "2026-06-17T00:00:00+00:00", "massive", 1.0, 1.0, True, "ok")
    sig, dq = EC.earnings_within_signal(r, "2026-06-18", U, horizon_days=7)
    assert sig == {"AAA": 1.0, "BBB": 0.0, "CCC": 1.0}                              # AAA/CCC report within 7d
    assert dq["feed_available"] and dq["missing_symbols_ratio"] == 0.0


def test_source_agnostic_fail_closed_end_to_end():
    # a low-coverage / not-ok calendar must make the FROZEN sleeve go flat, regardless of source
    bad = EC.CalendarResult({"AAA": ["2026-06-20"]}, "2026-06-17T00:00:00+00:00", "massive",
                            coverage=0.25, age_days=1.0, ok=False, reason="coverage")
    sig, dq = EC.earnings_within_signal(bad, "2026-06-18", U + ["DDD"])
    w, diag = PS.target_weights(sig, dq)
    assert diag["blocked"] and w == {}
    # a clean calendar produces a valid market-neutral book
    good = EC.CalendarResult(EV, "2026-06-17T00:00:00+00:00", "alpha_vantage", 1.0, 0.0, True, "ok")
    sig2, dq2 = EC.earnings_within_signal(good, "2026-06-18", U)
    w2, diag2 = PS.target_weights(sig2, dq2)
    assert not diag2["blocked"] and abs(diag2["gross"] - 1.0) < 1e-9


def test_no_live_trading_imports():
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "place_order", "run_v2", "live_trade", ".orders", "portfolio_optimizer")
    for f in ("earnings_calendar.py", "shadow_run.py"):
        mods = []
        for n in ast.walk(ast.parse((PKG / f).read_text())):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            assert not any(b in m.lower() for b in banned), f"{f} imports {m}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
