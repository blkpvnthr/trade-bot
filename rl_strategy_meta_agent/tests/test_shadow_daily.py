"""Checklist tests for the daily forward-shadow recorder: append-only + idempotent +
one completed session/day + fail-closed + off-Desktop + no broker imports. Network-free."""
import ast
import importlib.util
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "experiments" / "orthogonal_features"
sys.path.insert(0, str(PKG))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


STD = _load("shadow_daily_standalone")


def _row(date):
    return {"date": date, "blocked": False, "reason": "ok", "daily_pnl": 0.001, "turnover": 0.2,
            "rank_spread": 0.001, "daily_ic": 0.03}


# 2/3: append-only JSONL + refuses to duplicate a date already recorded
def test_append_only_and_idempotent_no_duplicate(tmp_path):
    log = tmp_path / "shadow_log.jsonl"
    assert STD.append_session(log, _row("2026-06-17")) is True
    assert STD.append_session(log, _row("2026-06-17")) is False        # duplicate date refused
    assert STD.append_session(log, _row("2026-06-18")) is True
    lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert [r["date"] for r in lines] == ["2026-06-17", "2026-06-18"]   # no dup, order preserved
    assert STD.already_recorded(log, "2026-06-17") and not STD.already_recorded(log, "2026-06-19")


# one-session-only: repeated calls for the same day never grow the log past one row
def test_one_session_only_per_day(tmp_path):
    log = tmp_path / "shadow_log.jsonl"
    for _ in range(5):
        STD.append_session(log, _row("2026-06-17"))
    assert len(log.read_text().splitlines()) == 1


# never rebuilds/overwrites: appending preserves the earlier line verbatim
def test_never_overwrites_existing_sessions(tmp_path):
    log = tmp_path / "shadow_log.jsonl"
    STD.append_session(log, _row("2026-06-17"))
    first = log.read_text().splitlines()[0]
    STD.append_session(log, _row("2026-06-18"))
    assert log.read_text().splitlines()[0] == first                    # byte-identical, untouched


# monotonic session_id + corruption detection (expected vs actual)
def test_session_id_monotonic_and_integrity_clean(tmp_path):
    log = tmp_path / "shadow_log.jsonl"
    for d in ("2026-06-17", "2026-06-18", "2026-06-19"):
        STD.append_session(log, _row(d))
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert [r["session_id"] for r in rows] == [1, 2, 3]                 # monotonic, contiguous
    integ = STD.session_integrity(log)
    assert integ["expected_session_count"] == 3 and integ["actual_session_count"] == 3
    assert integ["integrity_ok"] and integ["integrity_alert"] is None


def test_integrity_detects_a_lost_session(tmp_path):
    log = tmp_path / "shadow_log.jsonl"
    for d in ("2026-06-17", "2026-06-18", "2026-06-19"):
        STD.append_session(log, _row(d))
    lines = log.read_text().splitlines()
    log.write_text(lines[0] + "\n" + lines[2] + "\n")                  # drop the middle session (corruption)
    integ = STD.session_integrity(log)
    assert integ["expected_session_count"] == 3 and integ["actual_session_count"] == 2
    assert not integ["integrity_ok"] and integ["integrity_alert"]      # immediate alert


# 5: fail closed -> outage/flat on stale / missing-coverage / unavailable
def test_fail_closed_stale_coverage_unavailable():
    sig = {"A": 1.0, "B": 0.0, "C": 0.0}
    assert STD.frozen_weights(sig, {"earnings_age_days": 5, "missing_symbols_ratio": 0.0,
                                    "feed_available": True})[1]["blocked"] is True          # stale
    assert STD.frozen_weights(sig, {"earnings_age_days": 1, "missing_symbols_ratio": 0.10,
                                    "feed_available": True})[1]["blocked"] is True          # coverage
    assert STD.frozen_weights(sig, {"earnings_age_days": 1, "missing_symbols_ratio": 0.0,
                                    "feed_available": False})[1]["blocked"] is True         # unavailable
    ok = STD.frozen_weights(sig, {"earnings_age_days": 1, "missing_symbols_ratio": 0.0,
                                  "feed_available": True})
    assert not ok[1]["blocked"] and abs(ok[1]["gross"] - 1.0) < 1e-9                        # clean -> trades


# 1: writes outside ~/Desktop (TCC-exempt); no log-truncation in the forward runners
def test_log_dir_off_desktop_and_no_truncation():
    assert "/Desktop/" not in str(STD.DEFAULT_LOG_DIR)
    sd = _load("shadow_daily")
    assert "/Desktop/" not in str(sd.DEFAULT_LOG_DIR)
    for f in ("shadow_daily_standalone.py", "shadow_daily.py"):
        src = (PKG / f).read_text()
        assert "unlink" not in src and 'open("w")' not in src and '"w")' not in src         # never rebuilds


# 6: no broker / order / live execution imports
def test_no_broker_or_order_imports():
    banned = ("alpaca.trading", "trading_client", "executor", "broker", "submit_order",
              "execution", "place_order", "run_v2", "live_trade", ".orders", "portfolio_optimizer")
    for f in ("shadow_daily_standalone.py", "shadow_daily.py"):
        mods = []
        for n in ast.walk(ast.parse((PKG / f).read_text())):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        for m in mods:
            assert not any(b in m.lower() for b in banned), f"{f} imports {m}"


if __name__ == "__main__":
    import inspect
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            (fn(Path("/tmp")) if "tmp_path" in inspect.signature(fn).parameters else fn())
            print("ok", name)
