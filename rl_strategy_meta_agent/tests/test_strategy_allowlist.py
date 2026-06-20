"""Tests: the defensive allowlist blocks martingale and pin_bar when enabled."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import strategy_allowlist as A


def test_bad_families_and_sniper_are_blocked_when_enabled():
    a, b = A.filter_candidates(["martingale", "pin_bar_magic_v1", "sniper_strategy"], enabled=True)
    assert a == []
    fams = {r["strategy_family"] for r in b}
    assert fams == {"martingale", "pin_bar_magic_v1", "sniper_strategy"}
    assert all(r["blocked_by_strategy_family"] for r in b)


def test_sniper_is_blocked_after_invalid_audit():
    a, b = A.filter_candidates(["sniper_strategy"], enabled=True)
    assert a == []
    assert b[0]["strategy_family"] == "sniper_strategy"
    assert b[0]["blocked_by_strategy_family"] and "repaint" in b[0]["reason"].lower()


def test_only_validated_families_pass_when_enabled():
    a, b = A.filter_candidates(["linear_regression", "mean_reversion"], enabled=True)
    assert set(a) == {"linear_regression", "mean_reversion"} and b == []


def test_disabled_is_pass_through():
    a, b = A.filter_candidates(["martingale", "pin_bar_magic_v1", "sniper_strategy"], enabled=False)
    assert len(a) == 3 and b == []


def test_unknown_family_blocked_strict_allowlist():
    assert A.is_allowed("some_new_strategy", enabled=True) is False
    assert A.is_allowed("some_new_strategy", enabled=False) is True       # disabled -> allow


def test_family_status_after_audit():
    assert A.family_status("sniper_strategy") == "blocked"      # audit returned INVALID
    assert A.family_status("martingale") == "blocked"
    assert A.family_status("linear_regression") == "allowed"


def test_filter_works_on_dicts_and_logs_field():
    rows = [{"strategy_id": "martingale", "id": 1}, {"strategy_id": "linear_regression", "id": 2}]
    a, b = A.filter_candidates(rows, enabled=True)
    assert len(a) == 1 and a[0]["id"] == 2
    assert b[0]["strategy_family"] == "martingale" and b[0]["blocked_by_strategy_family"]


def test_env_flag_default_off():
    import os
    os.environ.pop(A.ENV_FLAG, None)
    assert A.allowlist_enabled() is False              # default disabled
    os.environ[A.ENV_FLAG] = "1"
    assert A.allowlist_enabled() is True
    os.environ.pop(A.ENV_FLAG, None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
