"""Correctness tests for the giveback-driven management policies."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import management_policies as M

# rise to +5% (peak at k=4) then give back to -1%; p crosses thresholds at k=4
R = np.array([0.0, 0.01, 0.03, 0.04, 0.05, 0.02, -0.01])
P = np.array([0.0, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90])


def test_first_trigger_and_no_trigger():
    assert M._first_trigger(P, 0.65) == 4
    assert M._first_trigger(P, 0.99) is None


def test_full_exit_locks_the_trigger_bar():
    assert M.full_exit(R, P, 0.65) == 0.05            # exits at k=4 = the peak
    assert M.full_exit(R, np.zeros_like(R), 0.65) == R[-1]   # never fires -> baseline


def test_partial_exit_blends_trigger_and_final():
    assert abs(M.partial_exit(R, P, 0.60, 0.5) - (0.5 * 0.05 + 0.5 * -0.01)) < 1e-9


def test_tighten_stop_trails_from_trigger():
    # from k=4 (peak 0.05), 0.5% trail exits at k=5 (0.05 - 0.02 = 0.03 >= 0.005)
    assert M.tighten_stop(R, P, 0.55, 0.005) == 0.02
    assert M.tighten_stop(R, P, 0.55, 0.005) > M.baseline(R)


def test_de_risk_trims_and_trails():
    # trim 0.5 at 0.05 + 0.5 * trailing-from-k4 (1% trail -> exits at k=5 = 0.02)
    assert abs(M.de_risk(R, P, 0.60, 0.5, 0.01) - (0.5 * 0.05 + 0.5 * 0.02)) < 1e-9


def test_no_trigger_every_policy_equals_baseline():
    flat = np.array([0.0, 0.01, 0.02, 0.03])
    zero = np.zeros(4)
    for fn in M.POLICIES.values():
        assert abs(fn(flat, zero) - flat[-1]) < 1e-9


def test_severe_giveback_detection():
    assert M.severe_giveback_happened(R)              # 0.05 -> -0.01 is severe
    assert not M.severe_giveback_happened(np.array([0.0, 0.01, 0.02, 0.025]))


def test_trail_from_rides_when_never_hit():
    rising = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    assert M._trail_from(rising, 1, 0.01) == 0.04     # monotone up -> final


def test_action_labels_when_exiting_is_right():
    # peak +5% at k=4, final -1%: exiting/tightening now beats holding
    al = M.action_labels(R, 4)
    assert al["exit_now_better"] == 1
    assert al["tighten_stop_better"] == 1
    assert al["take_partial_better"] == 1


def test_action_labels_when_holding_is_right():
    # a recoverable dip: +3% then dips to +1% then rips to +8% at the exit
    r = np.array([0.0, 0.03, 0.01, 0.05, 0.08])
    al = M.action_labels(r, 2)                         # at the +1% dip
    assert al["exit_now_better"] == 0                  # holding to +8% was better


def test_giveback_recovery_flags_recoverable_pullback():
    # severe giveback exists (peak ahead high, final lower) AND a new high forms
    r = np.array([0.0, 0.02, 0.04, 0.10, 0.03])        # peak 0.10, final 0.03
    g = M.giveback_recovery(r, 1)                       # decision-time peak = 0.02
    assert g["severe_giveback"] == 1
    assert g["new_high_after"] == 1                    # 0.10 > 0.02 -> a new high formed
    assert g["exit_now_better"] == 0                   # r[1]=0.02 < final 0.03


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
