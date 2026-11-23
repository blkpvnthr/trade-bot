"""Tests for SubscriptionManager.plan_allocated (Stage A5).

Risk-first pinning -> proportional 20/10 baseline/opportunity split of the
remainder -> opportunity exploit(70%)/explore(30)%, with fair baseline rotation,
deterministic rounding, dedup, a hard 30 cap, and rotation cursors that advance
only on committed rotations. Pure/deterministic; no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.baseline import BASELINE_UNIVERSE
from market_scan.subscription_manager import SubscriptionManager

REFS = ["SPY", "QQQ", "DIA", "IWM"]


def _mgr(cap=30):
    return SubscriptionManager(cap, mandatory_groups=3)


def test_mandatory_is_pinned_and_never_dropped():
    m = _mgr()
    plan = m.plan_allocated(
        positions=["HELD"], pending=["PEND"], references=REFS,
        baseline=list(BASELINE_UNIVERSE),
        opportunities=[f"O{i}" for i in range(50)],
        baseline_slots=20, opportunity_slots=10,
    )
    for s in ["HELD", "PEND", "SPY", "QQQ", "DIA", "IWM"]:
        assert s in plan.target
    assert plan.mandatory_dropped == []
    assert len(plan.target) <= 30


def test_hard_cap_never_exceeded():
    m = _mgr()
    plan = m.plan_allocated(
        positions=[f"P{i}" for i in range(5)], pending=[f"Q{i}" for i in range(5)],
        references=REFS, baseline=list(BASELINE_UNIVERSE),
        opportunities=[f"O{i}" for i in range(100)],
        baseline_slots=20, opportunity_slots=10,
    )
    assert len(plan.target) == 30
    assert len(set(plan.target)) == 30          # deduped


def test_twenty_ten_split_of_free_capacity():
    m = _mgr()
    base = [f"B{i}" for i in range(30)]
    opp = [f"O{i}" for i in range(30)]
    plan = m.plan_allocated(
        positions=[], pending=[], references=[],   # free == 30
        baseline=base, opportunities=opp,
        baseline_slots=20, opportunity_slots=10,
    )
    n_base = sum(1 for s in plan.target if s.startswith("B"))
    n_opp = sum(1 for s in plan.target if s.startswith("O"))
    assert n_base == 20 and n_opp == 10
    assert len(plan.target) == 30


def test_unused_baseline_capacity_redistributes_to_opportunities():
    m = _mgr()
    plan = m.plan_allocated(
        positions=[], pending=[], references=[],
        baseline=["B0", "B1", "B2"],               # only 3 baseline available (< 20)
        opportunities=[f"O{i}" for i in range(40)],
        baseline_slots=20, opportunity_slots=10,
    )
    n_base = sum(1 for s in plan.target if s.startswith("B"))
    n_opp = sum(1 for s in plan.target if s.startswith("O"))
    assert n_base == 3                             # all available baseline kept
    assert n_base + n_opp == 30                    # leftover went to opportunities


def test_opportunity_exploit_slots_go_to_highest_score():
    m = _mgr()
    base = [f"B{i}" for i in range(20)]            # baseline consumes its 20 share
    opp = [f"O{i}" for i in range(20)]
    scores = {f"O{i}": (20 - i) / 20.0 for i in range(20)}   # O0 best ... O19 worst
    plan = m.plan_allocated(
        positions=[], pending=[], references=[],   # free == 30 -> 20 base / 10 opp
        baseline=base, opportunities=opp, opp_score=scores,
        baseline_slots=20, opportunity_slots=10,
        exploration_fraction=0.3,                  # 7 exploit + 3 explore
    )
    chosen = [s for s in plan.target if s.startswith("O")]
    assert len(chosen) == 10                       # 7 exploit + 3 explore
    # The 7 exploit slots must be the 7 highest-scoring opportunities.
    assert {"O0", "O1", "O2", "O3", "O4", "O5", "O6"}.issubset(set(chosen))


def test_exploration_selects_beyond_the_top_scores():
    m = _mgr()
    base = [f"B{i}" for i in range(20)]
    opp = [f"O{i}" for i in range(20)]
    scores = {f"O{i}": (20 - i) / 20.0 for i in range(20)}
    plan = m.plan_allocated(
        positions=[], pending=[], references=[],
        baseline=base, opportunities=opp, opp_score=scores,
        baseline_slots=20, opportunity_slots=10, exploration_fraction=0.3,
    )
    chosen = set(s for s in plan.target if s.startswith("O"))
    # 3 explore slots come from BELOW the top-7, proving it's not pure greedy.
    assert len(chosen - {"O0", "O1", "O2", "O3", "O4", "O5", "O6"}) == 3


def test_baseline_rotates_fairly_across_commits():
    m = _mgr()
    base = [f"B{i}" for i in range(40)]            # 40 baseline, 20 slots -> 2 rounds to cover
    p1 = m.plan_allocated(positions=[], pending=[], references=[], baseline=base,
                          opportunities=[], baseline_slots=20, opportunity_slots=10)
    m.commit(p1)
    r1 = set(s for s in p1.target if s.startswith("B"))
    p2 = m.plan_allocated(positions=[], pending=[], references=[], baseline=base,
                          opportunities=[], baseline_slots=20, opportunity_slots=10)
    r2 = set(s for s in p2.target if s.startswith("B"))
    assert r1 != r2                                # cursor advanced -> different slice
    assert r1 | r2 == set(base)                    # two rounds cover ALL baseline


def test_cursor_advances_only_after_commit_rotation():
    # advance=False defers the cursor; commit_rotation() (called only after a
    # successful subscribe reconciliation) finalizes it.
    m = _mgr()
    base = [f"B{i}" for i in range(40)]
    p1 = m.plan_allocated(positions=[], pending=[], references=[], baseline=base,
                          opportunities=[], baseline_slots=20, opportunity_slots=10,
                          advance=False)
    # Without commit_rotation, a re-plan yields the SAME slice.
    p1b = m.plan_allocated(positions=[], pending=[], references=[], baseline=base,
                           opportunities=[], baseline_slots=20, opportunity_slots=10,
                           advance=False)
    assert [s for s in p1.target if s.startswith("B")] == \
           [s for s in p1b.target if s.startswith("B")]
    m.commit_rotation()        # reconciliation succeeded -> finalize cursor
    p2 = m.plan_allocated(positions=[], pending=[], references=[], baseline=base,
                          opportunities=[], baseline_slots=20, opportunity_slots=10,
                          advance=False)
    assert [s for s in p2.target if s.startswith("B")] != \
           [s for s in p1.target if s.startswith("B")]


def test_advance_false_does_not_rotate_cursor():
    m = _mgr()
    base = [f"B{i}" for i in range(40)]
    p1 = m.plan_allocated(positions=[], pending=[], references=[], baseline=base,
                          opportunities=[], baseline_slots=20, opportunity_slots=10,
                          advance=False)
    p2 = m.plan_allocated(positions=[], pending=[], references=[], baseline=base,
                          opportunities=[], baseline_slots=20, opportunity_slots=10,
                          advance=False)
    assert [s for s in p1.target if s.startswith("B")] == \
           [s for s in p2.target if s.startswith("B")]


def test_to_add_to_remove_against_current():
    m = _mgr()
    p1 = m.plan_allocated(positions=[], pending=[], references=REFS,
                          baseline=["B0", "B1"], opportunities=["O0"],
                          baseline_slots=20, opportunity_slots=10)
    m.commit(p1)
    assert set(p1.to_add) == set(p1.target)        # nothing was subscribed before
    p2 = m.plan_allocated(positions=[], pending=[], references=REFS,
                          baseline=["B0", "B1"], opportunities=["O0"],
                          baseline_slots=20, opportunity_slots=10)
    assert p2.to_add == [] and p2.to_remove == []  # stable target -> no churn


def test_existing_plan_interface_unchanged():
    # Back-compat: the original plan() still works.
    m = _mgr()
    plan = m.plan([["HELD"], ["PEND"], REFS, ["NVDA", "AMD"]])
    assert "HELD" in plan.target and "NVDA" in plan.target


# ------------------------------------------------------------------ config knobs
from market_scan.config import MarketScanConfig  # noqa: E402


def test_rotation_config_defaults():
    cfg = MarketScanConfig()
    assert cfg.rotation_interval_seconds == 60.0
    assert cfg.baseline_active_slots == 20
    assert cfg.opportunity_active_slots == 10
    assert cfg.opportunity_exploration_fraction == 0.3


def test_rotation_config_from_env(monkeypatch):
    monkeypatch.setenv("MARKET_SCAN_ROTATION_SECONDS", "90")
    monkeypatch.setenv("MARKET_SCAN_BASELINE_SLOTS", "18")
    monkeypatch.setenv("MARKET_SCAN_OPPORTUNITY_SLOTS", "12")
    monkeypatch.setenv("MARKET_SCAN_EXPLORATION_FRACTION", "0.25")
    cfg = MarketScanConfig.from_env()
    assert cfg.rotation_interval_seconds == 90.0
    assert cfg.baseline_active_slots == 18
    assert cfg.opportunity_active_slots == 12
    assert cfg.opportunity_exploration_fraction == 0.25


def test_exploration_fraction_is_clamped_to_unit_interval():
    assert MarketScanConfig(opportunity_exploration_fraction=1.7).opportunity_exploration_fraction == 1.0
    assert MarketScanConfig(opportunity_exploration_fraction=-0.5).opportunity_exploration_fraction == 0.0
