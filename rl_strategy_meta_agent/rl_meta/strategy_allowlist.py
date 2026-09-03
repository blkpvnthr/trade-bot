"""Paper-only defensive strategy-family allowlist.

Validated by walk-forward (scripts/strategy_family_walkforward.py): excluding
martingale + pin_bar_magic_v1 cut out-of-sample drawdown ~74% across 4/4 folds
(0.664 -> 0.169) while raising PF and total return. This is a DEFENSIVE exclusion
of structurally-bad strategies -- NOT proof of alpha.

Gated behind the env flag META_FAMILY_ALLOWLIST (default OFF): nothing is blocked
unless it is explicitly enabled, so existing behavior is unchanged by default.
Research / paper only -- this module never touches a broker, order, or risk path
and never changes position size.

  sniper_strategy is BLOCKED: rl_meta.repaint_audit returned INVALID (entries built
  on a repainting pre-v3 security() HTF call; PF ~19.9). See scripts/repaint_audit.py.
  The only validated paper families are linear_regression and mean_reversion.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("strategy_allowlist")

ALLOWED_FAMILIES = ("linear_regression", "mean_reversion")
BLOCKED_FAMILIES = ("martingale", "pin_bar_magic_v1", "sniper_strategy")
VERIFY_FAMILIES = {}                      # sniper cleared out: audit returned INVALID
BLOCK_REASON = {
    "martingale": "structural blow-up; the walk-forward drawdown driver",
    "pin_bar_magic_v1": "negative out-of-sample expectancy; suspected repaint",
    "sniper_strategy": "repaint audit INVALID: entries use a repainting pre-v3 "
                       "security() HTF call; PF ~19.9 implausible",
}
ENV_FLAG = "META_FAMILY_ALLOWLIST"


def allowlist_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


def family_status(family: str) -> str:
    """allowed | allowed_verify_repaint | blocked | blocked_not_in_allowlist."""
    if family in BLOCKED_FAMILIES:
        return "blocked"
    if family in VERIFY_FAMILIES:
        return "allowed_verify_repaint"
    if family in ALLOWED_FAMILIES:
        return "allowed"
    return "blocked_not_in_allowlist"          # strict allowlist: unknown -> blocked


def is_allowed(family: str, enabled: bool | None = None) -> bool:
    if enabled is None:
        enabled = allowlist_enabled()
    if not enabled:
        return True                            # disabled -> pass-through
    return family_status(family).startswith("allowed")


def _family_of(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("strategy") or item.get("strategy_id") or item.get("family")
    return getattr(item, "strategy", None) or getattr(item, "strategy_id", None)


def filter_candidates(items, enabled: bool | None = None, family_of=_family_of,
                      on_block=None):
    """Split candidates into (allowed, blocked). Each blocked record carries the
    field `blocked_by_strategy_family=True`. Disabled -> nothing blocked."""
    if enabled is None:
        enabled = allowlist_enabled()
    allowed, blocked = [], []
    for it in items:
        fam = _family_of(it) if family_of is _family_of else family_of(it)
        if is_allowed(fam, enabled):
            allowed.append(it)
        else:
            rec = {"candidate": it if isinstance(it, (str, int)) else repr(it),
                   "strategy_family": fam, "blocked_by_strategy_family": True,
                   "reason": BLOCK_REASON.get(fam, family_status(fam))}
            blocked.append(rec)
            log.warning("blocked_by_strategy_family family=%s reason=%s", fam, rec["reason"])
            if on_block:
                on_block(rec)
    return allowed, blocked


def _demo():
    cands = ["linear_regression", "mean_reversion", "sniper_strategy",
             "martingale", "pin_bar_magic_v1", "unknown_xyz"]
    # disabled -> pass-through
    a, b = filter_candidates(cands, enabled=False)
    assert len(a) == 6 and not b, "disabled allowlist must not block anything"
    # enabled -> martingale + pin_bar + sniper (invalid) + unknowns blocked
    a, b = filter_candidates(cands, enabled=True)
    blocked = {r["strategy_family"] for r in b}
    assert {"martingale", "pin_bar_magic_v1", "sniper_strategy"} <= blocked
    assert "unknown_xyz" in blocked                    # strict allowlist
    assert set(a) == {"linear_regression", "mean_reversion"}
    assert all(r["blocked_by_strategy_family"] for r in b)
    assert family_status("sniper_strategy") == "blocked"
    print("strategy_allowlist._demo OK  blocked:", sorted(blocked))


if __name__ == "__main__":
    _demo()
