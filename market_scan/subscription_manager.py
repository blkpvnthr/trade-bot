"""Dynamic IEX subscription allocation under a hard budget.

Alpaca permits a bounded number of symbols on one IEX websocket. This manager
allocates that budget across priority groups — conventionally
``[positions, pending, references, candidates]`` — keeping the mandatory
higher-priority groups (positions, pending) subscribed at all times and filling
the remaining budget from lower-priority groups with a rotating cursor so
coverage is fair and nothing is silently misrepresented as covered. Candidates
that do not fit this cycle are reported in ``dropped`` (no silent truncation).

The manager is pure/deterministic and holds only the currently-committed set and
a rotation cursor, so it is trivially unit-testable with no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubscriptionPlan:
    target: list[str] = field(default_factory=list)   # desired subscribed set (<= cap, + mandatory)
    to_add: list[str] = field(default_factory=list)    # target - current
    to_remove: list[str] = field(default_factory=list)  # current - target
    dropped: list[str] = field(default_factory=list)    # lower-priority symbols that didn't fit
    active: int = 0
    available: int = 0


class SubscriptionManager:
    def __init__(self, max_subscriptions: int, mandatory_groups: int = 2) -> None:
        self.max_subscriptions = max(1, int(max_subscriptions))
        self.mandatory_groups = max(0, int(mandatory_groups))
        self._current: set[str] = set()
        self._rotation = 0

    def plan(self, priority_groups) -> SubscriptionPlan:
        cap = self.max_subscriptions
        selected: list[str] = []
        seen: set[str] = set()
        overflow: list[str] = []

        for gi, group in enumerate(priority_groups or []):
            mandatory = gi < self.mandatory_groups
            for raw in group or []:
                s = str(raw).upper().strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                if mandatory:
                    # Mandatory symbols are ALWAYS kept, even past the soft cap.
                    selected.append(s)
                else:
                    overflow.append(s)

        free = max(0, cap - len(selected))
        dropped: list[str] = []
        if overflow:
            if free > 0:
                n = len(overflow)
                off = self._rotation % n
                rotated = overflow[off:] + overflow[:off]
                chosen = rotated[:free]
                dropped = rotated[free:]
                selected.extend(chosen)
                self._rotation = (self._rotation + free) % n
            else:
                dropped = list(overflow)

        target = list(dict.fromkeys(selected))
        tset = set(target)
        plan = SubscriptionPlan(
            target=target,
            to_add=[s for s in target if s not in self._current],
            to_remove=[s for s in self._current if s not in tset],
            dropped=dropped,
            active=len(target),
            available=max(0, cap - len(target)),
        )
        return plan

    def commit(self, plan: SubscriptionPlan) -> None:
        """Record ``plan.target`` as the now-subscribed set (call after the
        websocket subscribe/unsubscribe actually succeeds)."""
        self._current = set(plan.target)

    @property
    def current(self) -> set[str]:
        return set(self._current)
