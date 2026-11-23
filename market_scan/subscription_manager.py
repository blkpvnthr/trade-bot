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
    target: list[str] = field(default_factory=list)   # desired subscribed set, STRICTLY <= cap
    to_add: list[str] = field(default_factory=list)    # target - current
    to_remove: list[str] = field(default_factory=list)  # current - target
    dropped: list[str] = field(default_factory=list)    # symbols that did not fit the cap
    mandatory_dropped: list[str] = field(default_factory=list)  # mandatory symbols dropped (loud warn)
    overflow: bool = False                              # True if anything was dropped for the cap
    active: int = 0
    available: int = 0


class SubscriptionManager:
    def __init__(self, max_subscriptions: int, mandatory_groups: int = 2) -> None:
        self.max_subscriptions = max(1, int(max_subscriptions))
        self.mandatory_groups = max(0, int(mandatory_groups))
        self._current: set[str] = set()
        self._rotation = 0            # legacy plan() candidate cursor
        self._base_rotation = 0       # plan_allocated baseline rotation cursor
        self._opp_rotation = 0        # plan_allocated opportunity-exploration cursor
        self._pending_rotation: tuple[int, int] | None = None  # deferred cursor advance

    def plan(self, priority_groups) -> SubscriptionPlan:
        cap = self.max_subscriptions
        selected: list[str] = []
        seen: set[str] = set()
        candidates: list[str] = []
        mandatory_dropped: list[str] = []

        # Mandatory groups (positions > pending > references) are filled FIRST in
        # strict priority order, but NEVER past the cap -- the configured cap is
        # a hard provider limit and must not be silently exceeded. Anything that
        # does not fit is reported (mandatory_dropped) for a loud warning. (Open
        # position EXIT risk is handled by the executor's REST reconciliation, so
        # a position not on the stream only loses an order-flow exit *signal*,
        # never a risk control.)
        for gi, group in enumerate(priority_groups or []):
            mandatory = gi < self.mandatory_groups
            for raw in group or []:
                s = str(raw).upper().strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                if mandatory:
                    if len(selected) < cap:
                        selected.append(s)
                    else:
                        mandatory_dropped.append(s)
                else:
                    candidates.append(s)

        free = max(0, cap - len(selected))
        dropped: list[str] = list(mandatory_dropped)
        if candidates:
            if free > 0:
                n = len(candidates)
                off = self._rotation % n
                rotated = candidates[off:] + candidates[:off]
                selected.extend(rotated[:free])
                dropped.extend(rotated[free:])
                self._rotation = (self._rotation + free) % n
            else:
                dropped.extend(candidates)

        target = list(dict.fromkeys(selected))[:cap]
        tset = set(target)
        return SubscriptionPlan(
            target=target,
            to_add=[s for s in target if s not in self._current],
            to_remove=[s for s in self._current if s not in tset],
            dropped=dropped,
            mandatory_dropped=mandatory_dropped,
            overflow=bool(dropped),
            active=len(target),
            available=max(0, cap - len(target)),
        )

    # ------------------------------------------------------------ allocation
    @staticmethod
    def _dedup_upper(items) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in items or []:
            s = str(raw).upper().strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @staticmethod
    def _rotate_pick(items: list[str], cursor: int, n: int) -> tuple[list[str], int]:
        """Take ``n`` items starting at ``cursor`` (round-robin). Returns the pick
        and the next cursor. Fair across calls when the cursor is advanced."""
        if not items or n <= 0:
            return [], cursor
        m = len(items)
        off = cursor % m
        rotated = items[off:] + items[:off]
        pick = rotated[: min(n, m)]
        return pick, (cursor + len(pick)) % m

    @staticmethod
    def _split_free(free: int, baseline_slots: int, opportunity_slots: int,
                    n_base: int, n_opp: int) -> tuple[int, int]:
        """Split ``free`` capacity ~baseline_slots:opportunity_slots, capped by
        availability, with unused capacity redistributed to the other bucket.
        Deterministic round-half-up on the baseline share."""
        if free <= 0:
            return 0, 0
        total = max(1, int(baseline_slots) + int(opportunity_slots))
        b = int(free * int(baseline_slots) / total + 0.5)   # round-half-up
        b = min(b, free)
        o = free - b
        # Cap by what's actually available.
        b = min(b, max(0, n_base))
        o = min(o, max(0, n_opp))
        # Redistribute leftover capacity: opportunities first, then baseline.
        left = free - (b + o)
        if left > 0:
            add_o = min(left, max(0, n_opp - o))
            o += add_o
            left -= add_o
        if left > 0:
            add_b = min(left, max(0, n_base - b))
            b += add_b
            left -= add_b
        return b, o

    def _pick_opportunities(self, avail_opp: list[str], o_target: int,
                            exploration_fraction: float, opp_score,
                            advance: bool) -> tuple[list[str], int]:
        """Opportunity selection: exploit (top by score) + explore (rotating over
        the remainder). Returns (pick, next_opp_cursor)."""
        if o_target <= 0 or not avail_opp:
            return [], self._opp_rotation
        frac = min(1.0, max(0.0, float(exploration_fraction)))
        explore_n = int(o_target * frac + 0.5)          # round-half-up on explore
        explore_n = min(explore_n, o_target)
        exploit_n = o_target - explore_n

        def score_of(sym: str) -> float:
            if opp_score is None:
                return 0.0
            try:
                return float(opp_score(sym) if callable(opp_score) else opp_score.get(sym, 0.0))
            except Exception:  # noqa: BLE001
                return 0.0

        # Exploit: highest score first; deterministic tie-break on symbol.
        by_score = sorted(avail_opp, key=lambda s: (-score_of(s), s))
        exploit_pick = by_score[:exploit_n]
        chosen = set(exploit_pick)
        # Explore: rotate fairly over everything not exploited.
        remainder = [s for s in avail_opp if s not in chosen]
        explore_pick, next_cursor = self._rotate_pick(remainder, self._opp_rotation, explore_n)
        chosen.update(explore_pick)
        pick = exploit_pick + explore_pick
        # Redistribute any shortfall (pools too small) from the leftover by score.
        if len(pick) < o_target:
            for s in by_score:
                if s not in chosen:
                    pick.append(s)
                    chosen.add(s)
                    if len(pick) >= o_target:
                        break
        return pick[:o_target], next_cursor

    def plan_allocated(
        self,
        *,
        positions=(),
        pending=(),
        references=(),
        baseline=(),
        opportunities=(),
        baseline_slots: int = 20,
        opportunity_slots: int = 10,
        exploration_fraction: float = 0.3,
        opp_score=None,
        advance: bool = True,
    ) -> SubscriptionPlan:
        """Allocate the active IEX window risk-first, then ~20/10 baseline/
        opportunity of the remainder.

        Fill order (hard-capped at ``max_subscriptions``):
          1. open positions, 2. pending orders, 3. references  -- mandatory, never
             dropped (overflow -> ``mandatory_dropped`` + ``overflow``);
          4. rotating baseline subset (fair round-robin, never permanently removed
             from the master -- only rotated out of the active window);
          5. ranked opportunities: ``(1-exploration_fraction)`` exploited by
             ``opp_score`` + ``exploration_fraction`` explored by rotation.
        Unused baseline capacity redistributes to opportunities (and vice versa).
        Rotation cursors advance only when ``advance`` is True (so ranking-only
        reads never perturb rotation).
        """
        cap = self.max_subscriptions
        selected: list[str] = []
        seen: set[str] = set()
        mandatory_dropped: list[str] = []

        for group in (positions, pending, references):
            for s in self._dedup_upper(group):
                if s in seen:
                    continue
                seen.add(s)
                if len(selected) < cap:
                    selected.append(s)
                else:
                    mandatory_dropped.append(s)

        free = max(0, cap - len(selected))
        avail_base = [s for s in self._dedup_upper(baseline) if s not in seen]
        base_set = set(avail_base)
        avail_opp = [s for s in self._dedup_upper(opportunities)
                     if s not in seen and s not in base_set]

        b_target, o_target = self._split_free(
            free, baseline_slots, opportunity_slots, len(avail_base), len(avail_opp))

        base_pick, next_base = self._rotate_pick(avail_base, self._base_rotation, b_target)
        opp_pick, next_opp = self._pick_opportunities(
            avail_opp, o_target, exploration_fraction, opp_score, advance)

        for s in base_pick + opp_pick:
            if s not in seen and len(selected) < cap:
                seen.add(s)
                selected.append(s)

        target = selected[:cap]
        tset = set(target)
        dropped = list(mandatory_dropped) + [
            s for s in (avail_base + avail_opp) if s not in tset]

        if advance:
            self._base_rotation = next_base
            self._opp_rotation = next_opp
            self._pending_rotation = None
        else:
            # Defer the cursor advance until commit_rotation() confirms the
            # subscribe/unsubscribe reconciliation actually succeeded.
            self._pending_rotation = (next_base, next_opp)

        return SubscriptionPlan(
            target=target,
            to_add=[s for s in target if s not in self._current],
            to_remove=[s for s in self._current if s not in tset],
            dropped=dropped,
            mandatory_dropped=mandatory_dropped,
            overflow=bool(dropped),
            active=len(target),
            available=max(0, cap - len(target)),
        )

    def commit(self, plan: SubscriptionPlan) -> None:
        """Record ``plan.target`` as the now-subscribed set (call after the
        websocket subscribe/unsubscribe actually succeeds)."""
        self._current = set(plan.target)

    def commit_rotation(self) -> None:
        """Finalize a deferred rotation-cursor advance from the last
        ``plan_allocated(advance=False)``. Call only after the subscription
        reconciliation succeeded, so a failed apply does not skip a rotation
        slice."""
        if self._pending_rotation is not None:
            self._base_rotation, self._opp_rotation = self._pending_rotation
            self._pending_rotation = None

    @property
    def current(self) -> set[str]:
        return set(self._current)
