"""Canonical system state store (V6-Integrated spec section 3).

ONE source of truth that every component reads from and writes to, instead of each layer keeping a
private, drift-prone copy (paper 1: "All system components reference this centralized state"). The
store also detects the discrete EVENTS that drive the event-driven allocator (paper 1, section 10),
so the allocator runs only when something meaningful changed -- never every bar.

Pure data + change-detection. No broker calls, no network. JSON-persistable for restart recovery
and dashboard consumption.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = PROJECT_ROOT / "logs" / "execution" / "amps_v6_canonical_state.json"

# Reallocation event types (paper 1 section VII + spec section 10).
EV_REGIME_CHANGE = "confirmed_regime_change"
EV_ELIGIBLE_SET_CHANGE = "eligible_set_change"
EV_NEW_ALPHA_CANDIDATE = "new_alpha_confirmed_candidate"
EV_ALPHA_INVALIDATION = "entry_alpha_invalidation"
EV_CONFIDENCE_CHANGE = "material_confidence_change"
EV_EXPOSURE_LIMIT_CHANGE = "exposure_limit_change"
EV_POSITION_FILL = "position_fill"
EV_STOP_OR_EXIT = "stop_or_exit_trigger"
EV_RISK_POLICY = "risk_policy_event"
EV_EOD = "end_of_day_transition"


@dataclass
class CanonicalState:
    # --- market / regime context ---
    market_session: str = "CLOSED"            # PREMARKET|OPENING|REGULAR|CLOSING|CLOSED
    market_regime: str = "INDETERMINATE"      # BULL|BEAR|TRANSITION|CRISIS|INDETERMINATE
    regime_confidence: float = 0.0
    regime_locked: bool = False
    regime_lock_status: str = "UNLOCKED"      # UNLOCKED|FAST_LOCK|CONFIRMED_LOCK
    regime_lock_reason: str = ""
    # --- universe / permissions ---
    tradable_universe: list = field(default_factory=list)
    new_entries_allowed: bool = False
    reduce_only: bool = False
    symbol_eligibility: dict = field(default_factory=dict)   # symbol -> bool/reason
    # --- positions / orders / exposure ---
    positions: dict = field(default_factory=dict)            # symbol -> position snapshot
    open_orders: dict = field(default_factory=dict)          # order_id -> order snapshot
    pending_intents: list = field(default_factory=list)      # list of order-intent dicts
    buying_power: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    effective_leveraged_exposure: float = 0.0
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    # --- decision inputs ---
    agent_recommendations: dict = field(default_factory=dict)
    entry_alpha_evidence: dict = field(default_factory=dict)
    # --- versions / safety / health ---
    risk_policy_version: str = "unsigned"
    strategy_version: str = "amps_v6_integrated"
    kill_switch_state: dict = field(default_factory=dict)
    data_health: str = "UNKNOWN"              # OK|STALE|GAP|DISCONNECTED|UNKNOWN

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or STATE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CanonicalState":
        p = path or STATE_PATH
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        st = cls()
        for k, v in raw.items():
            if hasattr(st, k):
                setattr(st, k, v)
        return st


def diff_events(prev: CanonicalState, curr: CanonicalState,
                confidence_eps: float = 0.15) -> list[str]:
    """Detect the reallocation events between two states (spec section 10). Returns the list of
    event names that fired; empty list => no reallocation needed this tick (the whole point:
    'reacts only when meaningful changes occur')."""
    events: list[str] = []
    # confirmed regime change (only a CONFIRMED_LOCK flip counts, per paper 2 lock semantics)
    if (prev.market_regime != curr.market_regime
            and curr.regime_lock_status == "CONFIRMED_LOCK"):
        events.append(EV_REGIME_CHANGE)
    if set(prev.tradable_universe) != set(curr.tradable_universe):
        events.append(EV_ELIGIBLE_SET_CHANGE)
    # new alpha-confirmed candidate appeared
    prev_conf = {k for k, v in (prev.entry_alpha_evidence or {}).items() if v == "ALPHA_CONFIRMED"}
    curr_conf = {k for k, v in (curr.entry_alpha_evidence or {}).items() if v == "ALPHA_CONFIRMED"}
    if curr_conf - prev_conf:
        events.append(EV_NEW_ALPHA_CANDIDATE)
    if prev_conf - curr_conf:
        events.append(EV_ALPHA_INVALIDATION)
    if abs(curr.regime_confidence - prev.regime_confidence) >= confidence_eps:
        events.append(EV_CONFIDENCE_CHANGE)
    if abs(curr.effective_leveraged_exposure - prev.effective_leveraged_exposure) > 1e-9 and \
            prev.new_entries_allowed != curr.new_entries_allowed:
        events.append(EV_EXPOSURE_LIMIT_CHANGE)
    if set((prev.positions or {}).keys()) != set((curr.positions or {}).keys()):
        events.append(EV_POSITION_FILL)
    if not prev.reduce_only and curr.reduce_only:
        events.append(EV_RISK_POLICY)
    if prev.market_session != "CLOSING" and curr.market_session == "CLOSING":
        events.append(EV_EOD)
    return events
