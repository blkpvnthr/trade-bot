"""Platform readiness, horizon eligibility, operating modes, and kill switches (V6 spec 3, 42).

Three INDEPENDENT states gate live execution; ALL must be true to send a live order:
    PLATFORM_LIVE_READY       -- the harness (reconciliation, kill switches, tests) is sound.
    STRATEGY_LIVE_ELIGIBLE    -- at least one trading horizon has VALIDATED entry-alpha evidence.
    LIVE_ACTIVATION_APPROVED  -- an explicit human operator authorized live money (default false).

Horizon eligibility is tracked PER HORIZON so a validated swing strategy never auto-authorizes an
unvalidated scalp, and an unvalidated scalp never blocks an independently validated swing. Live
execution also requires mode == LIVE and new_entries_enabled and no active kill switch.

Everything here is pure + persist-to-JSON; no broker calls. Kill switches persist across restarts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "logs" / "execution"
PLATFORM_STATE_PATH = STATE_DIR / "amps_v6_platform_state.json"


class Mode(str, Enum):
    BACKTEST = "BACKTEST"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE_PILOT = "LIVE_PILOT"
    LIVE = "LIVE"


# Trading horizons (spec section 8). Max holding period in trading days (scalp/intraday < 1 day).
HORIZONS = ("MICRO_SCALP", "INTRADAY_MOMENTUM", "EXTENDED_INTRADAY", "SHORT_SWING", "EXTENDED_SWING")
HORIZON_ELIGIBILITY_FLAGS = tuple(f"{h}_ELIGIBLE" for h in HORIZONS)

# Kill switches (spec section 42). True == TRIPPED (blocks the scope it names).
KILL_SWITCHES = (
    "global", "strategy", "new_entries", "overnight_entries", "position_promotion",
    "leveraged_etf", "data_provider", "agent_system",
)


@dataclass
class PlatformState:
    mode: Mode = Mode.OBSERVE_ONLY
    live_activation_approved: bool = False
    new_entries_enabled: bool = False
    platform_live_ready: bool = False           # set True only after the readiness checklist passes
    # per-horizon eligibility (set True only by a VALIDATED evidence promotion)
    horizon_eligible: dict = field(default_factory=lambda: {h: False for h in HORIZONS})
    overnight_trading_eligible: bool = False
    # kill switches (default: global tripped == safe; nothing trades until explicitly cleared)
    kill: dict = field(default_factory=lambda: {k: (k == "global") for k in KILL_SWITCHES})
    # per-symbol and per-setup-family suppression
    symbol_suppressed: list = field(default_factory=list)
    setup_family_suppressed: list = field(default_factory=list)
    risk_policy_version: str = "unsigned"

    # ---- derived gates ----
    @property
    def strategy_live_eligible(self) -> bool:
        """True iff ANY horizon has been validated-and-enabled."""
        return any(self.horizon_eligible.get(h, False) for h in HORIZONS)

    def kill_active(self) -> bool:
        return any(self.kill.get(k, False) for k in KILL_SWITCHES if k in ("global",)) or self.kill.get("strategy", False)

    def can_send_live_order(self, horizon: str) -> tuple[bool, str]:
        """The full live-authorization conjunction for one horizon. Fail-closed with a reason."""
        if not self.platform_live_ready:
            return False, "PLATFORM_LIVE_READY=false"
        if not self.horizon_eligible.get(horizon, False):
            return False, f"{horizon}_ELIGIBLE=false (no validated evidence)"
        if not self.strategy_live_eligible:
            return False, "STRATEGY_LIVE_ELIGIBLE=false"
        if not self.live_activation_approved:
            return False, "LIVE_ACTIVATION_APPROVED=false (operator)"
        if self.mode != Mode.LIVE:
            return False, f"mode={self.mode.value} (not LIVE)"
        if not self.new_entries_enabled:
            return False, "new_entries_enabled=false"
        if self.kill.get("global", False):
            return False, "global kill switch TRIPPED"
        if self.kill.get("strategy", False):
            return False, "strategy kill switch TRIPPED"
        if self.kill.get("new_entries", False):
            return False, "new_entries kill switch TRIPPED"
        return True, "authorized"

    # ---- kill switch controls ----
    def trip(self, switch: str) -> None:
        if switch not in KILL_SWITCHES:
            raise ValueError(f"unknown kill switch {switch!r}")
        self.kill[switch] = True

    def clear(self, switch: str) -> None:
        if switch not in KILL_SWITCHES:
            raise ValueError(f"unknown kill switch {switch!r}")
        self.kill[switch] = False

    # ---- persistence ----
    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["strategy_live_eligible"] = self.strategy_live_eligible
        return d

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or PLATFORM_STATE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PlatformState":
        p = path or PLATFORM_STATE_PATH
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        st = cls()
        st.mode = Mode(raw.get("mode", Mode.OBSERVE_ONLY.value))
        st.live_activation_approved = bool(raw.get("live_activation_approved", False))
        st.new_entries_enabled = bool(raw.get("new_entries_enabled", False))
        st.platform_live_ready = bool(raw.get("platform_live_ready", False))
        st.horizon_eligible = {h: bool(raw.get("horizon_eligible", {}).get(h, False)) for h in HORIZONS}
        st.overnight_trading_eligible = bool(raw.get("overnight_trading_eligible", False))
        st.kill = {k: bool(raw.get("kill", {}).get(k, (k == "global"))) for k in KILL_SWITCHES}
        st.symbol_suppressed = list(raw.get("symbol_suppressed", []))
        st.setup_family_suppressed = list(raw.get("setup_family_suppressed", []))
        st.risk_policy_version = str(raw.get("risk_policy_version", "unsigned"))
        return st


def default_state() -> PlatformState:
    """Spec-mandated default: OBSERVE_ONLY, not approved, no new entries, global kill tripped."""
    return PlatformState()
