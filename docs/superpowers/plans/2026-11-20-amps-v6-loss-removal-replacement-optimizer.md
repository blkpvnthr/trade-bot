# AMPS V6 Loss-Removal & Replacement Optimizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continuously monitor every open position, deterministically close any whose canonical Total P/L (%) ≤ −0.5%, and replace it only with a universe.txt candidate that passes seller-exhaustion + positive-momentum + ALPHA_CONFIRMED + positive net expectancy + positive marginal-Sharpe + deterministic-risk gates — otherwise hold cash.

**Architecture:** New pure/testable modules under `amps_v6/` provide the canonical lifecycle Total P/L, the un-vetoable −0.5% exit (added to the existing `exit_stack`), the seller-exhaustion scorer, the positive-momentum gate, the marginal-Sharpe replacement optimizer, the replacement state machine, and anti-churn. They are wired into the **active** live engine (`trade.py` `ProductionOrderExecutor`) through its existing `exits` funnel (un-vetoable protective exits) and `rebalance` funnel (gated replacement buys), with `AlpacaBroker` extended for broker-native stop orders + reconciliation. All runtime state is PAPER/LIVE-namespaced.

**Tech Stack:** Python 3.11+, pandas/numpy, `alpaca-py` (`alpaca.trading`), pytest. Reuses `costs.py`, `metrics.py`, `portfolio_optimizer.py`, `order_flow_engine.py`, `universe.py`, and existing `amps_v6` primitives.

**Spec:** `docs/superpowers/specs/2026-11-20-amps-v6-loss-removal-replacement-optimizer-design.md`

## Global Constraints

- **Exit threshold is fixed:** `AMPS_TOTAL_PL_EXIT_THRESHOLD = -0.005`. Trigger iff `total_pl_pct <= -0.005` on a **valid** canonical computation. May not be widened, ignored, or vetoed by any analytical agent / optimizer / RL policy / LLM.
- **Canonical Total P/L is the single source:** one `compute_canonical_pnl()`; no other P/L definition may drive the rule (not previous-close, daily-account, raw last-trade, portfolio-wide, gross-ex-costs, agent-estimated, or stale/crossed/future/malformed quotes).
- **Exits are un-vetoable:** protective exits go through `exit_stack.evaluate` / the `exits` funnel, which take no agent input. Replacement buys go through `decision_fusion.fuse` + `risk_gateway.evaluate` + the `rebalance` funnel.
- **Cash is always valid:** never force a replacement.
- **Rank replacements by marginal portfolio Sharpe** (`delta_sharpe`) — never by exhaustion score, momentum, %change, volume, model confidence, standalone Sharpe, or raw expected return alone.
- **PAPER/LIVE isolation:** no shared executor locks, position state, order state, reconciliation files, runtime cursors, or audit identifiers. All new state paths are suffixed `_paper` / `_live`.
- **Do NOT modify:** `.claude/worktrees/`, `trader_rewrite/`, `vendor/`, `tradingview-mcp/`, `*.bak`/`*.backup`/`*.pre_killswitch.bak`, generated datasets, historical backtest outputs, unrelated research programs. Preserve all staged/unstaged/untracked work (incl. untracked `deploy/`, `scheduled_start_local.sh`, `trader/`).
- **Branch:** `amps-v6-loss-removal-optimizer` (already checked out). Commit after every task.
- **Non-claims:** the −0.5% exit is a risk control, not alpha. No claim that seller exhaustion guarantees reversal or that replacement improves Sharpe absent net-of-cost OOS + operational evidence.

## File Structure

**New files**
| Path | Responsibility |
|---|---|
| `amps_v6/configuration/replacement_config.py` | Versioned `ReplacementConfig` dataclass; env-overridable `AMPS_*` settings |
| `amps_v6/state/lifecycle_ledger.py` | `FillEvent`, `PositionLifecycle`, `CanonicalPnL`, `Quote`, `compute_canonical_pnl()`, `LifecycleLedger` (env-namespaced persistence + broker reconciliation) |
| `amps_v6/alpha/seller_exhaustion.py` | `SellerExhaustionScorer`, `SellerExhaustionResult`; 0..1 score, confirmations, persistence, `UNAVAILABLE` |
| `amps_v6/strategy/momentum_gate.py` | `PositiveMomentumGate`, `MomentumResult`; mandatory DI/BB/completed-bar + confirmation subset |
| `amps_v6/portfolio/marginal_sharpe.py` | `MarginalSharpeOptimizer`; `delta_sharpe` per candidate using covariance + costs |
| `amps_v6/portfolio/replacement_optimizer.py` | Orchestrator: pool, 19-gate eligibility, ranking, sizing, joint multi-stop, CASH_HELD |
| `amps_v6/execution/replacement_state_machine.py` | `ReplacementStateMachine` (HELD→…→PROTECTION_VERIFIED), env-namespaced persistence, restart-restore |
| `amps_v6/execution/anti_churn.py` | `AntiChurnRegistry`: cooldowns, per-cycle/session caps, hysteresis, turnover budget |
| `tests/test_amps_v6_loss_removal_replacement.py` | The 31 required tests (+ unit tests per module) |

**Modified files**
| Path | Change |
|---|---|
| `amps_v6/risk/exit_stack.py` | `TOTAL_PL_STOP_BREACHED` (priority 85) + `total_pl_pct`/`total_pl_valid` context + `total_pl_exit_threshold` |
| `amps_v6/execution/order_intent.py` | add `reduce_only: bool = False` |
| `order_flow_engine.py` | add `microprice`, `absorption_score`, `bid_replenishment_score`, `volume_climax_score` + snapshot fields |
| `universe.py` | persist the rotation cursor (env-namespaced) |
| `portfolio_rebalancer.py` | endpoint-routed paper/live (remove hardcoded `paper=True`) |
| `trade.py` | `AlpacaBroker`: stop orders + `get_order`/reconcile + cancel pending buys; `ProductionOrderExecutor`: loss-removal/replacement step, priority ordering, ledger feeding, env-namespaced state, diagnostics/report |

**Test conventions (match existing):** pytest; `from __future__ import annotations`; pure/offline where possible; broker/scanner via hand-written fakes (see `tests/test_di_bb.py`'s `_Broker`/`_Scanner`/`_Pos` and `tests/test_amps_v6_integrated.py`). Run a single test file with `python -m pytest tests/<file>.py -v`.

---

### Task 1: Versioned replacement config

**Files:**
- Create: `amps_v6/configuration/replacement_config.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Produces: `ReplacementConfig` dataclass with fields (and defaults): `config_version: str = "rc-2026-11-20"`, `total_pl_exit_threshold: float = -0.005`, `seller_exhaustion_min_score: float = 0.60`, `seller_exhaustion_min_confirmations: int = 3`, `seller_exhaustion_persistence_windows: int = 3`, `replacement_min_sharpe_improvement: float = 0.05`, `exited_symbol_cooldown_seconds: float = 1800.0`, `max_replacements_per_cycle: int = 2`, `max_symbol_replacements_per_session: int = 2`, `momentum_min_confirmations: int = 2`, `max_quote_age_s: float = 10.0`. Classmethod `ReplacementConfig.from_env(env: dict | None = None) -> ReplacementConfig`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_amps_v6_loss_removal_replacement.py
"""The 31 required tests for the AMPS V6 loss-removal + replacement optimizer
(spec 2026-11-20) plus per-module unit tests. Pure/offline; broker via fakes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import pytest

from amps_v6.configuration.replacement_config import ReplacementConfig


# ---------------------------------------------------------------- config
def test_config_defaults_and_fixed_threshold():
    cfg = ReplacementConfig()
    assert cfg.total_pl_exit_threshold == -0.005
    assert cfg.seller_exhaustion_min_confirmations == 3
    assert cfg.config_version


def test_config_env_override_but_threshold_locked():
    env = {
        "AMPS_SELLER_EXHAUSTION_MIN_SCORE": "0.75",
        "AMPS_MAX_REPLACEMENTS_PER_CYCLE": "5",
        "AMPS_TOTAL_PL_EXIT_THRESHOLD": "-0.02",  # attempt to widen
    }
    cfg = ReplacementConfig.from_env(env)
    assert cfg.seller_exhaustion_min_score == 0.75
    assert cfg.max_replacements_per_cycle == 5
    # threshold may be TIGHTENED but never widened past -0.005
    assert cfg.total_pl_exit_threshold == -0.005


def test_config_env_may_tighten_threshold():
    cfg = ReplacementConfig.from_env({"AMPS_TOTAL_PL_EXIT_THRESHOLD": "-0.003"})
    assert cfg.total_pl_exit_threshold == -0.003
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k config -v`
Expected: FAIL with `ModuleNotFoundError: amps_v6.configuration.replacement_config`

- [ ] **Step 3: Write the implementation**

```python
# amps_v6/configuration/replacement_config.py
"""Versioned, env-overridable config for the loss-removal + replacement optimizer.

Bridges the two config worlds in this repo: trade.py reads ALL_CAPS env vars,
amps_v6 uses dataclasses. This dataclass carries defaults and a config_version,
and `from_env` applies the AMPS_* overrides. The -0.5% exit threshold is a risk
control: env may TIGHTEN it (more negative not allowed past the floor) but may
NEVER widen it beyond -0.005.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

EXIT_THRESHOLD_FLOOR = -0.005   # most lenient allowed; env can only tighten


@dataclass
class ReplacementConfig:
    config_version: str = "rc-2026-11-20"
    total_pl_exit_threshold: float = -0.005
    seller_exhaustion_min_score: float = 0.60
    seller_exhaustion_min_confirmations: int = 3
    seller_exhaustion_persistence_windows: int = 3
    replacement_min_sharpe_improvement: float = 0.05
    exited_symbol_cooldown_seconds: float = 1800.0
    max_replacements_per_cycle: int = 2
    max_symbol_replacements_per_session: int = 2
    momentum_min_confirmations: int = 2
    max_quote_age_s: float = 10.0

    @classmethod
    def from_env(cls, env: dict | None = None) -> "ReplacementConfig":
        env = os.environ if env is None else env

        def _f(key: str, default: float) -> float:
            v = env.get(key)
            return float(v) if v not in (None, "") else default

        def _i(key: str, default: int) -> int:
            v = env.get(key)
            return int(v) if v not in (None, "") else default

        base = cls()
        raw_threshold = _f("AMPS_TOTAL_PL_EXIT_THRESHOLD", base.total_pl_exit_threshold)
        # tighten-only: never allow a value more lenient (less negative) than the floor
        threshold = min(raw_threshold, EXIT_THRESHOLD_FLOOR)
        return cls(
            total_pl_exit_threshold=threshold,
            seller_exhaustion_min_score=_f("AMPS_SELLER_EXHAUSTION_MIN_SCORE", base.seller_exhaustion_min_score),
            seller_exhaustion_min_confirmations=_i("AMPS_SELLER_EXHAUSTION_MIN_CONFIRMATIONS", base.seller_exhaustion_min_confirmations),
            seller_exhaustion_persistence_windows=_i("AMPS_SELLER_EXHAUSTION_PERSISTENCE_WINDOWS", base.seller_exhaustion_persistence_windows),
            replacement_min_sharpe_improvement=_f("AMPS_REPLACEMENT_MIN_SHARPE_IMPROVEMENT", base.replacement_min_sharpe_improvement),
            exited_symbol_cooldown_seconds=_f("AMPS_EXITED_SYMBOL_COOLDOWN_SECONDS", base.exited_symbol_cooldown_seconds),
            max_replacements_per_cycle=_i("AMPS_MAX_REPLACEMENTS_PER_CYCLE", base.max_replacements_per_cycle),
            max_symbol_replacements_per_session=_i("AMPS_MAX_SYMBOL_REPLACEMENTS_PER_SESSION", base.max_symbol_replacements_per_session),
            momentum_min_confirmations=_i("AMPS_MOMENTUM_MIN_CONFIRMATIONS", base.momentum_min_confirmations),
            max_quote_age_s=_f("AMPS_MAX_QUOTE_AGE_S", base.max_quote_age_s),
        )

    def as_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k config -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add amps_v6/configuration/replacement_config.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): versioned replacement config with env overrides and tighten-only exit threshold"
```

---

### Task 2: Canonical lifecycle Total P/L (the foundation)

**Files:**
- Create: `amps_v6/state/lifecycle_ledger.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Consumes: `costs.TransactionCostConfig`, `costs.BASELINE_COSTS`, `costs.total_cost_bps`.
- Produces:
  - `Quote(bid, ask, ts, bid_size=0.0, ask_size=0.0)`
  - `FillEvent(ts, symbol, side, qty, price, reference_price=0.0, commission=0.0, fees=0.0, client_order_id="")`
  - `PositionLifecycle(symbol, ...)` with `apply_fill(FillEvent)`, `apply_financing(float)`
  - `CanonicalPnL(...)` with `.valid`, `.status`, `.total_pl_pct`, `.position_total_pl`, `.bid`, `.quote_age_s`, `.as_dict()`
  - `validate_quote(q, now, max_quote_age_s, future_skew_s=2.0) -> str` returning one of `OK|UNKNOWN|STALE|CROSSED|MALFORMED`
  - `compute_canonical_pnl(lc, quote, now, cost_cfg=BASELINE_COSTS, max_quote_age_s=10.0, adv=0.0) -> CanonicalPnL`
  - `LifecycleLedger(env="paper")` with `.get(symbol)`, `.apply_fill(FillEvent)`, `.canonical(symbol, quote, now, **kw)`, `.reconcile(symbol, broker_qty, broker_avg_entry)`, `.save(path=None)`, classmethod `.load(env, path=None)`, property `.state_path`
- Status constants: `STATUS_OK`, `STATUS_UNKNOWN`, `STATUS_STALE`, `STATUS_CROSSED`, `STATUS_MALFORMED`.

- [ ] **Step 1: Write the failing tests** (covers required tests #1–#6 + reconciliation)

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
from amps_v6.state import lifecycle_ledger as LL
from costs import TransactionCostConfig

ZERO_COST = TransactionCostConfig(spread_bps=0.0, slippage_bps=0.0, commission_bps=0.0, impact_coefficient=0.0)


def _open(symbol="AAA", qty=10.0, price=100.0):
    lc = LL.PositionLifecycle(symbol=symbol)
    lc.apply_fill(LL.FillEvent(ts=0.0, symbol=symbol, side="buy", qty=qty, price=price, reference_price=price))
    return lc


def _q(bid, ask=None, ts=100.0):
    return LL.Quote(bid=bid, ask=ask if ask is not None else bid + 0.02, ts=ts)


def _pct(lc, bid, now=100.0, ts=100.0, cost=ZERO_COST):
    return LL.compute_canonical_pnl(lc, _q(bid, ts=ts), now=now, cost_cfg=cost, max_quote_age_s=10.0)


# ---- required #1/#2/#3: threshold boundary (zero costs, basis 1000) ----
def test_req01_minus_049_does_not_trigger():
    # (99.51-100)*10 = -4.9 -> -0.49%
    r = _pct(_open(), 99.51)
    assert r.valid and r.total_pl_pct == pytest.approx(-0.0049)
    assert not (r.valid and r.total_pl_pct <= -0.005)


def test_req02_exactly_minus_050_triggers():
    r = _pct(_open(), 99.50)   # -5.0 / 1000
    assert r.valid and r.total_pl_pct == pytest.approx(-0.005)
    assert r.valid and r.total_pl_pct <= -0.005


def test_req03_minus_051_triggers():
    r = _pct(_open(), 99.49)   # -5.1 / 1000
    assert r.valid and r.total_pl_pct == pytest.approx(-0.0051)
    assert r.valid and r.total_pl_pct <= -0.005


# ---- required #4: entry+exit costs affect Total P/L ----
def test_req04_costs_make_pnl_more_negative():
    lc = _open()
    gross = _pct(lc, 100.0, cost=ZERO_COST).total_pl_pct
    netted = _pct(lc, 100.0, cost=TransactionCostConfig(spread_bps=10, slippage_bps=5, commission_bps=1, impact_coefficient=10)).total_pl_pct
    assert gross == pytest.approx(0.0)
    assert netted < 0.0   # estimated remaining exit costs drag it negative at flat price


def test_req04b_entry_commissions_counted():
    lc = LL.PositionLifecycle(symbol="AAA")
    lc.apply_fill(LL.FillEvent(ts=0, symbol="AAA", side="buy", qty=10, price=100.0, reference_price=100.0, commission=2.0, fees=0.5))
    r = _pct(lc, 100.0)
    # flat price, zero exit-cost model, but 2.5 of entry costs -> -2.5/1000
    assert r.total_pl_pct == pytest.approx(-0.0025)


# ---- required #5: partial reductions do not hide lifecycle losses ----
def test_req05_partial_reduction_keeps_original_basis():
    lc = _open(qty=10, price=100.0)          # basis 1000
    lc.apply_fill(LL.FillEvent(ts=1, symbol="AAA", side="sell", qty=5, price=95.0, reference_price=95.0))  # realize -25
    r = _pct(lc, 95.0)                        # remaining 5 @ mark 95 -> unrealized -25
    assert lc.original_cost_basis == pytest.approx(1000.0)   # NOT reset to 500
    assert r.total_pl_pct == pytest.approx(-0.05)            # -50/1000, loss not hidden


# ---- required #6: stale/invalid data cannot fabricate a trigger ----
def test_req06_stale_quote_is_invalid():
    r = LL.compute_canonical_pnl(_open(), _q(50.0, ts=0.0), now=1000.0, cost_cfg=ZERO_COST, max_quote_age_s=10.0)
    assert r.status == LL.STATUS_STALE and not r.valid


def test_req06b_crossed_and_malformed_invalid():
    lc = _open()
    crossed = LL.compute_canonical_pnl(lc, LL.Quote(bid=101.0, ask=100.0, ts=100.0), now=100.0, cost_cfg=ZERO_COST)
    assert crossed.status == LL.STATUS_CROSSED and not crossed.valid
    future = LL.compute_canonical_pnl(lc, LL.Quote(bid=99.0, ask=99.1, ts=200.0), now=100.0, cost_cfg=ZERO_COST)
    assert future.status == LL.STATUS_MALFORMED and not future.valid
    nan = LL.compute_canonical_pnl(lc, LL.Quote(bid=float("nan"), ask=99.1, ts=100.0), now=100.0, cost_cfg=ZERO_COST)
    assert nan.status == LL.STATUS_MALFORMED and not nan.valid


# ---- ledger reconciliation + env-namespaced persistence ----
def test_ledger_reconcile_adopts_broker_truth():
    led = LL.LifecycleLedger(env="paper")
    led.apply_fill(LL.FillEvent(ts=0, symbol="AAA", side="buy", qty=10, price=100.0, reference_price=100.0))
    led.reconcile("AAA", broker_qty=8.0, broker_avg_entry=100.0)   # broker says 8 (2 filled elsewhere)
    assert led.get("AAA").qty == 8.0


def test_ledger_env_namespaced_paths_differ():
    assert "paper" in str(LL.LifecycleLedger(env="paper").state_path)
    assert "live" in str(LL.LifecycleLedger(env="live").state_path)
    assert LL.LifecycleLedger(env="paper").state_path != LL.LifecycleLedger(env="live").state_path


def test_ledger_roundtrip(tmp_path):
    led = LL.LifecycleLedger(env="paper")
    led.apply_fill(LL.FillEvent(ts=0, symbol="AAA", side="buy", qty=10, price=100.0, reference_price=100.0))
    p = tmp_path / "led.json"
    led.save(p)
    back = LL.LifecycleLedger.load("paper", p)
    assert back.get("AAA").qty == 10.0 and back.get("AAA").original_cost_basis == 1000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "req0 or ledger" -v`
Expected: FAIL with `ModuleNotFoundError: amps_v6.state.lifecycle_ledger`

- [ ] **Step 3: Write the implementation**

```python
# amps_v6/state/lifecycle_ledger.py
"""Canonical lifecycle Total P/L (%) for AMPS V6 — the single source of truth.

No "Total P/L" existed anywhere in the repo before this. The broker exposes only
gross unrealized_pl/plpc/cost_basis; this module adds a lifecycle fill/cost
ledger so the -0.5% rule is computed cost-inclusive, across partial reductions,
on validated quotes only. One formula, one function: compute_canonical_pnl().

  position_total_pl = realized_pnl_gross + unrealized_gross
                      - entry_costs - realized_exit_costs - est_remaining_exit_costs
                      - realized_slippage - financing
  total_pl_pct      = position_total_pl / original_cost_basis   # basis never shrinks on sells
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from costs import TransactionCostConfig, BASELINE_COSTS, total_cost_bps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "logs" / "execution"

STATUS_OK = "OK"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_STALE = "STALE"
STATUS_CROSSED = "CROSSED"
STATUS_MALFORMED = "MALFORMED"


@dataclass
class Quote:
    bid: float
    ask: float
    ts: float
    bid_size: float = 0.0
    ask_size: float = 0.0


@dataclass
class FillEvent:
    ts: float
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    price: float
    reference_price: float = 0.0   # intended/mid price at decision -> slippage = |price - ref|*qty
    commission: float = 0.0
    fees: float = 0.0
    client_order_id: str = ""


@dataclass
class PositionLifecycle:
    symbol: str
    qty: float = 0.0
    avg_entry: float = 0.0
    realized_pnl_gross: float = 0.0
    entry_costs: float = 0.0
    realized_exit_costs: float = 0.0
    realized_slippage: float = 0.0
    original_cost_basis: float = 0.0     # Σ buy notionals over lifecycle; never reduced by sells
    pending_qty: float = 0.0
    financing_accrued: float = 0.0
    fills: list = field(default_factory=list)

    def apply_fill(self, ev: FillEvent) -> None:
        if ev.reference_price and ev.reference_price > 0:
            self.realized_slippage += abs(ev.price - ev.reference_price) * ev.qty
        if ev.side == "buy":
            new_qty = self.qty + ev.qty
            if new_qty > 0:
                self.avg_entry = ((self.avg_entry * self.qty) + ev.price * ev.qty) / new_qty
            self.qty = new_qty
            self.original_cost_basis += ev.price * ev.qty
            self.entry_costs += ev.commission + ev.fees
        elif ev.side == "sell":
            reduce_qty = min(ev.qty, self.qty)
            self.realized_pnl_gross += (ev.price - self.avg_entry) * reduce_qty
            self.qty -= reduce_qty
            self.realized_exit_costs += ev.commission + ev.fees
        self.fills.append(asdict(ev))

    def apply_financing(self, amount: float) -> None:
        self.financing_accrued += amount


@dataclass
class CanonicalPnL:
    symbol: str
    status: str
    valid: bool
    position_total_pl: float
    total_pl_pct: float
    realized_pnl: float
    unrealized_pnl: float
    entry_costs: float
    realized_exit_costs: float
    estimated_remaining_exit_costs: float
    realized_slippage: float
    financing_costs: float
    cost_basis: float
    qty: float
    avg_entry: float
    mark: float
    bid: float
    quote_age_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def validate_quote(q: Optional[Quote], now: float, max_quote_age_s: float, future_skew_s: float = 2.0) -> str:
    if q is None:
        return STATUS_UNKNOWN
    for v in (q.bid, q.ask, q.ts):
        if v is None:
            return STATUS_MALFORMED
    try:
        if math.isnan(q.bid) or math.isnan(q.ask) or q.bid <= 0 or q.ask <= 0:
            return STATUS_MALFORMED
    except TypeError:
        return STATUS_MALFORMED
    if q.bid >= q.ask:
        return STATUS_CROSSED
    if q.ts > now + future_skew_s:
        return STATUS_MALFORMED            # future-dated
    if (now - q.ts) > max_quote_age_s:
        return STATUS_STALE
    return STATUS_OK


def compute_canonical_pnl(lc: PositionLifecycle, quote: Optional[Quote], now: float,
                          cost_cfg: TransactionCostConfig = BASELINE_COSTS,
                          max_quote_age_s: float = 10.0, adv: float = 0.0) -> CanonicalPnL:
    """THE canonical Total P/L (%). On an invalid quote, valid=False so the rule
    cannot fire; the returned numbers are informational only."""
    status = validate_quote(quote, now, max_quote_age_s)
    valid = status == STATUS_OK
    bid = float(quote.bid) if (quote is not None and valid) else 0.0
    # long position: liquidation mark is the bid; invalid -> fall back to avg_entry (valid=False anyway)
    mark = bid if valid else lc.avg_entry
    unrealized = (mark - lc.avg_entry) * lc.qty if lc.qty > 0 else 0.0
    remaining_notional = mark * lc.qty if lc.qty > 0 else 0.0
    est_exit_cost = (remaining_notional * total_cost_bps(remaining_notional, adv, cost_cfg) / 10_000.0) if lc.qty > 0 else 0.0
    position_total_pl = (
        lc.realized_pnl_gross + unrealized
        - lc.entry_costs - lc.realized_exit_costs - est_exit_cost
        - lc.realized_slippage - lc.financing_accrued
    )
    basis = lc.original_cost_basis
    total_pl_pct = (position_total_pl / basis) if basis > 0 else 0.0
    quote_age = (now - quote.ts) if (quote is not None and quote.ts) else float("inf")
    return CanonicalPnL(
        symbol=lc.symbol, status=status, valid=valid,
        position_total_pl=position_total_pl, total_pl_pct=total_pl_pct,
        realized_pnl=lc.realized_pnl_gross - lc.realized_exit_costs,
        unrealized_pnl=unrealized, entry_costs=lc.entry_costs,
        realized_exit_costs=lc.realized_exit_costs, estimated_remaining_exit_costs=est_exit_cost,
        realized_slippage=lc.realized_slippage, financing_costs=lc.financing_accrued,
        cost_basis=basis, qty=lc.qty, avg_entry=lc.avg_entry, mark=mark, bid=bid,
        quote_age_s=quote_age,
    )


class LifecycleLedger:
    """Manages per-symbol lifecycles with PAPER/LIVE-namespaced persistence and
    broker reconciliation. env must be 'paper' or 'live'."""

    def __init__(self, env: str = "paper"):
        if env not in ("paper", "live"):
            raise ValueError(f"env must be 'paper' or 'live', got {env!r}")
        self.env = env
        self._pos: dict[str, PositionLifecycle] = {}

    @property
    def state_path(self) -> Path:
        return STATE_DIR / f"amps_v6_lifecycle_ledger_{self.env}.json"

    def get(self, symbol: str) -> PositionLifecycle:
        return self._pos.setdefault(symbol, PositionLifecycle(symbol=symbol))

    def apply_fill(self, ev: FillEvent) -> None:
        self.get(ev.symbol).apply_fill(ev)

    def canonical(self, symbol: str, quote: Optional[Quote], now: float, **kw) -> CanonicalPnL:
        return compute_canonical_pnl(self.get(symbol), quote, now, **kw)

    def reconcile(self, symbol: str, broker_qty: float, broker_avg_entry: float) -> None:
        """Broker is truth for qty/avg_entry. On divergence, adopt the broker view
        (keeps cost/realized history; corrects drift). original_cost_basis is only
        raised to cover the broker qty, never lowered (never hides earlier losses)."""
        lc = self.get(symbol)
        if broker_avg_entry > 0:
            lc.avg_entry = broker_avg_entry
        lc.qty = broker_qty
        min_basis = broker_avg_entry * broker_qty if (broker_avg_entry > 0 and broker_qty > 0) else 0.0
        if min_basis > lc.original_cost_basis:
            lc.original_cost_basis = min_basis

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"env": self.env, "positions": {s: asdict(p) for s, p in self._pos.items()}}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, env: str, path: Optional[Path] = None) -> "LifecycleLedger":
        led = cls(env=env)
        path = path or led.state_path
        if not path.exists():
            return led
        data = json.loads(path.read_text(encoding="utf-8"))
        for sym, pd in data.get("positions", {}).items():
            fills = pd.pop("fills", [])
            lc = PositionLifecycle(**{k: v for k, v in pd.items() if k in PositionLifecycle.__dataclass_fields__ and k != "fills"})
            lc.fills = fills
            led._pos[sym] = lc
        return led
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "req0 or ledger" -v`
Expected: PASS (all req01–req06b + ledger tests)

- [ ] **Step 5: Commit**

```bash
git add amps_v6/state/lifecycle_ledger.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): canonical lifecycle Total P/L ledger (cost-inclusive, basis-preserving, validated quotes)"
```

---

### Task 3: Reduce-only flag on OrderIntent

**Files:**
- Modify: `amps_v6/execution/order_intent.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Produces: `OrderIntent(..., reduce_only: bool = False)` — serialized by `as_dict()`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
from amps_v6.execution.order_intent import OrderIntent


def test_order_intent_reduce_only_default_and_set():
    assert OrderIntent(symbol="AAA", side="sell", qty=10, reason="x").reduce_only is False
    oi = OrderIntent(symbol="AAA", side="sell", qty=10, reason="total_pl_stop", reduce_only=True)
    assert oi.reduce_only is True
    assert oi.as_dict()["reduce_only"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k reduce_only -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'reduce_only'`

- [ ] **Step 3: Add the field**

In `amps_v6/execution/order_intent.py`, add to the `OrderIntent` dataclass alongside the other execution-plumbing fields (`client_order_id`, `entry_price_hint`, `protective_stop`, `profit_target`):

```python
    reduce_only: bool = False      # exit/trim intent: may never increase exposure
```

(`as_dict()` already serializes all fields via `asdict`/dataclass reflection — no change needed there; verify by the test above.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k reduce_only -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add amps_v6/execution/order_intent.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): add reduce_only flag to OrderIntent"
```

---

### Task 4: TOTAL_PL_STOP_BREACHED in the exit stack (un-vetoable −0.5%)

**Files:**
- Modify: `amps_v6/risk/exit_stack.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Consumes: `ReplacementConfig.total_pl_exit_threshold` (passed into `ExitConfig.total_pl_exit_threshold`).
- Produces: `ExitConfig.total_pl_exit_threshold: float = -0.005`; `ExitContext.total_pl_pct: float = 0.0`, `ExitContext.total_pl_valid: bool = False`; a `TOTAL_PL_STOP_BREACHED` `ExitDecision(action=EXIT, priority=85)`.

- [ ] **Step 1: Write the failing tests** (covers required #7 — agents cannot veto)

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
from amps_v6.risk import exit_stack as XS


def test_total_pl_stop_fires_at_threshold():
    d = XS.evaluate(XS.ExitContext(price=100, avg_entry=100, total_pl_pct=-0.005, total_pl_valid=True))
    assert d.reason == "TOTAL_PL_STOP_BREACHED" and d.action == XS.EXIT and d.priority == 85


def test_total_pl_stop_not_at_minus_049():
    d = XS.evaluate(XS.ExitContext(price=100, avg_entry=100, total_pl_pct=-0.0049, total_pl_valid=True))
    assert d.reason != "TOTAL_PL_STOP_BREACHED"


def test_req06_exit_stack_ignores_invalid_total_pl():
    # even a catastrophic-looking total_pl cannot trigger when the data is invalid
    d = XS.evaluate(XS.ExitContext(price=100, avg_entry=100, total_pl_pct=-0.99, total_pl_valid=False))
    assert d.reason != "TOTAL_PL_STOP_BREACHED"


def test_req07_agents_cannot_veto_total_pl_exit():
    # ExitContext has NO agent/veto field; the exit is computed from price/pnl only.
    import dataclasses
    fields = {f.name for f in dataclasses.fields(XS.ExitContext)}
    assert not (fields & {"agent_veto", "adversarial_veto", "security_veto", "llm_override", "regime_veto"})
    # and the breach fires regardless of any regime/thesis signal
    d = XS.evaluate(XS.ExitContext(price=100, avg_entry=100, total_pl_pct=-0.006, total_pl_valid=True,
                                   regime_permits_hold=True, thesis_valid=True))
    assert d.reason == "TOTAL_PL_STOP_BREACHED"


def test_total_pl_stop_below_account_level_shutdowns():
    # account-level EMERGENCY/DRAWDOWN/DAILY_LOSS still dominate the -0.5% stop
    d = XS.evaluate(XS.ExitContext(price=100, avg_entry=100, total_pl_pct=-0.006, total_pl_valid=True,
                                   account_daily_pnl_pct=-0.05))
    assert d.reason == "DAILY_LOSS_SHUTDOWN"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "total_pl or req07" -v`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'total_pl_pct'`)

- [ ] **Step 3: Implement**

In `amps_v6/risk/exit_stack.py`:

a) Add to `ExitConfig` (after `regime_exit_enabled`):

```python
    total_pl_exit_threshold: float = -0.005   # canonical cost-inclusive Total P/L (%) hard stop
```

b) Add to `ExitContext` (after `emergency`):

```python
    total_pl_pct: float = 0.0        # canonical cost-inclusive Total P/L (%), fraction
    total_pl_valid: bool = False     # True only when computed from a validated quote
```

c) In `evaluate(...)`, insert the TOTAL_PL check immediately **after** the three account-level shutdowns (EMERGENCY/DRAWDOWN/DAILY_LOSS) and **before** the `avg_entry<=0 or price<=0` basis guard — so it dominates every per-position trigger but never overrides account-level flattening:

```python
    # --- canonical Total P/L (%) hard stop: deterministic, un-vetoable ---
    if ctx.total_pl_valid and ctx.total_pl_pct <= cfg.total_pl_exit_threshold:
        return ExitDecision(EXIT, "TOTAL_PL_STOP_BREACHED", 85)
```

d) Update the module docstring's exit-reason list to include `TOTAL_PL_STOP_BREACHED`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "total_pl or req07" -v && python -m pytest tests/test_amps_v6_integrated.py -k exit -v`
Expected: PASS (new tests pass; existing exit-stack tests still pass — the new branch only fires when `total_pl_valid`, which legacy callers leave `False`)

- [ ] **Step 5: Commit**

```bash
git add amps_v6/risk/exit_stack.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): TOTAL_PL_STOP_BREACHED -0.5% exit (priority 85, un-vetoable, validated-only)"
```

---

### Task 5: L1-computable microstructure primitives

**Files:**
- Modify: `order_flow_engine.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Produces (pure module functions in `order_flow_engine`):
  - `microprice(bid, ask, bid_size, ask_size) -> float | None` — size-weighted fair price; `None` on invalid inputs.
  - `absorption_score(sell_participation, downside_bps, cap_bps=20.0) -> float` — [0,1]; high = heavy aggressive selling, little downside.
  - `bid_replenishment_score(bid_sizes) -> float` — [0,1]; high = bid size dipped then recovered.
  - `volume_climax_score(sell_vol_series) -> float` — [0,1]; high = sell-volume spike followed by decline.
- Adds `microprice: float = 0.0` field to `OrderFlowSnapshot`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
import order_flow_engine as OFE


def test_microprice_size_weighted():
    # heavier ask size pulls microprice toward the bid (more size to sell into)
    mp = OFE.microprice(100.0, 100.10, bid_size=100, ask_size=900)
    assert mp is not None and 100.0 < mp < 100.05
    assert OFE.microprice(100.0, 100.0, 1, 1) is None        # not a valid 2-sided market
    assert OFE.microprice(0, 100, 1, 1) is None              # invalid
    assert OFE.microprice(100, 100.1, 0, 0) is None          # no size


def test_absorption_score_high_when_selling_without_downside():
    strong = OFE.absorption_score(sell_participation=0.9, downside_bps=1.0)
    weak = OFE.absorption_score(sell_participation=0.9, downside_bps=40.0)
    assert strong > 0.8 and weak == 0.0
    assert OFE.absorption_score(0.1, 1.0) < 0.2             # little selling -> little absorption


def test_bid_replenishment_dip_then_recover():
    assert OFE.bid_replenishment_score([1000, 200, 400, 900]) > 0.6   # dipped to 200, back to 900
    assert OFE.bid_replenishment_score([1000, 1000, 1000]) == 0.0     # never dipped
    assert OFE.bid_replenishment_score([500]) == 0.0                  # too short


def test_volume_climax_spike_then_decline():
    assert OFE.volume_climax_score([5000, 1000, 500, 200]) > 0.8      # climax then dry up
    assert OFE.volume_climax_score([100, 100, 100]) == 0.0            # flat, no climax
    assert OFE.volume_climax_score([]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "microprice or absorption or bid_replenishment or volume_climax" -v`
Expected: FAIL (`AttributeError: module 'order_flow_engine' has no attribute 'microprice'`)

- [ ] **Step 3: Implement** — add these pure functions to `order_flow_engine.py` (near the other `norm_*` signal primitives, e.g. after `norm_seller_exhaustion`):

```python
def microprice(bid, ask, bid_size, ask_size):
    """Size-weighted fair price (Stoikov microprice). Weight toward the side with
    LESS size: price = (bid*ask_size + ask*bid_size)/(bid_size+ask_size). Returns
    None for a crossed/locked/one-sided/empty book."""
    try:
        if bid is None or ask is None or bid <= 0 or ask <= 0 or bid >= ask:
            return None
        total = (bid_size or 0.0) + (ask_size or 0.0)
        if total <= 0:
            return None
        return float((bid * ask_size + ask * bid_size) / total)
    except (TypeError, ValueError):
        return None


def absorption_score(sell_participation, downside_bps, cap_bps=20.0):
    """Price absorption: aggressive selling continues without proportional
    downside. sell_participation in [0,1] (fraction of volume that is aggressive
    sells); downside_bps = realized drop over the window. High when participation
    is high and downside is small. 0 once downside exceeds cap_bps."""
    sp = max(0.0, min(1.0, float(sell_participation)))
    d = max(0.0, float(downside_bps))
    if d >= cap_bps:
        return 0.0
    return float(sp * (1.0 - d / cap_bps))


def bid_replenishment_score(bid_sizes):
    """Bid liquidity repeatedly replenishes after sell executions: the bid size
    dips then recovers. Score = recovery fraction from the trough back toward the
    peak. 0 if it never dipped or series too short."""
    s = [float(x) for x in (bid_sizes or []) if x is not None]
    if len(s) < 3:
        return 0.0
    peak = max(s)
    trough = min(s)
    if peak <= trough:
        return 0.0
    last = s[-1]
    return float(max(0.0, min(1.0, (last - trough) / (peak - trough))))


def volume_climax_score(sell_vol_series):
    """Unusually high sell volume (a climax) followed by reduced sell volume.
    Score = (peak - last)/peak, gated so a flat series (peak ~ mean) scores 0."""
    s = [float(x) for x in (sell_vol_series or []) if x is not None]
    if len(s) < 2:
        return 0.0
    peak = max(s)
    if peak <= 0:
        return 0.0
    mean = sum(s) / len(s)
    if peak <= mean * 1.5:          # no real spike
        return 0.0
    last = s[-1]
    return float(max(0.0, min(1.0, (peak - last) / peak)))
```

Then add the field to `OrderFlowSnapshot` (dataclass): `microprice: float = 0.0`, and in `SymbolOrderFlow.snapshot(...)` set it where `bid`/`ask` and their sizes are known:

```python
        mp = microprice(bid, ask, getattr(self, "_last_bid_size", 0.0), getattr(self, "_last_ask_size", 0.0))
        # ... include microprice=(mp if mp is not None else mid) in the OrderFlowSnapshot(...) construction
```

(If the engine does not retain last bid/ask sizes, store them in `on_quote`: `self._last_bid_size, self._last_ask_size = bid_size, ask_size`. This is additive and does not change existing fields.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "microprice or absorption or bid_replenishment or volume_climax" -v && python -m pytest tests/test_order_flow_engine.py -v`
Expected: PASS (new primitives pass; existing order-flow tests unaffected)

- [ ] **Step 5: Commit**

```bash
git add order_flow_engine.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(order_flow): add microprice, absorption, bid-replenishment, volume-climax primitives"
```

---

### Task 6: Seller-exhaustion scorer

**Files:**
- Create: `amps_v6/alpha/seller_exhaustion.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Consumes: `ReplacementConfig` (min_score, min_confirmations, persistence_windows); `order_flow_engine.{absorption_score,bid_replenishment_score,volume_climax_score}`; a time-ordered window of **snapshot-like** objects exposing (via `getattr`, all optional): `status` (str, "OK" required for validity), `mid`, `spread_bps`, `executed_imbalance`, `quote_imbalance`, `cvd_divergence`, `sell_pressure`, `microprice`, `sell_vol_60s`, `bid_size`.
- Produces:
  - `SellerExhaustionResult(symbol, available: bool, status: str, score: float, confirmations: int, categories: dict, persistence_ok: bool, confirmed: bool, reason: str)`
  - `SellerExhaustionScorer(cfg: ReplacementConfig)` with `score(symbol: str, window: list) -> SellerExhaustionResult`
  - `STATUS_UNAVAILABLE = "UNAVAILABLE"`

- [ ] **Step 1: Write the failing tests** (covers required #16, #17)

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
from types import SimpleNamespace
from amps_v6.alpha import seller_exhaustion as SE


def _snap(**kw):
    base = dict(status="OK", mid=100.0, spread_bps=5.0, executed_imbalance=0.0, quote_imbalance=0.0,
                cvd_divergence=0.0, sell_pressure=0.5, microprice=100.0, sell_vol_60s=100.0, bid_size=500.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _exhaustion_window(n=4):
    # selling weakens across windows: sell_pressure falls, executed/quote imbalance rise toward bid,
    # bullish CVD divergence, bid replenishes, sell volume climaxes then dries, spread contracts, price reclaims low
    return [
        _snap(sell_pressure=0.95, executed_imbalance=-0.6, quote_imbalance=-0.5, cvd_divergence=0.1,
              sell_vol_60s=5000, bid_size=1000, spread_bps=12, mid=99.5, microprice=99.5),
        _snap(sell_pressure=0.75, executed_imbalance=-0.3, quote_imbalance=-0.2, cvd_divergence=0.4,
              sell_vol_60s=1500, bid_size=300, spread_bps=9, mid=99.3, microprice=99.35),
        _snap(sell_pressure=0.55, executed_imbalance=0.0, quote_imbalance=0.1, cvd_divergence=0.6,
              sell_vol_60s=600, bid_size=600, spread_bps=7, mid=99.4, microprice=99.5),
        _snap(sell_pressure=0.35, executed_imbalance=0.2, quote_imbalance=0.3, cvd_divergence=0.7,
              sell_vol_60s=300, bid_size=950, spread_bps=5, mid=99.6, microprice=99.7),
    ]


def _cfg():
    return ReplacementConfig()


def test_seller_exhaustion_confirmed_on_clean_window():
    r = SE.SellerExhaustionScorer(_cfg()).score("AAA", _exhaustion_window())
    assert r.available and r.status == "OK"
    assert r.score >= 0.60 and r.confirmations >= 3 and r.persistence_ok
    assert r.confirmed is True


def test_req16_missing_orderflow_returns_unavailable():
    bad = [_snap(status="INSUFFICIENT_DATA") for _ in range(4)]
    r = SE.SellerExhaustionScorer(_cfg()).score("AAA", bad)
    assert not r.available and r.status == SE.STATUS_UNAVAILABLE and not r.confirmed
    # empty window -> unavailable too
    r2 = SE.SellerExhaustionScorer(_cfg()).score("AAA", [])
    assert not r2.available and r2.status == SE.STATUS_UNAVAILABLE


def test_req17_single_transient_imbalance_cannot_confirm():
    # mostly heavy selling, one bullish blip in the middle -> no persistent weakening
    w = [_snap(sell_pressure=0.9, executed_imbalance=-0.6, quote_imbalance=-0.5, cvd_divergence=-0.2, sell_vol_60s=4000),
         _snap(sell_pressure=0.2, executed_imbalance=0.5, quote_imbalance=0.4, cvd_divergence=0.6, sell_vol_60s=300),
         _snap(sell_pressure=0.92, executed_imbalance=-0.65, quote_imbalance=-0.55, cvd_divergence=-0.3, sell_vol_60s=4200),
         _snap(sell_pressure=0.94, executed_imbalance=-0.7, quote_imbalance=-0.6, cvd_divergence=-0.35, sell_vol_60s=4500)]
    r = SE.SellerExhaustionScorer(_cfg()).score("AAA", w)
    assert not r.persistence_ok and not r.confirmed


def test_seller_exhaustion_short_window_not_persistent():
    r = SE.SellerExhaustionScorer(_cfg()).score("AAA", _exhaustion_window(1)[:1])
    assert not r.confirmed  # length < persistence_windows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "seller_exhaustion or req16 or req17" -v`
Expected: FAIL (`ModuleNotFoundError: amps_v6.alpha.seller_exhaustion`)

- [ ] **Step 3: Implement**

```python
# amps_v6/alpha/seller_exhaustion.py
"""Seller-exhaustion scorer: is selling pressure weakening and are buyers
absorbing supply? Builds a normalized 0..1 score from INDEPENDENT evidence
categories computed over a time-ordered window of order-flow snapshots. Requires
a minimum score, a minimum number of independent confirmations, and persistence
across windows. Returns UNAVAILABLE (never a fabricated neutral/positive state)
when microstructure data is missing or stale.

Causal evidence (spec section "Seller-exhaustion requirement"). Categories that
need L2 (iceberg/stacked-imbalance) are intentionally absent -- IEX has no L2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from order_flow_engine import absorption_score, bid_replenishment_score, volume_climax_score

STATUS_UNAVAILABLE = "UNAVAILABLE"
_CONFIRM_THRESHOLD = 0.5   # a category "confirms" when its sub-score >= this


@dataclass
class SellerExhaustionResult:
    symbol: str
    available: bool
    status: str
    score: float
    confirmations: int
    categories: dict = field(default_factory=dict)
    persistence_ok: bool = False
    confirmed: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "available": self.available, "status": self.status,
                "score": round(self.score, 4), "confirmations": self.confirmations,
                "categories": {k: round(v, 3) for k, v in self.categories.items()},
                "persistence_ok": self.persistence_ok, "confirmed": self.confirmed, "reason": self.reason}


def _g(obj, name, default=None):
    v = getattr(obj, name, default)
    return default if v is None else v


def _series(window, name):
    return [_g(s, name) for s in window if _g(s, name) is not None]


class SellerExhaustionScorer:
    def __init__(self, cfg):
        self.cfg = cfg

    def _categories(self, window) -> dict:
        """Each category -> sub-score in [0,1]; omitted if not computable."""
        cats: dict[str, float] = {}
        mids = _series(window, "mid")
        sp = _series(window, "sell_pressure")
        ei = _series(window, "executed_imbalance")
        qi = _series(window, "quote_imbalance")
        cd = _series(window, "cvd_divergence")
        sv = _series(window, "sell_vol_60s")
        bs = _series(window, "bid_size")
        mp = _series(window, "microprice")
        spr = _series(window, "spread_bps")

        # 1. price absorption: strong selling, small realized downside
        if sp and mids:
            downside_bps = max(0.0, -(mids[-1] / mids[0] - 1.0)) * 10_000.0
            cats["absorption"] = absorption_score(sum(sp) / len(sp), downside_bps)
        # 2. failed lower low / failed breakdown: price broke the early low then reclaimed it
        if len(mids) >= 3:
            early_low = min(mids[:-1])
            broke = min(mids) < early_low
            reclaimed = mids[-1] > early_low
            cats["failed_lower_low"] = 1.0 if (broke and reclaimed) or (mids[-1] >= early_low) else 0.0
        # 3. executed-flow improvement: imbalance rising from negative toward neutral/positive
        if len(ei) >= 2:
            cats["executed_flow_improvement"] = max(0.0, min(1.0, (ei[-1] - ei[0] + 1.0) / 2.0)) if ei[-1] > ei[0] else 0.0
        # 4. declining sell intensity: sell pressure falling across the window
        if len(sp) >= 2:
            cats["declining_sell_intensity"] = max(0.0, min(1.0, (sp[0] - sp[-1]) / max(sp[0], 1e-9)))
        # 5. order-flow reversal: quote imbalance improving toward the bid
        if len(qi) >= 2:
            cats["orderflow_reversal"] = max(0.0, min(1.0, (qi[-1] - qi[0] + 1.0) / 2.0)) if qi[-1] > qi[0] else 0.0
        # 6. CVD divergence: bullish (positive) divergence present and rising
        if cd:
            cats["cvd_divergence"] = max(0.0, min(1.0, cd[-1]))
        # 7. bid replenishment
        if len(bs) >= 3:
            cats["bid_replenishment"] = bid_replenishment_score(bs)
        # 8. spread stabilization: spread contracting vs the window's earlier average
        if len(spr) >= 2:
            earlier = sum(spr[:-1]) / (len(spr) - 1)
            cats["spread_stabilization"] = 1.0 if spr[-1] <= earlier else 0.0
        # 9. microprice improvement: microprice above its window low
        if len(mp) >= 2 and max(mp) > min(mp):
            cats["microprice_improvement"] = max(0.0, min(1.0, (mp[-1] - min(mp)) / (max(mp) - min(mp))))
        # 10. volume climax: sell-volume spike then dry up
        if len(sv) >= 2:
            cats["volume_climax"] = volume_climax_score(sv)
        return cats

    def _persistence_ok(self, window) -> bool:
        n = self.cfg.seller_exhaustion_persistence_windows
        if len(window) < n:
            return False
        tail = window[-n:]
        sp = [_g(s, "sell_pressure", 1.0) for s in tail]
        # selling pressure must be non-increasing (weakening) across the last n windows
        return all(sp[i] <= sp[i - 1] + 0.05 for i in range(1, len(sp)))

    def score(self, symbol: str, window: list) -> SellerExhaustionResult:
        n = self.cfg.seller_exhaustion_persistence_windows
        # data availability: enough windows, all recent snapshots valid
        if not window or len(window) < n or any(_g(s, "status", "UNAVAILABLE") != "OK" for s in window[-n:]):
            return SellerExhaustionResult(symbol=symbol, available=False, status=STATUS_UNAVAILABLE,
                                          score=0.0, confirmations=0, reason="insufficient_or_stale_orderflow")
        cats = self._categories(window)
        if not cats:
            return SellerExhaustionResult(symbol=symbol, available=False, status=STATUS_UNAVAILABLE,
                                          score=0.0, confirmations=0, reason="no_computable_categories")
        score = sum(cats.values()) / len(cats)
        confirmations = sum(1 for v in cats.values() if v >= _CONFIRM_THRESHOLD)
        persistence_ok = self._persistence_ok(window)
        confirmed = (score >= self.cfg.seller_exhaustion_min_score
                     and confirmations >= self.cfg.seller_exhaustion_min_confirmations
                     and persistence_ok)
        return SellerExhaustionResult(symbol=symbol, available=True, status="OK", score=score,
                                      confirmations=confirmations, categories=cats,
                                      persistence_ok=persistence_ok, confirmed=confirmed,
                                      reason="confirmed" if confirmed else "below_threshold")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "seller_exhaustion or req16 or req17" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add amps_v6/alpha/seller_exhaustion.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): seller-exhaustion scorer (independent-evidence score, confirmations, persistence, UNAVAILABLE)"
```

---

### Task 7: Positive-momentum gate (completed 5m bars)

**Files:**
- Create: `amps_v6/strategy/momentum_gate.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Consumes: `ReplacementConfig.momentum_min_confirmations`; a 5-minute OHLCV `pandas.DataFrame` (columns `open,high,low,close,volume`; optional 1m/15m frames).
- Produces:
  - `MomentumResult(symbol, confirmed, mandatory_ok, plus_di, minus_di, close, bb_mid, momentum_up, confirmations: dict, confirmation_count: int, timeframe_disagreement: bool, confidence: float, reason: str)`
  - `PositiveMomentumGate(cfg)` with `evaluate(symbol, frame_5m, frame_1m=None, frame_15m=None, drop_last_incomplete=True) -> MomentumResult`.
- **Note (DRY):** the repo has DMI/BB math in `trade.di_bb_signal_for_frame`, `multi_timeframe_engine._dmi_adx`, and `backtest_di_bb`. Those are entangled with heavy imports; this gate uses small self-contained Wilder-DMI/SMA helpers to stay import-light and unit-testable offline. Keep the math identical in spirit (Wilder RMA, BB-mid = SMA20).

- [ ] **Step 1: Write the failing tests** (covers required #13, #14, #15)

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
import pandas as pd
from amps_v6.strategy import momentum_gate as MG


def _frame(closes, vols=None):
    vols = vols or [1000] * len(closes)
    rows = []
    for i, c in enumerate(closes):
        rows.append({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": vols[i]})
    return pd.DataFrame(rows)


def _uptrend(n=40):
    return _frame([100 + i * 0.3 for i in range(n)])


def _downtrend(n=40):
    return _frame([100 - i * 0.3 for i in range(n)])


def test_momentum_confirmed_on_uptrend():
    r = MG.PositiveMomentumGate(ReplacementConfig()).evaluate("AAA", _uptrend(), drop_last_incomplete=False)
    assert r.mandatory_ok and r.plus_di > r.minus_di and r.close > r.bb_mid and r.momentum_up
    assert r.confirmation_count >= 2 and r.confirmed


def test_req13_di_minus_dominant_rejected():
    r = MG.PositiveMomentumGate(ReplacementConfig()).evaluate("AAA", _downtrend(), drop_last_incomplete=False)
    assert r.plus_di <= r.minus_di and not r.mandatory_ok and not r.confirmed


def test_req14_price_at_or_below_bb_mid_rejected():
    # flat then a final close exactly at the SMA20 mid -> not strictly above -> reject
    closes = [100.0] * 40
    r = MG.PositiveMomentumGate(ReplacementConfig()).evaluate("AAA", _frame(closes), drop_last_incomplete=False)
    assert r.close <= r.bb_mid and not r.mandatory_ok and not r.confirmed


def test_req15_unfinished_bar_cannot_confirm():
    # 39 completed bars (downtrend) + 1 in-progress spike bar that WOULD look bullish.
    down = [100 - i * 0.3 for i in range(39)]
    frame = _frame(down + [250.0])     # last row is the forming bar
    gate = MG.PositiveMomentumGate(ReplacementConfig())
    with_incomplete = gate.evaluate("AAA", frame, drop_last_incomplete=False)
    completed_only = gate.evaluate("AAA", frame, drop_last_incomplete=True)
    # the forming bar alone must not be allowed to confirm momentum
    assert completed_only.close == pytest.approx(down[-1])   # used the last COMPLETED bar
    assert not completed_only.confirmed


def test_req15b_insufficient_completed_bars_not_confirmed():
    r = MG.PositiveMomentumGate(ReplacementConfig()).evaluate("AAA", _frame([100, 101, 102]), drop_last_incomplete=False)
    assert not r.confirmed and "insufficient" in r.reason


def test_timeframe_disagreement_reduces_confidence():
    gate = MG.PositiveMomentumGate(ReplacementConfig())
    agree = gate.evaluate("AAA", _uptrend(), frame_15m=_uptrend(), drop_last_incomplete=False)
    disagree = gate.evaluate("AAA", _uptrend(), frame_15m=_downtrend(), drop_last_incomplete=False)
    assert disagree.timeframe_disagreement and disagree.confidence < agree.confidence
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "momentum or req13 or req14 or req15 or timeframe" -v`
Expected: FAIL (`ModuleNotFoundError: amps_v6.strategy.momentum_gate`)

- [ ] **Step 3: Implement**

```python
# amps_v6/strategy/momentum_gate.py
"""Positive-momentum gate on COMPLETED 5-minute bars.

Mandatory (all three): DI+ > DI-, close > Bollinger middle (SMA20), and the
latest completed bar closes above the previous completed bar. Plus a configurable
minimum number of additional confirmations (ROC>0, SMA5>SMA20, positive 5m
return, positive 1m return, rising RSI, MACD histogram positive/rising, price>
VWAP, volume confirmation). Only completed bars are used; a forming bar is
dropped when drop_last_incomplete=True so an unfinished bar can never confirm.
Higher-timeframe (15m) contradiction is recorded and reduces confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DI_PERIOD = 14
BB_PERIOD = 20
_MIN_BARS = BB_PERIOD + DI_PERIOD + 1   # need enough completed bars for both


@dataclass
class MomentumResult:
    symbol: str
    confirmed: bool
    mandatory_ok: bool
    plus_di: float
    minus_di: float
    close: float
    bb_mid: float
    momentum_up: bool
    confirmations: dict = field(default_factory=dict)
    confirmation_count: int = 0
    timeframe_disagreement: bool = False
    confidence: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "confirmed": self.confirmed, "mandatory_ok": self.mandatory_ok,
                "plus_di": round(self.plus_di, 2), "minus_di": round(self.minus_di, 2),
                "close": round(self.close, 4), "bb_mid": round(self.bb_mid, 4),
                "momentum_up": self.momentum_up, "confirmations": self.confirmations,
                "confirmation_count": self.confirmation_count,
                "timeframe_disagreement": self.timeframe_disagreement,
                "confidence": round(self.confidence, 3), "reason": self.reason}


def _rma(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


def _dmi(h: pd.Series, l: pd.Series, c: pd.Series, n: int = DI_PERIOD):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, n).replace(0, np.nan)
    pdi = 100 * _rma(pdm, n) / atr
    mdi = 100 * _rma(mdm, n) / atr
    return pdi, mdi


def _completed(frame: pd.DataFrame, drop_last_incomplete: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    f = frame.reset_index(drop=True)
    return f.iloc[:-1].reset_index(drop=True) if (drop_last_incomplete and len(f) > 1) else f


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    gain = _rma(d.clip(lower=0), n)
    loss = _rma(-d.clip(upper=0), n).replace(0, np.nan)
    return 100 - 100 / (1 + gain / loss)


def _htf_bullish(frame: Optional[pd.DataFrame], drop_last_incomplete: bool) -> Optional[bool]:
    f = _completed(frame, drop_last_incomplete)
    if len(f) < _MIN_BARS:
        return None
    c = f["close"].astype(float)
    pdi, mdi = _dmi(f["high"].astype(float), f["low"].astype(float), c)
    bb_mid = c.rolling(BB_PERIOD).mean().iloc[-1]
    return bool(pdi.iloc[-1] > mdi.iloc[-1] and c.iloc[-1] > bb_mid)


class PositiveMomentumGate:
    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, symbol: str, frame_5m: pd.DataFrame, frame_1m: Optional[pd.DataFrame] = None,
                 frame_15m: Optional[pd.DataFrame] = None, drop_last_incomplete: bool = True) -> MomentumResult:
        f = _completed(frame_5m, drop_last_incomplete)
        if len(f) < _MIN_BARS:
            return MomentumResult(symbol, False, False, 0.0, 0.0, 0.0, 0.0, False,
                                  reason=f"insufficient_completed_bars({len(f)}<{_MIN_BARS})")
        c = f["close"].astype(float)
        h, l, v = f["high"].astype(float), f["low"].astype(float), f["volume"].astype(float)
        pdi, mdi = _dmi(h, l, c)
        plus_di, minus_di = float(pdi.iloc[-1]), float(mdi.iloc[-1])
        close = float(c.iloc[-1])
        bb_mid = float(c.rolling(BB_PERIOD).mean().iloc[-1])
        momentum_up = bool(close > float(c.iloc[-2]))

        mandatory_ok = (plus_di > minus_di) and (close > bb_mid) and momentum_up

        sma5 = float(c.rolling(5).mean().iloc[-1])
        sma20 = float(c.rolling(20).mean().iloc[-1])
        roc = float(c.iloc[-1] / c.iloc[-5] - 1.0) if len(c) >= 5 else 0.0
        rsi = _rsi(c)
        rsi_rising = bool(len(rsi.dropna()) >= 2 and rsi.iloc[-1] > rsi.iloc[-2])
        vwap = float((c * v).sum() / v.sum()) if v.sum() > 0 else close
        ret_5m = float(c.iloc[-1] / c.iloc[-2] - 1.0)
        vol_conf = bool(v.iloc[-1] >= v.rolling(20).mean().iloc[-1])
        conf = {
            "roc_positive": roc > 0,
            "sma5_gt_sma20": sma5 > sma20,
            "positive_5m_return": ret_5m > 0,
            "rsi_rising": rsi_rising,
            "price_above_vwap": close > vwap,
            "volume_confirmation": vol_conf,
        }
        if frame_1m is not None:
            f1 = _completed(frame_1m, drop_last_incomplete)
            if len(f1) >= 2:
                conf["positive_1m_return"] = float(f1["close"].iloc[-1]) > float(f1["close"].iloc[-2])
        confirmations = {k: bool(val) for k, val in conf.items()}
        confirmation_count = sum(1 for val in confirmations.values() if val)

        htf = _htf_bullish(frame_15m, drop_last_incomplete)
        disagreement = (htf is False)   # explicit contradiction only; None (no data) is not disagreement

        confirmed = mandatory_ok and confirmation_count >= self.cfg.momentum_min_confirmations
        confidence = max(0.0, min(1.0, 0.4 + 0.1 * confirmation_count - (0.2 if disagreement else 0.0)))
        return MomentumResult(symbol, confirmed, mandatory_ok, plus_di, minus_di, close, bb_mid,
                              momentum_up, confirmations, confirmation_count, disagreement, confidence,
                              reason="confirmed" if confirmed else ("mandatory_failed" if not mandatory_ok else "insufficient_confirmations"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "momentum or req13 or req14 or req15 or timeframe" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add amps_v6/strategy/momentum_gate.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): positive-momentum gate (completed-5m DI/BB/momentum + confirmations + HTF disagreement)"
```

---

### Task 8: Marginal-Sharpe optimizer

**Files:**
- Create: `amps_v6/portfolio/marginal_sharpe.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Consumes: `numpy`; optionally `portfolio_optimizer.build_shrinkage_covariance`, `costs`.
- Produces:
  - `PortfolioState(symbols, weights: dict, expected_daily_returns: dict, cov: np.ndarray, cov_order: list)`
  - `CandidateSharpe(symbol, delta_sharpe, sharpe_with_candidate, sharpe_cash, proposed_weight)`
  - `MarginalSharpeOptimizer(cfg)` with:
    - `portfolio_sharpe(weights: dict, mu: dict, cov, order) -> float` (annualized)
    - `evaluate_candidate(candidate_symbol, candidate_weight, candidate_mu, exited_symbol, state, round_trip_cost_frac=0.0) -> CandidateSharpe`
    - `rank(candidates: list[CandidateSharpe]) -> list[CandidateSharpe]` (descending `delta_sharpe`)
  - `build_state_from_history(weights, returns_df, min_lookback_days=20) -> PortfolioState` helper.

- [ ] **Step 1: Write the failing tests** (covers required #19, #20)

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
import numpy as np
from amps_v6.portfolio import marginal_sharpe as MS


def _state(symbols, weights, mu, cov):
    return MS.PortfolioState(symbols=symbols, weights=dict(zip(symbols, weights)),
                             expected_daily_returns=dict(zip(symbols, mu)),
                             cov=np.array(cov, dtype=float), cov_order=symbols)


def test_req19_sharpe_reducing_candidate_has_nonpositive_delta():
    # hold X (good). Candidate BAD has negative expected return and positive covariance -> hurts Sharpe.
    symbols = ["X", "BAD"]
    cov = [[0.0004, 0.0003], [0.0003, 0.0009]]
    st = _state(symbols, weights=[0.2, 0.0], mu=[0.001, -0.002], cov=cov)
    opt = MS.MarginalSharpeOptimizer(ReplacementConfig())
    cs = opt.evaluate_candidate("BAD", candidate_weight=0.2, candidate_mu=-0.002,
                                exited_symbol=None, state=st)
    assert cs.delta_sharpe <= 0.0


def test_req20_diversifier_beats_higher_standalone_return():
    # Exiting LOSER to cash. Two candidates:
    #   HICORR: higher standalone return but highly correlated with remaining holding KEEP
    #   DIVERSE: lower standalone return but negatively correlated with KEEP
    symbols = ["KEEP", "HICORR", "DIVERSE"]
    mu = [0.0010, 0.0012, 0.0008]
    cov = [
        [0.0004, 0.00038, -0.00030],   # KEEP
        [0.00038, 0.0004, -0.00005],   # HICORR ~ KEEP
        [-0.00030, -0.00005, 0.0004],  # DIVERSE anti-correlated with KEEP
    ]
    st = _state(symbols, weights=[0.2, 0.0, 0.0], mu=mu, cov=cov)
    opt = MS.MarginalSharpeOptimizer(ReplacementConfig())
    hi = opt.evaluate_candidate("HICORR", 0.2, mu[1], exited_symbol=None, state=st)
    dv = opt.evaluate_candidate("DIVERSE", 0.2, mu[2], exited_symbol=None, state=st)
    ranked = opt.rank([hi, dv])
    assert ranked[0].symbol == "DIVERSE"            # diversifier wins on marginal Sharpe
    assert dv.delta_sharpe > hi.delta_sharpe


def test_costs_reduce_delta_sharpe():
    symbols = ["X", "C"]
    cov = [[0.0004, 0.0], [0.0, 0.0004]]
    st = _state(symbols, weights=[0.2, 0.0], mu=[0.001, 0.001], cov=cov)
    opt = MS.MarginalSharpeOptimizer(ReplacementConfig())
    free = opt.evaluate_candidate("C", 0.2, 0.001, None, st, round_trip_cost_frac=0.0)
    costly = opt.evaluate_candidate("C", 0.2, 0.001, None, st, round_trip_cost_frac=0.01)
    assert costly.delta_sharpe < free.delta_sharpe
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "req19 or req20 or costs_reduce" -v`
Expected: FAIL (`ModuleNotFoundError: amps_v6.portfolio.marginal_sharpe`)

- [ ] **Step 3: Implement**

```python
# amps_v6/portfolio/marginal_sharpe.py
"""Marginal portfolio-Sharpe optimizer for replacement ranking.

For each qualifying candidate j:
    delta_sharpe[j] = sharpe(portfolio_after_candidate_j) - sharpe(portfolio_after_exit_to_cash)

Candidates are ranked by delta_sharpe -- NOT by standalone Sharpe, raw expected
return, momentum, or exhaustion. A diversifier (low/negative covariance with the
book) can beat a higher-standalone-return name. Costs enter as a return haircut
on the candidate. Cash (exit-to-cash) is the baseline, so a candidate that does
not raise Sharpe yields delta_sharpe <= 0 and is rejected downstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TRADING_DAYS = 252


@dataclass
class PortfolioState:
    symbols: list
    weights: dict
    expected_daily_returns: dict
    cov: np.ndarray
    cov_order: list


@dataclass
class CandidateSharpe:
    symbol: str
    delta_sharpe: float
    sharpe_with_candidate: float
    sharpe_cash: float
    proposed_weight: float


class MarginalSharpeOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def portfolio_sharpe(self, weights: dict, mu: dict, cov: np.ndarray, order: list) -> float:
        w = np.array([weights.get(s, 0.0) for s in order], dtype=float)
        if not np.any(w):
            return 0.0
        m = np.array([mu.get(s, 0.0) for s in order], dtype=float)
        daily_mean = float(w @ m)
        var = float(w @ np.asarray(cov, dtype=float) @ w)
        if var <= 1e-18:
            return 0.0
        return daily_mean / math.sqrt(var) * math.sqrt(TRADING_DAYS)

    def evaluate_candidate(self, candidate_symbol: str, candidate_weight: float, candidate_mu: float,
                           exited_symbol, state: PortfolioState, round_trip_cost_frac: float = 0.0) -> CandidateSharpe:
        order = list(state.cov_order)
        mu = dict(state.expected_daily_returns)
        # baseline: exit the stopped name to cash
        w_cash = {s: wv for s, wv in state.weights.items() if s != exited_symbol}
        sharpe_cash = self.portfolio_sharpe(w_cash, mu, state.cov, order)
        # with candidate added at proposed weight; cost drag haircuts its expected return
        w_cand = dict(w_cash)
        w_cand[candidate_symbol] = w_cand.get(candidate_symbol, 0.0) + candidate_weight
        mu_cand = dict(mu)
        haircut = (round_trip_cost_frac / max(candidate_weight, 1e-9)) if candidate_weight > 0 else 0.0
        mu_cand[candidate_symbol] = candidate_mu - haircut
        if candidate_symbol not in order:   # candidate absent from cov -> assume uncorrelated at book avg var
            order = order + [candidate_symbol]
            cov = _augment_cov(state.cov, state.cov_order, candidate_symbol)
        else:
            cov = np.asarray(state.cov, dtype=float)
        sharpe_cand = self.portfolio_sharpe(w_cand, mu_cand, cov, order)
        return CandidateSharpe(candidate_symbol, sharpe_cand - sharpe_cash, sharpe_cand, sharpe_cash, candidate_weight)

    def rank(self, candidates: list) -> list:
        return sorted(candidates, key=lambda c: c.delta_sharpe, reverse=True)


def _augment_cov(cov: np.ndarray, order: list, new_symbol: str) -> np.ndarray:
    """Append an uncorrelated row/col for a symbol missing from cov, using the mean
    diagonal variance as its variance and zero covariance with the book."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    var = float(np.mean(np.diag(cov))) if n else 0.0004
    out = np.zeros((n + 1, n + 1))
    out[:n, :n] = cov
    out[n, n] = var
    return out


def build_state_from_history(weights: dict, returns_df, min_lookback_days: int = 20) -> PortfolioState:
    """Build a PortfolioState from a returns-history DataFrame (columns = symbols).
    Uses portfolio_optimizer.build_shrinkage_covariance; mu = mean daily return."""
    import portfolio_optimizer as po
    result = po.build_shrinkage_covariance(returns_df, min_lookback_days=min_lookback_days)
    if result is None:
        symbols = list(returns_df.columns)
        cov = np.cov(returns_df.fillna(0.0).values, rowvar=False)
        cov = np.atleast_2d(cov)
        order = symbols
    else:
        cov, order = result
    mu = {s: float(returns_df[s].mean()) for s in order}
    return PortfolioState(symbols=list(weights.keys()), weights=dict(weights),
                          expected_daily_returns=mu, cov=np.asarray(cov, dtype=float), cov_order=list(order))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "req19 or req20 or costs_reduce" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add amps_v6/portfolio/marginal_sharpe.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): marginal portfolio-Sharpe optimizer (delta_sharpe ranking, covariance-aware, cost-haircut)"
```

---

### Task 9: Anti-churn registry

**Files:**
- Create: `amps_v6/execution/anti_churn.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Consumes: `ReplacementConfig` (`exited_symbol_cooldown_seconds`, `max_replacements_per_cycle`, `max_symbol_replacements_per_session`).
- Produces: `AntiChurnRegistry(cfg, env="paper")` with `start_cooldown(symbol, now)`, `in_cooldown(symbol, now) -> bool`, `exit_allowed(symbol) -> bool` (always True), `can_replace_symbol(symbol) -> bool`, `record_replacement(symbol)`, `begin_cycle()`, `cycle_budget_remaining() -> int`, `consume_cycle_slot()`, `save(path=None)`, classmethod `load(cfg, env, path=None)`, property `state_path`.

- [ ] **Step 1: Write the failing tests** (covers required #23, #24)

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
from amps_v6.execution import anti_churn as AC


def test_req23_cooldown_prevents_immediate_reentry():
    reg = AC.AntiChurnRegistry(ReplacementConfig(exited_symbol_cooldown_seconds=1800), env="paper")
    reg.start_cooldown("XYZ", now=1000.0)
    assert reg.in_cooldown("XYZ", now=1500.0) is True       # 500s < 1800s
    assert reg.in_cooldown("XYZ", now=3000.0) is False      # 2000s > 1800s
    assert reg.in_cooldown("OTHER", now=1500.0) is False


def test_req24_cooldown_never_blocks_exit():
    reg = AC.AntiChurnRegistry(ReplacementConfig(), env="paper")
    reg.start_cooldown("XYZ", now=1000.0)
    # a symbol in cooldown may still ALWAYS be exited
    assert reg.exit_allowed("XYZ") is True
    assert reg.in_cooldown("XYZ", now=1001.0) is True        # still cooling down for re-ENTRY


def test_session_replacement_cap():
    reg = AC.AntiChurnRegistry(ReplacementConfig(max_symbol_replacements_per_session=2), env="paper")
    assert reg.can_replace_symbol("ABC")
    reg.record_replacement("ABC"); reg.record_replacement("ABC")
    assert not reg.can_replace_symbol("ABC")


def test_cycle_budget():
    reg = AC.AntiChurnRegistry(ReplacementConfig(max_replacements_per_cycle=2), env="paper")
    reg.begin_cycle()
    assert reg.cycle_budget_remaining() == 2
    reg.consume_cycle_slot()
    assert reg.cycle_budget_remaining() == 1
    reg.begin_cycle()
    assert reg.cycle_budget_remaining() == 2   # reset each cycle


def test_anti_churn_env_namespaced():
    assert AC.AntiChurnRegistry(ReplacementConfig(), "paper").state_path != AC.AntiChurnRegistry(ReplacementConfig(), "live").state_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "req23 or req24 or session_replacement or cycle_budget or anti_churn_env" -v`
Expected: FAIL (`ModuleNotFoundError: amps_v6.execution.anti_churn`)

- [ ] **Step 3: Implement**

```python
# amps_v6/execution/anti_churn.py
"""Anti-churn controls: exited-symbol cooldown, per-cycle and per-session
replacement caps. Cooldowns block RE-ENTRY only -- they MUST NEVER block a
protective exit (exit_allowed is unconditionally True). PAPER/LIVE-namespaced
persistence so the two environments never share churn state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "logs" / "execution"


class AntiChurnRegistry:
    def __init__(self, cfg, env: str = "paper"):
        if env not in ("paper", "live"):
            raise ValueError(f"env must be 'paper' or 'live', got {env!r}")
        self.cfg = cfg
        self.env = env
        self._cooldown_until: dict[str, float] = {}
        self._session_counts: dict[str, int] = {}
        self._cycle_count: int = 0

    @property
    def state_path(self) -> Path:
        return STATE_DIR / f"amps_v6_anti_churn_{self.env}.json"

    # --- cooldown (re-entry only) ---
    def start_cooldown(self, symbol: str, now: float) -> None:
        self._cooldown_until[symbol] = now + self.cfg.exited_symbol_cooldown_seconds

    def in_cooldown(self, symbol: str, now: float) -> bool:
        return now < self._cooldown_until.get(symbol, 0.0)

    def exit_allowed(self, symbol: str) -> bool:
        """Protective exits are ALWAYS allowed, cooldown or not. Invariant."""
        return True

    # --- session cap ---
    def can_replace_symbol(self, symbol: str) -> bool:
        return self._session_counts.get(symbol, 0) < self.cfg.max_symbol_replacements_per_session

    def record_replacement(self, symbol: str) -> None:
        self._session_counts[symbol] = self._session_counts.get(symbol, 0) + 1

    # --- cycle cap ---
    def begin_cycle(self) -> None:
        self._cycle_count = 0

    def consume_cycle_slot(self) -> None:
        self._cycle_count += 1

    def cycle_budget_remaining(self) -> int:
        return max(0, self.cfg.max_replacements_per_cycle - self._cycle_count)

    # --- persistence (cooldowns + session counts survive restart; cycle count is ephemeral) ---
    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"env": self.env, "cooldown_until": self._cooldown_until,
                                    "session_counts": self._session_counts}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, cfg, env: str, path: Optional[Path] = None) -> "AntiChurnRegistry":
        reg = cls(cfg, env=env)
        path = path or reg.state_path
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            reg._cooldown_until = {k: float(v) for k, v in data.get("cooldown_until", {}).items()}
            reg._session_counts = {k: int(v) for k, v in data.get("session_counts", {}).items()}
        return reg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "req23 or req24 or session_replacement or cycle_budget or anti_churn_env" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add amps_v6/execution/anti_churn.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): anti-churn registry (cooldown re-entry only, cycle/session caps, env-namespaced)"
```

---

### Task 10: Replacement state machine

**Files:**
- Create: `amps_v6/execution/replacement_state_machine.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Produces:
  - State constants: `HELD, EXIT_REQUIRED, EXIT_SUBMITTED, EXIT_PARTIALLY_FILLED, EXIT_FILLED, REPLACEMENT_EVALUATION, REPLACEMENT_APPROVED, CASH_HELD, REPLACEMENT_SUBMITTED, REPLACEMENT_PARTIALLY_FILLED, REPLACEMENT_FILLED, PROTECTION_VERIFIED`.
  - `ReplacementStateMachine(env="paper")` with `open(position_id, symbol) -> dict`, `advance(position_id, to_state) -> bool`, `can_advance(position_id, to_state) -> bool`, `get(position_id) -> dict|None`, `exit_cid(position_id) -> str`, `replacement_cid(position_id) -> str`, `set_replacement(position_id, symbol)`, `unresolved() -> list[str]`, `save(path=None)`, classmethod `load(env, path=None)`, property `state_path`.

- [ ] **Step 1: Write the failing tests** (covers required #27, #28)

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
from amps_v6.execution import replacement_state_machine as RSM


def test_state_machine_happy_path():
    m = RSM.ReplacementStateMachine(env="paper")
    m.open("pos1", "XYZ")
    assert m.get("pos1")["state"] == RSM.HELD
    for s in (RSM.EXIT_REQUIRED, RSM.EXIT_SUBMITTED, RSM.EXIT_FILLED, RSM.REPLACEMENT_EVALUATION,
              RSM.REPLACEMENT_APPROVED, RSM.REPLACEMENT_SUBMITTED, RSM.REPLACEMENT_FILLED, RSM.PROTECTION_VERIFIED):
        assert m.advance("pos1", s)
    assert m.get("pos1")["state"] == RSM.PROTECTION_VERIFIED
    assert "pos1" not in m.unresolved()


def test_state_machine_illegal_transition_rejected():
    m = RSM.ReplacementStateMachine(env="paper")
    m.open("pos1", "XYZ")
    assert not m.advance("pos1", RSM.REPLACEMENT_FILLED)   # cannot jump from HELD


def test_state_machine_cash_held_can_reevaluate():
    m = RSM.ReplacementStateMachine(env="paper")
    m.open("pos1", "XYZ")
    m.advance("pos1", RSM.EXIT_REQUIRED); m.advance("pos1", RSM.EXIT_SUBMITTED)
    m.advance("pos1", RSM.EXIT_FILLED); m.advance("pos1", RSM.REPLACEMENT_EVALUATION)
    assert m.advance("pos1", RSM.CASH_HELD)
    assert m.advance("pos1", RSM.REPLACEMENT_EVALUATION)   # reconsider later


def test_req27_idempotent_client_order_ids():
    m = RSM.ReplacementStateMachine(env="paper")
    m.open("pos1", "XYZ")
    assert m.exit_cid("pos1") == m.exit_cid("pos1")           # stable
    assert m.replacement_cid("pos1") == m.replacement_cid("pos1")
    assert m.exit_cid("pos1") != m.replacement_cid("pos1")
    # re-open with same id does NOT reset state/cids
    before = m.get("pos1")["state"]
    m.advance("pos1", RSM.EXIT_REQUIRED)
    m.open("pos1", "XYZ")
    assert m.get("pos1")["state"] == RSM.EXIT_REQUIRED and before == RSM.HELD


def test_req28_restart_restores_unresolved_state(tmp_path):
    m = RSM.ReplacementStateMachine(env="paper")
    m.open("pos1", "XYZ")
    m.advance("pos1", RSM.EXIT_REQUIRED); m.advance("pos1", RSM.EXIT_SUBMITTED)
    p = tmp_path / "rsm.json"
    m.save(p)
    back = RSM.ReplacementStateMachine.load("paper", p)
    assert back.get("pos1")["state"] == RSM.EXIT_SUBMITTED
    assert "pos1" in back.unresolved()


def test_state_machine_env_namespaced():
    assert RSM.ReplacementStateMachine("paper").state_path != RSM.ReplacementStateMachine("live").state_path
    assert "paper" in RSM.ReplacementStateMachine("paper").exit_cid_for("pos1")  # env in the audit id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "state_machine or req27 or req28" -v`
Expected: FAIL (`ModuleNotFoundError: amps_v6.execution.replacement_state_machine`)

- [ ] **Step 3: Implement**

```python
# amps_v6/execution/replacement_state_machine.py
"""Replacement lifecycle state machine (one record per stopped position).

HELD -> EXIT_REQUIRED -> EXIT_SUBMITTED -> [EXIT_PARTIALLY_FILLED] -> EXIT_FILLED
     -> REPLACEMENT_EVALUATION -> (REPLACEMENT_APPROVED | CASH_HELD)
     -> REPLACEMENT_SUBMITTED -> [REPLACEMENT_PARTIALLY_FILLED] -> REPLACEMENT_FILLED
     -> PROTECTION_VERIFIED (terminal). CASH_HELD loops back to REPLACEMENT_EVALUATION.

Idempotent, deterministic, env-namespaced client-order-ids (also isolate the
PAPER/LIVE audit trail). Persisted so a restart restores unresolved work.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "logs" / "execution"

HELD = "HELD"
EXIT_REQUIRED = "EXIT_REQUIRED"
EXIT_SUBMITTED = "EXIT_SUBMITTED"
EXIT_PARTIALLY_FILLED = "EXIT_PARTIALLY_FILLED"
EXIT_FILLED = "EXIT_FILLED"
REPLACEMENT_EVALUATION = "REPLACEMENT_EVALUATION"
REPLACEMENT_APPROVED = "REPLACEMENT_APPROVED"
CASH_HELD = "CASH_HELD"
REPLACEMENT_SUBMITTED = "REPLACEMENT_SUBMITTED"
REPLACEMENT_PARTIALLY_FILLED = "REPLACEMENT_PARTIALLY_FILLED"
REPLACEMENT_FILLED = "REPLACEMENT_FILLED"
PROTECTION_VERIFIED = "PROTECTION_VERIFIED"

TRANSITIONS: dict[str, set[str]] = {
    HELD: {EXIT_REQUIRED},
    EXIT_REQUIRED: {EXIT_SUBMITTED},
    EXIT_SUBMITTED: {EXIT_PARTIALLY_FILLED, EXIT_FILLED},
    EXIT_PARTIALLY_FILLED: {EXIT_PARTIALLY_FILLED, EXIT_FILLED},
    EXIT_FILLED: {REPLACEMENT_EVALUATION},
    REPLACEMENT_EVALUATION: {REPLACEMENT_APPROVED, CASH_HELD},
    REPLACEMENT_APPROVED: {REPLACEMENT_SUBMITTED},
    CASH_HELD: {REPLACEMENT_EVALUATION},
    REPLACEMENT_SUBMITTED: {REPLACEMENT_PARTIALLY_FILLED, REPLACEMENT_FILLED},
    REPLACEMENT_PARTIALLY_FILLED: {REPLACEMENT_PARTIALLY_FILLED, REPLACEMENT_FILLED},
    REPLACEMENT_FILLED: {PROTECTION_VERIFIED},
    PROTECTION_VERIFIED: set(),
}
TERMINAL = {PROTECTION_VERIFIED}


class ReplacementStateMachine:
    def __init__(self, env: str = "paper"):
        if env not in ("paper", "live"):
            raise ValueError(f"env must be 'paper' or 'live', got {env!r}")
        self.env = env
        self._records: dict[str, dict] = {}

    @property
    def state_path(self) -> Path:
        return STATE_DIR / f"amps_v6_replacement_state_{self.env}.json"

    def exit_cid_for(self, position_id: str) -> str:
        return f"amps-exit-{self.env}-{position_id}"

    def replacement_cid_for(self, position_id: str) -> str:
        return f"amps-repl-{self.env}-{position_id}"

    def open(self, position_id: str, symbol: str) -> dict:
        """Idempotent: an existing record is returned unchanged (never reset)."""
        if position_id in self._records:
            return self._records[position_id]
        rec = {"position_id": position_id, "symbol": symbol, "state": HELD,
               "replacement_symbol": None, "exit_cid": self.exit_cid_for(position_id),
               "replacement_cid": self.replacement_cid_for(position_id), "history": []}
        self._records[position_id] = rec
        return rec

    def get(self, position_id: str) -> Optional[dict]:
        return self._records.get(position_id)

    def can_advance(self, position_id: str, to_state: str) -> bool:
        rec = self._records.get(position_id)
        return bool(rec) and to_state in TRANSITIONS.get(rec["state"], set())

    def advance(self, position_id: str, to_state: str) -> bool:
        if not self.can_advance(position_id, to_state):
            return False
        rec = self._records[position_id]
        rec["history"].append((rec["state"], to_state))
        rec["state"] = to_state
        return True

    def set_replacement(self, position_id: str, symbol: str) -> None:
        if position_id in self._records:
            self._records[position_id]["replacement_symbol"] = symbol

    def exit_cid(self, position_id: str) -> str:
        return self.open(position_id, "").get("exit_cid", self.exit_cid_for(position_id))

    def replacement_cid(self, position_id: str) -> str:
        return self.open(position_id, "").get("replacement_cid", self.replacement_cid_for(position_id))

    def unresolved(self) -> list[str]:
        return [pid for pid, r in self._records.items() if r["state"] not in TERMINAL]

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"env": self.env, "records": self._records}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, env: str, path: Optional[Path] = None) -> "ReplacementStateMachine":
        m = cls(env=env)
        path = path or m.state_path
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            m._records = data.get("records", {})
        return m
```

Note: `exit_cid`/`replacement_cid` call `open(position_id, "")`; since `open` is idempotent it will NOT overwrite an existing record (so `test_req27` where `open("pos1","XYZ")` precedes is safe — the `""` symbol only applies if the record did not already exist).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "state_machine or req27 or req28" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add amps_v6/execution/replacement_state_machine.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): replacement state machine (HELD..PROTECTION_VERIFIED, idempotent env cids, restart-restore)"
```

---

### Task 11: Replacement optimizer orchestrator

**Files:**
- Create: `amps_v6/portfolio/replacement_optimizer.py`
- Test: `tests/test_amps_v6_loss_removal_replacement.py`

**Interfaces:**
- Consumes: `ReplacementConfig`; `marginal_sharpe.MarginalSharpeOptimizer` (duck-typed: `.evaluate_candidate(sym, weight, mu, exited, state, round_trip_cost_frac) -> CandidateSharpe`); `anti_churn.AntiChurnRegistry`; `amps_v6.alpha.entry_alpha.ALPHA_CONFIRMED`; `universe.load_canonical_universe` + `universe.classify_symbol` (for `build_pool`); an optional `risk_approver(candidate, proposed_weight) -> (approved: bool, approved_weight: float, reason: str)`.
- Produces:
  - `CandidateInputs(symbol, seller_exhaustion, momentum, alpha_decision, expected_net_return, candidate_mu, proposed_weight, spread_bps=10.0, adv=1e9, exposure_group="equity", is_leveraged_or_inverse=False, broker_restricted=False, regime_permitted=True, data_valid=True, signal_valid_at_submit=True, in_universe=True, round_trip_cost_frac=0.0)`
  - `ReplacementDecision(exited_symbol, candidate, action, delta_sharpe, proposed_weight, rejected: dict, reason)`
  - `ReplacementOptimizer(cfg, marginal_sharpe_opt, anti_churn, risk_approver=None, max_spread_bps=50.0, min_adv=0.0, leveraged_cap=0.15)` with:
    - `evaluate_replacement(exited_symbol, candidates, state, now, exit_proceeds_confirmed, leveraged_exposure_current=0.0) -> ReplacementDecision`
    - `evaluate_joint(exits, candidates_by_exit, state, now, buying_power, equity, exit_proceeds_confirmed=True, leveraged_exposure_current=0.0) -> list[ReplacementDecision]`
  - `build_pool(exited_symbol, now, anti_churn, loader=None) -> list[str]` (universe.txt only, minus exited + cooldown)

- [ ] **Step 1: Write the failing tests** (covers required #10, #11, #12, #18, #21, #22, #25, #26)

```python
# append to tests/test_amps_v6_loss_removal_replacement.py
from amps_v6.portfolio import replacement_optimizer as RO
from amps_v6.alpha import entry_alpha as EA


class _FakeSharpe:
    """Duck-typed MarginalSharpeOptimizer: delta_sharpe preset per symbol."""
    def __init__(self, deltas):
        self.deltas = deltas

    def evaluate_candidate(self, sym, weight, mu, exited, state, round_trip_cost_frac=0.0):
        from amps_v6.portfolio.marginal_sharpe import CandidateSharpe
        d = self.deltas.get(sym, 0.0) - round_trip_cost_frac
        return CandidateSharpe(sym, d, d, 0.0, weight)


def _good_se():
    return SE.SellerExhaustionResult("C", available=True, status="OK", score=0.8, confirmations=4,
                                     persistence_ok=True, confirmed=True, reason="confirmed")


def _bad_se():
    return SE.SellerExhaustionResult("C", available=True, status="OK", score=0.2, confirmations=1,
                                     persistence_ok=False, confirmed=False, reason="below_threshold")


def _good_mom():
    return MG.MomentumResult("C", confirmed=True, mandatory_ok=True, plus_di=28, minus_di=18, close=101,
                             bb_mid=100, momentum_up=True, confirmation_count=3, confidence=0.7, reason="confirmed")


def _bad_mom():
    return MG.MomentumResult("C", confirmed=False, mandatory_ok=False, plus_di=15, minus_di=25, close=99,
                             bb_mid=100, momentum_up=False, confirmation_count=0, confidence=0.2, reason="mandatory_failed")


def _cand(symbol="C", **kw):
    base = dict(symbol=symbol, seller_exhaustion=_good_se(), momentum=_good_mom(),
                alpha_decision=EA.ALPHA_CONFIRMED, expected_net_return=0.004, candidate_mu=0.001,
                proposed_weight=0.1)
    base.update(kw)
    return RO.CandidateInputs(**base)


def _opt(deltas=None, **kw):
    return RO.ReplacementOptimizer(ReplacementConfig(), _FakeSharpe(deltas or {"C": 0.2}),
                                   AC.AntiChurnRegistry(ReplacementConfig(), "paper"), **kw)


def _state():
    return MS.PortfolioState(["KEEP"], {"KEEP": 0.2}, {"KEEP": 0.001}, np.array([[0.0004]]), ["KEEP"])


def test_fully_qualified_candidate_is_replaced():
    d = _opt().evaluate_replacement("LOSER", [_cand()], _state(), now=1000.0, exit_proceeds_confirmed=True)
    assert d.action == "REPLACE" and d.candidate == "C" and d.delta_sharpe > 0


def test_req10_replacement_waits_for_confirmed_proceeds():
    d = _opt().evaluate_replacement("LOSER", [_cand()], _state(), now=1000.0, exit_proceeds_confirmed=False)
    assert d.action == "HOLD_CASH" and d.reason == "AWAITING_EXIT_PROCEEDS"


def test_req11_exhaustion_without_momentum_rejected():
    d = _opt().evaluate_replacement("LOSER", [_cand(momentum=_bad_mom())], _state(), 1000.0, True)
    assert d.action == "HOLD_CASH" and "momentum" in d.rejected["C"]


def test_req12_momentum_without_exhaustion_rejected():
    d = _opt().evaluate_replacement("LOSER", [_cand(seller_exhaustion=_bad_se())], _state(), 1000.0, True)
    assert d.action == "HOLD_CASH" and "seller_exhaustion" in d.rejected["C"]


def test_req18_alpha_must_be_confirmed():
    d = _opt().evaluate_replacement("LOSER", [_cand(alpha_decision=EA.ALPHA_WEAK)], _state(), 1000.0, True)
    assert d.action == "HOLD_CASH" and d.rejected["C"] == "alpha_not_confirmed"


def test_nonpositive_net_expectancy_rejected():
    d = _opt().evaluate_replacement("LOSER", [_cand(expected_net_return=-0.001)], _state(), 1000.0, True)
    assert d.rejected["C"] == "nonpositive_net_expectancy"


def test_req21_cash_when_nothing_qualifies():
    d = _opt().evaluate_replacement("LOSER", [_cand(momentum=_bad_mom()), _cand(symbol="D", alpha_decision=EA.NO_ALPHA_EVIDENCE)],
                                    _state(), 1000.0, True)
    assert d.action == "HOLD_CASH" and d.reason == "NO_QUALIFYING_REPLACEMENT" and d.candidate is None


def test_sharpe_below_min_improvement_rejected():
    d = _opt(deltas={"C": 0.01}).evaluate_replacement("LOSER", [_cand()], _state(), 1000.0, True)   # < 0.05 default
    assert d.action == "HOLD_CASH" and d.rejected["C"] == "insufficient_marginal_sharpe"


def test_req25_leveraged_inverse_capped():
    d = _opt().evaluate_replacement("LOSER", [_cand(is_leveraged_or_inverse=True, proposed_weight=0.1)],
                                    _state(), 1000.0, True, leveraged_exposure_current=0.10)   # 0.10+0.10 > 0.15 cap
    assert d.rejected["C"] == "leveraged_inverse_cap"


def test_req26_broker_restriction_blocks():
    d = _opt().evaluate_replacement("LOSER", [_cand(broker_restricted=True)], _state(), 1000.0, True)
    assert d.rejected["C"] == "broker_restricted"


def test_req17_risk_can_reduce_or_reject():
    def _risk(cand, w):
        return (False, 0.0, "risk_rejected")
    d = _opt(risk_approver=_risk).evaluate_replacement("LOSER", [_cand()], _state(), 1000.0, True)
    assert d.action == "HOLD_CASH" and d.rejected["C"] == "risk_rejected"


def test_req22_joint_multi_exit_no_bp_double_spend():
    # two exits, each wants a full-BP replacement; only one can be funded
    opt = _opt(deltas={"A": 0.3, "B": 0.2})
    cands = {
        "LOSER1": [_cand(symbol="A", proposed_weight=1.0)],
        "LOSER2": [_cand(symbol="B", proposed_weight=1.0)],
    }
    decs = opt.evaluate_joint(["LOSER1", "LOSER2"], cands, _state(), now=1000.0,
                              buying_power=10_000.0, equity=10_000.0, exit_proceeds_confirmed=True)
    replaced = [d for d in decs if d.action == "REPLACE"]
    cash = [d for d in decs if d.action == "HOLD_CASH"]
    assert len(replaced) == 1 and replaced[0].candidate == "A"   # higher delta funded first
    assert len(cash) == 1 and cash[0].reason == "INSUFFICIENT_BUYING_POWER"


def test_build_pool_excludes_exited_and_cooldown():
    reg = AC.AntiChurnRegistry(ReplacementConfig(), "paper")
    reg.start_cooldown("TQQQ", now=1000.0)
    pool = RO.build_pool("INTC", now=1000.0, anti_churn=reg, loader=lambda: ["INTC", "TQQQ", "AMD", "NVDA"])
    assert "INTC" not in pool and "TQQQ" not in pool and "AMD" in pool and "NVDA" in pool
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "qualified_candidate or req10 or req11 or req12 or req18 or req21 or req22 or req25 or req26 or req17 or net_expectancy or sharpe_below or build_pool" -v`
Expected: FAIL (`ModuleNotFoundError: amps_v6.portfolio.replacement_optimizer`)

- [ ] **Step 3: Implement**

```python
# amps_v6/portfolio/replacement_optimizer.py
"""Replacement optimizer orchestrator.

Builds the candidate pool from universe.txt ONLY, runs the full 19-point
eligibility conjunction per candidate, ranks the survivors by MARGINAL portfolio
Sharpe (delta_sharpe) -- never by exhaustion/momentum/return alone -- lets
deterministic risk reduce/reject, and returns a replacement or CASH. Cash is
always valid; replacements are never forced. Joint multi-exit evaluation funds
by descending delta_sharpe without double-spending buying power.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from amps_v6.alpha import entry_alpha as EA


@dataclass
class CandidateInputs:
    symbol: str
    seller_exhaustion: object          # SellerExhaustionResult
    momentum: object                   # MomentumResult
    alpha_decision: str
    expected_net_return: float         # after costs, fraction
    candidate_mu: float                # expected daily return (for marginal Sharpe)
    proposed_weight: float
    spread_bps: float = 10.0
    adv: float = 1e9
    exposure_group: str = "equity"
    is_leveraged_or_inverse: bool = False
    broker_restricted: bool = False
    regime_permitted: bool = True
    data_valid: bool = True
    signal_valid_at_submit: bool = True
    in_universe: bool = True
    round_trip_cost_frac: float = 0.0


@dataclass
class ReplacementDecision:
    exited_symbol: str
    candidate: Optional[str]
    action: str                        # "REPLACE" | "HOLD_CASH"
    delta_sharpe: float = 0.0
    proposed_weight: float = 0.0
    rejected: dict = field(default_factory=dict)
    reason: str = ""


def build_pool(exited_symbol: str, now: float, anti_churn, loader: Optional[Callable] = None) -> list[str]:
    """Pool = universe.txt ONLY, minus the exited symbol and any symbol in cooldown.
    No S&P/Nasdaq, no bullish_*.csv, no hard-coded fallback."""
    if loader is None:
        import universe
        loader = universe.load_canonical_universe
    syms = loader()
    return [s for s in syms if s != exited_symbol and not anti_churn.in_cooldown(s, now)]


class ReplacementOptimizer:
    def __init__(self, cfg, marginal_sharpe_opt, anti_churn, risk_approver: Optional[Callable] = None,
                 max_spread_bps: float = 50.0, min_adv: float = 0.0, leveraged_cap: float = 0.15):
        self.cfg = cfg
        self.ms = marginal_sharpe_opt
        self.anti_churn = anti_churn
        self.risk_approver = risk_approver
        self.max_spread_bps = max_spread_bps
        self.min_adv = min_adv
        self.leveraged_cap = leveraged_cap

    def _eligible(self, c: CandidateInputs, now: float, leveraged_exposure_current: float) -> tuple[bool, str]:
        """19-point conjunction (gates 1-11,13-16,18-19). Marginal-Sharpe (12),
        concentration/cov (15) and deterministic risk (17) are applied after ranking."""
        if not c.in_universe:
            return False, "not_in_universe"
        if self.anti_churn.in_cooldown(c.symbol, now):
            return False, "cooldown"
        if not self.anti_churn.can_replace_symbol(c.symbol):
            return False, "session_replacement_cap"
        if not c.data_valid:
            return False, "data_invalid"
        se = c.seller_exhaustion
        if not getattr(se, "available", False):
            return False, "seller_exhaustion_unavailable"
        if not getattr(se, "confirmed", False):
            return False, "seller_exhaustion_not_confirmed"
        mom = c.momentum
        if not getattr(mom, "mandatory_ok", False):
            return False, f"momentum_mandatory_failed({getattr(mom, 'reason', '')})"
        if not getattr(mom, "confirmed", False):
            return False, "momentum_not_confirmed"
        if c.alpha_decision != EA.ALPHA_CONFIRMED:
            return False, "alpha_not_confirmed"
        if c.expected_net_return <= 0:
            return False, "nonpositive_net_expectancy"
        if not c.regime_permitted:
            return False, "regime_blocked"
        if c.spread_bps > self.max_spread_bps:
            return False, "spread_too_wide"
        if c.adv < self.min_adv:
            return False, "insufficient_liquidity"
        if c.is_leveraged_or_inverse and (leveraged_exposure_current + c.proposed_weight) > self.leveraged_cap:
            return False, "leveraged_inverse_cap"
        if c.broker_restricted:
            return False, "broker_restricted"
        if not c.signal_valid_at_submit:
            return False, "signal_expired"
        return True, "eligible"

    def evaluate_replacement(self, exited_symbol: str, candidates: list, state, now: float,
                             exit_proceeds_confirmed: bool, leveraged_exposure_current: float = 0.0) -> ReplacementDecision:
        if not exit_proceeds_confirmed:
            return ReplacementDecision(exited_symbol, None, "HOLD_CASH", reason="AWAITING_EXIT_PROCEEDS")
        rejected: dict = {}
        scored = []
        for c in candidates:
            ok, reason = self._eligible(c, now, leveraged_exposure_current)
            if not ok:
                rejected[c.symbol] = reason
                continue
            cs = self.ms.evaluate_candidate(c.symbol, c.proposed_weight, c.candidate_mu, exited_symbol,
                                            state, round_trip_cost_frac=c.round_trip_cost_frac)
            if cs.delta_sharpe < self.cfg.replacement_min_sharpe_improvement:
                rejected[c.symbol] = "insufficient_marginal_sharpe"
                continue
            scored.append((cs.delta_sharpe, c, cs))
        scored.sort(key=lambda t: t[0], reverse=True)   # rank by marginal Sharpe
        for delta, c, cs in scored:
            weight = c.proposed_weight
            if self.risk_approver is not None:
                approved, approved_weight, risk_reason = self.risk_approver(c, weight)
                if not approved or approved_weight <= 0:
                    rejected[c.symbol] = risk_reason or "risk_rejected"
                    continue
                weight = approved_weight
            return ReplacementDecision(exited_symbol, c.symbol, "REPLACE", delta_sharpe=delta,
                                       proposed_weight=weight, rejected=rejected, reason="REPLACE")
        return ReplacementDecision(exited_symbol, None, "HOLD_CASH", rejected=rejected,
                                   reason="NO_QUALIFYING_REPLACEMENT")

    def evaluate_joint(self, exits: list, candidates_by_exit: dict, state, now: float, buying_power: float,
                       equity: float, exit_proceeds_confirmed: bool = True,
                       leveraged_exposure_current: float = 0.0) -> list:
        """One joint pass over simultaneous stop-outs. Funds by descending
        delta_sharpe; each funded replacement consumes its notional from the
        shared buying power so two exits can never double-spend it. Respects the
        per-cycle replacement budget."""
        # First, get each exit's best candidate decision independently.
        prelim = []
        for ex in exits:
            d = self.evaluate_replacement(ex, candidates_by_exit.get(ex, []), state, now,
                                          exit_proceeds_confirmed, leveraged_exposure_current)
            prelim.append(d)
        # Fund REPLACE decisions by descending delta_sharpe within shared BP + cycle budget.
        replace = sorted([d for d in prelim if d.action == "REPLACE"], key=lambda d: d.delta_sharpe, reverse=True)
        cash = [d for d in prelim if d.action != "REPLACE"]
        remaining_bp = buying_power
        funded = []
        for d in replace:
            notional = d.proposed_weight * equity
            if self.anti_churn.cycle_budget_remaining() <= 0:
                funded.append(ReplacementDecision(d.exited_symbol, None, "HOLD_CASH", rejected=d.rejected,
                                                  reason="CYCLE_REPLACEMENT_LIMIT"))
                continue
            if notional > remaining_bp + 1e-6:
                funded.append(ReplacementDecision(d.exited_symbol, None, "HOLD_CASH", rejected=d.rejected,
                                                  reason="INSUFFICIENT_BUYING_POWER"))
                continue
            remaining_bp -= notional
            self.anti_churn.consume_cycle_slot()
            self.anti_churn.record_replacement(d.candidate)
            funded.append(d)
        return funded + cash
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_replacement.py -k "qualified_candidate or req10 or req11 or req12 or req18 or req21 or req22 or req25 or req26 or req17 or net_expectancy or sharpe_below or build_pool" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add amps_v6/portfolio/replacement_optimizer.py tests/test_amps_v6_loss_removal_replacement.py
git commit -m "feat(amps_v6): replacement optimizer orchestrator (19-gate conjunction, delta_sharpe rank, joint multi-stop, cash-valid)"
```

---

### Task 12: AlpacaBroker — stop orders, symbol-scoped cancel, reconciliation

**Files:**
- Modify: `trade.py` (imports at line 89; `AlpacaBroker` methods after `get_order_by_id`, `trade.py:2717`)
- Test: `tests/test_amps_v6_broker_execution.py` (new focused file)

**Interfaces:**
- Produces on `AlpacaBroker`:
  - `submit_stop_order(symbol, qty, stop_price, client_order_id, tif=TimeInForce.GTC)` — protective SELL stop for a long (reduce-only in effect); verifies `qty <= held`.
  - `get_open_orders_for_symbol(symbol) -> list`
  - `cancel_open_orders_for_symbol(symbol, side=None) -> int` (count cancelled; `side` filters e.g. `"buy"`)
  - `find_protective_stop(symbol) -> order | None`
  - `reconcile_order(client_order_id) -> str` (broker status UPPER, or `"UNKNOWN"` on timeout/exception)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_amps_v6_broker_execution.py
"""AlpacaBroker protective-stop / cancel / reconcile extensions — fake broker, no network."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
import trade
from alpaca.trading.enums import TimeInForce


class _Order:
    def __init__(self, oid, symbol, side, otype="market", status="new", coid=None):
        self.id = oid
        self.symbol = symbol
        self.side = types.SimpleNamespace(value=side)
        self.type = types.SimpleNamespace(value=otype)
        self.status = types.SimpleNamespace(value=status)
        self.client_order_id = coid


class _Pos:
    def __init__(self, symbol, qty):
        self.symbol = symbol
        self.qty = str(qty)
        self.side = types.SimpleNamespace(value="long")
        self.qty_available = str(qty)


class _FakeTrading:
    def __init__(self, orders=None, positions=None):
        self._orders = orders or []
        self._positions = positions or []
        self.submitted = []
        self.cancelled = []

    def get_orders(self, filter=None):
        syms = getattr(filter, "symbol_or_symbols", None) or getattr(filter, "symbols", None)
        if syms:
            return [o for o in self._orders if o.symbol in syms]
        return list(self._orders)

    def get_all_positions(self):
        return self._positions

    def submit_order(self, order_data=None):
        self.submitted.append(order_data)
        return order_data

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)

    def get_order_by_client_id(self, coid):
        for o in self._orders:
            if o.client_order_id == coid:
                return o
        raise RuntimeError("not found")


def _broker(trading):
    b = object.__new__(trade.AlpacaBroker)
    b.trading = trading
    b._fractionable_cache = {}
    return b


def test_submit_stop_order_builds_sell_stop():
    t = _FakeTrading(positions=[_Pos("XYZ", 10)])
    b = _broker(t)
    b.submit_stop_order("XYZ", qty=10, stop_price=99.5, client_order_id="amps-stop-paper-1")
    assert len(t.submitted) == 1
    req = t.submitted[0]
    assert str(getattr(req.side, "value", req.side)).lower() == "sell"
    assert float(getattr(req, "stop_price")) == 99.5


def test_submit_stop_order_rejects_qty_above_held():
    b = _broker(_FakeTrading(positions=[_Pos("XYZ", 3)]))
    try:
        b.submit_stop_order("XYZ", qty=10, stop_price=99.5, client_order_id="x")
        assert False, "expected long-only rejection"
    except RuntimeError as e:
        assert "LONG_ONLY" in str(e)


def test_cancel_open_orders_for_symbol_filters_buys():
    t = _FakeTrading(orders=[_Order("1", "XYZ", "buy"), _Order("2", "XYZ", "sell"), _Order("3", "ABC", "buy")])
    b = _broker(t)
    n = b.cancel_open_orders_for_symbol("XYZ", side="buy")
    assert n == 1 and t.cancelled == ["1"]   # only the XYZ buy, not the XYZ sell or ABC


def test_find_protective_stop():
    t = _FakeTrading(orders=[_Order("9", "XYZ", "sell", otype="stop")])
    assert _broker(t).find_protective_stop("XYZ").id == "9"
    assert _broker(_FakeTrading()).find_protective_stop("XYZ") is None


def test_reconcile_order_status_and_unknown():
    t = _FakeTrading(orders=[_Order("1", "XYZ", "sell", status="filled", coid="cid-1")])
    b = _broker(t)
    assert b.reconcile_order("cid-1") == "FILLED"
    assert b.reconcile_order("missing") == "UNKNOWN"   # timeout/exception -> UNKNOWN, not failure
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_broker_execution.py -v`
Expected: FAIL (`AttributeError: 'AlpacaBroker' object has no attribute 'submit_stop_order'`)

- [ ] **Step 3: Implement**

a) Extend the imports at `trade.py:89`:

```python
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, StopOrderRequest
```

b) Add these methods to `AlpacaBroker` immediately after `get_order_by_id` (`trade.py:2717`):

```python
    def submit_stop_order(self, symbol: str, qty: float, stop_price: float, client_order_id: str,
                          tif: TimeInForce = TimeInForce.GTC):
        """Protective SELL stop for a long position (reduce-only in effect). Verifies
        qty does not exceed verified held long shares. Rests at the broker until
        triggered; the engine cancels it before any market exit (replace-safe)."""
        symbol = safe_symbol(symbol)
        q = Decimal(str(qty))
        if not q.is_finite() or q <= 0:
            raise ValueError("Stop quantity must be finite and positive")
        sp = Decimal(str(stop_price))
        if not sp.is_finite() or sp <= 0:
            raise ValueError("Stop price must be finite and positive")
        with self._order_lock:
            held = Decimal(0)
            for pos in self.trading.get_all_positions():
                pq = Decimal(str(pos.qty))
                ps = str(getattr(pos.side, "value", pos.side)).lower()
                if not pq.is_finite() or pq < 0 or ps == "short":
                    raise RuntimeError(f"LONG_ONLY: existing short or invalid position: {pos.symbol}")
                if safe_symbol(pos.symbol) == symbol:
                    held = pq
            if q > held:
                raise RuntimeError(f"LONG_ONLY: stop qty {q} exceeds held long shares {held} for {symbol}")
            order = StopOrderRequest(
                symbol=symbol,
                qty=float(q),
                side=OrderSide.SELL,
                time_in_force=tif,
                stop_price=float(sp),
                client_order_id=client_order_id,
            )
            return self.trading.submit_order(order_data=order)

    def get_open_orders_for_symbol(self, symbol: str) -> list:
        symbol = safe_symbol(symbol)
        try:
            return list(self.trading.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[symbol], limit=500, nested=True)))
        except Exception:
            return []

    def cancel_open_orders_for_symbol(self, symbol: str, side: str | None = None) -> int:
        """Cancel this symbol's open orders (optionally only one side, e.g. 'buy' to
        pull pending entry/increase orders before an exit). Returns count cancelled."""
        n = 0
        for o in self.get_open_orders_for_symbol(symbol):
            try:
                if side is not None and str(getattr(o.side, "value", o.side)).lower() != side.lower():
                    continue
                self.trading.cancel_order_by_id(o.id)
                n += 1
            except Exception:
                continue
        return n

    def find_protective_stop(self, symbol: str):
        """Return the resting protective SELL stop order for symbol, or None.
        Used to CONFIRM protection exists at the broker (never trust local state)."""
        for o in self.get_open_orders_for_symbol(symbol):
            try:
                t = str(getattr(o.type, "value", getattr(o, "type", ""))).lower()
                s = str(getattr(o.side, "value", getattr(o, "side", ""))).lower()
                if "stop" in t and s == "sell":
                    return o
            except Exception:
                continue
        return None

    def reconcile_order(self, client_order_id: str) -> str:
        """Broker status for a client order id (UPPERCASE), or 'UNKNOWN' on
        timeout/exception. A timeout is UNKNOWN, never a confirmed failure."""
        try:
            o = self.trading.get_order_by_client_id(client_order_id)
            return str(getattr(o.status, "value", getattr(o, "status", "UNKNOWN"))).upper()
        except Exception:
            return "UNKNOWN"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_broker_execution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trade.py tests/test_amps_v6_broker_execution.py
git commit -m "feat(trade): AlpacaBroker protective stop orders, symbol-scoped cancel, order reconciliation"
```

---

### Task 13: Executor wiring — canonical-P/L exit detection + broker-native protection

**Files:**
- Modify: `trade.py` — `ScannerConfig` (add fields near `amps_v6_enabled`, ~`trade.py:476`); `ProductionOrderExecutor` (new methods after `_positions_by_symbol`, `trade.py:5016`); `run_cycle_from_files` (insert after `equity=` at `trade.py:7185` and amend the exits concat at `trade.py:7214`); `build_default_config` env overrides (near `trade.py:7565`).
- Test: `tests/test_amps_v6_loss_removal_engine.py` (new focused file, fakes only)

**Interfaces:**
- Consumes: Tasks 1–12 modules; `BASELINE_COSTS` (import `from costs import BASELINE_COSTS` at top of trade.py if not present).
- Produces on `ProductionOrderExecutor`: `_amps_env` (`"paper"|"live"`), `_ensure_loss_removal_engine()`, `_latest_quote(symbol) -> Quote|None`, `_run_loss_removal_overlay(equity) -> pd.DataFrame` (exit rows, reduce-only full sells, reason `"total_pl_stop"`), `_ensure_protective_stops(equity) -> None`, `_log_position_exit_breach(cp, threshold, quote) -> None`. On `ScannerConfig`: `loss_removal_enabled: bool = True`, `loss_removal_shadow_mode: bool = False`.

- [ ] **Step 1: Write the failing tests** (covers required #8 at integration level + env selection)

```python
# tests/test_amps_v6_loss_removal_engine.py
"""Executor-level loss-removal wiring: canonical -0.5% exit detection, pending-buy
cancellation, shadow mode, PAPER/LIVE env selection. Fakes only, no network."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
import pandas as pd
import pytest

import trade
from amps_v6.state import lifecycle_ledger as LL
from amps_v6.execution import anti_churn as AC
from amps_v6.execution import replacement_state_machine as RSM
from amps_v6.configuration.replacement_config import ReplacementConfig


class _BPos:
    def __init__(self, symbol, qty, avg, cur):
        self.symbol = symbol; self.qty = qty
        self.avg_entry_price = avg; self.current_price = cur
        self.market_value = qty * cur; self.unrealized_pl = (cur - avg) * qty
        self.unrealized_plpc = cur / avg - 1


class _FakeBroker:
    def __init__(self, positions):
        self._positions = positions
        self.cancelled = []
        self.stops = []
    def get_all_open_positions(self):
        return self._positions
    def cancel_open_orders_for_symbol(self, symbol, side=None):
        self.cancelled.append((symbol, side)); return 1
    def find_protective_stop(self, symbol):
        return None
    def submit_stop_order(self, symbol, qty, stop_price, client_order_id, tif=None):
        self.stops.append((symbol, qty, stop_price, client_order_id)); return object()


class _FakeData:
    """get_stock_latest_quote returns a bid below entry so canonical P/L breaches."""
    def __init__(self, bid, ask, ts):
        self._bid, self._ask, self._ts = bid, ask, ts
    def get_stock_latest_quote(self, req):
        sym = (req.symbol_or_symbols if isinstance(req.symbol_or_symbols, str) else req.symbol_or_symbols[0])
        q = types.SimpleNamespace(bid_price=self._bid, ask_price=self._ask, bid_size=100, ask_size=100,
                                  timestamp=types.SimpleNamespace(timestamp=lambda: self._ts))
        return {sym: q}


def _exec(monkeypatch, tmp_path, positions, bid, shadow=False, env="paper"):
    for mod in (LL, AC, RSM):
        monkeypatch.setattr(mod, "STATE_DIR", tmp_path, raising=False)
    e = object.__new__(trade.ProductionOrderExecutor)
    e.config = trade.ExecutorConfig()
    e.config.paper = (env == "paper")
    e.config.scanner.loss_removal_enabled = True
    e.config.scanner.loss_removal_shadow_mode = shadow
    e.broker = _FakeBroker(positions)
    e.scanner = types.SimpleNamespace(data=_FakeData(bid=bid, ask=bid + 0.02, ts=1_000_000.0))
    # pre-wire the loss-removal engine against tmp state (bypass _ensure_* network bits)
    cfg = ReplacementConfig()
    e._amps_env = env
    e._rc = cfg
    e._ledger = LL.LifecycleLedger(env)
    e._anti_churn = AC.AntiChurnRegistry(cfg, env)
    e._repl_sm = RSM.ReplacementStateMachine(env)
    e._order_flow_engine = None
    e._se_window = {}
    e._loss_removal_ready = True
    # seed the ledger so canonical basis exists (10 @ 100)
    e._ledger.apply_fill(LL.FillEvent(ts=0, symbol=positions[0].symbol, side="buy", qty=positions[0].qty,
                                      price=positions[0].avg_entry_price, reference_price=positions[0].avg_entry_price))
    monkeypatch.setattr(trade, "now_utc", trade.now_utc, raising=False)
    return e


def test_minus_060_breach_produces_reduce_only_exit_row(monkeypatch, tmp_path):
    pos = [_BPos("XYZ", 10, 100.0, 99.4)]
    e = _exec(monkeypatch, tmp_path, pos, bid=99.4)   # -0.6% gross, more negative after costs
    rows = e._run_loss_removal_overlay(equity=100000.0)
    assert not rows.empty and rows.iloc[0]["symbol"] == "XYZ"
    assert rows.iloc[0]["side"] == "sell" and rows.iloc[0]["reason"] == "total_pl_stop"
    # required #8: pending BUY/increase orders cancelled before exit
    assert ("XYZ", "buy") in e.broker.cancelled
    assert e._repl_sm.get("XYZ")["state"] == RSM.EXIT_REQUIRED


def test_minus_040_does_not_breach(monkeypatch, tmp_path):
    pos = [_BPos("XYZ", 10, 100.0, 99.7)]   # -0.3% gross; with small costs still > -0.5%? use clearly-safe level
    e = _exec(monkeypatch, tmp_path, pos, bid=99.95)   # -0.05% gross -> above threshold even after costs
    rows = e._run_loss_removal_overlay(equity=100000.0)
    assert rows.empty


def test_shadow_mode_adds_no_rows_but_logs(monkeypatch, tmp_path):
    pos = [_BPos("XYZ", 10, 100.0, 99.4)]
    e = _exec(monkeypatch, tmp_path, pos, bid=99.4, shadow=True)
    rows = e._run_loss_removal_overlay(equity=100000.0)
    assert rows.empty   # shadow: no enforcement rows ...
    assert e._repl_sm.get("XYZ")["state"] == RSM.EXIT_REQUIRED   # ... but state/intent still recorded


def test_req30_paper_live_env_selection(monkeypatch, tmp_path):
    e_paper = _exec(monkeypatch, tmp_path, [_BPos("XYZ", 10, 100.0, 99.4)], bid=99.4, env="paper")
    e_live = _exec(monkeypatch, tmp_path, [_BPos("XYZ", 10, 100.0, 99.4)], bid=99.4, env="live")
    assert e_paper._amps_env == "paper" and e_live._amps_env == "live"
    assert e_paper._ledger.state_path != e_live._ledger.state_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_engine.py -v`
Expected: FAIL (`AttributeError: 'ProductionOrderExecutor' object has no attribute '_run_loss_removal_overlay'` / `ScannerConfig` has no `loss_removal_enabled`)

- [ ] **Step 3: Implement**

a) Ensure `from costs import BASELINE_COSTS` is importable in trade.py (add near the other top-level imports if absent).

b) Add to `ScannerConfig` (near `amps_v6_enabled`, `trade.py:476`):

```python
    loss_removal_enabled: bool = True
    loss_removal_shadow_mode: bool = False
```

c) Add these methods to `ProductionOrderExecutor` right after `_positions_by_symbol` (`trade.py:5016`):

```python
    _LOSS_REMOVAL_COLS = ["symbol", "side", "delta_shares", "delta_dollars", "price", "reason",
                          "target_weight", "current_weight", "delta_weight", "trade_confidence"]

    def _ensure_loss_removal_engine(self) -> None:
        if getattr(self, "_loss_removal_ready", False):
            return
        from amps_v6.configuration.replacement_config import ReplacementConfig
        from amps_v6.state.lifecycle_ledger import LifecycleLedger
        from amps_v6.execution.anti_churn import AntiChurnRegistry
        from amps_v6.execution.replacement_state_machine import ReplacementStateMachine
        from amps_v6.alpha.seller_exhaustion import SellerExhaustionScorer
        from amps_v6.strategy.momentum_gate import PositiveMomentumGate
        from amps_v6.portfolio.marginal_sharpe import MarginalSharpeOptimizer
        from amps_v6.portfolio.replacement_optimizer import ReplacementOptimizer
        env = "paper" if self.config.paper else "live"
        cfg = ReplacementConfig.from_env()
        self._amps_env = env
        self._rc = cfg
        self._ledger = LifecycleLedger.load(env)
        self._anti_churn = AntiChurnRegistry.load(cfg, env)
        self._repl_sm = ReplacementStateMachine.load(env)
        self._se_scorer = SellerExhaustionScorer(cfg)
        self._mom_gate = PositiveMomentumGate(cfg)
        self._ms_opt = MarginalSharpeOptimizer(cfg)
        self._repl_opt = ReplacementOptimizer(cfg, self._ms_opt, self._anti_churn)
        try:
            self._ensure_profit_exit_engines()   # shares self._order_flow_engine
        except Exception:
            self._order_flow_engine = None
        self._se_window = {}
        self._loss_removal_ready = True

    def _latest_quote(self, symbol: str):
        from amps_v6.state.lifecycle_ledger import Quote
        data = getattr(self.scanner, "data", None) if self.scanner is not None else None
        if data is None:
            return None
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            resp = data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=[symbol]))
            q = resp[symbol]
            ts = getattr(q, "timestamp", None)
            ts_epoch = ts.timestamp() if ts is not None and hasattr(ts, "timestamp") else time.time()
            return Quote(bid=float(getattr(q, "bid_price", 0) or 0), ask=float(getattr(q, "ask_price", 0) or 0),
                         ts=float(ts_epoch), bid_size=float(getattr(q, "bid_size", 0) or 0),
                         ask_size=float(getattr(q, "ask_size", 0) or 0))
        except Exception:
            return None

    def _log_position_exit_breach(self, cp, threshold: float, quote) -> None:
        print(f"[position-exit]\nsymbol={cp.symbol}\ntotal_pl_pct={cp.total_pl_pct:.4f}\n"
              f"threshold={threshold:.4f}\ncondition=\"<=\"\naction=EXIT_REQUIRED")
        try:
            write_event("total_pl_stop_breached", {
                "symbol": cp.symbol, "env": self._amps_env, "total_pl_pct": cp.total_pl_pct,
                "position_total_pl": cp.position_total_pl, "threshold": threshold,
                "realized_pnl": cp.realized_pnl, "unrealized_pnl": cp.unrealized_pnl,
                "cost_basis": cp.cost_basis, "qty": cp.qty, "avg_entry": cp.avg_entry,
                "bid": cp.bid, "quote_age_s": cp.quote_age_s, "status": cp.status,
                "timestamp": now_utc().isoformat()})
        except Exception:
            pass

    def _run_loss_removal_overlay(self, equity: float) -> pd.DataFrame:
        """Detect canonical Total P/L (%) <= -0.5% breaches and emit reduce-only full-exit
        rows into the exits funnel. Un-vetoable (exit_stack takes no agent input). Shadow
        mode records state/intent but emits no rows. Fail-safe: any error -> no rows."""
        from amps_v6.risk import exit_stack as XS
        from amps_v6.execution import replacement_state_machine as RSM
        empty = pd.DataFrame(columns=self._LOSS_REMOVAL_COLS)
        try:
            self._ensure_loss_removal_engine()
        except Exception as exc:
            print(f"[loss-removal] engine init failed (non-fatal): {exc}")
            return empty
        now = time.time()
        xcfg = XS.ExitConfig(total_pl_exit_threshold=self._rc.total_pl_exit_threshold)
        rows = []
        for p in self.broker.get_all_open_positions():
            sym = safe_symbol(p.symbol)
            if not sym or p.qty <= 0:
                continue
            self._ledger.reconcile(sym, p.qty, p.avg_entry_price)
            q = self._latest_quote(sym)
            cp = self._ledger.canonical(sym, q, now, cost_cfg=BASELINE_COSTS,
                                        max_quote_age_s=self._rc.max_quote_age_s)
            ctx = XS.ExitContext(price=(cp.mark or p.current_price), avg_entry=p.avg_entry_price,
                                 total_pl_pct=cp.total_pl_pct, total_pl_valid=cp.valid)
            decision = XS.evaluate(ctx, xcfg)
            if decision.reason != "TOTAL_PL_STOP_BREACHED":
                continue
            pid = sym
            self._repl_sm.open(pid, sym)
            self._repl_sm.advance(pid, RSM.EXIT_REQUIRED)
            try:
                self.broker.cancel_open_orders_for_symbol(sym, side="buy")   # block added exposure
            except Exception as exc:
                print(f"[loss-removal] cancel pending buys failed for {sym}: {exc}")
            self._log_position_exit_breach(cp, self._rc.total_pl_exit_threshold, q)
            if not self.config.scanner.loss_removal_shadow_mode:
                px = cp.bid or p.current_price
                rows.append({"symbol": sym, "side": "sell", "delta_shares": int(p.qty),
                             "delta_dollars": round(float(p.qty) * float(px), 2), "price": float(px),
                             "reason": "total_pl_stop", "target_weight": 0.0, "current_weight": 0.0,
                             "delta_weight": 0.0, "trade_confidence": 1.0})
        for obj in (self._ledger, self._repl_sm, self._anti_churn):
            try:
                obj.save()
            except Exception:
                pass
        return pd.DataFrame(rows, columns=self._LOSS_REMOVAL_COLS) if rows else empty

    def _ensure_protective_stops(self, equity: float) -> None:
        """After entries, ensure a broker-native protective SELL stop exists at the -0.5%
        level and is CONFIRMED at the broker (never trust local state). Backstopped by the
        continuous monitor. Skips positions already exiting. Fail-safe."""
        try:
            self._ensure_loss_removal_engine()
        except Exception:
            return
        for p in self.broker.get_all_open_positions():
            sym = safe_symbol(p.symbol)
            if not sym or p.qty <= 0 or p.avg_entry_price <= 0:
                continue
            rec = self._repl_sm.get(sym)
            if rec and rec["state"] != "HELD":
                continue   # exiting/replacing -> do not add new protection
            try:
                if self.broker.find_protective_stop(sym) is not None:
                    continue   # confirmed at broker
                stop_px = round(p.avg_entry_price * (1.0 + self._rc.total_pl_exit_threshold), 2)
                cid = f"amps-stop-{self._amps_env}-{sym}-{int(round(stop_px * 100))}"
                self.broker.submit_stop_order(sym, qty=p.qty, stop_price=stop_px, client_order_id=cid)
            except Exception as exc:
                print(f"[protection] stop ensure failed for {sym}: {exc}")
```

d) Wire into `run_cycle_from_files`. Immediately after `equity = max(float(acct.equity), 1.0)` (`trade.py:7184`), insert:

```python
                # --- loss-removal: canonical Total P/L (%) <= -0.5% exits (priority 4), and
                #     broker-native protective stops (priority 3). Runs BEFORE deterioration. ---
                loss_exits = pd.DataFrame()
                if self.config.scanner.loss_removal_enabled:
                    try:
                        self._ensure_protective_stops(equity)
                        loss_exits = self._run_loss_removal_overlay(equity)
                    except Exception as exc:  # noqa: BLE001 - overlay must never break the cycle
                        print(f"[loss-removal] overlay failed (non-fatal): {exc}")
                        write_event("loss_removal_overlay_error", {"error": str(exc)})
```

And change the exits concat at `trade.py:7214` from:

```python
                exits = pd.concat([monitor_exits, scanner_exits], ignore_index=True)
```

to (loss-removal exits first → highest funnel priority):

```python
                exits = pd.concat([loss_exits, monitor_exits, scanner_exits], ignore_index=True)
```

e) Add env overrides in `build_default_config` near the existing `AMPS_V6` overrides (`trade.py:7565`):

```python
    config.scanner.loss_removal_enabled = os.getenv("AMPS_LOSS_REMOVAL", "1").strip().lower() not in ("0", "false", "no")
    config.scanner.loss_removal_shadow_mode = os.getenv("AMPS_LOSS_REMOVAL_SHADOW", "0").strip().lower() in ("1", "true", "yes")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trade.py tests/test_amps_v6_loss_removal_engine.py
git commit -m "feat(trade): wire canonical -0.5% exit detection + broker-native protective stops into the live cycle"
```

---

### Task 14: Executor wiring — replacement evaluation (reconcile → optimize → emit buys or cash)

**Files:**
- Modify: `trade.py` — `ScannerConfig` (`loss_removal_replace_enabled`); `ProductionOrderExecutor` (methods after Task 13's); `run_cycle_from_files` (insert after the AMPS V6 overlay at `trade.py:7293`); `build_default_config` env override.
- Test: `tests/test_amps_v6_loss_removal_engine.py` (extend)

**Interfaces:**
- Consumes: Task 11 `replacement_optimizer` (`build_pool`, `ReplacementOptimizer`, `CandidateInputs`); `replacement_state_machine` states.
- Produces on `ProductionOrderExecutor`: `_reconcile_exit_states() -> None`, `_build_candidate_inputs(symbol) -> CandidateInputs | None` (overridable), `_run_replacement_evaluation(rebalance, equity, buying_power) -> pd.DataFrame`, `_universe_loader` (callable). On `ScannerConfig`: `loss_removal_replace_enabled: bool = True`.

- [ ] **Step 1: Write the failing tests** (covers required #9, #10, #21 at integration level)

```python
# append to tests/test_amps_v6_loss_removal_engine.py
from amps_v6.portfolio import replacement_optimizer as RO
from amps_v6.portfolio import marginal_sharpe as MS
from amps_v6.alpha import entry_alpha as EA
from amps_v6.alpha import seller_exhaustion as SE
from amps_v6.strategy import momentum_gate as MG
import numpy as np


def _flat_broker_after_exit(symbol):
    class _B(_FakeBroker):
        def get_position(self, sym):
            return _BPos(sym, 0.0, 100.0, 0.0)   # flat
    return _B([])


def _repl_exec(monkeypatch, tmp_path, broker, env="paper"):
    for mod in (LL, AC, RSM):
        monkeypatch.setattr(mod, "STATE_DIR", tmp_path, raising=False)
    e = object.__new__(trade.ProductionOrderExecutor)
    e.config = trade.ExecutorConfig()
    e.config.paper = (env == "paper")
    e.config.scanner.loss_removal_enabled = True
    e.config.scanner.loss_removal_replace_enabled = True
    e.config.scanner.loss_removal_shadow_mode = False
    e.broker = broker
    e.scanner = types.SimpleNamespace(data=None)
    cfg = ReplacementConfig()
    e._amps_env = env; e._rc = cfg
    e._ledger = LL.LifecycleLedger(env)
    e._anti_churn = AC.AntiChurnRegistry(cfg, env)
    e._repl_sm = RSM.ReplacementStateMachine(env)
    e._ms_opt = MS.MarginalSharpeOptimizer(cfg)
    e._repl_opt = RO.ReplacementOptimizer(cfg, e._ms_opt, e._anti_churn)
    e._order_flow_engine = None; e._se_window = {}
    e._universe_loader = lambda: ["CAND"]
    e._loss_removal_ready = True
    return e


def _good_cand(symbol="CAND"):
    se = SE.SellerExhaustionResult(symbol, True, "OK", 0.8, 4, persistence_ok=True, confirmed=True)
    mom = MG.MomentumResult(symbol, True, True, 28, 18, 101, 100, True, confirmation_count=3, confidence=0.7)
    return RO.CandidateInputs(symbol=symbol, seller_exhaustion=se, momentum=mom,
                              alpha_decision=EA.ALPHA_CONFIRMED, expected_net_return=0.004,
                              candidate_mu=0.002, proposed_weight=0.1)


def test_req09_partial_fill_keeps_unresolved(monkeypatch, tmp_path):
    # position still has shares after EXIT_SUBMITTED -> stays unresolved, no replacement
    broker = _FakeBroker([_BPos("XYZ", 5, 100.0, 99.0)])   # 5 of 10 remain
    broker.get_position = lambda sym: _BPos(sym, 5, 100.0, 99.0)
    e = _repl_exec(monkeypatch, tmp_path, broker)
    e._repl_sm.open("XYZ", "XYZ"); e._repl_sm.advance("XYZ", RSM.EXIT_REQUIRED); e._repl_sm.advance("XYZ", RSM.EXIT_SUBMITTED)
    rb = e._run_replacement_evaluation(pd.DataFrame(), equity=100000.0, buying_power=100000.0)
    assert "XYZ" in e._repl_sm.unresolved()
    assert e._repl_sm.get("XYZ")["state"] in (RSM.EXIT_SUBMITTED, RSM.EXIT_PARTIALLY_FILLED)
    assert rb.empty   # no replacement while not flat


def test_req10_replacement_only_after_flat(monkeypatch, tmp_path):
    e = _repl_exec(monkeypatch, tmp_path, _flat_broker_after_exit("XYZ"))
    e._repl_sm.open("XYZ", "XYZ"); e._repl_sm.advance("XYZ", RSM.EXIT_REQUIRED); e._repl_sm.advance("XYZ", RSM.EXIT_SUBMITTED)
    e._build_candidate_inputs = lambda sym: _good_cand(sym)
    rb = e._run_replacement_evaluation(pd.DataFrame(), equity=100000.0, buying_power=100000.0)
    # flat -> advanced through EXIT_FILLED -> evaluated -> a qualifying buy emitted
    assert not rb.empty and rb.iloc[0]["symbol"] == "CAND" and rb.iloc[0]["side"] == "buy"
    assert e._repl_sm.get("XYZ")["state"] in (RSM.REPLACEMENT_SUBMITTED, RSM.REPLACEMENT_APPROVED)


def test_req21_cash_when_no_candidate_qualifies(monkeypatch, tmp_path):
    e = _repl_exec(monkeypatch, tmp_path, _flat_broker_after_exit("XYZ"))
    e._repl_sm.open("XYZ", "XYZ"); e._repl_sm.advance("XYZ", RSM.EXIT_REQUIRED); e._repl_sm.advance("XYZ", RSM.EXIT_SUBMITTED)
    e._build_candidate_inputs = lambda sym: None   # nothing computable/qualifies
    rb = e._run_replacement_evaluation(pd.DataFrame(), equity=100000.0, buying_power=100000.0)
    assert rb.empty and e._repl_sm.get("XYZ")["state"] == RSM.CASH_HELD
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_amps_v6_loss_removal_engine.py -k "req09 or req10 or req21_cash" -v`
Expected: FAIL (`AttributeError: ... '_run_replacement_evaluation'`)

- [ ] **Step 3: Implement**

a) `ScannerConfig`: add `loss_removal_replace_enabled: bool = True` (near Task 13's fields).

b) Add to `ProductionOrderExecutor` (after Task 13's methods). Add required imports at method top as shown.

```python
    def _reconcile_exit_states(self) -> None:
        """Advance exit lifecycle from broker truth. EXIT_REQUIRED->EXIT_SUBMITTED (the
        reduce-only row was handed to run_once_from_frames last step); then, when the broker
        position is flat, EXIT_SUBMITTED/PARTIALLY_FILLED -> EXIT_FILLED -> REPLACEMENT_EVALUATION
        and start the exited-symbol cooldown. A still-held quantity stays EXIT_PARTIALLY_FILLED
        (unresolved) -- never evaluated for replacement until flat."""
        from amps_v6.execution import replacement_state_machine as RSM
        now = time.time()
        for pid in list(self._repl_sm.unresolved()):
            rec = self._repl_sm.get(pid)
            st, sym = rec["state"], rec["symbol"]
            if st == RSM.EXIT_REQUIRED:
                self._repl_sm.advance(pid, RSM.EXIT_SUBMITTED)
                st = RSM.EXIT_SUBMITTED
            if st in (RSM.EXIT_SUBMITTED, RSM.EXIT_PARTIALLY_FILLED):
                pos = self.broker.get_position(sym)
                if float(getattr(pos, "qty", 0.0)) <= 0:
                    if st == RSM.EXIT_SUBMITTED:
                        self._repl_sm.advance(pid, RSM.EXIT_PARTIALLY_FILLED)
                    self._repl_sm.advance(pid, RSM.EXIT_FILLED)
                    self._repl_sm.advance(pid, RSM.REPLACEMENT_EVALUATION)
                    self._anti_churn.start_cooldown(sym, now)
                elif st == RSM.EXIT_SUBMITTED:
                    self._repl_sm.advance(pid, RSM.EXIT_PARTIALLY_FILLED)   # still holding -> partial

    def _build_candidate_inputs(self, symbol: str):
        """Build CandidateInputs for a pool symbol from live data. Defensive: returns None
        when microstructure/bars/alpha are unavailable (a None candidate is simply skipped,
        never fabricated). Overridden in tests. Real wiring:
          - seller exhaustion: self._se_scorer.score(symbol, self._se_window.get(symbol, []))
          - momentum: self._mom_gate.evaluate(symbol, <5m frame>, <1m>, <15m>)
          - alpha: evidence registry classification / entry_alpha decision
          - candidate_mu / returns: from recent bars; exposure_group via universe.classify_symbol
        """
        from amps_v6.portfolio import replacement_optimizer as RO
        from amps_v6.alpha import entry_alpha as EA
        try:
            import universe
            window = self._se_window.get(symbol, [])
            se = self._se_scorer.score(symbol, window)
            if not se.available:
                return None
            frames = self.scanner._download_intraday_bars([symbol], 5) if self.scanner is not None else None
            if frames is None or frames.empty:
                return None
            f5 = frames[frames["symbol"] == symbol].copy()
            mom = self._mom_gate.evaluate(symbol, f5, drop_last_incomplete=True)
            info = universe.classify_symbol(symbol)
            exposure_group = getattr(info, "exposure_group", "equity")
            is_lev = exposure_group in ("leveraged", "inverse", "leveraged_inverse")
            # alpha + expected return are supplied by the existing entry-alpha path; conservative defaults here
            return RO.CandidateInputs(
                symbol=symbol, seller_exhaustion=se, momentum=mom,
                alpha_decision=EA.NO_ALPHA_EVIDENCE,   # replaced by real evidence lookup in integration
                expected_net_return=0.0, candidate_mu=0.0, proposed_weight=0.0,
                exposure_group=exposure_group, is_leveraged_or_inverse=is_lev)
        except Exception:
            return None

    def _run_replacement_evaluation(self, rebalance: pd.DataFrame, equity: float, buying_power: float) -> pd.DataFrame:
        """Reconcile exits, then for every flat (EXIT_FILLED->REPLACEMENT_EVALUATION) position run
        the replacement optimizer over the universe.txt pool and emit a buy row (REPLACE) or hold
        cash (CASH_HELD). Never forces a replacement. Fail-safe. Returns the (possibly augmented)
        rebalance frame."""
        from amps_v6.portfolio import replacement_optimizer as RO
        from amps_v6.portfolio import marginal_sharpe as MS
        from amps_v6.execution import replacement_state_machine as RSM
        import numpy as np
        try:
            self._ensure_loss_removal_engine()
            self._reconcile_exit_states()
        except Exception as exc:
            print(f"[replacement] reconcile failed (non-fatal): {exc}")
            return rebalance
        if not getattr(self.config.scanner, "loss_removal_replace_enabled", True):
            return rebalance
        now = time.time()
        pending = [pid for pid in self._repl_sm.unresolved()
                   if self._repl_sm.get(pid)["state"] == RSM.REPLACEMENT_EVALUATION]
        if not pending:
            return rebalance
        self._anti_churn.begin_cycle()
        # minimal portfolio state (held names at current weights); covariance augmented per-candidate
        held = {safe_symbol(p.symbol): (p.market_value / max(equity, 1.0))
                for p in self.broker.get_all_open_positions() if p.qty > 0}
        order = list(held.keys()) or ["__CASH__"]
        cov = np.eye(len(order)) * 0.0004
        state = MS.PortfolioState(symbols=list(held.keys()), weights=held,
                                  expected_daily_returns={s: 0.0 for s in order}, cov=cov, cov_order=order)
        new_rows = []
        exits = [self._repl_sm.get(pid)["symbol"] for pid in pending]
        cands_by_exit = {}
        for ex in exits:
            pool = RO.build_pool(ex, now, self._anti_churn, loader=getattr(self, "_universe_loader", None))
            cands = [c for c in (self._build_candidate_inputs(s) for s in pool) if c is not None]
            cands_by_exit[ex] = cands
        decisions = self._repl_opt.evaluate_joint(exits, cands_by_exit, state, now, buying_power, equity,
                                                  exit_proceeds_confirmed=True)
        dec_by_exit = {d.exited_symbol: d for d in decisions}
        for pid in pending:
            sym = self._repl_sm.get(pid)["symbol"]
            d = dec_by_exit.get(sym)
            if d is None:
                continue
            if d.action == "REPLACE" and d.candidate:
                self._repl_sm.set_replacement(pid, d.candidate)
                self._repl_sm.advance(pid, RSM.REPLACEMENT_APPROVED)
                self._log_replacement(d)
                if not self.config.scanner.loss_removal_shadow_mode:
                    px_dollars = round(d.proposed_weight * equity, 2)
                    new_rows.append({"symbol": d.candidate, "side": "buy", "delta_shares": 0,
                                     "delta_dollars": px_dollars, "price": 0.0, "reason": "amps_replacement",
                                     "target_weight": d.proposed_weight, "current_weight": 0.0,
                                     "delta_weight": d.proposed_weight, "trade_confidence": 1.0})
                    self._repl_sm.advance(pid, RSM.REPLACEMENT_SUBMITTED)
            else:
                self._repl_sm.advance(pid, RSM.CASH_HELD)
                self._log_replacement(d)
        for obj in (self._repl_sm, self._anti_churn):
            try:
                obj.save()
            except Exception:
                pass
        if not new_rows:
            return rebalance
        add = pd.DataFrame(new_rows)
        return add if (rebalance is None or rebalance.empty) else pd.concat([rebalance, add], ignore_index=True)

    def _log_replacement(self, d) -> None:
        if d.action == "REPLACE":
            print(f"[replacement]\nexited={d.exited_symbol}\ncandidate={d.candidate}\n"
                  f"delta_sharpe={d.delta_sharpe:.3f}\nrisk=APPROVED\naction=REPLACE")
        else:
            print(f"[replacement]\nexited={d.exited_symbol}\ncandidate=NONE\n"
                  f"reason={d.reason}\naction=HOLD_CASH")
        try:
            write_event("amps_replacement_decision", {"env": self._amps_env, **{
                "exited": d.exited_symbol, "candidate": d.candidate, "action": d.action,
                "delta_sharpe": d.delta_sharpe, "reason": d.reason, "rejected": d.rejected}})
        except Exception:
            pass
```

c) Wire into `run_cycle_from_files` immediately after the AMPS V6 overlay block (`trade.py:7293`, before the `if not monitor_exits.empty:` log at `7295`):

```python
                # --- replacement evaluation (priority 7): only for exits broker-confirmed flat ---
                if self.config.scanner.loss_removal_enabled:
                    try:
                        bp = float(acct.buying_power)
                        rebalance = self._run_replacement_evaluation(rebalance, equity, bp)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[replacement] evaluation failed (non-fatal): {exc}")
                        write_event("replacement_evaluation_error", {"error": str(exc)})
```

d) `build_default_config`: add `config.scanner.loss_removal_replace_enabled = os.getenv("AMPS_LOSS_REMOVAL_REPLACE", "1").strip().lower() not in ("0", "false", "no")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_amps_v6_loss_removal_engine.py -v`
Expected: PASS (all loss-removal engine tests)

- [ ] **Step 5: Commit**

```bash
git add trade.py tests/test_amps_v6_loss_removal_engine.py
git commit -m "feat(trade): wire replacement evaluation (reconcile->optimize->buy|cash) into the live cycle"
```

---
