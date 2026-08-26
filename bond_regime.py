#!/usr/bin/env python3
"""
Bond-market regime overlay for V2.9.8 (OPT-IN, default OFF).
============================================================

Turns bond-market data into two BOUNDED risk multipliers that tilt the strategy's
existing risk levers -- the volatility budget and the growth-gate ceiling -- WITHOUT
changing any strategy parameter or the optimizer math:

    vol_budget    *= vol_budget_mult      (de-risk in risk-off, mild add in risk-on)
    growth_ceiling *= growth_ceiling_mult (tighten the equity gate in risk-off only)

Both multipliers are exactly 1.0 in a neutral regime, and the whole overlay is
OFF by default, so the backtest and the live decision are byte-identical unless it
is explicitly enabled (env ``BOND_REGIME_OVERLAY``).

Signals (Alpaca ETF proxies -- no yield indices needed; all causal, z-scored on a
trailing window):
  * flight-to-quality : momentum(IEF) - momentum(HYG)   (Treasuries up, credit down)
  * credit spread     : momentum(LQD) - momentum(HYG)   (IG over HY = spreads widen)
  * bond volatility    : realized vol(TLT)                (a MOVE-index proxy)
  * curve slope proxy : momentum(TLT) - momentum(SHY)    (telemetry only, not in the score)

HIGHER composite z = MORE risk-off.

IMPORTANT (research policy): this repeats the shape of the already-REJECTED
Fed/macro liquidity regime overlay. It ships OFF and OWES a standalone PIT
event/IC study (bootstrap + placebo) before being trusted on real money. The
overlay-on-vs-off backtest is the falsification harness; see
``scripts``/`make_backtest_tearsheet` usage in LIVE_TRADING.md.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


@dataclass
class BondOverlayConfig:
    enabled: bool = False
    # ETFs needed for the signals (must be present in the bond price frame).
    bond_assets: tuple[str, ...] = ("SHY", "IEF", "TLT", "LQD", "HYG")
    mom_lookback: int = 63
    vol_lookback: int = 21
    z_lookback: int = 252
    # signal weights (normalized internally)
    w_flight_to_quality: float = 0.40
    w_credit: float = 0.40
    w_bondvol: float = 0.20
    tilt_scale: float = 1.0          # z-score units for the tanh squash
    max_vol_cut: float = 0.30        # up to -30% vol budget in max risk-off
    max_vol_add: float = 0.10        # up to +10% vol budget in max risk-on
    max_growth_cut: float = 0.20     # up to -20% growth ceiling in max risk-off

    @classmethod
    def from_env(cls) -> "BondOverlayConfig":
        on = os.environ.get("BOND_REGIME_OVERLAY", "").strip().lower() in ("1", "true", "yes", "on")
        return cls(enabled=on)


@dataclass
class BondRegime:
    as_of: pd.Timestamp
    risk_off_z: float
    vol_budget_mult: float
    growth_ceiling_mult: float
    curve_slope_proxy: float
    components: dict = field(default_factory=dict)


def _zscore(series: pd.Series, lookback: int) -> pd.Series:
    m = series.rolling(lookback, min_periods=max(20, lookback // 3)).mean()
    s = series.rolling(lookback, min_periods=max(20, lookback // 3)).std(ddof=0)
    z = (series - m) / s.where(s > 1e-12)
    return z.clip(-3.0, 3.0)


def regime_frame(bond_prices: pd.DataFrame, cfg: BondOverlayConfig) -> pd.DataFrame:
    """
    Vectorized per-date regime frame: risk_off_z, vol_budget_mult,
    growth_ceiling_mult, curve_slope_proxy. All quantities are causal (use only
    past/current prices), so slicing the input at a date and reading the last row
    equals what a live decision would see at that date.
    """
    need = list(cfg.bond_assets)
    missing = [a for a in need if a not in bond_prices.columns]
    if missing:
        raise ValueError(f"bond price frame missing {missing}")
    px = bond_prices[need].astype(float)
    rets = px.pct_change()

    def mom(sym):
        return px[sym] / px[sym].shift(cfg.mom_lookback) - 1.0

    ftq = mom("IEF") - mom("HYG")
    credit = mom("LQD") - mom("HYG")
    bondvol = rets["TLT"].rolling(cfg.vol_lookback,
                                  min_periods=max(5, cfg.vol_lookback // 2)).std(ddof=0) * math.sqrt(TRADING_DAYS)
    curve = mom("TLT") - mom("SHY")

    z_ftq = _zscore(ftq, cfg.z_lookback)
    z_credit = _zscore(credit, cfg.z_lookback)
    z_bondvol = _zscore(bondvol, cfg.z_lookback)

    wsum = cfg.w_flight_to_quality + cfg.w_credit + cfg.w_bondvol
    risk_off = (cfg.w_flight_to_quality * z_ftq
                + cfg.w_credit * z_credit
                + cfg.w_bondvol * z_bondvol) / max(wsum, 1e-9)

    t = np.tanh(risk_off / max(cfg.tilt_scale, 1e-9))          # [-1, 1]
    pos = t.clip(lower=0.0)     # risk-off amount
    neg = (-t).clip(lower=0.0)  # risk-on amount
    vol_mult = 1.0 - cfg.max_vol_cut * pos + cfg.max_vol_add * neg
    growth_mult = 1.0 - cfg.max_growth_cut * pos               # only tightens

    out = pd.DataFrame({
        "risk_off_z": risk_off,
        "vol_budget_mult": vol_mult,
        "growth_ceiling_mult": growth_mult,
        "curve_slope_proxy": curve,
        "z_ftq": z_ftq, "z_credit": z_credit, "z_bondvol": z_bondvol,
    })
    # Neutral (mult=1.0) wherever the signal is undefined (insufficient history).
    out["vol_budget_mult"] = out["vol_budget_mult"].fillna(1.0)
    out["growth_ceiling_mult"] = out["growth_ceiling_mult"].fillna(1.0)
    out["risk_off_z"] = out["risk_off_z"].fillna(0.0)
    return out


def compute_bond_regime(bond_prices: pd.DataFrame, cfg: BondOverlayConfig,
                        as_of=None) -> BondRegime:
    """Single-date regime (uses only data through as_of; the rest is ignored)."""
    bp = bond_prices.sort_index()
    if as_of is not None:
        bp = bp.loc[:pd.Timestamp(as_of)]
    frame = regime_frame(bp, cfg)
    row = frame.iloc[-1]
    return BondRegime(
        as_of=bp.index[-1],
        risk_off_z=float(row["risk_off_z"]),
        vol_budget_mult=float(row["vol_budget_mult"]),
        growth_ceiling_mult=float(row["growth_ceiling_mult"]),
        curve_slope_proxy=float(row["curve_slope_proxy"]) if np.isfinite(row["curve_slope_proxy"]) else 0.0,
        components={"z_ftq": float(row["z_ftq"]) if np.isfinite(row["z_ftq"]) else 0.0,
                    "z_credit": float(row["z_credit"]) if np.isfinite(row["z_credit"]) else 0.0,
                    "z_bondvol": float(row["z_bondvol"]) if np.isfinite(row["z_bondvol"]) else 0.0})


def demo() -> None:
    """Self-check: neutral history -> mults ~1.0; a credit crash -> de-risk (<1)."""
    import numpy as np
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(0)
    base = {s: 100 * np.cumprod(1 + 0.0002 + 0.003 * rng.standard_normal(len(idx)))
            for s in ("SHY", "IEF", "TLT", "LQD", "HYG")}
    bp = pd.DataFrame(base, index=idx)
    cfg = BondOverlayConfig(enabled=True)
    r0 = compute_bond_regime(bp, cfg)
    assert 0.6 <= r0.vol_budget_mult <= 1.11, r0.vol_budget_mult
    # inject a credit crash at the end: HYG collapses, IEF/LQD hold
    bp2 = bp.copy()
    bp2.iloc[-30:, bp2.columns.get_loc("HYG")] *= np.linspace(1.0, 0.80, 30)
    r1 = compute_bond_regime(bp2, cfg)
    assert r1.risk_off_z > r0.risk_off_z, (r0.risk_off_z, r1.risk_off_z)
    assert r1.vol_budget_mult < 1.0 and r1.growth_ceiling_mult <= 1.0
    print(f"bond_regime demo OK: neutral mult={r0.vol_budget_mult:.3f}  "
          f"crash mult={r1.vol_budget_mult:.3f} (risk_off {r1.risk_off_z:+.2f})")


if __name__ == "__main__":
    demo()
