"""
Deterministic synthetic price fixtures for V2.9.8 strategy tests.

Offline, seeded, no network. Produces a price panel in the exact schema
`strat_runner.load_prices` expects (10 ETFs + CASH_RATE_3M as an annualized
percent), engineered to exercise the full recovery-state machine including a
2008-style crash (CRISIS / severe-bear) and a recovery into BULL.

This is NOT a test module (no `test_` prefix) so pytest will not collect it;
tests and the golden-capture harness both import `make_synthetic_prices` so
there is a single fixture definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RISK_ASSETS = ("SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM", "VNQ")

# Extra bond ETFs for the (optional) bond-regime overlay. They are NOT part of the
# strategy universe -- the allocator only ever selects RISK_ASSETS -- so adding
# them here leaves every strategy/backtest number byte-identical, provided their
# RNG draws happen AFTER the 10 risk assets (they do; see the second loop below).
BOND_EXTRA = ("SHY", "LQD", "HYG")

# Market beta and daily idiosyncratic drift/vol per asset. Defensive assets get
# negative/low beta so they rally (or hold) when the equity market crashes.
_ASSET_SPEC = {
    "SPY": (1.00, 0.00030, 0.006),
    "QQQ": (1.15, 0.00040, 0.009),
    "IWM": (1.20, 0.00025, 0.011),
    "EFA": (1.05, 0.00015, 0.008),
    "EEM": (1.30, 0.00020, 0.013),
    "VNQ": (1.10, 0.00015, 0.012),
    "TLT": (-0.40, 0.00018, 0.007),
    "IEF": (-0.20, 0.00010, 0.003),
    "GLD": (0.10, 0.00030, 0.008),
    "DBC": (0.50, 0.00000, 0.010),
}

# Bond-overlay ETFs (processed AFTER the 10 so risk-asset draws are unchanged).
# HYG (high yield) carries real equity/credit beta so it sells off with stocks in
# the crash -> the LQD-vs-HYG credit spread widens and IEF-vs-HYG flight-to-quality
# rises, which is exactly what a risk-off bond overlay should detect.
_BOND_EXTRA_SPEC = {
    "SHY": (-0.05, 0.00005, 0.0010),
    "LQD": (0.20, 0.00012, 0.0040),
    "HYG": (0.55, 0.00010, 0.0060),
}


def make_synthetic_prices(seed: int = 7,
                          start: str = "2005-01-01",
                          end: str = "2015-12-31") -> pd.DataFrame:
    """
    Build a deterministic (date x [10 ETFs + CASH_RATE_3M]) price panel.

    A common market factor drives all assets via per-asset betas, with an
    engineered crash 2008-09-15..2009-03-06 (strong negative drift, high vol)
    and a one-year recovery, so `build_gate_schedule` visits CRISIS, BOTTOMING,
    RECOVERING, and BULL. Prices are total-return-style cumulative products of
    (1 + daily return), matching the auto_adjust series the backtest assumes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # --- market factor with three regimes -------------------------------------
    mkt_drift = np.full(n, 0.00035)
    mkt_vol = np.full(n, 0.010)
    crash = (dates >= "2008-09-15") & (dates <= "2009-03-06")
    recovery = (dates > "2009-03-06") & (dates <= "2010-03-06")
    mkt_drift[crash] = -0.0045
    mkt_vol[crash] = 0.030
    mkt_drift[recovery] = 0.0011
    mkt_vol[recovery] = 0.014
    market = mkt_drift + mkt_vol * rng.standard_normal(n)

    out = {}
    for asset, (beta, drift, idio) in _ASSET_SPEC.items():
        r = drift + beta * market + idio * rng.standard_normal(n)
        # Defensive flight-to-quality kick during the crash.
        if asset in ("TLT", "IEF", "GLD"):
            r[crash] += 0.0010
        prices = 100.0 * np.cumprod(1.0 + r)
        out[asset] = prices

    # Bond-overlay ETFs -- drawn AFTER the 10, so the risk assets stay byte-identical.
    for asset, (beta, drift, idio) in _BOND_EXTRA_SPEC.items():
        r = drift + beta * market + idio * rng.standard_normal(n)
        if asset == "SHY":
            r[crash] += 0.0005          # short Treasuries: mild flight-to-quality
        if asset == "HYG":
            r[crash] -= 0.0015          # high yield: extra credit stress in the crash
        out[asset] = 100.0 * np.cumprod(1.0 + r)

    df = pd.DataFrame(out, index=dates)[list(RISK_ASSETS) + list(BOND_EXTRA)]

    # --- cash rate (annualized percent), ZIRP after the crash -----------------
    rate = np.full(n, 4.5)
    rate[dates >= "2007-09-01"] = 4.0
    rate[dates >= "2008-01-01"] = 2.5
    rate[dates >= "2008-10-01"] = 0.3
    rate[dates >= "2009-01-01"] = 0.15
    # tiny wiggle so it is not perfectly constant
    rate = rate + 0.02 * np.sin(np.arange(n) / 40.0)
    df["CASH_RATE_3M"] = np.clip(rate, 0.01, None)

    return df
