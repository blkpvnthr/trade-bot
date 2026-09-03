"""Deterministic synthetic panel for demo/tests - a planted alpha + pure noise, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_panel(n_days: int = 400, n_syms: int = 30, seed: int = 0, persistence: float = 0.0):
    """Returns (closes, volumes, factor). Returns are driven by a hidden cross-sectional factor f_t
    (known at close t) plus noise, so a signal that recovers f_t is a REAL alpha; noise is not.

    persistence = AR(1) coefficient on the factor. 0 = i.i.d. (full daily turnover); ->1 = slow
    factor (low turnover), needed to fixture a *promotable* alpha (turnover not pathological).
    factor is returned explicitly (NOT via .attrs) - a DataFrame in .attrs propagates to every
    derived frame and breaks pandas' attrs-equality check inside stack().
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n_days, freq="B", tz="UTC")
    syms = [f"S{i:02d}" for i in range(n_syms)]
    raw = rng.standard_normal((n_days, n_syms))                # hidden factor, known at close t
    if persistence:
        f = raw.copy()
        for t in range(1, n_days):
            f[t] = persistence * f[t - 1] + np.sqrt(1 - persistence ** 2) * raw[t]
    else:
        f = raw
    noise = rng.standard_normal((n_days, n_syms))
    ret = np.zeros((n_days, n_syms))
    ret[1:] = 0.0040 * f[:-1] + 0.020 * noise[1:]              # return t->t+1 loads on f_t
    closes = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=syms)
    vol = pd.DataFrame(1e6 * np.exp(0.3 * rng.standard_normal((n_days, n_syms))),
                       index=dates, columns=syms)
    factor = pd.DataFrame(f, index=dates, columns=syms)
    return closes, vol, factor
