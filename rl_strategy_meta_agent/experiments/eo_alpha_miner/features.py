"""Leak-free engineered features from OHLCV, plus the next-period-return label.

The evolutionary search composes these ARITHMETICALLY (the C++ EO core does
+,-,*,safe_div,min,max,abs,sign,neg); the time-series operators the spec lists
(rolling_mean/std, zscore, rank, delay, delta) live here as precomputed feature
columns. Everything uses only past/contemporaneous data -- ONLY `target` looks
forward (the label), so there is no lookahead in any feature.

Panel convention: MultiIndex (symbol, timestamp); per-symbol time-series ops via
groupby(level=0), cross-sectional ops via groupby(level=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def _by_sym(panel, col):
    return panel[col].groupby(level=0)


def build_features(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """ohlcv: columns open/high/low/close/volume (+ optional vwap), MultiIndex
    (symbol, timestamp). Returns (panel_with_features, feature_names, 'target')."""
    p = ohlcv.sort_index().copy()
    g = lambda c: p[c].groupby(level=0)
    ts = lambda s: s.groupby(level=1)                    # cross-section per timestamp

    p["ret"] = g("close").pct_change()
    p["ret_hl"] = (p["high"] - p["low"]) / p["close"].replace(0, np.nan)
    for w in (5, 10, 20):
        p[f"sma_{w}"] = g("close").transform(lambda s: s.rolling(w).mean()) / p["close"] - 1.0
        p[f"vol_sma_{w}"] = g("volume").transform(lambda s: s.rolling(w).mean())
    for w in (10, 20):
        p[f"ret_std_{w}"] = p.groupby(level=0)["ret"].transform(lambda s: s.rolling(w).std())
    # zscore_20 of close and volume (rolling, per symbol)
    for c in ("close", "volume"):
        m = g(c).transform(lambda s: s.rolling(20).mean())
        sd = g(c).transform(lambda s: s.rolling(20).std())
        p[f"z20_{c}"] = (p[c] - m) / (sd + 1e-9)
    # cross-sectional rank (per timestamp) of return and volume
    p["rank_ret"] = ts(p["ret"]).rank(pct=True)
    p["rank_vol"] = ts(p["volume"]).rank(pct=True)
    # delay + delta
    for d in (1, 5):
        p[f"delay_ret_{d}"] = p.groupby(level=0)["ret"].shift(d)
        p[f"delta_close_{d}"] = g("close").transform(lambda s: s - s.shift(d)) / p["close"]
    if "vwap" in p.columns:
        p["vwap_dist"] = p["close"] / p["vwap"] - 1.0

    # label: next-period return (the ONLY forward-looking column)
    p["target"] = p.groupby(level=0)["ret"].shift(-1)

    feats = [c for c in p.columns if c not in OHLCV + ["vwap", "target"]]
    return p, feats, "target"


def export_for_eo(panel: pd.DataFrame, feats: list[str], target: str, path: str) -> int:
    """Write 'group,target,<feats>' CSV (rows sorted by timestamp so the C++ side
    sees contiguous cross-sections). group = factorized timestamp id."""
    sub = panel[feats + [target]].dropna()
    sub = sub.sort_index(level=1)
    grp = pd.factorize(sub.index.get_level_values(1))[0]
    out = pd.DataFrame({"group": grp, "target": sub[target].values})
    for f in feats:
        out[f] = sub[f].values
    out.to_csv(path, index=False)
    return len(out)


def _demo():
    rng = np.random.default_rng(0)
    rows = []
    for s in ("AAA", "BBB", "CCC"):
        px = 100.0
        for t in pd.date_range("2024-01-02 14:30", periods=60, freq="15min", tz="UTC"):
            px *= 1 + rng.normal(0, 0.01)
            rows.append((s, t, px, px * 1.01, px * 0.99, px, rng.integers(1e4, 1e5)))
    df = pd.DataFrame(rows, columns=["symbol", "timestamp", *OHLCV]).set_index(["symbol", "timestamp"])
    p, feats, tgt = build_features(df)
    assert len(feats) > 10 and tgt == "target"
    assert p["target"].groupby(level=0).apply(lambda s: s.iloc[-1]).isna().all()  # last bar has no future
    n = export_for_eo(p, feats, tgt, "/tmp/_eo_feat_demo.csv")
    assert n > 0
    print(f"features._demo OK  {len(feats)} features, exported {n} rows")


if __name__ == "__main__":
    _demo()
