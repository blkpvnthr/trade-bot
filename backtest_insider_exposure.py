#!/usr/bin/env python3
"""backtest_insider_exposure.py - Track-1-step-1: continuous market/regime-vol EXPOSURE
scaling for the vol-gated insider-shares book (the validated live edge).

Exposure scaling = pure position SIZING. It multiplies the per-trade capital (ALLOC) by a
point-in-time market scalar computed BEFORE entry. It never touches the trade's realized
return, its exit, or whether the trade is taken - so it cannot clip the right tail (the
reason exit logic is deferred to Track-1-step-3). Trade selection and count are invariant
across variants by construction; only the size-weighting of the book changes.

Why the normalization matters: Sharpe is scale-invariant, so a scalar that merely
deleverages looks identical on a risk-adjusted basis. To isolate the TIMING of exposure (not
its level), every variant's scalar series is normalized to the SAME mean deployed exposure as
the baseline. A variant only wins if it puts capital to work at better times, not less of it.

Variants:
  1. incumbent baseline        - fixed ALLOC (today's book: vol-gate entry + fixed 3d timer)
  2. trend exposure scalar      - SPY/QQQ distance above/below 200-SMA (PIT)
  3. realized-vol regime scalar - inverse SPY 20d realized vol vs a fixed 16% reference (PIT)
  4. combined trend x vol

Reuses backtest_vol_gated_insider._collect (trades) + backtest_insider_copier.load_bars
(market series). Research-only; writes JSON/CSV/MD; touches no live/order/broker path.

  python backtest_insider_exposure.py
  python backtest_insider_exposure.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

HOLD = 3
COST = 0.010            # 100 bps round-trip (the insider-copier slippage assumption)
ALLOC = 0.10            # baseline per-position fraction
MAX_OPEN = 3
TRAIN_END = _date(2024, 1, 1)
START_EQUITY = 10_000.0
OUT = base.PROJECT_ROOT / "backtest_outputs"

# exposure-scalar knobs
TREND_BAND = 0.10       # +/-10% from 200-SMA spans the [0,1] trend scalar
VOL_REF = 0.16          # fixed long-run SPY annualized vol (no-lookahead reference)
VOL_CLIP = (0.30, 1.80)  # inverse-vol scalar clamp


# ---- PIT market scalars ------------------------------------------------------------
def _indicator_frame(sym: str) -> pd.DataFrame | None:
    bars = base.load_bars(sym)
    if bars is None or bars.empty:
        return None
    df = pd.DataFrame(index=pd.Index(list(bars.index)))
    df["close"] = bars["close"].values
    df["sma200"] = df["close"].rolling(200).mean()
    ret = df["close"].pct_change()
    df["rv20"] = ret.rolling(20).std() * np.sqrt(252)
    return df


def _row_before(df: pd.DataFrame, d: _date):
    """Last indicator row STRICTLY before entry date d (point-in-time, no lookahead)."""
    import bisect
    i = bisect.bisect_left(list(df.index), d) - 1
    return df.iloc[i] if i >= 0 else None


def build_scalars(trades: list[dict]) -> dict[str, np.ndarray]:
    """Return raw (un-normalized) scalar arrays aligned to `trades`, plus the regime label."""
    spy = _indicator_frame("SPY")
    qqq = _indicator_frame("QQQ")
    trend, vol, regime = [], [], []
    for t in trades:
        d = t["date"]
        tr_parts = []
        for df in (spy, qqq):
            if df is None:
                continue
            r = _row_before(df, d)
            if r is None or not np.isfinite(r["sma200"]) or r["sma200"] <= 0:
                continue
            z = r["close"] / r["sma200"] - 1.0
            tr_parts.append(float(np.clip(0.5 + z / TREND_BAND, 0.0, 1.0)))
        trend.append(float(np.mean(tr_parts)) if tr_parts else 1.0)

        rs = _row_before(spy, d) if spy is not None else None
        if rs is not None and np.isfinite(rs["rv20"]) and rs["rv20"] > 1e-6:
            vol.append(float(np.clip(VOL_REF / rs["rv20"], *VOL_CLIP)))
        else:
            vol.append(1.0)

        rg = "n/a"
        if rs is not None and np.isfinite(rs["sma200"]) and rs["sma200"] > 0:
            rg = "bull" if rs["close"] >= rs["sma200"] else "bear"
        regime.append(rg)

    trend = np.array(trend); vol = np.array(vol)
    return {"trend": trend, "vol": vol, "combined": trend * vol,
            "baseline": np.ones(len(trades)), "regime": np.array(regime)}


def _normalize(raw: np.ndarray) -> np.ndarray:
    """Match mean deployed exposure to the baseline (mean 1.0) so the test isolates TIMING."""
    m = raw.mean()
    return raw / m if m > 1e-9 else np.ones_like(raw)


# ---- portfolio sim (tearsheet_insider.equity_curve, ALLOC scaled per trade) --------
def equity_curve(trades: list[dict], size: np.ndarray):
    daily: dict = {}
    open_until: list = []
    pt, taken_size = [], []                 # per-trade net return + its size multiplier
    for r, sz in zip(trades, size):
        d0 = r["date"]
        open_until = [x for x in open_until if x > d0]
        if len(open_until) >= MAX_OPEN:
            continue                         # capacity full -> skip (identical across variants)
        net = r["gross"] - COST
        pt.append(net); taken_size.append(sz)
        per_day = (1 + net) ** (1 / HOLD) - 1
        for k in range(HOLD):
            dd = d0 + timedelta(days=k)
            daily[dd] = daily.get(dd, 0.0) + ALLOC * sz * per_day
        open_until.append(d0 + timedelta(days=HOLD))
    s = pd.Series(daily).sort_index()
    s.index = pd.to_datetime(s.index)
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full, fill_value=0.0)
    eq = START_EQUITY * (1 + s).cumprod()
    return eq, s, np.array(pt), np.array(taken_size)


def _drop_topn_total(trades, size, n):
    """Total return after removing the n trades with the largest scaled contribution."""
    contrib = [( (r["gross"] - COST) * sz, i) for i, (r, sz) in enumerate(zip(trades, size))]
    drop = {i for _, i in sorted(contrib, reverse=True)[:n]}
    kept = [(r, sz) for j, (r, sz) in enumerate(zip(trades, size)) if j not in drop]
    if not kept:
        return 0.0
    eq, *_ = equity_curve([r for r, _ in kept], np.array([sz for _, sz in kept]))
    return float(eq.iloc[-1] / eq.iloc[0] - 1)


def metrics(trades, size) -> dict:
    eq, ret, pt, tsz = equity_curve(trades, size)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    tot = float(eq.iloc[-1] / eq.iloc[0] - 1)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else 0.0
    maxdd = float((eq / eq.cummax() - 1).min())
    act = ret[ret != 0]
    sharpe = float(act.mean() / act.std() * np.sqrt(252)) if len(act) and act.std() > 0 else 0.0
    # size-weighted dollar PF + concentration (where sizing actually bites)
    wnet = pt * tsz
    w, l = wnet[wnet > 0], wnet[wnet < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) and l.sum() != 0 else float("inf")
    order = np.argsort(wnet)[::-1]
    topc = {f"top{n}_contribution": (round(float(wnet[order[:n]].sum() / wnet.sum()), 4)
                                     if wnet.sum() != 0 else None) for n in (1, 3, 5)}
    dropn = {f"total_drop_top{n}": round(_drop_topn_total(trades, size, n), 4) for n in (1, 3, 5)}
    return {
        "trades_taken": int(len(pt)),
        "avg_deployed_exposure": round(float(ALLOC * tsz.mean()), 4),   # ~ALLOC if normalized
        "win_rate": round(float((pt > 0).mean()), 4),                   # trade-level (invariant)
        "avg_R_net": round(float(pt.mean()), 5),                        # trade-level (invariant)
        "CAGR": round(cagr, 4),
        "total_return": round(tot, 4),
        "sharpe_active": round(sharpe, 3),
        "max_drawdown": round(maxdd, 4),
        "calmar": round(cagr / abs(maxdd), 3) if maxdd < 0 else None,
        "pf_size_weighted": ("inf" if pf == float("inf") else round(pf, 3)),
        **topc, **dropn,
    }


def regime_breakdown(trades, regime) -> dict:
    out = {}
    for rg in ("bull", "bear", "n/a"):
        idx = [i for i, x in enumerate(regime) if x == rg]
        if not idx:
            continue
        g = np.array([trades[i]["gross"] - COST for i in idx])
        out[rg] = {"n": len(idx), "mean_net": round(float(g.mean()), 5),
                   "win_rate": round(float((g > 0).mean()), 4)}
    return out


def run() -> dict:
    ins = vg._collect(hold=HOLD, min_adv=1_000_000)
    vthr = float(np.percentile([r["rv20"] for r in ins if r["date"] < TRAIN_END], 80))
    trades = sorted([r for r in ins if r["rv20"] >= vthr], key=lambda r: r["date"])
    sc = build_scalars(trades)
    regime = sc.pop("regime")

    variants = {"1_baseline": "baseline", "2_trend": "trend",
                "3_vol": "vol", "4_combined": "combined"}
    results, sizes = {}, {}
    for label, key in variants.items():
        size = _normalize(sc[key])
        sizes[label] = size
        results[label] = metrics(trades, size)

    payload = {
        "strategy": "vol_gated_insider_shares",
        "params": {"hold_days": HOLD, "cost_bps": COST * 1e4, "alloc": ALLOC,
                   "max_open": MAX_OPEN, "vol_gate_pctile": 80, "train_end": str(TRAIN_END),
                   "trend_band": TREND_BAND, "vol_ref": VOL_REF, "vol_clip": VOL_CLIP},
        "n_trades_universe": len(trades),
        "note": "exposure scaling is pure sizing (trade selection/count invariant); every "
                "variant normalized to the baseline's mean deployed exposure to isolate "
                "TIMING of risk, not level. give-back EXITS deliberately NOT included.",
        "variants": results,
        "regime_breakdown": regime_breakdown(trades, regime),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "insider_exposure_backtest.json").write_text(json.dumps(payload, indent=2, default=str))
    # per-trade CSV (date, sym, gross, net, regime, each variant's normalized size)
    with open(OUT / "insider_exposure_trades.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "sym", "gross", "net", "regime"] + list(variants))
        for i, t in enumerate(trades):
            w.writerow([t["date"], t["sym"], round(t["gross"], 5), round(t["gross"] - COST, 5),
                        regime[i]] + [round(float(sizes[v][i]), 4) for v in variants])
    (OUT / "insider_exposure_backtest.md").write_text(_md(payload))
    return payload


def _md(p: dict) -> str:
    keys = ["trades_taken", "avg_deployed_exposure", "win_rate", "avg_R_net", "CAGR",
            "sharpe_active", "max_drawdown", "calmar", "pf_size_weighted",
            "top1_contribution", "top5_contribution", "total_return",
            "total_drop_top1", "total_drop_top3", "total_drop_top5"]
    L = ["# Insider book - market/regime-vol EXPOSURE scaling (Track-1-step-1)", "",
         f"_{p['n_trades_universe']} vol-gated trades, {p['params']['cost_bps']:.0f}bps, "
         f"{p['params']['hold_days']}d hold. Pure sizing; mean exposure matched across variants._", "",
         "| Metric | " + " | ".join(p["variants"]) + " |",
         "|---|" + "---|" * len(p["variants"])]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(str(p["variants"][v].get(k)) for v in p["variants"]) + " |")
    L += ["", "## Regime breakdown (trade-level, where the edge concentrates)",
          "| Regime | n | mean net | win rate |", "|---|---|---|---|"]
    for rg, s in p["regime_breakdown"].items():
        L.append(f"| {rg} | {s['n']} | {s['mean_net']*100:+.2f}% | {s['win_rate']*100:.0f}% |")
    L += ["", "## Read",
          "- `win_rate`/`avg_R_net` are trade-level and IDENTICAL across variants by design "
          "(pure sizing). The discriminators are `sharpe_active`, `max_drawdown`, `calmar`, "
          "and the `drop_topN` robustness.",
          "- `avg_deployed_exposure` ~ alloc confirms the mean-exposure normalization (timing test, not leverage).",
          "- A variant only wins if it lifts Sharpe/Calmar or cuts drawdown WITHOUT the gain "
          "vanishing under drop-top-N (i.e. not just concentration in a few lucky names)."]
    return "\n".join(L)


def _selftest() -> None:
    # constant scalar == baseline (sizing identity); mean-normalization matches exposure.
    trades = [{"date": _date(2024, 1, 2) + timedelta(days=7 * i), "sym": f"S{i}",
               "gross": (0.05 if i % 2 else -0.03), "rv20": 0.3, "adv": 2e6} for i in range(20)]
    base_eq, *_ = equity_curve(trades, np.ones(len(trades)))
    raw = np.array([2.0] * len(trades))            # constant 2x, mean-normalized -> 1.0
    same_eq, *_ = equity_curve(trades, _normalize(raw))
    assert np.allclose(base_eq.values, same_eq.values), "normalized constant scalar must equal baseline"
    wobble = _normalize(np.array([0.5, 1.5] * 10))
    assert abs(wobble.mean() - 1.0) < 1e-9, "normalization must set mean exposure to 1.0"
    # a timing scalar that sizes UP winners must beat baseline Sharpe (sanity of the mechanism)
    tilt = _normalize(np.array([1.5 if t["gross"] > 0 else 0.5 for t in trades]))
    m_base, m_tilt = metrics(trades, np.ones(len(trades))), metrics(trades, tilt)
    assert m_tilt["sharpe_active"] > m_base["sharpe_active"], "oracle timing tilt should lift Sharpe"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        out = run()
        print((OUT / "insider_exposure_backtest.md").read_text())
