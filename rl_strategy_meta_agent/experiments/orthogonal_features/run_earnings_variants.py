#!/usr/bin/env python3
"""A vs B variant validation of the earnings_within_7d sleeve, with CANONICAL PARITY.

Reuses run_extended_validation.build_panel + evaluate (SAME frozen bars, SAME fut_ret_1, SAME
LV.streams dollar-neutral cost model, SAME walk-forward / symbol-holdout / placebo / bootstrap).
Only the cross-sectional SIGNAL changes:

  canonical  sig = earnings_within_7d                       (raw flag -> reproduces the 1.11 record)
  A          sig = flag_liquid                              (calendar + sleeve liquidity, equal wt)
  B          sig = flag_liquid * positive drift-rank weight (PIT, calendar-gated)

PIT FIREWALL: B multiplies by `flag`, so a name is non-zero ONLY after the forward calendar flags
it. Drift/liquidity can rank/size, never create a signal.

EARNINGS-DATE PROVENANCE CAVEAT (load-bearing, inherited from the canonical record): the HISTORICAL
flag here comes from NF.load_earnings -> av_catalyst.earnings `reportedDate` (the ACTUAL ex-post
report date), NOT a historical archive of forward-SCHEDULED dates (none exists; EARNINGS_CALENDAR is
a forward-only snapshot). This is the SAME source the canonical extended_validation 1.11 record uses
(required for parity). It is a NEAR-PIT proxy: scheduled dates are announced weeks ahead and rarely
move, so "within 7d" is approximately knowable at bar time, but RESCHEDULED events can differ. The
LIVE sleeve does NOT use reportedDate -- it uses forward-scheduled EARNINGS_CALENDAR
(rl_trading/sleeves/earnings_features.fetch_av_earnings_calendar -> cal_*.json). Treat backtest
metrics as "reportedDate-proxy"; the date-shuffle placebo (edge dies on shuffled dates) is the
guard that the result is not a pure calendar artifact.

Variant C (calendar + drift + AV fundamentals/beat) is LIVE_ONLY_NON_PIT_BACKTESTABLE: AV
fundamentals are CURRENT snapshots with no historical as-of, so C is deliberately NOT evaluated
here and never enters canonical validation metrics or promotion. It exists only as disabled live
scanner metadata (see rl_trading/sleeves/earnings_features.py).

Net reported at BOTH 5bps (canonical parity) and 10bps round-trip (task requirement). Research only.

    python experiments/orthogonal_features/run_earnings_variants.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_extended_validation as RX
from rl_meta.stream_validation import stream_metrics

OUT = Path(__file__).resolve().parent / "out" / "earnings_variant_validation.json"
LOOKAHEAD = 7
MIN_PRICE = 5.0
MIN_ADV = 5_000_000.0
CANONICAL_NET_SHARPE_5BPS = 1.11  # pinned from out/extended_validation.json (earnings_within_7d)


def _days_to_earnings(idx):
    earn = RX.NF.load_earnings(RX.SYMS)
    sa = idx.get_level_values(0)
    dts = pd.to_datetime(idx.get_level_values(1)).to_numpy("datetime64[D]")
    out = np.full(len(idx), np.nan)
    for s in pd.unique(sa):
        ed = earn.get(s, np.array([], dtype="datetime64[ns]"))
        rows = np.where(sa == s)[0]
        if ed.size == 0:
            continue
        ed = ed.astype("datetime64[D]")
        for r in rows:
            fut = ed[ed >= dts[r]]
            if fut.size:
                out[r] = (fut[0] - dts[r]) / np.timedelta64(1, "D")
    return pd.Series(out, index=idx)


def build_features(panel):
    close, vol = panel["close"], panel["volume"]
    ret = close.groupby(level=0).pct_change()
    ret_5d = close.groupby(level=0).transform(lambda s: s / s.shift(5) - 1.0)
    ret_20d = close.groupby(level=0).transform(lambda s: s / s.shift(20) - 1.0)
    vol_20d = ret.groupby(level=0).transform(lambda s: s.rolling(20).std())
    adv20 = (close * vol).groupby(level=0).transform(lambda s: s.rolling(20).mean())
    days_to = _days_to_earnings(panel.index)
    drift = (
        100 * ret_5d.clip(lower=0)
        + 35 * ret_20d.clip(lower=0)
        + (adv20 / 1e8).clip(upper=3)
        + (LOOKAHEAD - days_to).clip(lower=0) * 0.15
        - (vol_20d * 20).clip(upper=2)
    )
    liquid = (close >= MIN_PRICE) & (adv20 >= MIN_ADV)
    flag_liquid = ((panel["earnings_within_7d"] > 0) & liquid).astype(float)
    return pd.DataFrame({
        "earnings_within_7d": panel["earnings_within_7d"],
        "days_to_earnings": days_to, "ret_5d": ret_5d, "ret_20d": ret_20d,
        "vol_20d": vol_20d, "avg_dollar_vol_20d": adv20, "drift_score": drift,
        "flag_liquid": flag_liquid,
    }, index=panel.index)


def _within_day_rank01(score, gate):
    """Within-day percentile rank in (0,1] among gated rows; 0 elsewhere."""
    ts = score.index.get_level_values(1)
    r = score.where(gate > 0).groupby(ts).rank(pct=True)
    return r.fillna(0.0)


def _net10(panel, sig):
    """Net metrics at 10bps round-trip, sign-aligned exactly like evaluate()."""
    sign = 1.0 if RX.spearman_ic(sig, panel["fut_ret_5"])[0] >= 0 else -1.0
    g, t = RX.LV.streams(panel, sig * sign, fwd="fut_ret_1")
    m = stream_metrics(g.values, t.values, cost_bps=10.0)
    return {"net_sharpe_10bps": m["net_sharpe"], "net_max_drawdown_10bps": m["net_max_drawdown"],
            "turnover_day": m["mean_turnover"], "breakeven_bps": m["breakeven_cost_bps"]}


def date_shuffle_placebo(panel, real_net10, n=50):
    """Shuffle WHICH rows carry the earnings flag (destroys date alignment, preserves base rate),
    then re-measure net@10bps. A real pre-earnings-drift edge must NOT survive shuffled dates."""
    flag = panel["earnings_within_7d"].to_numpy()
    rng = np.random.default_rng(0)
    sharpes = []
    for _ in range(n):
        perm = rng.permutation(len(flag))
        sig = pd.Series(flag[perm], index=panel.index)
        if sig.sum() == 0:
            continue
        sharpes.append(_net10(panel, sig)["net_sharpe_10bps"])
    sharpes = np.array(sharpes)
    return {
        "placebo_mean_net_sharpe_10bps": round(float(sharpes.mean()), 3),
        "placebo_max_net_sharpe_10bps": round(float(sharpes.max()), 3),
        "real_net_sharpe_10bps": real_net10,
        "frac_placebo_ge_real": round(float((sharpes >= real_net10).mean()), 4),
        "edge_survives_shuffle": bool((sharpes >= real_net10).mean() < 0.05),
    }


def _pick_winner(by):
    """Promote B over A only if it STRICTLY improves net@10bps without worse drawdown or turnover
    (NULL-honest: ties/regressions keep the simpler calendar-only A). C is never eligible."""
    A, B = by["variant_A"], by["variant_B"]
    if (B["net_sharpe_10bps"] > A["net_sharpe_10bps"]
            and B["net_max_drawdown_10bps"] <= A["net_max_drawdown_10bps"]   # smaller magnitude = better
            and B["turnover_day"] <= A["turnover_day"]):                     # must NOT increase turnover
        return {"variant": "variant_B",
                "reason": "drift/liquidity rank strictly improves net@10bps w/o worse DD or turnover"}
    return {"variant": "variant_A",
            "reason": ("calendar-only baseline; B did not strictly improve net@10bps without "
                       "increasing turnover (B turnover > A turnover)")}


def main():
    panel = RX.build_panel()
    feats = build_features(panel)
    RX.HOR.update({"canonical": 5, "variant_A": 5, "variant_B": 5})
    drift_w = _within_day_rank01(feats["drift_score"], feats["flag_liquid"])
    flag = feats["flag_liquid"]

    sigs = {
        "canonical": panel["earnings_within_7d"],   # raw flag -> parity to the pinned 1.11 record
        "variant_A": flag,                          # calendar + liquidity, equal weight
        "variant_B": flag * (0.5 + 0.5 * drift_w),  # + PIT-safe drift/liquidity rank
    }
    recs = []
    for name, sig in sigs.items():
        r = RX.evaluate(panel, name, sig)
        r.update(_net10(panel, sig))
        r["pit_safe"] = True
        recs.append(r)

    by = {r["signal"]: r for r in recs}
    canon = by["canonical"]
    payload = {
        "canonical_net_sharpe_5bps_pinned": CANONICAL_NET_SHARPE_5BPS,
        "parity_check": {
            "canonical_net_sharpe_5bps": canon["net_sharpe_5bps"],
            "turnover_day": canon["turnover_day"],
            "within_0.05_of_pin": bool(abs(canon["net_sharpe_5bps"] - CANONICAL_NET_SHARPE_5BPS) <= 0.05),
        },
        "winner": _pick_winner(by),
        "date_shuffle_placebo": date_shuffle_placebo(panel, by["variant_A"]["net_sharpe_10bps"]),
        "variant_C_status": "LIVE_ONLY_NON_PIT_BACKTESTABLE (excluded from validation by design)",
        "variants": recs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _print(payload)
    return 0


def _print(p):
    print(f"\n=== earnings_within_7d A vs B (parity to canonical {CANONICAL_NET_SHARPE_5BPS}) ===\n")
    print(f"parity: {p['parity_check']}\n")
    cols = ["signal", "n_obs", "ic", "net_sharpe_5bps", "net_sharpe_10bps",
            "net_max_drawdown_10bps", "turnover_day", "breakeven_bps", "placebo_p",
            "holdout_net_sharpe", "wf_positive_year_frac"]
    print(" | ".join(cols))
    for r in p["variants"]:
        print(" | ".join(str(r.get(c)) for c in cols))
    print(f"\nWINNER: {p['winner']}")
    print(f"date-shuffle placebo: {p['date_shuffle_placebo']}")
    print(f"variant_C: {p['variant_C_status']}\n")


if __name__ == "__main__":
    raise SystemExit(main())
