"""Entry-Alpha engine (V6 spec sections 13-15, 34).

The mandatory entry gate. It does NOT look at indicators to decide "is this a nice setup"; it asks
the only question that matters: does this entry set belong to a feature-state subgroup whose FORWARD
executable price behavior is directionally ASYMMETRIC (MFE materially > |MAE|) and net-positive
after cost -- and does it beat matched random-entry and simple-momentum controls?

A key falsification guard (spec section 34): if MFE and |MAE| grow proportionally as the holding
period lengthens, that is NOT alpha -- it is just a larger symmetric random walk. The decision logic
tests the RATIO and the net drift, not the raw MFE.

Pure/deterministic. No I/O, no network, no LLM. Reused by the multi-horizon diagnostic and (later)
by the live EntryAlphaAgent wrapper.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

import numpy as np

# decisions
ALPHA_CONFIRMED = "ALPHA_CONFIRMED"
ALPHA_WEAK = "ALPHA_WEAK"
NO_ALPHA_EVIDENCE = "NO_ALPHA_EVIDENCE"
OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STALE_EVIDENCE = "STALE_EVIDENCE"
VETO = "VETO"


@dataclass
class EntrySet:
    """One symbol's entries for excursion measurement.
    highs/lows/opens/closes: 1-D float arrays of the timeframe's bars (5m OR daily).
    entry_idx: integer indices where an entry fires (fill at idx+1 open).
    atr: per-bar ATR array (same length) or None.
    cost_rt_bps: modeled round-trip cost for this symbol.
    """
    symbol: str
    highs: np.ndarray
    lows: np.ndarray
    opens: np.ndarray
    closes: np.ndarray
    entry_idx: np.ndarray
    cost_rt_bps: float
    atr: Optional[np.ndarray] = None


def _excursion_one(es: EntrySet, horizon: int, target_bps: Optional[float], stop_bps: Optional[float]):
    """Return lists (mfe_bps, mae_bps, hold_net_bps, tgt_first_flags) for one EntrySet/horizon.
    hold_net = (horizon-end close / entry-open - 1)*1e4 - cost  (pure directional drift, net).
    tgt_first_flags: +1 target-before-stop, -1 stop-first, 0 neither (only if target & stop given).
    Conservative intrabar: the low is tested against the stop before the high can reach the target.
    """
    h, l, o, c = es.highs, es.lows, es.opens, es.closes
    n = len(c)
    rt = es.cost_rt_bps
    mfe, mae, hold_net, tgt = [], [], [], []
    for i in es.entry_idx:
        if i + 1 >= n:
            continue
        ep = o[i + 1]
        if ep <= 0:
            continue
        end = min(i + 1 + horizon, n - 1)
        if end <= i + 1:
            continue
        seg_h = h[i + 1:end + 1]
        seg_l = l[i + 1:end + 1]
        mfe.append((seg_h.max() / ep - 1.0) * 1e4)
        mae.append((seg_l.min() / ep - 1.0) * 1e4)
        hold_net.append((c[end] / ep - 1.0) * 1e4 - rt)
        if target_bps is not None and stop_bps is not None:
            tgt_p = ep * (1 + target_bps / 1e4)
            stop_p = ep * (1 - stop_bps / 1e4)
            flag = 0
            for j in range(i + 1, end + 1):
                if l[j] <= stop_p:
                    flag = -1
                    break
                if h[j] >= tgt_p:
                    flag = 1
                    break
            tgt.append(flag)
    return mfe, mae, hold_net, tgt


def horizon_profile(entry_sets: list[EntrySet], horizon: int,
                    target_bps: Optional[float] = None, stop_bps: Optional[float] = None) -> dict:
    """Aggregate excursion stats across all symbols for one horizon."""
    mfe, mae, hold_net, tgt = [], [], [], []
    for es in entry_sets:
        a, b, hn, t = _excursion_one(es, horizon, target_bps, stop_bps)
        mfe += a; mae += b; hold_net += hn; tgt += t
    n = len(mfe)
    if n == 0:
        return {"n": 0}
    avg_mfe = statistics.mean(mfe)
    avg_mae = statistics.mean(mae)
    ratio = avg_mfe / abs(avg_mae) if abs(avg_mae) > 1e-9 else float("inf")
    out = {
        "n": n,
        "avg_mfe_bps": round(avg_mfe, 2),
        "avg_mae_bps": round(avg_mae, 2),
        "median_mfe_bps": round(statistics.median(mfe), 2),
        "median_mae_bps": round(statistics.median(mae), 2),
        "mfe_mae_ratio": round(ratio, 3) if ratio != float("inf") else "inf",
        "hold_to_horizon_net_bps": round(statistics.mean(hold_net), 3),
        "hold_net_win_rate": round(sum(1 for x in hold_net if x > 0) / n, 3),
    }
    if tgt:
        decided = [x for x in tgt if x != 0]
        out["target_before_stop"] = round(sum(1 for x in tgt if x == 1) / len(decided), 3) if decided else None
        out["target_or_stop_decided_frac"] = round(len(decided) / len(tgt), 3)
    return out


def random_control(entry_sets: list[EntrySet], n_per_symbol_source: list[EntrySet],
                   horizon: int, seed: int = 12345) -> dict:
    """Matched random-entry control: for each symbol, draw the SAME number of entries uniformly at
    random from valid indices, then measure the horizon profile. Isolates whether the real entries
    beat 'enter any day' on the same instruments/period."""
    rng = np.random.default_rng(seed)
    ctrl = []
    for es, src in zip(entry_sets, n_per_symbol_source):
        k = len(src.entry_idx)
        valid_hi = len(es.closes) - horizon - 2
        if k <= 0 or valid_hi <= 1:
            continue
        idx = rng.integers(0, valid_hi, size=k)
        ctrl.append(EntrySet(es.symbol, es.highs, es.lows, es.opens, es.closes,
                             np.asarray(sorted(set(idx.tolist()))), es.cost_rt_bps, es.atr))
    return horizon_profile(ctrl, horizon)


@dataclass
class AlphaDecision:
    decision: str
    horizon: int
    ratio: float
    hold_net_bps: float
    net_vs_control_bps: float
    n: int
    reason: str


def decide(profile: dict, control_profile: dict, horizon: int,
           min_n: int = 200, min_ratio: float = 1.15, min_net_bps: float = 0.0,
           min_edge_vs_control_bps: float = 1.0) -> AlphaDecision:
    """Map evidence -> decision. ALPHA_CONFIRMED requires: enough samples AND asymmetric excursion
    (ratio >= min_ratio) AND net-positive drift (>= min_net_bps) AND beating the random control by
    >= min_edge_vs_control_bps. Anything less is graded down honestly -- never rounded up."""
    n = profile.get("n", 0)
    if n < min_n:
        return AlphaDecision(INSUFFICIENT_DATA, horizon, 0.0, 0.0, 0.0, n,
                             f"n={n} < min_n={min_n}")
    ratio = profile.get("mfe_mae_ratio")
    ratio = float(ratio) if ratio not in (None, "inf") else 999.0
    net = float(profile.get("hold_to_horizon_net_bps", 0.0))
    ctrl_net = float(control_profile.get("hold_to_horizon_net_bps", 0.0)) if control_profile.get("n") else 0.0
    edge = net - ctrl_net
    # CONFIRMED (deployable): favorable asymmetry AND net-positive AND genuinely beats the control.
    confirmed = (ratio >= min_ratio and net >= min_net_bps and edge >= min_edge_vs_control_bps)
    # WEAK (CONTINUE_RESEARCH): a real selection SIGNAL -- favorable asymmetry AND beats the control
    # by the required margin -- but NOT net-positive. This is evidence of selection, not of profit;
    # it is honestly graded down, never rounded up to CONFIRMED.
    signal = (ratio >= min_ratio and edge >= min_edge_vs_control_bps)
    marginal = (ratio >= 1.05 and net > 0 and edge > 0)
    if confirmed:
        dec, reason = ALPHA_CONFIRMED, f"ratio={ratio:.2f}>={min_ratio}, net={net:.2f}bps>=0, edge_vs_ctrl={edge:.2f}bps"
    elif signal:
        dec, reason = ALPHA_WEAK, (f"selection signal (ratio={ratio:.2f}, beats control by {edge:.2f}bps) "
                                   f"but net={net:.2f}bps NOT positive -> not deployable")
    elif marginal:
        dec, reason = ALPHA_WEAK, f"marginal (ratio={ratio:.2f}, net={net:.2f}bps, edge={edge:.2f}bps)"
    else:
        dec, reason = NO_ALPHA_EVIDENCE, f"ratio={ratio:.2f}, net={net:.2f}bps, edge_vs_ctrl={edge:.2f}bps"
    return AlphaDecision(dec, horizon, ratio, round(net, 3), round(edge, 3), n, reason)
