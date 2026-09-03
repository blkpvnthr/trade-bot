"""Exit-quality / trade-management modeling -- leakage-controlled.

The trade-level during/exit features (whole-trade MFE/MAE, profit_capture_ratio)
are FUTURE information relative to any mid-trade decision, so they cannot be
model inputs. Instead we reconstruct each trade's bar-close return path and
build DECISION-POINT examples: at bar k, features use only the path up to k
(entry..k) plus the pre-entry features known at entry; targets describe what
happens AFTER k (k+1..exit). That is the only honest way to ask "should I exit,
reduce, or hold now?".

  inputs  (known at k): pre_entry_features + run_* (ret/mfe/mae/giveback/velocity)
  targets (future of k): early_exit_good, hold_longer_good, severe_giveback,
                         future_giveback
"""
from __future__ import annotations

import numpy as np

RUN_FEATURES = ["ret_now", "run_mfe", "run_mae", "run_giveback", "run_drawdown",
                "bars_held", "ret_velocity", "above_run_peak"]
DECISION_FLAGS = ["early_exit_good", "hold_longer_good", "severe_giveback"]
DECISION_REG = ["future_giveback"]

SEVERE_GIVEBACK = 0.01      # >=1% of return given back from peak-from-here to exit
HOLD_MARGIN = 0.005         # >=0.5% more profit ahead to call holding "good"
MAX_POINTS = 4              # decision snapshots per trade (bounds dataset size)


def return_path(ts_ns, close, entry_ns: int, exit_ns: int,
                entry_price: float, is_long: bool):
    """Bar-close return path over [entry_ns, exit_ns], signed by direction.
    `ts_ns`/`close` are pre-sorted arrays (int64 epoch-ns, float). Uses ONLY
    bars inside the trade. None if too short for a decision."""
    lo = np.searchsorted(ts_ns, entry_ns, "left")
    hi = np.searchsorted(ts_ns, exit_ns, "right")
    c = close[lo:hi]
    if len(c) < 3 or entry_price <= 0:
        return None
    r = (c - entry_price) / entry_price
    return r if is_long else -r


def sample_decision_ks(last: int, max_points: int = MAX_POINTS) -> list:
    """Decision bars k in 1..last-1 (can't decide at entry or final bar)."""
    if last < 2:
        return []
    ks = list(range(1, last))
    if len(ks) <= max_points:
        return ks
    idx = np.linspace(0, len(ks) - 1, max_points).round().astype(int)
    return [ks[i] for i in sorted(set(idx))]


def running_features(r: np.ndarray, k: int) -> dict:
    """Features at decision bar k -- ONLY uses r[0..k] (no look-ahead)."""
    upto = r[:k + 1]
    rk = float(r[k])
    mfe = float(upto.max()); mae = float(upto.min())
    return {
        "ret_now": rk, "run_mfe": mfe, "run_mae": mae,
        "run_giveback": mfe - rk, "run_drawdown": rk - mae,
        "bars_held": float(k), "ret_velocity": rk - float(r[k - 1]),
        "above_run_peak": float(rk >= mfe - 1e-12),
    }


def decision_targets(r: np.ndarray, k: int, severe: float = SEVERE_GIVEBACK,
                     margin: float = HOLD_MARGIN) -> dict:
    """Targets describing the FUTURE of the trade after bar k (r[k+1..exit])."""
    rk = float(r[k]); final = float(r[-1])
    fut = r[k + 1:]
    fut_max = float(fut.max())
    peak_from_here = max(rk, fut_max)
    return {
        "early_exit_good": int(rk > final),               # locking now beats actual exit
        "hold_longer_good": int(fut_max > rk + margin),   # more profit ahead
        "severe_giveback": int(peak_from_here - final >= severe),
        "future_giveback": round(peak_from_here - final, 6),
        "final_ret": round(final, 6),
    }


def build_decision_rows(r: np.ndarray, pre_feats: dict, meta: dict,
                        max_points: int = MAX_POINTS) -> list:
    rows = []
    for k in sample_decision_ks(len(r) - 1, max_points):
        row = dict(pre_feats)
        row.update(meta)
        row.update(running_features(r, k))
        row.update(decision_targets(r, k))
        row["k"] = k
        rows.append(row)
    return rows


# -- exit policies (for the policy-comparison backtest) --------------------
def trailing_stop_return(r: np.ndarray, trail: float = 0.01) -> float:
    """Exit at the first bar whose drop from the running peak exceeds `trail`."""
    peak = r[0]
    for k in range(1, len(r)):
        peak = max(peak, float(r[k - 1]))
        if peak - float(r[k]) >= trail:
            return float(r[k])
    return float(r[-1])


# -- mid-trade management policy derived from the giveback model -----------
# The severe_giveback classifier outputs P(the trade is about to surrender >=1%
# of its peak). This maps that probability + the current path state to ONE
# management action. It is a research decision rule, not an order: nothing here
# touches a broker. Thresholds are the backtest's swept operating points.
EXIT_THR = 0.60       # P(giveback) at/above this -> get out
TIGHTEN_THR = 0.40    # elevated risk -> tighten the stop, keep the runner
MANAGEMENT_ACTIONS = ("hold", "scale_out", "tighten_stop", "exit")


def management_action(p_giveback: float, ret_now: float, run_giveback: float,
                      exit_thr: float = EXIT_THR, tighten_thr: float = TIGHTEN_THR) -> str:
    """hold | scale_out | tighten_stop | exit from the giveback probability.

    - exit         : model says a severe giveback is likely now.
    - tighten_stop : risk is elevated but not decisive -> protect the open profit.
    - scale_out    : in profit and the model sees the peak forming (giveback has
                     started) -> bank part, ride the rest.
    - hold         : low giveback risk -> let it run.
    """
    if not np.isfinite(p_giveback):
        return "hold"
    if p_giveback >= exit_thr:
        return "exit"
    if p_giveback >= tighten_thr:
        return "tighten_stop"
    if ret_now > 0 and run_giveback > 0:
        return "scale_out"
    return "hold"


def _demo():
    r = np.array([0.0, 0.01, 0.03, 0.02, 0.05, 0.01, -0.01])   # up then give back
    f1 = running_features(r, 3)
    r2 = r.copy(); r2[4:] = -9.0                                # corrupt the future
    assert running_features(r2, 3) == f1, "running features must ignore future bars"
    t = decision_targets(r, 3)
    assert t["future_giveback"] >= 0 and "final_ret" in t
    rows = build_decision_rows(r, {"pre_x": 1.0}, {"trade_id": "t"})
    assert rows and all("run_giveback" in x for x in rows)
    assert trailing_stop_return(r, 0.01) < r.max()             # stop locks before the dump
    assert management_action(0.9, 0.03, 0.02) == "exit"
    assert management_action(0.5, 0.03, 0.02) == "tighten_stop"
    assert management_action(0.1, 0.03, 0.02) == "scale_out"
    assert management_action(0.1, 0.03, 0.0) == "hold"
    print("exit_quality._demo OK", len(rows), "decision rows")


if __name__ == "__main__":
    _demo()
