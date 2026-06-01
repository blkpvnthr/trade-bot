#!/usr/bin/env python3
"""
rl_strategy_overlay.py — Realistic trade-selection + risk-management overlay.
=============================================================================

Honest answer to "how do we lift win-rate / profit-factor and cut drawdown?". The replay
outcomes are a FIXED 3:1 bracket (≈ +3R win / −1R loss), so by identity:

        profit_factor = 3 · win_rate / (1 − win_rate)

i.e. PF 25 *requires* a 89.3% win rate and win-rate 0.65 *forces* PF 5.6 — they cannot both
hold. So we do NOT chase those numbers. The only honest levers (confirmed on the data and in
the literature) are:

  1. SELECTION — concentrate into higher-quality setups using ONLY ex-ante features
     (rel_volume, mtf_alignment, catalyst_strength, sector_regime, setup_score). Never uses
     realized_R / peak_R to decide (that would be look-ahead).
  2. RISK MANAGEMENT — lower base risk, conviction (½-Kelly-style) sizing, volatility
     targeting, a daily loss cap, and drawdown de-risking. These cut max drawdown and lift
     risk-adjusted return without inventing edge.

The selection threshold is tuned IN-SAMPLE (first 70% chronological) and reported
OUT-OF-SAMPLE (last 30%) so the gains are not curve-fit. OFFLINE; read-only; no orders,
no live wiring, no gate/risk-control changes to the live system.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import glob
import json

import rl_common as rc

DATASET = "alpha_entry_dataset"      # ex-ante features + attached realized_R


# ── ex-ante quality score (NO look-ahead) ────────────────────────────────────────────
def quality_score(row: Dict[str, Any]) -> float:
    """Composite ex-ante setup quality in ~[0, 1]. Uses only pre-trade signals."""
    rvol = rc.to_f(row.get("rel_volume"), 1.0)
    rvol_c = max(0.0, min(1.0, (rvol - 1.0) / 2.0))           # rvol 1→0, 3→1
    mtf = max(0.0, rc.to_f(row.get("mtf_alignment")))          # only reward up-alignment
    cat = max(0.0, min(1.0, rc.to_f(row.get("catalyst_strength"))))
    sec = max(0.0, rc.to_f(row.get("sector_regime")))
    setup = rc.to_f(row.get("setup_score")) / 100.0
    return round(0.35 * rvol_c + 0.30 * mtf + 0.20 * cat + 0.10 * sec + 0.05 * setup, 6)


# ── "Generating Alpha" downside gates (sentiment / regime / confluence) ──────────────
# Pillai, Ajith & Sumesh (ComSIA 2026): treat FinBERT sentiment and market regime as GATES
# that SUPPRESS entries (never as buy signals), and require multi-modal CONFLUENCE (score ≥ 2)
# before taking a trade. These are pure downside filters on the existing selection — they can
# only remove trades, never add or force one — so they preserve every live safety control.
def load_sentiment_map() -> Dict[tuple, float]:
    """Offline per-(date_tag, symbol) sentiment St ∈ [−1, 1] from the LLM universe reports.
    Reads system-data/*/*_llm_universe_sentiment_report.json (read-only, no network). The
    report's sentiment_score is [0,1]; sentiment_to_st rescales it to the paper's [−1,1].
    Missing report/symbol → caller defaults to neutral 0.0 (never reads as bearish)."""
    smap: Dict[tuple, float] = {}
    pattern = str(rc.PROJECT_ROOT / "system-data" / "*" / "*_llm_universe_sentiment_report.json")
    for f in sorted(glob.glob(pattern)):
        try:
            doc = json.loads(open(f, encoding="utf-8").read())
        except Exception:
            continue
        tag = str(doc.get("data_tag") or doc.get("date_tag") or "")
        for r in doc.get("reports", []) or []:
            sym = str(r.get("symbol") or "").upper()
            if not sym:
                continue
            smap[(tag, sym)] = rc.sentiment_to_st(r.get("sentiment_score"))
    return smap


def sentiment_st(row: Dict[str, Any], sent_map: Dict[tuple, float]) -> float:
    """St for this row: prefer an enriched ga_sentiment_st, else the offline map, else 0.0."""
    if row.get("ga_sentiment_st") is not None:
        return rc.sentiment_to_st(row.get("ga_sentiment_st"))
    key = (str(row.get("date_tag", "")), str(row.get("symbol", "")).upper())
    return sent_map.get(key, 0.0) if sent_map else 0.0


def passes_gates(row: Dict[str, Any], params: Dict[str, Any],
                 sent_map: Dict[tuple, float] = None) -> bool:
    """True if the row survives the GA downside gates enabled in `params`. Suppressive only."""
    if params.get("sentiment_gate"):
        if sentiment_st(row, sent_map or {}) < params.get("sent_gate_min", rc.SENT_GATE_MIN):
            return False                                          # paper §D: suspend on bearish news
    if params.get("regime_gate"):
        bearish = (rc.to_f(row.get("ga_regime_switch")) < 0) or \
                  (rc.to_f(row.get("spy_regime")) < 0 and rc.to_f(row.get("sector_regime")) < 0)
        if bearish:
            return False                                          # paper §F: long-only off in bear regime
    if params.get("confluence_gate"):
        row = {**row, "ga_sentiment_st": sentiment_st(row, sent_map or {})}
        if rc.generating_alpha_confluence(row)["confluence"] < params.get("min_confluence", rc.MIN_CONFLUENCE):
            return False                                          # paper Alg.1: entry needs score ≥ 2
    return True


# ── policy parameters ────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "base_risk": float(os.getenv("STRAT_BASE_RISK", "0.0025")),      # 0.25%/trade (was 0.5%)
    "max_trades_per_day": int(os.getenv("STRAT_MAX_TRADES_PER_DAY", "5")),
    "daily_loss_cap": float(os.getenv("STRAT_DAILY_LOSS_CAP", "0.01")),   # stop the day at −1%
    "derisk_k": float(os.getenv("STRAT_DERISK_K", "3.0")),          # cut size as equity DD grows
    "target_atr": float(os.getenv("STRAT_TARGET_ATR", "0.01")),     # vol-target anchor
    "quality_sizing": os.getenv("STRAT_QUALITY_SIZING", "1") != "0",
    "vol_target": os.getenv("STRAT_VOL_TARGET", "1") != "0",
    # "Generating Alpha" downside gates (suppressive only; default on for the improved policy)
    "sentiment_gate": os.getenv("STRAT_SENTIMENT_GATE", "1") != "0",
    "regime_gate": os.getenv("STRAT_REGIME_GATE", "1") != "0",
    "confluence_gate": os.getenv("STRAT_CONFLUENCE_GATE", "1") != "0",
    "sent_gate_min": float(os.getenv("STRAT_SENT_GATE_MIN", str(rc.SENT_GATE_MIN))),
    "min_confluence": int(os.getenv("STRAT_MIN_CONFLUENCE", str(rc.MIN_CONFLUENCE))),
}
BASELINE_PARAMS = {**DEFAULT_PARAMS, "base_risk": float(os.getenv("SIM_RISK_PCT", "0.005")),
                   "daily_loss_cap": 1.0, "derisk_k": 0.0, "quality_sizing": False,
                   "vol_target": False, "sentiment_gate": False, "regime_gate": False,
                   "confluence_gate": False}


def position_risk(params: Dict[str, Any], quality: float, atr_pct: float, equity_dd: float) -> float:
    """Risk fraction for one trade: base × conviction × vol-target × drawdown-de-risk."""
    risk = params["base_risk"]
    if params.get("quality_sizing"):
        risk *= 0.5 + max(0.0, min(1.0, quality))             # 0.5×..1.5× by conviction (½-Kelly-ish)
    if params.get("vol_target") and atr_pct > 0:
        risk *= max(0.5, min(1.5, params["target_atr"] / atr_pct))
    if params.get("derisk_k"):
        risk *= 1.0 - min(0.7, params["derisk_k"] * max(0.0, equity_dd))
    return max(0.0, risk)


# ── selection ────────────────────────────────────────────────────────────────────────
def select(trades: List[dict], quality_min: float, max_per_day: int,
           params: Dict[str, Any] = None, sent_map: Dict[tuple, float] = None) -> List[dict]:
    """Keep trades with quality ≥ quality_min, ranked, capped to max_per_day per date_tag.
    When `params` enables the GA gates, also drop rows that fail them (suppressive only)."""
    scored = [{**t, "_q": quality_score(t)} for t in trades]
    scored = [t for t in scored if t["_q"] >= quality_min]
    if params:
        scored = [t for t in scored if passes_gates(t, params, sent_map)]
    scored.sort(key=lambda t: (str(t.get("date_tag", "")), str(t.get("timestamp", "")), -t["_q"]))
    by_day: Dict[str, int] = {}
    out: List[dict] = []
    for t in scored:
        d = str(t.get("date_tag", ""))
        if by_day.get(d, 0) >= max_per_day:
            continue
        by_day[d] = by_day.get(d, 0) + 1
        out.append(t)
    return out


def quality_threshold_for_percentile(trades: List[dict], pct: float) -> float:
    qs = sorted(quality_score(t) for t in trades)
    if not qs:
        return 0.0
    return qs[min(len(qs) - 1, int(len(qs) * pct))]


# ── simulation (equity loop with sizing + daily loss cap) ────────────────────────────
def simulate(trades: List[dict], params: Dict[str, Any], start_equity: float = 100_000.0) -> Dict[str, Any]:
    equity = peak = start_equity
    max_dd = 0.0
    gross_win = gross_loss = 0.0
    wins = 0
    day = None
    day_start_equity = equity
    day_pnl = 0.0
    day_locked = False
    daily: Dict[str, float] = {}
    rs: List[float] = []
    taken = 0
    for t in trades:
        d = str(t.get("date_tag", ""))
        if d != day:
            day, day_start_equity, day_pnl, day_locked = d, equity, 0.0, False
        if day_locked:
            continue
        equity_dd = (peak - equity) / peak if peak > 0 else 0.0
        risk = position_risk(params, t.get("_q", quality_score(t)),
                             rc.to_f(t.get("atr_pct")), equity_dd)
        r = rc.to_f(t.get("realized_R"))
        pnl = r * equity * risk
        equity += pnl
        day_pnl += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        rs.append(r)
        taken += 1
        if r > 0:
            wins += 1; gross_win += pnl
        elif r < 0:
            gross_loss += abs(pnl)
        daily[d] = daily.get(d, 0.0) + pnl
        if day_pnl <= -params["daily_loss_cap"] * day_start_equity:
            day_locked = True                                 # daily loss cap hit → stop the day
    n = len(rs)
    day_pnls = list(daily.values())
    return {
        "trades": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0),
        "mean_R": round(sum(rs) / n, 4) if n else 0.0,
        "total_return_pct": round((equity / start_equity - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "calmar": round((equity / start_equity - 1.0) / max_dd, 3) if max_dd > 0 else 0.0,
        "trading_days": len(day_pnls),
        "worst_day": round(min(day_pnls), 2) if day_pnls else 0.0,
        "best_day": round(max(day_pnls), 2) if day_pnls else 0.0,
        "final_equity": round(equity, 2),
    }


# ── IS/OOS evaluation ────────────────────────────────────────────────────────────────
def _enriched_sorted() -> List[dict]:
    rows = rc.load_dataset(DATASET)
    rows = [r for r in rows if rc.to_f(r.get("mtf_alignment")) != 0.0]   # real bar context only
    return sorted(rows, key=lambda r: (str(r.get("date_tag", "")), str(r.get("timestamp", ""))))


def tune_quality_percentile(is_trades: List[dict],
                            grid=(0.0, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9), min_trades: int = 300) -> float:
    """Pick the IS quality percentile maximising IS mean realized_R (NO per-day cap, so the
    statistic is on thousands of trades), subject to a minimum trade count. IS-only → no leak."""
    best_pct, best_obj = 0.0, -1e9
    for pct in grid:
        thr = quality_threshold_for_percentile(is_trades, pct)
        sel = [t for t in is_trades if quality_score(t) >= thr]
        if len(sel) < min_trades:
            continue
        mr = sum(rc.to_f(t.get("realized_R")) for t in sel) / len(sel)
        if mr > best_obj:
            best_obj, best_pct = mr, pct
    return best_pct


def selection_edge(is_t: List[dict], oos_t: List[dict], thr: float) -> Dict[str, Any]:
    """Pure SELECTION effect (no sizing, no per-day cap) on the full samples: R-based metrics
    for take-all vs quality-gated, in-sample and out-of-sample. This is the statistically
    meaningful test of whether the ex-ante quality score has predictive value."""
    def metrics(trades, gated):
        rs = [rc.to_f(t.get("realized_R")) for t in trades
              if (not gated or quality_score(t) >= thr)]
        m = rc.compute_metrics(rs)
        return {"trades": m["trades"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                "expectancy": m["expectancy"], "apr": m["apr"], "max_drawdown_R": m["max_drawdown"]}
    return {"in_sample": {"take_all": metrics(is_t, False), "gated": metrics(is_t, True)},
            "out_of_sample": {"take_all": metrics(oos_t, False), "gated": metrics(oos_t, True)}}


def gate_edge(is_t: List[dict], oos_t: List[dict], thr: float,
              sent_map: Dict[tuple, float]) -> Dict[str, Any]:
    """Isolated effect of the GA downside gates: quality-gated rows WITHOUT vs WITH the
    sentiment/regime/confluence gates (R-based, no sizing/cap). Shows whether the gates lift
    expectancy / cut drawdown out-of-sample. Pure suppression, so trade count only falls."""
    def metrics(trades, with_gates):
        rs = [rc.to_f(t.get("realized_R")) for t in trades
              if quality_score(t) >= thr and (not with_gates or passes_gates(t, DEFAULT_PARAMS, sent_map))]
        m = rc.compute_metrics(rs)
        return {"trades": m["trades"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                "expectancy": m["expectancy"], "apr": m["apr"], "max_drawdown_R": m["max_drawdown"]}
    return {"in_sample": {"no_gates": metrics(is_t, False), "gated": metrics(is_t, True)},
            "out_of_sample": {"no_gates": metrics(oos_t, False), "gated": metrics(oos_t, True)}}


def evaluate(train_frac: float = 0.7) -> Dict[str, Any]:
    trades = _enriched_sorted()
    if not trades:
        return {"error": "no enriched trades"}
    k = max(1, int(len(trades) * train_frac))
    is_t, oos_t = trades[:k], trades[k:]
    sent_map = load_sentiment_map()

    # baseline: take all, flat 0.5% risk, no caps, no gates
    base_is = simulate(select(is_t, 0.0, BASELINE_PARAMS["max_trades_per_day"]), BASELINE_PARAMS)
    base_oos = simulate(select(oos_t, 0.0, BASELINE_PARAMS["max_trades_per_day"]), BASELINE_PARAMS)

    # improved: tune quality gate IN-SAMPLE, apply OUT-OF-SAMPLE; risk-managed sizing + GA gates
    pct = tune_quality_percentile(is_t)
    thr = quality_threshold_for_percentile(is_t, pct)        # threshold from IS only (no leak)
    imp_is = simulate(select(is_t, thr, DEFAULT_PARAMS["max_trades_per_day"], DEFAULT_PARAMS, sent_map), DEFAULT_PARAMS)
    imp_oos = simulate(select(oos_t, thr, DEFAULT_PARAMS["max_trades_per_day"], DEFAULT_PARAMS, sent_map), DEFAULT_PARAMS)

    return {"total_enriched": len(trades), "is_trades": len(is_t), "oos_trades": len(oos_t),
            "tuned_quality_percentile": pct, "tuned_quality_threshold": round(thr, 4),
            "sentiment_keys_loaded": len(sent_map),
            "selection_edge": selection_edge(is_t, oos_t, thr),
            "gate_edge": gate_edge(is_t, oos_t, thr, sent_map),
            "baseline_account": {"in_sample": base_is, "out_of_sample": base_oos},
            "improved_account": {"in_sample": imp_is, "out_of_sample": imp_oos},
            "params": {"baseline": BASELINE_PARAMS, "improved": DEFAULT_PARAMS}}


def account_table(thr: float, accounts=(25_000.0, 50_000.0, 100_000.0)) -> Dict[str, Any]:
    """OOS improved-policy metrics at each account size (dollar figures scale; %-metrics do not)."""
    trades = _enriched_sorted()
    k = max(1, int(len(trades) * 0.7))
    sel = select(trades[k:], thr, DEFAULT_PARAMS["max_trades_per_day"], DEFAULT_PARAMS, load_sentiment_map())
    return {f"${int(a/1000)}K": simulate(sel, DEFAULT_PARAMS, start_equity=a) for a in accounts}


if __name__ == "__main__":
    import json
    res = evaluate()
    if "error" in res:
        print(res["error"]); raise SystemExit(0)
    print(f"Enriched trades={res['total_enriched']}  IS={res['is_trades']}  OOS={res['oos_trades']}  "
          f"tuned gate: top {int((1-res['tuned_quality_percentile'])*100)}% (q≥{res['tuned_quality_threshold']})")
    print("\n[A] SELECTION EDGE — full sample, R-based, no sizing/cap (statistically meaningful):")
    se = res["selection_edge"]
    se_cols = ["trades", "win_rate", "profit_factor", "expectancy", "apr"]
    print(f"{'split/policy':24s}" + "".join(f"{c:>14s}" for c in se_cols))
    for split in ("in_sample", "out_of_sample"):
        for pol in ("take_all", "gated"):
            m = se[split][pol]
            print(f"{split+'/'+pol:24s}" + "".join(f"{m[c]:>14}" for c in se_cols))
    print(f"\n[A2] GENERATING-ALPHA GATE EDGE — quality-gated rows, no-gates vs +sentiment/regime/"
          f"confluence gates (sentiment keys loaded={res.get('sentiment_keys_loaded', 0)}):")
    ge = res["gate_edge"]
    print(f"{'split/policy':24s}" + "".join(f"{c:>14s}" for c in se_cols))
    for split in ("in_sample", "out_of_sample"):
        for pol in ("no_gates", "gated"):
            m = ge[split][pol]
            print(f"{split+'/'+pol:24s}" + "".join(f"{m[c]:>14}" for c in se_cols))
    print("\n[B] ACCOUNT POLICY — ≤5 trades/day + risk management (short-horizon, illustrative):")
    cols = ["trades", "win_rate", "profit_factor", "total_return_pct", "max_drawdown_pct", "calmar"]
    print(f"{'policy/split':24s}" + "".join(f"{c:>16s}" for c in cols))
    for pol in ("baseline_account", "improved_account"):
        for split in ("in_sample", "out_of_sample"):
            m = res[pol][split]
            print(f"{pol+'/'+split:24s}" + "".join(f"{m[c]:>16}" for c in cols))
    res["account_table_oos"] = account_table(res["tuned_quality_threshold"])
    print("\nOOS improved-policy account table:")
    acc_cols = ["total_return_pct", "max_drawdown_pct", "win_rate", "profit_factor", "trades", "worst_day", "best_day"]
    print(f"{'account':9s}" + "".join(f"{c:>16s}" for c in acc_cols))
    for acct, m in res["account_table_oos"].items():
        print(f"{acct:9s}" + "".join(f"{m[c]:>16}" for c in acc_cols))
    rc.write_json(rc.DAILY_OUTPUTS / rc.data_tag() / "rl_strategy_overlay.json", res)
