"""End-to-end demonstration of the research platform on REAL decisions we already understand.

This is also the platform's self-test: the machine must return the verdicts we've already proven
by hand —
  • REJECT the broad-ETF mean-rev challenger (we showed breadth dilutes the edge),
  • REJECT/CONTINUE insider conviction-bucket sizing (we showed flat wins),
  • surface the full-book insider challenger as a real lead (Part-3 Sharpe +4.07 vs gated +1.34).

Run:  python -m research_platform.demo
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path

import numpy as np

import backtest_insider_copier as base
import backtest_insider_sizing as ins
import backtest_meanrev_expansion as mr

from . import (ExperimentSpec, Registry, EventLog, FillModel, PortfolioConfig,
               PromotionCriteria, evaluate, nightly_report)
from .portfolios import flat_weights, bucket_weights, zscore_weights

ROOT = base.PROJECT_ROOT / "research_outputs"
REG = Registry(str(ROOT / "registry"))
FILL = FillModel(slippage_bps=5.0)          # ONE shared fill model for every portfolio
TRAIN_END = _date(2024, 1, 1)
CRIT = PromotionCriteria(min_trades=30, min_sharpe_improvement=0.0, max_p_value=0.10,
                         min_year_trades=8, max_year_loss=-0.01, burn_in_days=60,
                         max_drawdown=-0.40, train_end="2024-01-01")
ORIG10 = {"SPY", "IWM", "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI", "XLC"}


def run_insider():
    pool = ins.collect()                                       # real insider candidates
    train = [r for r in pool if r["date"] < TRAIN_END]
    vthr = float(np.percentile([r["rv20"] for r in train], 80))
    vthr50 = float(np.percentile([r["rv20"] for r in train], 50))
    fits = {k: (float(np.mean([r[k] for r in train])),
                float(np.std([r[k] for r in train]) or 1.0))
            for k in ("rv20", "breadth")}

    champ = PortfolioConfig("insider_volgate_flat", "champion",
                            lambda r: r["rv20"] >= vthr, flat_weights,
                            "Production: vol-gated (rv20≥train-P80), flat sizing.")
    challengers = [
        # the LADDER: isolate gate-threshold vs concentration vs tilt-vs-gate
        PortfolioConfig("insider_gate_p50", "challenger",
                        lambda r: r["rv20"] >= vthr50, flat_weights,
                        "Relaxed vol-gate (P50): more concurrent names, keeps mild vol selection."),
        PortfolioConfig("insider_voltilt_light", "challenger",
                        lambda r: True, zscore_weights("rv20", *fits["rv20"], lam=0.25),
                        "Full book + LIGHT vol tilt (vol as a lean, not a hard gate)."),
        PortfolioConfig("insider_voltilt_med", "challenger",
                        lambda r: True, zscore_weights("rv20", *fits["rv20"], lam=0.5),
                        "Full book + medium vol tilt."),
        PortfolioConfig("insider_fullbook_flat", "challenger",
                        lambda r: True, flat_weights,
                        "Full diversified book, flat sizing (the prior PROMOTE lead)."),
        PortfolioConfig("insider_volgate_buckets", "challenger",
                        lambda r: r["rv20"] >= vthr,
                        bucket_weights([("rv20", *fits["rv20"]), ("breadth", *fits["breadth"])]),
                        "Conviction A+8/A6/B4/C2 buckets on vol+breadth (expected: no gain)."),
    ]
    spec = ExperimentSpec(
        name="insider_sizing_and_gate",
        description="Insider sleeve: gate-threshold ladder (P80/P50/none) × sizing (flat/tilt/buckets).",
        config={"hold": ins.HOLD, "min_adv": ins.MIN_ADV, "vol_pctile": 80, "fill_bps": 5.0},
        search_space={"gate": ["volgate_p80", "gate_p50", "fullbook"],
                      "sizing": ["flat", "voltilt_light", "voltilt_med", "buckets"]},
        promotion_criteria=CRIT.__dict__,
        data_paths=[str(base.PIT_EVENTS_PATH), str(base.BARS_DIR)])
    REG.register(spec)
    log = EventLog(str(ROOT / "events" / f"{spec.spec_hash}_insider.jsonl"))
    results = [evaluate(champ, c, pool, FILL, CRIT, spec.spec_hash, return_key="gross",
                        log=log, log_champion=(i == 0))
               for i, c in enumerate(challengers)]
    nightly_report("Insider — gate & sizing", spec.spec_hash, results,
                   str(ROOT / "nightly_insider.md"))
    return spec, results


def run_meanrev():
    regime = mr._spy_regime()
    universe = [s for s in mr.ETF_UNIVERSE if mr._load(s) is not None]
    raw = mr.collect_core_trades(universe, regime)             # real mean-rev candidates (broad ETF)
    pool = [{**t, "gross": t["ret"]} for t in raw]             # normalize return key

    champ = PortfolioConfig("meanrev_orig10_flat", "champion",
                            lambda r: r["sym"] in ORIG10, flat_weights,
                            "Production: validated ~10 US sector/broad ETFs, flat sizing.")
    challengers = [
        PortfolioConfig("meanrev_broad_etf_flat", "challenger",
                        lambda r: True, flat_weights,
                        "Broad ETF universe (all classes), flat (expected: REJECT — breadth dilutes)."),
    ]
    spec = ExperimentSpec(
        name="meanrev_universe",
        description="Mean-rev: validated ~10 ETFs vs broad ETF universe.",
        config={"rsi_len": mr.RSI_LEN, "rsi_buy": mr.RSI_BUY, "max_hold": mr.MAX_HOLD,
                "cost_bps": mr.COST * 1e4, "fill_bps": 5.0},
        search_space={"universe": ["orig10", "broad_etf"]},
        promotion_criteria=CRIT.__dict__,
        data_paths=[str(base.BARS_DIR)])
    REG.register(spec)
    log = EventLog(str(ROOT / "events" / f"{spec.spec_hash}_meanrev.jsonl"))
    results = [evaluate(champ, c, pool, FILL, CRIT, spec.spec_hash, return_key="gross",
                        log=log, log_champion=(i == 0))
               for i, c in enumerate(challengers)]
    nightly_report("Mean-reversion — universe", spec.spec_hash, results,
                   str(ROOT / "nightly_meanrev.md"))
    return spec, results


def main():
    print("=== Research platform demo: replaying real decisions ===")
    print("Insider experiment...")
    si, ri = run_insider()
    print(f"  registered {si.spec_hash} (git {si.git_commit[:8]}, data {si.data_snapshot})")
    for r in ri:
        print(f"  {r['challenger'].portfolio_id:32s} -> {r['verdict']:9s} "
              f"ΔSharpe={r['d_sharpe']:+.3f} p={r['p_value']}")
    print("Mean-reversion experiment...")
    sm, rm = run_meanrev()
    print(f"  registered {sm.spec_hash} (git {sm.git_commit[:8]}, data {sm.data_snapshot})")
    for r in rm:
        print(f"  {r['challenger'].portfolio_id:32s} -> {r['verdict']:9s} "
              f"ΔSharpe={r['d_sharpe']:+.3f} p={r['p_value']}")

    nev = sum(EventLog(str(p)).count() for p in (ROOT / "events").glob("*.jsonl"))
    print(f"\nLedger: {nev:,} decision events across {len(list((ROOT/'events').glob('*.jsonl')))} logs")
    print(f"Registry: {len(REG.list())} immutable experiment specs")
    print(f"Reports : {ROOT/'nightly_insider.md'} , {ROOT/'nightly_meanrev.md'}")


if __name__ == "__main__":
    main()
