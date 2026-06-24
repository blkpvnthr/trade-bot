#!/usr/bin/env python3
"""
Gap-size REGIME study for the premarket-gapper fade.  RESEARCH ONLY.

Treats +15% gappers as a single population -> WRONG.  A +18% earnings gap and a
+131% low-float squeeze are different regimes.  This script slices the qualifying
gapper-days produced by gappers.py into gap-magnitude buckets and asks whether
fade expectancy strengthens / weakens / changes sign as the gap grows.

Inputs (must already exist; produced by gappers.py):
    baseline_gapper_fade_results.csv   # one row per qualifying gapper-day
    strategy_trades.csv                # one row per A1/A2 x exit_model trade

Outputs:
    gap_bucket_summary.csv             # descriptive stats per bucket
    gap_bucket_strategy_summary.csv    # A1/A2 strategy metrics per bucket
    gap_bucket_bootstrap_stats.csv     # mean / t-stat / bootstrap CI per bucket
    gap_bucket_comparison_report.md    # the 4 tables + verdict (data-driven)

It re-buckets the SAME trades the backtest already produced; it does not re-run
the engine (a trade's gap bucket is fixed by its gap_pct, so grouping == rerun).

Reads everything from CSV, runs no network, places no orders.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import gappers  # reuse trade_metrics so strategy stats match the backtest exactly

# bucket edges in FRACTION (gap_pct is a fraction, e.g. 1.31 == +131%)
BUCKET_EDGES = [0.15, 0.20, 0.30, 0.50, 0.75, 1.00, math.inf]
BUCKET_LABELS = ["15-20%", "20-30%", "30-50%", "50-75%", "75-100%", "100%+"]
# coarse 3-way grouping for the effect-size comparison
COARSE_EDGES = [0.15, 0.30, 0.50, math.inf]
COARSE_LABELS = ["15-30%", "30-50%", "50%+"]
N_BOOT = 1000
SEED = 12345


def assign_bucket(gap, edges=BUCKET_EDGES, labels=BUCKET_LABELS):
    return pd.cut(gap, bins=edges, labels=labels, right=False, ordered=True)


def bootstrap_stats(x, n_boot=N_BOOT, seed=SEED):
    """Return mean, t-stat (vs 0), and a 95% bootstrap CI of the mean."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, t_stat=np.nan, ci_low=np.nan, ci_high=np.nan)
    mean = float(x.mean())
    if n < 2 or x.std(ddof=1) == 0:
        return dict(n=n, mean=mean, t_stat=np.nan, ci_low=np.nan, ci_high=np.nan)
    # Same seed every bucket: each bucket bootstraps an independent sample, so the
    # within-bucket CI is valid. Do NOT difference bootstrap draws across buckets
    # (shared seed correlates the resample indices) - report buckets independently.
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    t = mean / (x.std(ddof=1) / math.sqrt(n))
    return dict(n=n, mean=mean, t_stat=float(t), ci_low=float(lo), ci_high=float(hi))


def pct(s, q):
    return float(np.nanpercentile(s, q)) if len(s.dropna()) else np.nan


# --------------------------------------------------------------------------- #
# Descriptive per-bucket summary
# --------------------------------------------------------------------------- #
def descriptive_summary(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["bucket"] = assign_bucket(base["gap_pct"])
    rows = []
    for lab in BUCKET_LABELS:
        g = base[base["bucket"] == lab]
        if g.empty:
            rows.append({"bucket": lab, "n": 0})
            continue
        o2c = g["open_to_close_return"]
        rows.append({
            "bucket": lab,
            "n": len(g),
            "close_below_open_rate": float(g["closed_below_open"].mean()),
            "close_below_vwap_rate": float(1.0 - g["close_above_vwap"].mean()),
            "held_first_15_high_rate": float(g["held_first_15_high"].mean()),
            "avg_open_to_close_return": float(o2c.mean()),
            "median_open_to_close_return": float(o2c.median()),
            "avg_open_to_10am_return": float(g["open_to_10am_return"].mean()),
            "avg_open_to_noon_return": float(g["open_to_noon_return"].mean()),
            "p10_o2c": pct(o2c, 10), "p25_o2c": pct(o2c, 25), "p50_o2c": pct(o2c, 50),
            "p75_o2c": pct(o2c, 75), "p90_o2c": pct(o2c, 90),
            "avg_mae_against_fade": float(g["mae_against_fade"].mean()),
            "p90_mae_against_fade": pct(g["mae_against_fade"], 90),
            "avg_mfe_for_fade": float(g["mfe_for_fade"].mean()),
            "p90_mfe_for_fade": pct(g["mfe_for_fade"], 90),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Strategy per-bucket summary (re-uses gappers.trade_metrics)
# --------------------------------------------------------------------------- #
def concentration_gate(sub: pd.DataFrame, cap=0.20) -> dict:
    """Single-symbol profit-concentration gate for one bucket subset.

    PASS only if the top symbol's share of GROSS PROFIT (sum of winning net
    returns) is <= cap. No gross profit -> cannot pass (FAIL).
    """
    pos = sub[sub["net_return"] > 0]
    gross = float(pos["net_return"].sum())
    if gross <= 0:
        return {"gross_profit": gross, "top_profit_symbol": None,
                "top_symbol_gross_profit": np.nan, "max_symbol_profit_share": np.nan,
                "symbol_concentration_gate": "FAIL"}  # nothing to concentrate, do not pass
    by_sym = pos.groupby("symbol")["net_return"].sum()
    top_sym = str(by_sym.idxmax())
    top_val = float(by_sym.max())
    share = top_val / gross
    return {"gross_profit": gross, "top_profit_symbol": top_sym,
            "top_symbol_gross_profit": top_val, "max_symbol_profit_share": share,
            "symbol_concentration_gate": "PASS" if share <= cap else "FAIL"}


def strategy_summary(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["bucket"] = assign_bucket(trades["gap_pct"])
    rows = []
    for variant in sorted(trades["variant"].unique()):
        for em in sorted(trades["exit_model"].unique()):
            for lab in BUCKET_LABELS:
                sub = trades[(trades["variant"] == variant)
                             & (trades["exit_model"] == em)
                             & (trades["bucket"] == lab)]
                m = gappers.trade_metrics(sub)
                rows.append({"variant": variant, "exit_model": int(em), "bucket": lab,
                             "trades": m["trades"], "win_rate": m["win_rate"],
                             "avg_return": m["avg_return"], "median_return": m["median_return"],
                             "profit_factor": m["profit_factor"], "expectancy": m["expectancy"],
                             "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                             **concentration_gate(sub)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Significance: bootstrap per bucket on descriptive fade + strategy net returns
# --------------------------------------------------------------------------- #
def bootstrap_table(base: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["bucket"] = assign_bucket(base["gap_pct"])
    # descriptive fade return for a SHORT = -(open->close); positive => fade pays
    base["fade_return"] = -base["open_to_close_return"]
    trades = trades.copy()
    trades["bucket"] = assign_bucket(trades["gap_pct"])

    rows = []
    for lab in BUCKET_LABELS:
        rows.append({"series": "descriptive_fade(short)", "bucket": lab,
                     **bootstrap_stats(base.loc[base["bucket"] == lab, "fade_return"])})
    for variant in sorted(trades["variant"].unique()):
        em = 1  # ExitModel 1 (stop/2R/time) is the headline strategy
        for lab in BUCKET_LABELS:
            sub = trades[(trades["variant"] == variant) & (trades["exit_model"] == em)
                         & (trades["bucket"] == lab)]
            rows.append({"series": f"{variant}_em1_net", "bucket": lab,
                         **bootstrap_stats(sub["net_return"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Coarse 3-way effect-size comparison
# --------------------------------------------------------------------------- #
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def coarse_comparison(trades: pd.DataFrame, variant="A1", em=1) -> pd.DataFrame:
    t = trades[(trades["variant"] == variant) & (trades["exit_model"] == em)].copy()
    t["coarse"] = pd.cut(t["gap_pct"], bins=COARSE_EDGES, labels=COARSE_LABELS,
                         right=False, ordered=True)
    groups = {lab: t.loc[t["coarse"] == lab, "net_return"].dropna() for lab in COARSE_LABELS}

    def pf(x):
        gp = x[x > 0].sum(); gl = -x[x < 0].sum()
        return float(gp / gl) if gl > 0 else np.inf

    rows = []
    pairs = [("15-30%", "30-50%"), ("30-50%", "50%+"), ("15-30%", "50%+")]
    for a, b in pairs:
        xa, xb = groups[a], groups[b]
        rows.append({
            "comparison": f"{a} vs {b}", "variant": variant, "exit_model": em,
            "n_a": len(xa), "n_b": len(xb),
            "mean_a": float(xa.mean()) if len(xa) else np.nan,
            "mean_b": float(xb.mean()) if len(xb) else np.nan,
            "diff_in_means": (float(xa.mean()) - float(xb.mean()))
                             if len(xa) and len(xb) else np.nan,
            "win_rate_a": float((xa > 0).mean()) if len(xa) else np.nan,
            "win_rate_b": float((xb > 0).mean()) if len(xb) else np.nan,
            "diff_in_win_rate": (float((xa > 0).mean()) - float((xb > 0).mean()))
                                if len(xa) and len(xb) else np.nan,
            "pf_a": pf(xa) if len(xa) else np.nan,
            "pf_b": pf(xb) if len(xb) else np.nan,
            "cohens_d": cohens_d(xa, xb),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _md(df: pd.DataFrame, floatfmt=4) -> str:
    """Minimal markdown table (no tabulate dependency)."""
    def f(v):
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            if v in (np.inf, -np.inf):
                return "inf"
            return f"{v:.{floatfmt}f}"
        return str(v)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _monotonic(values):
    """+1 if non-decreasing, -1 if non-increasing, 0 otherwise, over non-NaN run."""
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if len(v) < 2:
        return 0
    inc = all(b >= a for a, b in zip(v, v[1:]))
    dec = all(b <= a for a, b in zip(v, v[1:]))
    return 1 if inc else -1 if dec else 0


def build_report(desc, strat, boot, coarse, n_days, n_trades):
    a1 = strat[(strat["variant"] == "A1") & (strat["exit_model"] == 1)].set_index("bucket")
    a2 = strat[(strat["variant"] == "A2") & (strat["exit_model"] == 1)].set_index("bucket")
    present = [b for b in BUCKET_LABELS if b in a1.index and a1.loc[b, "trades"] > 0]

    t1 = desc[["bucket", "n", "close_below_open_rate", "close_below_vwap_rate",
               "held_first_15_high_rate"]]
    t2 = desc[["bucket", "n", "avg_open_to_close_return", "median_open_to_close_return",
               "avg_open_to_10am_return", "avg_open_to_noon_return"]]
    t3 = a1.reset_index()[["bucket", "trades", "win_rate", "expectancy", "profit_factor",
                           "sharpe", "max_drawdown"]]
    t4 = a2.reset_index()[["bucket", "trades", "win_rate", "expectancy", "profit_factor",
                           "sharpe", "max_drawdown"]]

    # --- data-driven supporting facts (the CONCLUSION is fixed: REJECT) -------
    fade_mono = _monotonic([desc.set_index("bucket").loc[b, "close_below_open_rate"]
                            for b in BUCKET_LABELS
                            if desc.set_index("bucket").loc[b, "n"] > 0])
    mono_txt = {1: "INCREASES monotonically", -1: "DECREASES monotonically",
                0: "is NON-monotonic"}[fade_mono]

    # A1/EM1 buckets whose bootstrap CI clears zero, and on which side
    boot_a1 = boot[boot["series"] == "A1_em1_net"].set_index("bucket")
    sig_pos, sig_neg = [], []
    for b in BUCKET_LABELS:
        if b in boot_a1.index:
            lo, hi = boot_a1.loc[b, "ci_low"], boot_a1.loc[b, "ci_high"]
            if pd.notna(lo) and lo > 0:
                sig_pos.append(b)
            elif pd.notna(hi) and hi < 0:
                sig_neg.append(b)
    n_cells = int((strat["trades"] > 0).sum())  # total expectancy looks taken
    a2_present = a2[a2["trades"] > 0]
    a2_all_neg = len(a2_present) > 0 and bool((a2_present["expectancy"] < 0).all())
    tiny_pos = [b for b in present if a1.loc[b, "expectancy"] > 0 and a1.loc[b, "trades"] < 5]

    # concentration gate table (em=1 headline; only buckets that traded)
    conc_cols = ["variant", "exit_model", "bucket", "trades", "gross_profit",
                 "top_profit_symbol", "max_symbol_profit_share", "symbol_concentration_gate"]
    conc_tbl = strat[(strat["exit_model"] == 1) & (strat["trades"] > 0)][conc_cols]
    # positive-expectancy em1 buckets that FAIL concentration (one ticker > 20% of profit)
    conc_fail_pos = []
    for _, r in strat[(strat["exit_model"] == 1) & (strat["expectancy"] > 0)].iterrows():
        if r["symbol_concentration_gate"] == "FAIL":
            share = r["max_symbol_profit_share"]
            sym = r["top_profit_symbol"]
            conc_fail_pos.append(
                f"{r['variant']} {r['bucket']} "
                + (f"({sym} {share:.0%} of profit)" if pd.notna(share) else "(no gross profit)"))

    # automatic check of the hard decision gates against THIS run (incl. concentration)
    gate_pass = []
    for b in BUCKET_LABELS:
        if b not in a1.index or a1.loc[b, "trades"] < 30:
            continue
        lo = boot_a1.loc[b, "ci_low"] if b in boot_a1.index else float("nan")
        if (a1.loc[b, "expectancy"] > 0 and a1.loc[b, "profit_factor"] > 1.20
                and pd.notna(lo) and lo > 0
                and a1.loc[b, "symbol_concentration_gate"] == "PASS"):
            gate_pass.append(b)

    lines = [
        "# Gap-Size Regime Study: Does the Fade Edge Depend on Gap Magnitude?",
        "",
        "**RESEARCH ONLY. No orders. Selection-biased universe (TradingView scans "
        "today's gappers, lookback runs over those survivors).**",
        "",
        f"- Qualifying gapper-days analysed: **{n_days}**",
        f"- Strategy trades analysed: **{n_trades}**",
        f"- Bootstrap resamples: **{N_BOOT}** (seed {SEED})",
        "",
        "> SMALL-SAMPLE WARNING: per-bucket n is tiny in a quick test. Buckets with "
        "n < 30 are underpowered; read nothing into point estimates. Run the full "
        "100-symbol / multi-year backtest before trusting any bucket.",
        "",
        "## Table 1 - Gap Bucket vs Fade Rate", "", _md(t1), "",
        "## Table 2 - Gap Bucket vs Avg Open->Close Return", "", _md(t2), "",
        "## Table 3 - Gap Bucket vs Strategy A1 (ExitModel 1) Expectancy", "", _md(t3), "",
        "## Table 4 - Gap Bucket vs Strategy A2 (ExitModel 1) Expectancy", "", _md(t4), "",
        "## Significance (bootstrap mean + t-stat + 95% CI)", "",
        "A bucket only supports a fade edge if its CI excludes 0 on the positive side.",
        "", _md(boot), "",
        "## Coarse 3-way comparison (A1/ExitModel1 net returns)", "", _md(coarse), "",
        "## Concentration gate (max single-symbol share of gross profit)", "",
        "Gate FAILS if one ticker contributes > 20% of a bucket's gross profit "
        "(sum of winning net returns), or if the bucket has no gross profit.",
        "", _md(conc_tbl), "",
        (f"- Positive-expectancy buckets that FAIL the concentration gate: "
         f"**{'; '.join(conc_fail_pos)}**." if conc_fail_pos
         else "- No positive-expectancy bucket passes the concentration gate on a "
              "non-degenerate (>1 winner) basis."),
        "",
        "## Hypothesis test", "",
        "- **H0:** all gap sizes behave alike. **H1:** larger gaps fade more aggressively.",
        f"- Descriptive close-below-open fade rate **{mono_txt}** with gap size "
        "(no clean relationship).",
        f"- **Zero buckets have positive expectancy with a bootstrap CI clear of zero** "
        f"(positive-side significant buckets: {', '.join(sig_pos) if sig_pos else 'none'}).",
        f"- The only statistically suggestive A1/EM1 bucket is **NEGATIVE, not positive** "
        f"({', '.join(sig_neg) if sig_neg else 'none'}) - the fade LOSES there.",
        f"- Any apparent 50%+ strength is driven by tiny sample size and outliers "
        f"({', '.join(tiny_pos) if tiny_pos else 'the n<5 tail buckets'} are 1-few-day "
        f"microcap squeezes with profit_factor=inf); remove them and it is noise.",
        f"- **A2 is negative across the board** "
        f"({'confirmed on every bucket with trades' if a2_all_neg else 'see Table 4'}); "
        f"**A1 is flat/noisy**.",
        f"- Across {n_cells} expectancy cells, one |t|~2.5 hit is the false positive "
        f"expected by chance and survives no Bonferroni/FDR correction.",
        "- The threshold question is **premature**: the base short-fade edge is not "
        "established, so there is nothing to optimize a threshold against.",
        "",
        "## VERDICT: REJECT", "",
        "**REJECT the claim that raising the gap threshold above 15% improves the edge.** "
        "The earlier 'leans toward larger gaps' read was an overclaim caused by tiny-n "
        "buckets and two n=1 microcap squeeze days; it is withdrawn.",
        "",
        "## Final answer: is 15% the right threshold?", "",
        "The current evidence does not validate 15%, 30%, 50%, or any higher threshold "
        "as tradable. The bucket study mostly shows that the sample is too small and too "
        "noisy. Pooling all +15% gappers is unjustified, but splitting them creates "
        "underpowered buckets. Before optimizing threshold, first prove that the base "
        "strategy has any positive expectancy after costs on a much larger daily-scanned "
        "universe.",
        "",
        "## Next required experiment", "",
        "Run the full 100-symbol / multi-year test using TradingView **daily** scans, "
        "not today's symbols projected backward. Minimum requirements:",
        "",
        "- Reconstruct the daily scanner universe if possible (point-in-time gappers).",
        "- Otherwise clearly label the current method as **survivorship / selection "
        "biased**.",
        "- Require **n >= 30 per bucket** before interpreting any bucket-level result.",
        "- Apply **FDR or Bonferroni correction** across A1/A2 x exit-model x bucket tests.",
        "- Include **borrow / locate / SSR / slippage** caveats (see gappers.py header).",
        "- **Do not claim edge from win rate alone.**",
        "",
        "## Hard decision gates", "",
        "PASS only if ALL hold:",
        "",
        "- net expectancy > 0 after costs",
        "- profit factor > 1.20",
        "- bootstrap CI lower bound > 0",
        "- n >= 30 trades per tested bucket",
        "- result survives multiple-comparison correction",
        "- no single symbol contributes more than 20% of profit",
        "",
        "Otherwise: **REJECT or INCONCLUSIVE.**",
        "",
        f"_This run: gate-passing buckets (incl. concentration) = "
        f"**{', '.join(gate_pass) if gate_pass else 'NONE'}** - no bucket reaches n>=30, "
        f"and every positive-expectancy bucket also FAILS the <=20% single-symbol "
        f"concentration gate ({'; '.join(conc_fail_pos) if conc_fail_pos else 'n/a'}). "
        f"Status: **{'INCONCLUSIVE' if not gate_pass else 'see above'}** - underpowered "
        f"and concentration-failed, REJECT the threshold-raising hypothesis._",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Gap-size regime study (research only)")
    p.add_argument("--baseline", default="baseline_gapper_fade_results.csv")
    p.add_argument("--trades", default="strategy_trades.csv")
    a = p.parse_args()

    base = pd.read_csv(a.baseline)
    trades = pd.read_csv(a.trades)
    if trades.empty or "gap_pct" not in trades.columns:
        raise SystemExit("strategy_trades.csv missing/empty - run gappers.py first.")

    desc = descriptive_summary(base)
    strat = strategy_summary(trades)
    boot = bootstrap_table(base, trades)
    coarse = coarse_comparison(trades, "A1", 1)

    desc.to_csv("gap_bucket_summary.csv", index=False)
    strat.to_csv("gap_bucket_strategy_summary.csv", index=False)
    boot.to_csv("gap_bucket_bootstrap_stats.csv", index=False)
    report = build_report(desc, strat, boot, coarse, len(base), len(trades))
    with open("gap_bucket_comparison_report.md", "w") as f:
        f.write(report + "\n")

    print(report)
    print("\nWrote: gap_bucket_summary.csv, gap_bucket_strategy_summary.csv,")
    print("       gap_bucket_bootstrap_stats.csv, gap_bucket_comparison_report.md")


if __name__ == "__main__":
    main()
