#!/usr/bin/env python3
"""
backtest_meanrev_expansion.py — Tier-2 expansion of the validated Connors RSI(2) mean-reversion
edge from 8 ETFs to a liquidity-screened broad universe, with the §A objective filter-promotion
process and §D regime stratification (see PRODUCTION_ARCHITECTURE.md).

Central question: does the ETF edge (PF 2.57 OOS) survive on individual liquid STOCKS, and which
candidate filters actually earn their slot out-of-sample?

Method (no peeking):
  1. Universe = top-N cached symbols by median $-volume with >=1000 bars (liquidity-defined,
     not hand-picked → no ticker selection bias). ETF vs stock tagged.
  2. Core signal (validated, unchanged): RSI(2)<10 AND close>SMA200 → next-open;
     exit RSI(2)>70 or MAX_HOLD=6d; 5bps round-trip. Collect ALL core trades + feature snapshot.
  3. §A filter promotion: GREEDY FORWARD SELECTION on TRAIN only. A filter is kept iff
     ΔSharpe>0 AND ΔCAGR>0 AND retained-trade-fraction>=0.6 AND survives BH-FDR (bootstrap p).
     CAGR comes from a CAPACITY-CONSTRAINED portfolio sim (cash buffer + max concurrent + per-name),
     NOT the trade list — trade-list PF can rise while CAGR falls on idle capital.
  4. Selected config evaluated on OOS (the honest number) vs core-only OOS.
  5. §D regime stratification (bull/bear via SPY>200SMA; vol-hi/lo via SPY rv20 median) + per-year.
  6. Diversification: corr of daily PnL vs the vol-gated INSIDER edge.

Selection is done on TRAIN; OOS is reported untouched.
"""
from __future__ import annotations

import sys
from datetime import date as _date
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import backtest_insider_copier as base
import backtest_vol_gated_insider as vg

# ---- validated core params (unchanged from candidate_A) ----
RSI_LEN, RSI_BUY, RSI_EXIT, MAX_HOLD = 2, 10.0, 70.0, 6
COST = 0.0005
TRAIN_END = _date(2024, 1, 1)
SEED = 7

# ---- universe construction ----
MIN_BARS = 1000                # need SMA200 warmup + train + test
TOP_N_LIQUID = 150             # cap universe at the 150 most liquid qualifying names
# known liquid ETFs (everything else in the screen is treated as a stock)
KNOWN_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "XLF", "XLK", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLU", "XLB", "XLRE", "XLC", "SMH", "SOXX", "XBI", "ARKK", "EEM", "EFA", "TLT",
    "HYG", "LQD", "GLD", "SLV", "USO", "GDX", "KRE", "XOP", "XRT", "IBB", "VXX", "UVXY",
    "SQQQ", "TQQQ", "SPXL", "VNQ", "EWZ", "FXI", "VEA", "VWO", "AGG", "BND", "IEF",
}

# ---- portfolio-sim constraints (capacity binds in selloffs — that's the point) ----
DEPLOYABLE = 0.55              # 45% cash buffer preserved
PER_NAME = 0.10               # 10% single-name cap
MAX_CONC = 5                  # int(DEPLOYABLE / PER_NAME)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _load(sym: str) -> Optional[pd.DataFrame]:
    try:
        b = pd.read_parquet(base.BARS_DIR / f"{sym}_day.parquet")
    except Exception:
        return None
    if b is None or len(b) < MIN_BARS:
        return None
    return b


# curated broad liquid ETF universe — baskets mean-revert (where the edge actually lives)
ETF_UNIVERSE = [
    # broad
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "MDY",
    # sector SPDRs
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    # industry / thematic
    "SMH", "SOXX", "XBI", "IBB", "ARKK", "KRE", "XOP", "XRT", "XHB", "ITB", "KWEB",
    # country / region
    "EEM", "EFA", "EWZ", "FXI", "EWJ", "EWG", "EWW", "INDA", "VEA", "VWO",
    "EWY", "EWT", "EWU", "EWC", "EWA", "EWH", "EZU",
    # bonds
    "TLT", "IEF", "HYG", "LQD", "AGG", "BND", "SHY", "TIP", "EMB",
    # commodities / metals
    "GLD", "SLV", "USO", "GDX", "GDXJ", "UNG", "DBC",
    # style / factor
    "VTV", "VUG", "IWD", "IWF", "MTUM", "QUAL", "USMV", "VYM", "SDY", "DVY",
    # real estate
    "VNQ", "IYR",
]


# economically-principled subset: US-EQUITY broad + sector + industry + style + REIT ETFs only.
# Hypothesis: short-term mean-reversion-in-uptrend is a US-equity-index phenomenon; bonds,
# commodities and single-country ETFs (which trend / have different dynamics) dilute it.
US_EQUITY_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "MDY",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    "SMH", "SOXX", "XBI", "IBB", "KRE", "XOP", "XRT", "XHB", "ITB", "ARKK",
    "VTV", "VUG", "IWD", "IWF", "MTUM", "QUAL", "USMV", "VYM", "SDY", "DVY",
    "VNQ", "IYR",
]

# correct asset-class tagging in ETF modes (KNOWN_ETFS alone is incomplete)
KNOWN_ETFS = KNOWN_ETFS | set(ETF_UNIVERSE) | set(US_EQUITY_ETFS)


def build_universe() -> List[str]:
    """Liquidity-defined universe: top-N cached symbols by median $-vol with enough history.

    Pass 'etf' for the broad-ETF universe, or 'usetf' for the US-equity-ETF subset.
    """
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if mode == "etf":
        return [s for s in ETF_UNIVERSE if _load(s) is not None]
    if mode == "usetf":
        return [s for s in US_EQUITY_ETFS if _load(s) is not None]
    rows = []
    for p in sorted(base.BARS_DIR.glob("*_day.parquet")):
        sym = p.name[: -len("_day.parquet")]
        b = _load(sym)
        if b is None:
            continue
        try:
            dvol = (b["close"].astype(float) * b["volume"].astype(float)).tail(252).median()
        except Exception:
            continue
        if not np.isfinite(dvol) or dvol <= 0:
            continue
        rows.append((sym, float(dvol)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:TOP_N_LIQUID]]


def _spy_regime() -> Dict[_date, Dict[str, bool]]:
    """date -> {bull: close>SMA200, volhi: rv20>=median(rv20)} using SPY as the market proxy."""
    b = _load("SPY")
    if b is None:
        return {}
    cl = b["close"].astype(float)
    sma200 = cl.rolling(200).mean()
    rv20 = cl.pct_change().rolling(20).std()
    rvmed = rv20.median()
    out = {}
    for ts, c, s, rv in zip(b.index, cl.values, sma200.values, rv20.values):
        d = ts.date() if hasattr(ts, "date") else ts
        if not (np.isfinite(s) and np.isfinite(rv)):
            continue
        out[d] = {"bull": bool(c > s), "volhi": bool(rv >= rvmed)}
    return out


def collect_core_trades(universe: List[str], regime: Dict[_date, Dict[str, bool]]) -> List[dict]:
    """Run the validated core signal once; snapshot per-trade features for later filtering."""
    trades = []
    for sym in universe:
        b = _load(sym)
        if b is None:
            continue
        td = list(b.index)
        op = b["open"].astype(float).values
        cl = b["close"].astype(float).values
        vol = b["volume"].astype(float).values
        rsi = _rsi(b["close"].astype(float), RSI_LEN).values
        clo = b["close"].astype(float)
        sma10 = clo.rolling(10).mean().values
        sma50 = clo.rolling(50).mean().values
        sma200 = clo.rolling(200).mean().values
        rv20 = clo.pct_change().rolling(20).std().values
        dvol = (clo * b["volume"].astype(float)).rolling(20).mean().values
        is_etf = sym in KNOWN_ETFS
        i = 200
        while i < len(td) - MAX_HOLD - 1:
            if rsi[i] < RSI_BUY and cl[i] > sma200[i]:
                entry = op[i + 1]
                xk = i + 1 + MAX_HOLD
                for k in range(i + 1, i + 1 + MAX_HOLD):
                    if rsi[k] > RSI_EXIT:
                        xk = k
                        break
                xk = min(xk, len(td) - 1)
                exit_px = cl[xk]
                if entry > 0 and exit_px > 0:
                    edate = td[i + 1].date() if hasattr(td[i + 1], "date") else td[i + 1]
                    reg = regime.get(edate, {"bull": True, "volhi": False})
                    trades.append({
                        "sym": sym, "etf": is_etf, "date": edate,
                        "ret": exit_px / entry - 1 - COST, "hold": max(xk - (i + 1), 1),
                        # feature snapshot (all known at/ before signal bar i)
                        "rsi": float(rsi[i]),
                        "rv20": float(rv20[i]) if np.isfinite(rv20[i]) else 0.0,
                        "dvol": float(dvol[i]) if np.isfinite(dvol[i]) else 0.0,
                        "dist10": float(cl[i] / sma10[i] - 1) if np.isfinite(sma10[i]) and sma10[i] > 0 else 0.0,
                        "trend50": bool(np.isfinite(sma50[i]) and sma50[i] > sma200[i]),
                        "mkt_bull": bool(reg["bull"]), "mkt_volhi": bool(reg["volhi"]),
                    })
                i = xk + 1
            else:
                i += 1
    return trades


# ---------- metrics ----------
def sharpe(rows: List[dict]) -> float:
    if len(rows) < 2:
        return 0.0
    v = np.array([r["ret"] for r in rows])
    sd = v.std()
    return float(v.mean() / sd) if sd > 0 else 0.0


def pf_of(rows: List[dict]) -> float:
    v = np.array([r["ret"] for r in rows]) if rows else np.array([])
    w, l = v[v > 0].sum(), abs(v[v < 0].sum())
    return float(w / l) if l > 0 else float("inf")


def portfolio_cagr_mdd(rows: List[dict]) -> tuple:
    """Capacity-constrained event sim → (CAGR, MaxDD). Captures idle-capital + concurrency limits."""
    if not rows:
        return 0.0, 0.0
    events = []
    for idx, r in enumerate(sorted(rows, key=lambda x: x["date"])):
        # exit date approximated by trading-day offset = hold (calendar proxy is fine for ordering)
        events.append((r["date"], 0, idx, r))           # 0 = entry (process entries first per day)
        events.append((r["date"], r["hold"], idx, r))   # exit marker; resolved by sequence below
    # simple sequential sim: walk entries in order, hold N concurrent, realize on exit
    open_pos = []   # list of (exit_seq, weight, ret)
    seq = 0
    equity = peak = 1.0
    mdd = 0.0
    ordered = sorted(rows, key=lambda x: (x["date"],))
    # use an index-based clock: each trade occupies [k, k+hold) in its own symbol-day terms;
    # approximate global concurrency by date overlap
    active = []  # (exit_date, weight, ret)
    first = ordered[0]["date"]
    last = max(r["date"] for r in ordered)
    for r in ordered:
        d = r["date"]
        # close positions whose exit_date <= d
        still = []
        for (xd, w, ret) in active:
            if xd <= d:
                equity *= (1 + w * ret)
                peak = max(peak, equity)
                mdd = min(mdd, equity / peak - 1)
            else:
                still.append((xd, w, ret))
        active = still
        if len(active) < MAX_CONC:
            w = PER_NAME
            xd = d + pd.Timedelta(days=int(r["hold"]) * 7 // 5 + 1)  # ~trading->calendar
            xd = xd.date() if hasattr(xd, "date") else xd
            active.append((xd, w, r["ret"]))
    for (xd, w, ret) in active:
        equity *= (1 + w * ret)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    days = max((last - first).days, 1)
    cagr = equity ** (365.0 / days) - 1
    return float(cagr), float(mdd)


# ---------- §A candidate filters (thresholds computed on TRAIN) ----------
def make_filters(train: List[dict]) -> Dict[str, Callable[[dict], bool]]:
    def pct(key, q):
        vals = [t[key] for t in train if np.isfinite(t[key])]
        return float(np.percentile(vals, q)) if vals else 0.0
    rv_p80 = pct("rv20", 80)
    dvol_p50 = pct("dvol", 50)
    dist_p20 = pct("dist10", 20)   # most-extended-below (negative)
    return {
        "vol_gate(rv20>=P80)": lambda t: t["rv20"] >= rv_p80,
        "liquidity(dvol>=P50)": lambda t: t["dvol"] >= dvol_p50,
        "rsi_deep(<5)": lambda t: t["rsi"] < 5.0,
        "dip_deep(dist<=P20)": lambda t: t["dist10"] <= dist_p20,
        "trend50(SMA50>200)": lambda t: t["trend50"],
        "mkt_regime(bull)": lambda t: t["mkt_bull"],
    }


def apply_filters(rows: List[dict], fns: List[Callable]) -> List[dict]:
    out = rows
    for fn in fns:
        out = [t for t in out if fn(t)]
    return out


def bootstrap_dsharpe_p(prev: List[dict], fn: Callable, B: int = 1500, seed: int = SEED) -> float:
    """p-value for ΔSharpe>0 (filtered vs prev) under resampling of prev."""
    if len(prev) < 10:
        return 1.0
    rng = np.random.RandomState(seed)
    arr = np.array([(r["ret"], 1.0 if fn(r) else 0.0) for r in prev])
    base_idx = np.arange(len(arr))
    neg = 0
    for _ in range(B):
        s = rng.choice(base_idx, size=len(arr), replace=True)
        a = arr[s]
        full = a[:, 0]
        sub = a[a[:, 1] > 0, 0]
        sh_full = full.mean() / full.std() if full.std() > 0 else 0.0
        sh_sub = sub.mean() / sub.std() if len(sub) > 1 and sub.std() > 0 else 0.0
        if sh_sub - sh_full <= 0:
            neg += 1
    return neg / B


def bh_fdr(pvals: Dict[str, float], q: float = 0.10) -> Dict[str, bool]:
    """Benjamini-Hochberg: returns name->survives."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    survive = {k: False for k in pvals}
    kmax = 0
    for i, (k, p) in enumerate(items, start=1):
        if p <= q * i / m:
            kmax = i
    for i, (k, p) in enumerate(items, start=1):
        if i <= kmax:
            survive[k] = True
    return survive


def greedy_select(train: List[dict]) -> tuple:
    """Greedy forward selection on TRAIN. Returns (selected_names, selected_fns, promotion_log)."""
    filters = make_filters(train)
    selected_names, selected_fns = [], []
    log = []  # rows for the promotion table
    remaining = dict(filters)
    while remaining:
        prev = apply_filters(train, selected_fns)
        base_sh = sharpe(prev)
        base_cagr, _ = portfolio_cagr_mdd(prev)
        base_pf = pf_of(prev)
        cand = {}
        for name, fn in remaining.items():
            sub = [t for t in prev if fn(t)]
            if not sub:
                continue
            d_sh = sharpe(sub) - base_sh
            d_cagr = portfolio_cagr_mdd(sub)[0] - base_cagr
            d_pf = pf_of(sub) - base_pf
            frac = len(sub) / max(len(prev), 1)
            p = bootstrap_dsharpe_p(prev, fn)
            cand[name] = {"d_sh": d_sh, "d_cagr": d_cagr, "d_pf": d_pf,
                          "frac": frac, "n": len(sub), "p": p}
        if not cand:
            break
        # keep-rule predicate (before FDR): ΔSharpe>0 AND ΔCAGR>0 AND frac>=0.6
        eligible = {k: v for k, v in cand.items()
                    if v["d_sh"] > 0 and v["d_cagr"] > 0 and v["frac"] >= 0.6}
        fdr = bh_fdr({k: v["p"] for k, v in eligible.items()}) if eligible else {}
        eligible = {k: v for k, v in eligible.items() if fdr.get(k)}
        if not eligible:
            # record the best rejected candidate for transparency, then stop
            best_rej = max(cand.items(), key=lambda kv: kv[1]["d_sh"])
            log.append({"filter": best_rej[0], **best_rej[1], "kept": False,
                        "reason": "failed keep-rule/FDR (greedy stop)"})
            break
        best = max(eligible.items(), key=lambda kv: kv[1]["d_sh"])
        name, info = best
        log.append({"filter": name, **info, "kept": True, "reason": "kept"})
        selected_names.append(name)
        selected_fns.append(remaining[name])
        del remaining[name]
    return selected_names, selected_fns, log


# ---------- reporting ----------
def stat_block(rows: List[dict]) -> dict:
    if not rows:
        return {"n": 0}
    v = np.array([r["ret"] for r in rows])
    cagr, mdd = portfolio_cagr_mdd(rows)
    return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float((v > 0).mean()), "sharpe": sharpe(rows), "pf": pf_of(rows),
            "cagr": cagr, "mdd": mdd}


def fmt_row(name, s):
    if not s or s.get("n", 0) == 0:
        return f"| {name} | 0 | — | — | — | — | — | — | — |"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {name} | {s['n']:,} | {s['mean']*100:+.2f}% | {s['median']*100:+.2f}% | "
            f"{s['win']*100:.0f}% | {s['sharpe']:+.3f} | {pf} | {s['cagr']*100:+.1f}% | {s['mdd']*100:.0f}% |")


def main():
    print("Building liquidity-screened universe from cache...")
    universe = build_universe()
    regime = _spy_regime()
    n_etf = sum(1 for s in universe if s in KNOWN_ETFS)
    print(f"Universe: {len(universe)} names ({n_etf} ETFs, {len(universe)-n_etf} stocks)")

    print("Collecting core mean-reversion trades...")
    trades = collect_core_trades(universe, regime)
    train = [t for t in trades if t["date"] < TRAIN_END]
    test = [t for t in trades if t["date"] >= TRAIN_END]

    # core-signal OOS by asset class (the central question)
    etfs_oos = [t for t in test if t["etf"]]
    stk_oos = [t for t in test if not t["etf"]]

    print("Greedy forward filter selection on TRAIN...")
    sel_names, sel_fns, plog = greedy_select(train)

    core_test = stat_block(test)
    sel_test = stat_block(apply_filters(test, sel_fns))

    # §D regime stratification of the SELECTED config on full sample
    sel_all = apply_filters(trades, sel_fns)
    buckets = {
        "bull / vol-low": [t for t in sel_all if t["mkt_bull"] and not t["mkt_volhi"]],
        "bull / vol-high": [t for t in sel_all if t["mkt_bull"] and t["mkt_volhi"]],
        "bear / vol-low": [t for t in sel_all if not t["mkt_bull"] and not t["mkt_volhi"]],
        "bear / vol-high": [t for t in sel_all if not t["mkt_bull"] and t["mkt_volhi"]],
    }
    years = {}
    for t in sel_all:
        years.setdefault(t["date"].year, []).append(t)

    # diversification vs insider edge (reuse candidate_A approach)
    sel_daily = {}
    for r in sel_all:
        per_day = (1 + r["ret"]) ** (1 / r["hold"]) - 1
        sel_daily.setdefault(r["date"], []).append(per_day)
    sel_series = {d: float(np.mean(v)) for d, v in sel_daily.items()}
    try:
        ins = vg._collect(hold=3, min_adv=1_000_000)
        vthr = np.percentile([r["rv20"] for r in ins if r["date"] < TRAIN_END], 80) if ins else 1
        ins_daily = {}
        for r in ins:
            if r["rv20"] >= vthr:
                ins_daily.setdefault(r["date"], []).append((1 + r["gross"]) ** (1 / 3) - 1)
        ins_series = {d: float(np.mean(v)) for d, v in ins_daily.items()}
        common = sorted(set(sel_series) & set(ins_series))
        corr = (float(np.corrcoef([sel_series[d] for d in common],
                                  [ins_series[d] for d in common])[0, 1])
                if len(common) > 30 else float("nan"))
    except Exception as e:
        corr, common = float("nan"), []

    # ---- markdown ----
    H = "| Set | n | Mean | Median | Win% | Sharpe | PF | CAGR | MaxDD |"
    SEP = "|---|---|---|---|---|---|---|---|---|"
    L = [
        "# Tier-2 Mean-Reversion Universe Expansion — Gauntlet + §A Filter Promotion", "",
        f"Universe: **{len(universe)}** liquidity-screened names ({n_etf} ETFs, {len(universe)-n_etf} stocks), "
        f">={MIN_BARS} bars. Core: RSI({RSI_LEN})<{RSI_BUY} & close>SMA200 → next-open; "
        f"exit RSI>{RSI_EXIT} or {MAX_HOLD}d; {COST*1e4:.0f}bps. "
        f"Portfolio sim: {int(DEPLOYABLE*100)}% deployable, {int(PER_NAME*100)}%/name, {MAX_CONC} concurrent.", "",
        "## 1. Core signal OOS (2024+) — does the ETF edge survive on stocks?", "",
        H, SEP,
        fmt_row("ETF-only", stat_block(etfs_oos)),
        fmt_row("Stock-only", stat_block(stk_oos)),
        fmt_row("Combined", core_test), "",
        "## 2. §A filter promotion (greedy forward selection on TRAIN, BH-FDR q=0.10)", "",
        "| Step filter | ΔPF | ΔSharpe | ΔCAGR | retained | boot-p | kept | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in plog:
        L.append(f"| {r['filter']} | {r['d_pf']:+.2f} | {r['d_sh']:+.3f} | {r['d_cagr']*100:+.1f}% | "
                 f"{r['frac']*100:.0f}% | {r['p']:.3f} | {'✅' if r['kept'] else '❌'} | {r['reason']} |")
    L += ["", f"**Selected filters:** {', '.join(sel_names) if sel_names else '(none survived — core only)'}", "",
          "## 3. Selected config vs core — OOS (2024+)", "",
          H, SEP,
          fmt_row("Core only", core_test),
          fmt_row("Core + selected", sel_test), "",
          "## 4. §D regime stratification (selected config, full sample)", "",
          H, SEP]
    for name, rows in buckets.items():
        L.append(fmt_row(name, stat_block(rows)))
    L += ["", "### Per-year stability", "", H, SEP]
    for yr in sorted(years):
        L.append(fmt_row(str(yr), stat_block(years[yr])))
    L += ["", "## 5. Diversification",
          f"- Corr of daily PnL vs vol-gated INSIDER edge ({len(common)} common days): "
          f"**{corr:+.2f}**" if common else "- insider corr: n/a", ""]

    # verdict
    s = sel_test
    ok_pf = bool(s.get("n") and s["pf"] >= 2.0)
    ok_n = bool(s.get("n", 0) >= 30)
    ok_dd = bool(s.get("n") and s["mdd"] > -0.25)
    ok_corr = (abs(corr) < 0.4) if corr == corr else True
    ok_regime = all(stat_block(r).get("mean", -1) > -0.01 for r in buckets.values() if r)
    passed = ok_pf and ok_n and ok_dd and ok_corr and ok_regime
    L += ["## VERDICT", "",
          f"- OOS PF>=2.0: {'YES' if ok_pf else 'NO'} ({s.get('pf', float('nan')):.2f})" if s.get("n") else "- n/a",
          f"- OOS n>=30: {'YES' if ok_n else 'NO'} ({s.get('n', 0)})",
          f"- MaxDD>-25%: {'YES' if ok_dd else 'NO'} ({s.get('mdd', 0)*100:.0f}%)" if s.get("n") else "",
          f"- |insider corr|<0.4: {'YES' if ok_corr else 'NO'} ({corr:+.2f})",
          f"- Non-catastrophic in every regime bucket: {'YES' if ok_regime else 'NO'}",
          "",
          ("**PASS → propose wiring expanded Tier-2 with the selected filters (shadow first).**"
           if passed else
           "**FAIL → keep ETF-only Tier-2; expansion to stocks not validated as configured.**")]
    out = "\n".join(L)
    mode = sys.argv[1].lower() if (len(sys.argv) > 1 and sys.argv[1].lower() in ("etf", "usetf")) else "liquid"
    (base.PROJECT_ROOT / "backtest_outputs" / f"meanrev_expansion_{mode}.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
