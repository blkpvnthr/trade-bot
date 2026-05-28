#!/usr/bin/env python3
"""
scripts/evaluate_variants.py

Rank all strategy variants by a multi-factor composite score derived from
execution metrics collected in logs/execution/{strategy_id}/.

Optimization objective: maximize risk-adjusted returns — NOT raw return alone.
Strategies are ranked by composite_score, which rewards Sharpe, consistency,
profit factor, and Calmar ratio while penalizing drawdowns, return volatility,
overtrading, skipped-signal noise, and alpha decay.

Metrics computed:
  Existing: total_return, sharpe_ratio, max_drawdown, profit_factor, win_rate,
            mae_avg, mfe_avg, trade_count, exposure_pct, daily_loss_violations
  New:      calmar_ratio, daily_pnl_consistency, return_volatility, avg_daily_pnl,
            trade_frequency_per_day, recent_sharpe_7d, alpha_decay,
            skipped_signal_count, walk_forward_oos_sharpe, composite_score

Writes:
  system-data/variant_rankings_{date}.csv
  system-data/variant_rankings_{date}.json
  daily_system_outputs/research_rankings.json  (for dashboard)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).strftime("%Y%m%d")

# Default composite score weights — overridden by evaluation.composite_score_weights in YAML.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "sharpe":           0.30,
    "consistency":      0.20,
    "profit_factor":    0.20,
    "calmar":           0.15,
    "return_volatility": 0.05,
    "alpha_decay":      0.05,
    "overtrading":      0.03,
    "skipped_noise":    0.02,
    "walk_forward_oos": 0.00,   # 0 until walk-forward data accumulates
}


# ─────────────────────────────────────────────────────────────────────────────
# Config + data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_event_files(strategy_id: str, lookback_days: int) -> list[Path]:
    """Find JSONL execution event files for a strategy variant."""
    log_root = PROJECT_ROOT / "logs" / "execution"
    cutoff = datetime.now(ET) - timedelta(days=lookback_days)
    files = []
    for date_dir in sorted(log_root.glob("*")):
        try:
            dir_date = datetime.strptime(date_dir.name, "%Y%m%d").replace(tzinfo=ET)
        except ValueError:
            continue
        if dir_date < cutoff:
            continue
        for pattern in [
            f"{date_dir.name}_execution_events.jsonl",
            f"{date_dir.name}_executed_orders.jsonl",
        ]:
            candidate = date_dir / strategy_id / pattern
            if not candidate.exists():
                candidate = date_dir / pattern
                if not candidate.exists():
                    continue
            files.append(candidate)
    return files


def load_trades(strategy_id: str, lookback_days: int) -> pd.DataFrame:
    """Load completed trade records for a strategy variant."""
    files = find_event_files(strategy_id, lookback_days)
    records = []
    for f in files:
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid = ev.get("strategy_id", ev.get("strategy", ""))
                    if strategy_id not in sid and sid:
                        continue
                    records.append(ev)
        except OSError:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "pnl" not in df.columns and "realized_pnl" in df.columns:
        df["pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    elif "pnl" in df.columns:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    else:
        return pd.DataFrame()

    df = df.dropna(subset=["pnl"])
    return df


def load_skipped_signals(strategy_id: str, lookback_days: int) -> int:
    """Count skipped signals logged within the lookback window."""
    path = PROJECT_ROOT / "logs" / "execution" / strategy_id / "skipped_signals.jsonl"
    if not path.exists():
        return 0
    cutoff = datetime.now(ET) - timedelta(days=lookback_days)
    count = 0
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    ts_str = ev.get("timestamp", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts.astimezone(ET) < cutoff:
                                continue
                        except Exception:
                            pass
                    count += 1
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return count


def load_walk_forward_oos_sharpe(strategy_id: str) -> float | None:
    """
    Load the most recent out-of-sample average Sharpe from walk-forward results.
    Returns None if no walk-forward data exists yet.
    """
    wf_dir = PROJECT_ROOT / "daily_system_outputs" / "walkforward"
    if not wf_dir.exists():
        return None
    candidates = sorted(wf_dir.glob(f"*_walkforward_{strategy_id}.json"), reverse=True)
    for path in candidates[:3]:
        try:
            data = json.loads(path.read_text())
            agg = data.get("aggregate", {})
            sharpe = agg.get("avg_sharpe") or agg.get("median_sharpe")
            if sharpe is not None:
                return float(sharpe)
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame, lookback_days: int) -> dict[str, Any]:
    """
    Compute all ranking metrics from a trade DataFrame.

    Returns both the original 10 metrics and the extended set used by
    compute_composite_score(). No live data is fetched — metrics are
    derived solely from the trade records passed in.
    """
    if trades.empty or len(trades) == 0:
        return _empty_metrics()

    pnl = trades["pnl"].values

    # ── Basic returns ─────────────────────────────────────────────────────────
    total_return = float(pnl.sum())

    # ── Daily P&L series ──────────────────────────────────────────────────────
    if "date" in trades.columns:
        daily_pnl = trades.groupby("date")["pnl"].sum()
    else:
        daily_pnl = pd.Series(pnl)

    mu = float(daily_pnl.mean())
    sigma_val = daily_pnl.std(ddof=1)
    sigma = float(sigma_val) if not np.isnan(sigma_val) else 0.0

    # ── Sharpe (annualised, 252 trading days) ─────────────────────────────────
    sharpe = float(mu / sigma * math.sqrt(252)) if sigma > 0 else 0.0

    # ── Max drawdown ──────────────────────────────────────────────────────────
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = float(drawdowns.min())

    # ── Profit factor ─────────────────────────────────────────────────────────
    gross_profit = float(pnl[pnl > 0].sum()) if any(pnl > 0) else 0.0
    gross_loss = abs(float(pnl[pnl < 0].sum())) if any(pnl < 0) else 1e-8
    profit_factor = gross_profit / gross_loss

    # ── Win rate ──────────────────────────────────────────────────────────────
    win_rate = float((pnl > 0).sum() / len(pnl))

    # ── MAE / MFE ─────────────────────────────────────────────────────────────
    mae_avg = float(trades["mae"].mean()) if "mae" in trades.columns else float("nan")
    mfe_avg = float(trades["mfe"].mean()) if "mfe" in trades.columns else float("nan")

    # ── Trade count ───────────────────────────────────────────────────────────
    trade_count = len(pnl)

    # ── Exposure ──────────────────────────────────────────────────────────────
    exposure_pct = float(trades["exposure_pct"].mean()) if "exposure_pct" in trades.columns else float("nan")

    # ── Daily loss violations ─────────────────────────────────────────────────
    trading_days = len(daily_pnl)
    if "date" in trades.columns:
        daily_loss_violations = int((daily_pnl < 0).sum())
    else:
        daily_loss_violations = 0

    # ── Calmar ratio (annualised return / absolute max drawdown) ─────────────
    # Rewards capital efficiency — penalises large drawdowns relative to return.
    annualised_return = (total_return / max(lookback_days, 1)) * 252
    calmar_ratio = annualised_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # ── Daily P&L consistency (fraction of trading days with positive P&L) ───
    profitable_days = int((daily_pnl > 0).sum())
    daily_pnl_consistency = profitable_days / trading_days if trading_days > 0 else 0.0

    # ── Return volatility (std of daily P&L) ─────────────────────────────────
    return_volatility = sigma

    # ── Average daily P&L ────────────────────────────────────────────────────
    avg_daily_pnl = mu

    # ── Trade frequency per trading day ──────────────────────────────────────
    trade_frequency_per_day = trade_count / max(trading_days, 1)

    # ── Recent 7-day Sharpe and alpha decay ──────────────────────────────────
    # alpha_decay = recent_sharpe / full_period_sharpe.
    # < 1.0 → strategy degrading; > 1.0 → improving.
    recent_sharpe_7d: float | None = None
    alpha_decay: float | None = None
    if "date" in trades.columns:
        recent_cutoff = (datetime.now(ET) - timedelta(days=7)).strftime("%Y-%m-%d")
        try:
            recent_pnl = daily_pnl[daily_pnl.index >= recent_cutoff]
        except Exception:
            recent_pnl = pd.Series(dtype=float)
        if len(recent_pnl) >= 3:
            r_mu = float(recent_pnl.mean())
            r_sigma_val = recent_pnl.std(ddof=1)
            r_sigma = float(r_sigma_val) if not np.isnan(r_sigma_val) else 0.0
            if r_sigma > 0:
                recent_sharpe_7d = float(r_mu / r_sigma * math.sqrt(252))
                if abs(sharpe) > 1e-6:
                    alpha_decay = recent_sharpe_7d / sharpe

    return {
        # Original metrics (preserved for backward-compatibility)
        "total_return":           round(total_return, 2),
        "sharpe_ratio":           round(sharpe, 3),
        "max_drawdown":           round(max_drawdown, 2),
        "profit_factor":          round(profit_factor, 3),
        "win_rate":               round(win_rate, 4),
        "mae_avg":                round(mae_avg, 4) if not math.isnan(mae_avg) else None,
        "mfe_avg":                round(mfe_avg, 4) if not math.isnan(mfe_avg) else None,
        "trade_count":            trade_count,
        "exposure_pct":           round(exposure_pct, 4) if not math.isnan(exposure_pct) else None,
        "daily_loss_violations":  daily_loss_violations,
        # Extended metrics
        "calmar_ratio":           round(calmar_ratio, 3),
        "daily_pnl_consistency":  round(daily_pnl_consistency, 4),
        "return_volatility":      round(return_volatility, 2),
        "avg_daily_pnl":          round(avg_daily_pnl, 2),
        "trade_frequency_per_day": round(trade_frequency_per_day, 2),
        "recent_sharpe_7d":       round(recent_sharpe_7d, 3) if recent_sharpe_7d is not None else None,
        "alpha_decay":            round(alpha_decay, 3) if alpha_decay is not None else None,
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "total_return": None, "sharpe_ratio": None, "max_drawdown": None,
        "profit_factor": None, "win_rate": None, "mae_avg": None, "mfe_avg": None,
        "trade_count": 0, "exposure_pct": None, "daily_loss_violations": 0,
        "calmar_ratio": None, "daily_pnl_consistency": None, "return_volatility": None,
        "avg_daily_pnl": None, "trade_frequency_per_day": None,
        "recent_sharpe_7d": None, "alpha_decay": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_score(
    metrics: dict[str, Any],
    skipped_signals: int,
    walk_forward_oos_sharpe: float | None,
    weights: dict[str, float],
) -> float:
    """
    Multi-factor composite score. Higher is better.

    Optimization objective: maximize risk-adjusted returns — NOT raw return alone.

    Rewards: Sharpe, daily P&L consistency, profit factor, Calmar ratio,
             walk-forward OOS performance, improving alpha.
    Penalizes: return volatility relative to avg P&L, overtrading,
               skipped-signal noise, alpha decay.

    All components are bounded via tanh so a single outlier metric cannot
    dominate. Weights are config-driven (composite_score_weights in YAML).

    This function makes no assumptions about guaranteed profits. A strategy
    with zero trades returns 0.0.
    """
    trade_count = metrics.get("trade_count") or 0
    if trade_count == 0:
        return 0.0

    def _t(x: float) -> float:
        return float(math.tanh(x))

    # ── Sharpe component ──────────────────────────────────────────────────────
    # tanh(1.5 Sharpe / 1.5) ≈ 0.76 — a Sharpe of 1.5 is strong signal.
    sharpe = float(metrics.get("sharpe_ratio") or 0.0)
    sharpe_signal = _t(sharpe / 1.5)

    # ── Consistency component ─────────────────────────────────────────────────
    # Maps [0, 1] profitable-day fraction to [-1, 1].
    consistency = float(metrics.get("daily_pnl_consistency") or 0.5)
    consistency_signal = consistency * 2.0 - 1.0

    # ── Profit factor component ───────────────────────────────────────────────
    # PF=1.0 (break-even) → 0, PF=2.0 → 0.76, PF=0.5 → -0.46.
    pf = float(metrics.get("profit_factor") or 1.0)
    pf_signal = _t(pf - 1.0)

    # ── Calmar component (return/drawdown efficiency) ─────────────────────────
    # Calmar ≥ 1.0 → strong signal; 0 or negative → no signal.
    calmar = float(metrics.get("calmar_ratio") or 0.0)
    calmar_signal = _t(calmar / 1.0)

    # ── Return volatility penalty ─────────────────────────────────────────────
    # Penalises strategies where daily P&L variance >> average daily P&L.
    avg_daily = abs(float(metrics.get("avg_daily_pnl") or 0.0)) + 1e-6
    return_vol = float(metrics.get("return_volatility") or 0.0)
    vol_ratio = return_vol / avg_daily
    volatility_signal = -_t(vol_ratio / 3.0)

    # ── Alpha decay component ─────────────────────────────────────────────────
    # > 1.0 = improving (positive signal); < 1.0 = decaying (negative signal).
    alpha_decay = metrics.get("alpha_decay")
    if alpha_decay is not None:
        alpha_signal = _t(float(alpha_decay) - 1.0)
    else:
        alpha_signal = 0.0  # neutral when insufficient recent data

    # ── Overtrading penalty ───────────────────────────────────────────────────
    # Fires when trade frequency exceeds 8 per trading day.
    trade_freq = float(metrics.get("trade_frequency_per_day") or 0.0)
    overtrading_signal = -_t(max(0.0, trade_freq - 8.0) / 5.0)

    # ── Skipped-signal noise penalty ─────────────────────────────────────────
    # Penalises excessive skipped signals relative to actual trades executed.
    skipped_per_trade = skipped_signals / (trade_count + 1)
    skip_signal = -_t(skipped_per_trade / 10.0)

    # ── Walk-forward OOS Sharpe component ────────────────────────────────────
    # Only contributes when walk-forward data exists (weight is 0.0 by default
    # until enough history accumulates). Set weight to 0.10+ once WF runs.
    if walk_forward_oos_sharpe is not None:
        wf_signal = _t(float(walk_forward_oos_sharpe) / 1.5)
    else:
        wf_signal = 0.0

    # ── Weighted composite ────────────────────────────────────────────────────
    w = weights
    score = (
        w.get("sharpe",            0.30) * sharpe_signal
        + w.get("consistency",     0.20) * consistency_signal
        + w.get("profit_factor",   0.20) * pf_signal
        + w.get("calmar",          0.15) * calmar_signal
        + w.get("return_volatility", 0.05) * volatility_signal
        + w.get("alpha_decay",     0.05) * alpha_signal
        + w.get("overtrading",     0.03) * overtrading_signal
        + w.get("skipped_noise",   0.02) * skip_signal
        + w.get("walk_forward_oos", 0.00) * wf_signal
    )
    return round(float(score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_variants(
    config_path: Path = PROJECT_ROOT / "configs" / "strategy_experiments.yaml",
    lookback_days: int = 30,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    eval_cfg = cfg.get("evaluation", {})
    lookback_days = eval_cfg.get("lookback_days", lookback_days)
    min_trades = eval_cfg.get("min_trades_for_ranking", 5)
    weights = {**_DEFAULT_WEIGHTS, **eval_cfg.get("composite_score_weights", {})}

    rows = []
    for variant_id, variant_cfg in cfg.get("variants", {}).items():
        trades = load_trades(variant_id, lookback_days)
        metrics = compute_metrics(trades, lookback_days)
        skipped = load_skipped_signals(variant_id, lookback_days)
        wf_oos = load_walk_forward_oos_sharpe(variant_id)

        composite = compute_composite_score(metrics, skipped, wf_oos, weights)

        metrics["variant_id"] = variant_id
        metrics["strategy_id"] = variant_cfg.get("strategy_id", variant_id)
        metrics["credential_set"] = variant_cfg.get("credential_set", "unknown")
        metrics["rankable"] = (metrics.get("trade_count") or 0) >= min_trades
        metrics["skipped_signal_count"] = skipped
        metrics["walk_forward_oos_sharpe"] = wf_oos
        metrics["composite_score"] = composite
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rankable = df[df["rankable"]].copy()
    if not rankable.empty:
        rankable["rank"] = (
            rankable["composite_score"]
            .fillna(-999)
            .rank(ascending=False, method="min")
            .astype(int)
        )
        df = df.merge(rankable[["variant_id", "rank"]], on="variant_id", how="left")
    else:
        df["rank"] = None

    cols = [
        "rank", "variant_id", "strategy_id", "credential_set",
        "composite_score", "sharpe_ratio", "calmar_ratio", "max_drawdown",
        "profit_factor", "win_rate", "daily_pnl_consistency", "avg_daily_pnl",
        "return_volatility", "trade_frequency_per_day", "alpha_decay",
        "recent_sharpe_7d", "skipped_signal_count", "walk_forward_oos_sharpe",
        "total_return", "trade_count", "mae_avg", "mfe_avg",
        "exposure_pct", "daily_loss_violations", "rankable",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("rank", na_position="last")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Rank strategy variants by composite score")
    parser.add_argument("--config", default="configs/strategy_experiments.yaml")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--out-dir", default="system-data")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    rankings = rank_variants(config_path, args.lookback_days)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"variant_rankings_{TODAY}.csv"
    json_path = out_dir / f"variant_rankings_{TODAY}.json"

    rankings.to_csv(csv_path, index=False)
    rankings.to_json(json_path, orient="records", indent=2)

    daily_out = PROJECT_ROOT / "daily_system_outputs"
    daily_out.mkdir(parents=True, exist_ok=True)
    rankings.to_json(daily_out / "research_rankings.json", orient="records", indent=2)

    print(f"[evaluate] ranked {len(rankings)} variants → {csv_path}")
    print(f"[evaluate] research rankings → {daily_out / 'research_rankings.json'}")
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    main()
