#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

from portfolio_optimizer import (
    OptimizerConfig,
    optimize_portfolio,
    print_portfolio_summary,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "daily_system_outputs"
OUT.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")

MODEL_PLAN_PATH = OUT / "regime_position_plan.csv"
BREAKOUT_CANDIDATES_PATH = OUT / "latest_breakout_candidates.csv"
SCANNER_SCORES_PATH = OUT / "latest_scanner_scores.csv"

TARGETS_OUT = OUT / "optimized_portfolio_targets.csv"
REBALANCE_OUT = OUT / "daily_walkforward_emulated_rebalance_orders.csv"
EXIT_OUT = OUT / "daily_walkforward_emulated_exit_orders.csv"


def norm(symbol: Any) -> str:
    return str(symbol or "").upper().strip().replace(".", "-")


def f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[read] failed {path}: {exc}")
        return pd.DataFrame()


def load_candidates() -> pd.DataFrame:
    for path in [MODEL_PLAN_PATH, BREAKOUT_CANDIDATES_PATH, SCANNER_SCORES_PATH]:
        df = safe_read_csv(path)

        if not df.empty and "symbol" in df.columns:
            print(f"[load] candidates={path}")
            df["symbol"] = df["symbol"].map(norm)
            return df.drop_duplicates("symbol").reset_index(drop=True)

    raise FileNotFoundError(
        "No candidate file found. Expected one of: "
        f"{MODEL_PLAN_PATH}, {BREAKOUT_CANDIDATES_PATH}, {SCANNER_SCORES_PATH}"
    )


def get_account_and_positions() -> tuple[object, pd.DataFrame]:
    client = TradingClient(API_KEY, SECRET_KEY, paper=True)

    account = client.get_account()

    rows: list[dict[str, Any]] = []

    for p in client.get_all_positions():
        rows.append(
            {
                "symbol": norm(getattr(p, "symbol", "")),
                "qty": f(getattr(p, "qty", 0)),
                "avg_entry_price": f(getattr(p, "avg_entry_price", 0)),
                "current_price": f(getattr(p, "current_price", 0)),
                "market_value": f(getattr(p, "market_value", 0)),
                "cost_basis": f(getattr(p, "cost_basis", 0)),
                "unrealized_pl": f(getattr(p, "unrealized_pl", 0)),
                "unrealized_plpc": f(getattr(p, "unrealized_plpc", 0)),
            }
        )

    return account, pd.DataFrame(rows)


def ensure_optimizer_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy()
    df["symbol"] = df["symbol"].map(norm)

    rename_map = {
        "close": "price",
        "explosive_alpha_probability": "continuation_prob",
        "explosive_continuation_probability": "continuation_prob",
        "continuation_probability": "continuation_prob",
        "confidence": "trade_confidence",
        "confidence_score": "trade_confidence",
        "setup_score": "model_score",
        "score": "model_score",
        "expected_return": "expected_return_5d",
        "expected_sharpe": "expected_sharpe_score",
        "atr_pct": "volatility20",
        "late_reversal_risk": "confluence_reversal_risk",
    }

    for old, new in rename_map.items():
        if new not in df.columns and old in df.columns:
            df[new] = df[old]

    defaults = {
        "price": 0.0,
        "trade_confidence": 0.50,
        "continuation_prob": 0.50,
        "model_score": 0.50,
        "expected_return_5d": 0.0,
        "expected_sharpe_score": 0.0,
        "trend_score": 0.50,
        "momentum_score": 0.50,
        "volatility20": 0.03,
        "confluence_reversal_risk": 0.0,
        "bear": 0.0,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [c for c in defaults.keys() if c != "symbol"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(defaults[col])

    df = df[df["price"] > 0].copy()

    if df.empty:
        raise RuntimeError("No candidates with valid price after optimizer column normalization.")

    return df.reset_index(drop=True)


def clean_order_outputs(rebalance: pd.DataFrame, exits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = rebalance.copy() if rebalance is not None else pd.DataFrame()
    ex = exits.copy() if exits is not None else pd.DataFrame()

    if not rb.empty:
        rb["symbol"] = rb["symbol"].map(norm)

        if "side" not in rb.columns:
            rb["side"] = rb["delta_dollars"].map(lambda x: "buy" if f(x) > 0 else "sell")

        rb["side"] = rb["side"].astype(str).str.lower()
        rb["delta_dollars"] = pd.to_numeric(rb.get("delta_dollars", 0), errors="coerce").fillna(0.0)

        # portfolio_optimizer can return both buy and sell deltas here.
        # Split sells into exit file so alpaca_executor can sell first cleanly.
        sell_from_rb = rb[rb["side"] == "sell"].copy()
        buy_from_rb = rb[rb["side"] == "buy"].copy()

        if not sell_from_rb.empty:
            if "reason" not in sell_from_rb.columns:
                sell_from_rb["reason"] = "optimizer_reduce_position"

            ex = pd.concat([ex, sell_from_rb], ignore_index=True)

        rb = buy_from_rb.copy()

    if not rb.empty:
        if "reason" not in rb.columns:
            rb["reason"] = "optimizer_add_or_open"

        rb["delta_dollars"] = rb["delta_dollars"].abs().round(2)

    if not ex.empty:
        ex["symbol"] = ex["symbol"].map(norm)

        if "side" not in ex.columns:
            ex["side"] = "sell"

        ex["side"] = "sell"

        if "reason" not in ex.columns:
            ex["reason"] = "optimizer_reduce_or_exit"

        if "delta_shares" in ex.columns:
            ex["delta_shares"] = pd.to_numeric(ex["delta_shares"], errors="coerce").fillna(0).abs().astype(int)

        if "delta_dollars" in ex.columns:
            ex["delta_dollars"] = pd.to_numeric(ex["delta_dollars"], errors="coerce").fillna(0.0).abs().round(2)

    return rb.reset_index(drop=True), ex.reset_index(drop=True)


def main() -> None:
    print("===== PORTFOLIO REBALANCER =====")

    raw_candidates = load_candidates()
    candidates = ensure_optimizer_columns(raw_candidates)

    account, positions = get_account_and_positions()

    equity = max(f(getattr(account, "equity", 0)), 1.0)
    cash = f(getattr(account, "cash", 0))

    print(f"[account] equity=${equity:,.2f} cash=${cash:,.2f}")
    print(f"[positions] open={len(positions)}")
    print(f"[candidates] rows={len(candidates)}")

    config = OptimizerConfig(starting_equity=equity)

    targets, rebalance_orders, exit_orders, summary = optimize_portfolio(
        candidates=candidates,
        equity=equity,
        current_positions=positions,
        config=config,
    )

    rebalance_orders, exit_orders = clean_order_outputs(
        rebalance_orders,
        exit_orders,
    )

    targets.to_csv(TARGETS_OUT, index=False)
    rebalance_orders.to_csv(REBALANCE_OUT, index=False)
    exit_orders.to_csv(EXIT_OUT, index=False)

    print_portfolio_summary(summary)

    print(f"[saved] {TARGETS_OUT} rows={len(targets)}")
    print(f"[saved] {REBALANCE_OUT} rows={len(rebalance_orders)}")
    print(f"[saved] {EXIT_OUT} rows={len(exit_orders)}")
    print("[done] portfolio rebalance orders generated")


if __name__ == "__main__":
    main()