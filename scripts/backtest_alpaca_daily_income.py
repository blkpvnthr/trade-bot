#!/usr/bin/env python3
"""Backtest ``alpaca_daily_income`` against Alpaca historical minute bars.

The $400 daily value is an optimization benchmark and entry lock, not a
performance promise. This is an offline research command and never submits
orders.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.replay_market_day import _load_alpaca_bars
from scripts.walkforward_strategy import WFConfig, _compute_metrics, _simulate_portfolio

OUTPUT_DIR = PROJECT_ROOT / "daily_system_outputs" / "backtests"
STRATEGY_ID = "alpaca_daily_income"


def _trading_dates(start: date, end: date) -> list[date]:
    result: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            result.append(day)
        day += timedelta(days=1)
    return result


def load_alpaca_range(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    for trading_day in _trading_dates(start, end):
        daily = _load_alpaca_bars(symbols, trading_day)
        for symbol, bars in daily.items():
            if not bars.empty:
                frames[symbol].append(bars)
    return {
        symbol: pd.concat(parts).sort_index()
        for symbol, parts in frames.items()
        if parts
    }


def run_backtest(
    symbols: list[str],
    start: date,
    end: date,
    initial_cash: float,
    risk_per_trade_pct: float,
    daily_profit_target: float,
    max_daily_loss_pct: float,
) -> dict:
    bars_by_symbol = load_alpaca_range(symbols, start, end)
    if not bars_by_symbol:
        raise RuntimeError(
            "No Alpaca historical bars loaded. Configure APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY for a paper account and verify the requested dates."
        )
    config = WFConfig(
        starting_cash=initial_cash,
        max_position_weight=0.12,
        max_gross_exposure=0.70,
        max_order_dollars=initial_cash * 0.12,
        cash_buffer_pct=0.08,
        risk_per_trade_pct=risk_per_trade_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        daily_profit_target_dollars=daily_profit_target,
        max_open_positions=5,
    )
    portfolio = _simulate_portfolio(STRATEGY_ID, bars_by_symbol, {}, config)
    metrics = _compute_metrics(portfolio)
    return {
        "strategy_id": STRATEGY_ID,
        "objective_note": (
            "Daily profit target is an aspirational paper-trading entry lock, "
            "not a guarantee of profit."
        ),
        "data_source": "alpaca_historical_bars",
        "date_range": {"start": str(start), "end": str(end)},
        "symbols": sorted(bars_by_symbol),
        "config": {
            "initial_cash": initial_cash,
            "risk_per_trade_pct": risk_per_trade_pct,
            "daily_profit_target_dollars": daily_profit_target,
            "max_daily_loss_pct": max_daily_loss_pct,
            "max_position_weight": config.max_position_weight,
            "max_gross_exposure": config.max_gross_exposure,
            "max_open_positions": config.max_open_positions,
        },
        "metrics": metrics,
        "trades": portfolio.trades,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca historical backtest for daily-income paper strategy")
    parser.add_argument("--start", required=True, help="First session, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last session, YYYY-MM-DD")
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,NVDA,AMD,TSLA")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.0035)
    parser.add_argument("--daily-profit-target", type=float, default=400.0)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.015)
    args = parser.parse_args()

    result = run_backtest(
        symbols=[item.strip().upper() for item in args.symbols.split(",") if item.strip()],
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        initial_cash=args.initial_cash,
        risk_per_trade_pct=args.risk_per_trade_pct,
        daily_profit_target=args.daily_profit_target,
        max_daily_loss_pct=args.max_daily_loss_pct,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{datetime.now().strftime('%Y%m%d')}_{STRATEGY_ID}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    metrics = result["metrics"]
    print(f"[backtest] output={path}")
    print(
        f"[backtest] pnl=${metrics['net_pnl']:,.2f} avg_daily=${metrics['average_daily_pnl']:,.2f} "
        f"trades={metrics['total_trades']} win_rate={metrics['win_rate_pct']:.1f}% "
        f"profit_factor={metrics['profit_factor']:.2f} sharpe={metrics['sharpe_ratio']:.2f} "
        f"max_dd={metrics['max_drawdown_pct']:.2f}% avg_R={metrics['average_r_per_trade']:.2f}"
    )
    print(
        f"[backtest] target_days={metrics['days_hitting_profit_target']} "
        f"max_loss_days={metrics['days_hitting_max_loss']} "
        "(objective only; profits are not guaranteed)"
    )


if __name__ == "__main__":
    main()
