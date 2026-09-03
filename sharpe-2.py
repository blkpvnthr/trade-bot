#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


TICKER_FILE = "tickers.txt"

FALLBACK_TICKERS = [
    "ASX", "CRDO", "TSM", "AMD", "LRCX", "SE", "PDD", "META", "NFLX",
    "MSFT", "NVDA", "AMZN", "GOOGL", "SHOP", "AAPL", "CRM", "AVGO", "ORCL"
]

LOOKBACK_PERIOD = "1y"
RISK_FREE_RATE = 0.05

MIN_RETURN = 0.30
MIN_SHARPE = 0.75
MAX_VOLATILITY = 1.20

TOP_N = 5
MAX_WEIGHT = 0.40

RETURN_POWER = 3.0
SHARPE_POWER = 1.0

TRADING_DAYS = 252


def load_tickers() -> list[str]:
    path = Path(TICKER_FILE)

    if not path.exists():
        return FALLBACK_TICKERS

    text = path.read_text()

    tickers = []

    for item in text.replace("\n", ",").split(","):
        ticker = item.strip().upper().replace('"', "").replace("'", "")

        if ticker and not ticker.endswith(".TXT"):
            tickers.append(ticker)

    return sorted(set(tickers))


def download_prices(tickers: list[str]) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        period=LOOKBACK_PERIOD,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise ValueError("No price data downloaded.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers[:1]

    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna()

    if prices.empty:
        raise ValueError("No usable price data after cleaning.")

    return prices


def calculate_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()

    annual_return = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    individual_sharpe = (annual_return - RISK_FREE_RATE) / annual_volatility

    metrics = pd.DataFrame(
        {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "individual_sharpe": individual_sharpe,
        }
    )

    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna()

    return metrics


def drop_weak_names(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.copy()

    selected = selected[selected["annual_return"] >= MIN_RETURN]
    selected = selected[selected["individual_sharpe"] >= MIN_SHARPE]
    selected = selected[selected["annual_volatility"] <= MAX_VOLATILITY]

    if selected.empty:
        return selected

    selected["rank_score"] = (
        selected["annual_return"].clip(lower=0) ** RETURN_POWER
    ) * (
        selected["individual_sharpe"].clip(lower=0) ** SHARPE_POWER
    )

    selected = selected.sort_values("rank_score", ascending=False)

    if TOP_N and TOP_N > 0:
        selected = selected.head(TOP_N)

    return selected


def cap_and_normalize_weights(scores: pd.Series) -> pd.Series:
    scores = scores.clip(lower=0)

    if scores.sum() <= 0:
        scores = pd.Series(1.0, index=scores.index)

    weights = scores / scores.sum()

    for _ in range(100):
        capped = weights > MAX_WEIGHT

        if not capped.any():
            break

        weights[capped] = MAX_WEIGHT

        remaining_weight = 1.0 - weights[capped].sum()
        uncapped = ~capped

        if uncapped.sum() == 0:
            break

        remaining_scores = scores[uncapped]

        if remaining_scores.sum() <= 0:
            weights[uncapped] = remaining_weight / uncapped.sum()
        else:
            weights[uncapped] = remaining_weight * remaining_scores / remaining_scores.sum()

    return weights / weights.sum()


def build_portfolio(selected: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scores = selected["rank_score"]

    weights = cap_and_normalize_weights(scores)

    returns = prices[selected.index].pct_change().dropna()
    portfolio_daily_returns = returns @ weights

    portfolio_return = float(portfolio_daily_returns.mean() * TRADING_DAYS)
    portfolio_volatility = float(portfolio_daily_returns.std() * np.sqrt(TRADING_DAYS))

    if portfolio_volatility > 0:
        portfolio_sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    else:
        portfolio_sharpe = np.nan

    portfolio = selected.copy()
    portfolio["weight"] = weights

    summary = {
        "portfolio_return": portfolio_return,
        "portfolio_volatility": portfolio_volatility,
        "portfolio_sharpe": portfolio_sharpe,
        "names_selected": len(portfolio),
    }

    return portfolio.sort_values("weight", ascending=False), summary


def main() -> int:
    try:
        tickers = load_tickers()

        if not tickers:
            raise ValueError("No tickers found.")

        print(f"\nLoaded {len(tickers)} tickers:")
        print(", ".join(tickers))

        prices = download_prices(tickers)
        metrics = calculate_metrics(prices)
        portfolio_candidates = drop_weak_names(metrics)

        if portfolio_candidates.empty:
            print("\nNo tickers passed filters.")
            print("\nAll ticker metrics:")
            print(
                metrics
                .sort_values("annual_return", ascending=False)
                .to_string(float_format=lambda x: f"{x:.4f}")
            )
            return 1

        portfolio, summary = build_portfolio(portfolio_candidates, prices)

        dropped = metrics.index.difference(portfolio.index)

        print("\nSelected Return-Biased Portfolio")
        print("=" * 100)
        print(
            portfolio[
                [
                    "annual_return",
                    "annual_volatility",
                    "individual_sharpe",
                    "rank_score",
                    "weight",
                ]
            ].to_string(float_format=lambda x: f"{x:.4f}")
        )

        print("=" * 100)
        print(f"Names selected:        {summary['names_selected']}")
        print(f"Portfolio return:      {summary['portfolio_return']:.2%}")
        print(f"Portfolio volatility:  {summary['portfolio_volatility']:.2%}")
        print(f"Portfolio Sharpe:      {summary['portfolio_sharpe']:.3f}")

        if len(dropped) > 0:
            print("\nDropped Names")
            print("=" * 100)
            print(
                metrics
                .loc[dropped]
                .sort_values("annual_return", ascending=False)
                .to_string(float_format=lambda x: f"{x:.4f}")
            )
        with open("tickers3.txt", "w") as f:
            f.write(
                ",".join(
                    f'"{ticker}"'
                    for ticker in portfolio.index.tolist()
                )
            )
        portfolio.to_csv("optimized_portfolio.csv")
        metrics.to_csv("all_ticker_metrics.csv")

        print("\nSaved:")
        print("optimized_portfolio.csv")
        print("all_ticker_metrics.csv")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())