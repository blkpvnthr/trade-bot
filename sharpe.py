#!/usr/bin/env python3

import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

LOOKBACK_PERIOD = "1y"
RISK_FREE_RATE = 0.05
TRADING_DAYS = 252

MIN_RETURN = 0.25
MIN_SHARPE = 0.75
MAX_VOLATILITY = 1.50

TOP_N = 50

RETURN_POWER = 3.0
SHARPE_POWER = 1.0

BATCH_SIZE = 75
SLEEP_SECONDS = 1


def clean_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    ticker = ticker.replace(".", "-")
    ticker = ticker.replace("^", "")
    ticker = ticker.replace("/", "-")
    return ticker


def get_nasdaq_composite_universe() -> list[str]:
    print("Downloading NASDAQ listed symbols...")

    df = pd.read_csv(
        NASDAQ_LISTED_URL,
        sep="|",
        dtype=str,
    )

    if "Symbol" not in df.columns:
        raise ValueError(f"Symbol column missing. Columns: {df.columns.tolist()}")

    df = df[df["Symbol"].notna()]
    df = df[df["Symbol"] != "File Creation Time"]

    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].str.upper() != "Y"]

    if "ETF" in df.columns:
        df = df[df["ETF"].str.upper() != "Y"]

    tickers = []

    for symbol in df["Symbol"].tolist():
        ticker = clean_ticker(symbol)

        if not ticker:
            continue

        if "$" in ticker:
            continue

        if len(ticker) > 5:
            continue

        tickers.append(ticker)

    tickers = sorted(set(tickers))

    if not tickers:
        raise ValueError("No NASDAQ symbols found.")

    print(f"NASDAQ Composite universe: {len(tickers)} symbols")

    pd.Series(tickers, name="symbol").to_csv("nasdaq_composite_universe.csv", index=False)

    return tickers


def download_prices(tickers: list[str]) -> pd.DataFrame:
    frames = []

    total = len(tickers)

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]

        print(f"Downloading prices {i + 1}-{min(i + BATCH_SIZE, total)} of {total}")

        try:
            data = yf.download(
                tickers=batch,
                period=LOOKBACK_PERIOD,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception as exc:
            print(f"Batch failed: {exc}")
            time.sleep(SLEEP_SECONDS)
            continue

        if data.empty:
            time.sleep(SLEEP_SECONDS)
            continue

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" not in data.columns.get_level_values(0):
                time.sleep(SLEEP_SECONDS)
                continue

            prices = data["Close"]
        else:
            if "Close" not in data.columns:
                time.sleep(SLEEP_SECONDS)
                continue

            prices = data[["Close"]]
            prices.columns = batch[:1]

        frames.append(prices)

        time.sleep(SLEEP_SECONDS)

    if not frames:
        raise ValueError("No price data downloaded.")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill()
    prices = prices.dropna(axis=0, how="all")

    if prices.empty:
        raise ValueError("No usable price data after cleaning.")

    return prices


def calculate_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).dropna(how="all")

    annual_return = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    individual_sharpe = (annual_return - RISK_FREE_RATE) / annual_volatility

    metrics = pd.DataFrame(
        {
            "annual_return": annual_return,
            "individual_sharpe": individual_sharpe,
            "annual_volatility": annual_volatility,
        }
    )

    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    metrics = metrics.dropna()

    return metrics


def filter_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (metrics["annual_return"] >= MIN_RETURN)
        & (metrics["individual_sharpe"] >= MIN_SHARPE)
        & (metrics["annual_volatility"] <= MAX_VOLATILITY)
    ].copy()

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


def main() -> int:
    try:
        tickers = get_nasdaq_composite_universe()

        prices = download_prices(tickers)
        metrics = calculate_metrics(prices)
        selected = filter_metrics(metrics)

        print("\nFiltered NASDAQ Composite Winners")
        print("=" * 100)

        if selected.empty:
            print("No tickers passed filters.")
            print("\nTop names by annual return:")
            print(
                metrics
                .sort_values("annual_return", ascending=False)
                .head(50)
                .to_string(float_format=lambda x: f"{x:.4f}")
            )
        else:
            print(selected.to_string(float_format=lambda x: f"{x:.4f}"))

        metrics.sort_values("annual_return", ascending=False).to_csv(
            "nasdaq_composite_metrics.csv"
        )

        selected.to_csv("filtered_nasdaq_composite.csv")

        print("\nSaved:")
        print("nasdaq_composite_universe.csv")
        print("nasdaq_composite_metrics.csv")
        print("filtered_nasdaq_composite.csv")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())