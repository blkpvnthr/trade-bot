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
BENCHMARK = "QQQ"

MIN_RETURN = 0.05
MIN_SHARPE = 0.25
MAX_VOLATILITY = 2.50
MAX_BETA = -0.10
MIN_AVG_VOLUME = 1_000_000

TOP_N = 50

RETURN_POWER = 2.0
SHARPE_POWER = 1.0
NEGATIVE_BETA_POWER = 2.0

BATCH_SIZE = 75
SLEEP_SECONDS = 1

INVERSE_ETFS = [
    "SQQQ", "PSQ", "QID",
    "SPXU", "SH", "SDS",
    "TZA", "RWM", "TWM",
    "DOG", "DXD", "SDOW",
    "SARK", "NVDS", "TSLS",
    "SOXS", "LABD", "TECS",
    "WEBS", "FAZ", "DRV",
    "YANG", "EDZ", "ERY",
    "UVXY", "VIXY",
]


def clean_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    ticker = ticker.replace(".", "-")
    ticker = ticker.replace("^", "")
    ticker = ticker.replace("/", "-")
    return ticker


def get_nasdaq_stock_universe() -> list[str]:
    print("Downloading NASDAQ listed stock symbols...")

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
        raise ValueError("No NASDAQ stock symbols found.")

    print(f"NASDAQ stock universe: {len(tickers)} symbols")

    return tickers


def get_universe() -> list[str]:
    nasdaq_stocks = get_nasdaq_stock_universe()

    universe = sorted(set(nasdaq_stocks + INVERSE_ETFS + [BENCHMARK]))

    print(f"Inverse ETFs added: {len(INVERSE_ETFS)}")
    print(f"Total scan universe: {len(universe)}")

    pd.Series(universe, name="symbol").to_csv(
        "negative_beta_inverse_universe.csv",
        index=False,
    )

    return universe


def download_price_volume(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_frames = []
    volume_frames = []

    total = len(tickers)

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]

        print(
            f"Downloading prices/volume "
            f"{i + 1}-{min(i + BATCH_SIZE, total)} of {total}"
        )

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
            fields = data.columns.get_level_values(0)

            if "Close" not in fields or "Volume" not in fields:
                time.sleep(SLEEP_SECONDS)
                continue

            prices = data["Close"]
            volumes = data["Volume"]

        else:
            if "Close" not in data.columns or "Volume" not in data.columns:
                time.sleep(SLEEP_SECONDS)
                continue

            prices = data[["Close"]].copy()
            volumes = data[["Volume"]].copy()

            prices.columns = batch[:1]
            volumes.columns = batch[:1]

        price_frames.append(prices)
        volume_frames.append(volumes)

        time.sleep(SLEEP_SECONDS)

    if not price_frames:
        raise ValueError("No price data downloaded.")

    prices = pd.concat(price_frames, axis=1)
    volumes = pd.concat(volume_frames, axis=1)

    prices = prices.loc[:, ~prices.columns.duplicated()]
    volumes = volumes.loc[:, ~volumes.columns.duplicated()]

    prices = prices.dropna(axis=1, how="all")
    volumes = volumes.dropna(axis=1, how="all")

    common_cols = sorted(set(prices.columns) & set(volumes.columns))

    prices = prices[common_cols]
    volumes = volumes[common_cols]

    prices = prices.ffill().dropna(axis=0, how="all")
    volumes = volumes.fillna(0)

    if prices.empty:
        raise ValueError("No usable price data after cleaning.")

    if BENCHMARK not in prices.columns:
        raise ValueError(f"Benchmark {BENCHMARK} was not downloaded.")

    return prices, volumes


def calculate_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    aligned = pd.concat(
        [asset_returns, benchmark_returns],
        axis=1,
    ).dropna()

    if aligned.shape[0] < 30:
        return np.nan

    asset = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]

    benchmark_variance = benchmark.var()

    if benchmark_variance == 0:
        return np.nan

    return float(asset.cov(benchmark) / benchmark_variance)


def calculate_metrics(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).dropna(how="all")

    if BENCHMARK not in returns.columns:
        raise ValueError(f"Benchmark {BENCHMARK} missing from returns.")

    benchmark_returns = returns[BENCHMARK]

    annual_return = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    individual_sharpe = (
        annual_return - RISK_FREE_RATE
    ) / annual_volatility

    avg_daily_volume = volumes.mean()

    betas = {}

    for ticker in returns.columns:
        if ticker == BENCHMARK:
            continue

        betas[ticker] = calculate_beta(
            returns[ticker],
            benchmark_returns,
        )

    metrics = pd.DataFrame(
        {
            "annual_return": annual_return,
            "individual_sharpe": individual_sharpe,
            "annual_volatility": annual_volatility,
            "avg_daily_volume": avg_daily_volume,
        }
    )

    metrics["beta_to_qqq"] = pd.Series(betas)
    metrics = metrics.drop(index=[BENCHMARK], errors="ignore")
    metrics["is_inverse_etf"] = metrics.index.isin(INVERSE_ETFS)

    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    metrics = metrics.dropna()

    return metrics


def filter_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (
            (metrics["beta_to_qqq"] <= MAX_BETA)
            | (metrics["is_inverse_etf"])
        )
        & (metrics["annual_return"] >= MIN_RETURN)
        & (metrics["individual_sharpe"] >= MIN_SHARPE)
        & (metrics["annual_volatility"] <= MAX_VOLATILITY)
        & (metrics["avg_daily_volume"] >= MIN_AVG_VOLUME)
    ].copy()

    if selected.empty:
        return selected

    selected["negative_beta_score"] = (
        selected["beta_to_qqq"].abs()
        ** NEGATIVE_BETA_POWER
    )

    selected["rank_score"] = (
        selected["annual_return"].clip(lower=0) ** RETURN_POWER
    ) * (
        selected["individual_sharpe"].clip(lower=0) ** SHARPE_POWER
    ) * (
        selected["negative_beta_score"]
    )

    selected = selected.sort_values(
        "rank_score",
        ascending=False,
    )

    if TOP_N and TOP_N > 0:
        selected = selected.head(TOP_N)

    return selected


def main() -> int:
    try:
        tickers = get_universe()

        prices, volumes = download_price_volume(tickers)

        metrics = calculate_metrics(
            prices=prices,
            volumes=volumes,
        )

        selected = filter_metrics(metrics)

        print("\nFiltered Negative-Beta / Inverse ETF Winners")
        print("=" * 130)

        display_cols = [
            "annual_return",
            "individual_sharpe",
            "annual_volatility",
            "avg_daily_volume",
            "beta_to_qqq",
            "is_inverse_etf",
        ]

        if not selected.empty:
            display_cols += [
                "negative_beta_score",
                "rank_score",
            ]

        if selected.empty:
            print("No tickers passed filters.")
            print("\nTop negative-beta liquid names:")
            print(
                metrics[
                    metrics["avg_daily_volume"] >= MIN_AVG_VOLUME
                ]
                .sort_values("beta_to_qqq", ascending=True)
                .head(50)
                .to_string(float_format=lambda x: f"{x:.4f}")
            )
        else:
            print(
                selected[display_cols]
                .to_string(float_format=lambda x: f"{x:.4f}")
            )

        metrics.sort_values(
            "beta_to_qqq",
            ascending=True,
        ).to_csv("negative_beta_metrics.csv")

        selected.to_csv("filtered_negative_beta_inverse.csv")

        with open("tickers2.txt", "w") as f:
            f.write(
                ",".join(
                    f'"{ticker}"'
                    for ticker in selected.index.tolist()
                )
            )

        print("\nSaved:")
        print("negative_beta_inverse_universe.csv")
        print("negative_beta_metrics.csv")
        print("filtered_negative_beta_inverse.csv")
        print("tickers2.txt")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())