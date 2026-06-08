#!/usr/bin/env python3

from tradingview_screener import Query, col

# Quality-growth screen requested by the user. Edit values here to retune.
DEFAULT_FILTERS = {
    "market_cap_min": 300_000_000,
    "market_cap_max": 10_000_000_000,
    "price_min": 3.99,
    "volume_min": 1_000_000,
    "relative_volume_min": 1.0,
    "revenue_growth_ttm_yoy_min": 0.0,     # %
    "eps_diluted_growth_ttm_yoy_min": 8.0,  # %
    "debt_to_equity_max": 1.0,
    "price_above_vwap": True,
    "analyst_rating": ("buy", "strong_buy"),
}

# TradingView "Analyst rating" categories → upper bound of recommendation_mark
# (scale: 1.0 = Strong Buy … 5.0 = Strong Sell; lower = more bullish).
_RATING_UPPER = {"strong_buy": 1.5, "buy": 2.5, "neutral": 3.5, "hold": 3.5,
                 "sell": 4.5, "strong_sell": 5.5}


def _rating_max_mark(analyst_rating):
    """Resolve the recommendation_mark ceiling from an analyst-rating spec.

    Accepts a string, an iterable of strings (e.g. ("buy","strong_buy")), or the legacy
    int count 2 (= the two bullish buckets). Returns the least-restrictive bullish ceiling.
    """
    if analyst_rating is None:
        return None
    if isinstance(analyst_rating, int):
        # N most-bullish buckets, ordered strong_buy, buy, neutral, sell, strong_sell
        order = ["strong_buy", "buy", "neutral", "sell", "strong_sell"]
        names = order[:max(1, analyst_rating)]
    elif isinstance(analyst_rating, str):
        names = [analyst_rating]
    else:
        names = list(analyst_rating)
    bounds = [_RATING_UPPER[str(n).lower().replace(" ", "_")] for n in names
              if str(n).lower().replace(" ", "_") in _RATING_UPPER]
    return max(bounds) if bounds else None


def scan_quality_growth(filters=None, limit=200, symbols_only=True, write=True):
    """Quality-growth screen — small/mid-cap, liquid, profitable-growth, analyst-favoured,
    trading above VWAP. Maps the user's filters dict to validated TradingView fields and
    writes tv_screener_tickers.txt (consumed by load_candidate_universe()).

    Fields: market_cap_basic, close, volume, relative_volume_10d_calc,
            total_revenue_yoy_growth_ttm, earnings_per_share_diluted_yoy_growth_ttm,
            debt_to_equity_fq, VWAP, recommendation_mark.
    NOTE: VWAP / relative volume are intraday — most meaningful during RTH.
    """
    f = {**DEFAULT_FILTERS, **(filters or {})}
    conditions = [
        col("market_cap_basic") >= f["market_cap_min"],
        col("market_cap_basic") <= f["market_cap_max"],
        col("close") > f["price_min"],
        col("volume") > f["volume_min"],
        col("relative_volume_10d_calc") > f["relative_volume_min"],
        col("total_revenue_yoy_growth_ttm") > f["revenue_growth_ttm_yoy_min"],
        col("earnings_per_share_diluted_yoy_growth_ttm") > f["eps_diluted_growth_ttm_yoy_min"],
        col("debt_to_equity_fq") < f["debt_to_equity_max"],
    ]
    if f.get("price_above_vwap"):
        conditions.append(col("close") > col("VWAP"))
    mark_ceiling = _rating_max_mark(f.get("analyst_rating"))
    if mark_ceiling is not None:
        conditions.append(col("recommendation_mark") <= mark_ceiling)

    query = (
        Query()
        .select("exchange", "name", "close", "volume", "relative_volume_10d_calc",
                "market_cap_basic", "total_revenue_yoy_growth_ttm",
                "earnings_per_share_diluted_yoy_growth_ttm", "debt_to_equity_fq",
                "VWAP", "recommendation_mark", "sector")
        .where(*conditions)
        .order_by("relative_volume_10d_calc", ascending=False)
        .limit(limit)
    )
    _, df = query.get_scanner_data()
    if df is None or df.empty:
        print("scan_quality_growth: no matches (off-hours VWAP/rel-vol, or filters too tight).")
        return [] if symbols_only else df

    print(f"\n=== Quality-growth screen: {len(df)} names (mark<= {mark_ceiling}) ===\n")
    print(df.to_string(index=False))
    tickers = df["name"].dropna().astype(str).unique().tolist()
    if write:
        with open("tv_screener_tickers.txt", "w") as fh:
            fh.write(",".join(f'"{t}"' for t in tickers))
        print(f"\nSaved {len(tickers)} tickers to tv_screener_tickers.txt")
    return tickers if symbols_only else df


def scan_stocks(limit=200, symbols_only=True):
    query = (
        Query()
        .select(
            "exchange",
            "name",
            "close",
            "volume",
            "relative_volume_10d_calc",
            "Perf.W",
            "Volatility.W",
            "SMA50",
            "SMA200",
            "sector",
        )
        .where(
            col("close") > 1,
            col("volume") > 1_000_000,
            col("Perf.W") > 0,
            col("Volatility.W") > 5,
            col("SMA50") > col("SMA200"),
        )
        .order_by("relative_volume_10d_calc", ascending=False)
        .limit(limit)
    )

    _, df = query.get_scanner_data()

    if df.empty:
        print("No matching stocks found.")
        return df

    if "exchange" in df.columns and "name" in df.columns:
        df["tv_ticker"] = df["exchange"].astype(str) + ":" + df["name"].astype(str)

    print("\n=== Screener Results ===\n")
    print(df.to_string(index=False))

    if symbols_only:
        tickers = df["name"].dropna().astype(str).unique().tolist()
    else:
        tickers = df["tv_ticker"].dropna().astype(str).unique().tolist()

    ticker_string = ",".join(f'"{ticker}"' for ticker in tickers)

    print("\n=== Quoted Comma-Separated Tickers ===\n")
    print(ticker_string)

    with open("tv_screener_tickers.txt", "w") as f:
        f.write(ticker_string)

    print("\nSaved to: tv_screener_tickers.txt")

    return df


def scan_premarket_momentum(limit=100, rating_col="Recommend.All", symbols_only=True, write=True):
    """Institutional premarket-momentum screen (drives premarket_levels.py watchlist).

    Filters: vol>1M · price>premarket-high · chg>0% · mkt-cap>2B · premarket-vol>5K ·
             Momentum>0 · Summary rating in {buy, strong buy}  (Recommend.All >= 0.1)
    NOTE: premarket_high / premarket_volume / change are LIVE fields — meaningful only during
    a trading session (premarket 4:00-9:30 ET, then RTH). Off-hours returns few/no names.
    """
    cols = ["exchange", "name", "close", "change", "volume", "premarket_high",
            "premarket_volume", "market_cap_basic", "Mom", rating_col, "Recommend.All"]
    query = (
        Query()
        .select(*cols)
        .where(
            col("volume") > 1_000_000,
            col("close") > col("premarket_high"),
            col("change") > 0,
            col("market_cap_basic") > 2_000_000_000,
            col("premarket_volume") > 5_000,
            col("Mom") > 0,
            col(rating_col) >= 0.1,                 # buy (>=0.1) or strong buy (>=0.5)
        )
        .order_by("Mom", ascending=False)
        .limit(limit)
    )
    _, df = query.get_scanner_data()
    if df.empty:
        print("scan_premarket_momentum: no matches (off-hours or no qualifying setups).")
        return [] if symbols_only else df
    tickers = df["name"].dropna().astype(str).unique().tolist()
    if write:
        with open("tv_screener_tickers.txt", "w") as f:
            f.write(",".join(f'"{t}"' for t in tickers))
    print(f"scan_premarket_momentum: {len(tickers)} names -> {tickers[:15]}{' ...' if len(tickers) > 15 else ''}")
    return tickers if symbols_only else df


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "quality"
    if arg == "premarket":
        scan_premarket_momentum(limit=100)
    elif arg == "legacy":
        scan_stocks(limit=200, symbols_only=True)
    else:
        scan_quality_growth(limit=200, symbols_only=True)