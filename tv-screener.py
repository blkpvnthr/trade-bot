#!/usr/bin/env python3

from tradingview_screener import Query, col

# Single canonical universe screen requested by the user (2026-06-13). Edit here to retune.
#   price > $5 | EPS growth YoY TTM > 0% | analyst rating buy/strong_buy |
#   volume > 1M | oscillator (OS) rating buy/strong_buy | EV/EBIT TTM > 20
#   sorted by EV/EBIT TTM descending.
DEFAULT_FILTERS = {
    "price_min": 5.0,                            # close > $5
    "eps_diluted_growth_ttm_yoy_min": 0.0,       # EPS growth YoY TTM > 0%
    "analyst_rating": ("buy", "strong_buy"),     # recommendation_mark <= 2.5
    "volume_min": 1_000_000,                     # volume > 1M shares
    "oscillator_rating": ("buy", "strong_buy"),  # Recommend.Other > 0.1
    "ev_ebit_ttm_min": 20.0,                     # EV/EBIT TTM > 20
}

# TradingView "Analyst rating" categories → upper bound of recommendation_mark
# (scale: 1.0 = Strong Buy … 5.0 = Strong Sell; lower = more bullish).
_RATING_UPPER = {"strong_buy": 1.5, "buy": 2.5, "neutral": 3.5, "hold": 3.5,
                 "sell": 4.5, "strong_sell": 5.5}

# TradingView technical gauge (Oscillators / MAs / Summary) ∈ [-1, 1]:
# Strong Buy > 0.5, Buy 0.1..0.5, Neutral -0.1..0.1, Sell -0.5..-0.1, Strong Sell < -0.5.
_GAUGE_LOWER = {"strong_buy": 0.5, "buy": 0.1, "neutral": -0.1, "sell": -0.5, "strong_sell": -1.0}


def _gauge_min(rating):
    """Least-restrictive lower bound on a TradingView technical gauge for a rating spec
    (e.g. ('buy','strong_buy') → 0.1, so the condition `gauge > 0.1` admits both)."""
    if rating is None:
        return None
    names = [rating] if isinstance(rating, str) else list(rating)
    bounds = [_GAUGE_LOWER[str(n).lower().replace(" ", "_")] for n in names
              if str(n).lower().replace(" ", "_") in _GAUGE_LOWER]
    return min(bounds) if bounds else None


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
        col("close") > f["price_min"],
        col("volume") > f["volume_min"],
        col("earnings_per_share_diluted_yoy_growth_ttm") > f["eps_diluted_growth_ttm_yoy_min"],
        col("enterprise_value_to_ebit_ttm") > f["ev_ebit_ttm_min"],
    ]
    mark_ceiling = _rating_max_mark(f.get("analyst_rating"))
    if mark_ceiling is not None:
        conditions.append(col("recommendation_mark") <= mark_ceiling)
    osc_min = _gauge_min(f.get("oscillator_rating"))
    if osc_min is not None:
        conditions.append(col("Recommend.Other") > osc_min)

    query = (
        Query()
        .select("exchange", "name", "close", "volume",
                "earnings_per_share_diluted_yoy_growth_ttm", "recommendation_mark",
                "Recommend.Other", "enterprise_value_to_ebit_ttm", "sector")
        .where(*conditions)
        .order_by("enterprise_value_to_ebit_ttm", ascending=False)
        .limit(limit)
    )
    _, df = query.get_scanner_data()
    if df is None or df.empty:
        print("scan_quality_growth: no matches (filters too tight).")
        return [] if symbols_only else df

    print(f"\n=== Universe screen: {len(df)} names "
          f"(analyst mark<= {mark_ceiling}, OS gauge> {osc_min}, EV/EBIT desc) ===\n")
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