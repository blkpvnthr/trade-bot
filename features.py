import os
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from config.env_loader import load_instance_env

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from market_context_builder import TODAY_TAG


def normalize_symbol(sym: str) -> str:
    return str(sym).replace(".", "-").upper().strip()


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_instance_env()

OUT_DIR = PROJECT_ROOT / "system-data" / f"{TODAY_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW = OUT_DIR / f"{TODAY_TAG}_daily_raw_bars.csv"
OUT_FEATURES = OUT_DIR / f"{TODAY_TAG}_daily_features.csv"
OUT_SYMBOLS = OUT_DIR / f"{TODAY_TAG}_download_universe_symbols.csv"

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

# These are required by trade.py / run_daily_system_walkforward_emulated.py.
REGIME_SYMBOLS = list(dict.fromkeys([
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "VIXY",
]))

CORE_SYMBOLS = pd.read_csv("data/bullish_2026-05-13.csv")["Symbol"].to_list()

# Set to None to use the full combined universe.
# Use 500/1000 if Alpaca rate limits are an issue.
MAX_SYMBOLS: int | None = 30


def get_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }


def get_sp500() -> list[str]:
    print("[setup] fetching S&P 500 symbols...")

    resp = requests.get(WIKI_SP500_URL, headers=get_headers(), timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))

    if not tables:
        raise RuntimeError("Could not parse S&P 500 table from Wikipedia.")

    df = tables[0].copy()

    if "Symbol" not in df.columns:
        raise RuntimeError("Wikipedia S&P 500 table does not contain a Symbol column.")

    symbols = (
        df["Symbol"]
        .astype(str)
        .map(normalize_symbol)
        .tolist()
    )

    symbols = sorted(set(s for s in symbols if s and s != "NAN"))

    print(f"[setup] S&P 500 symbols={len(symbols)}")
    return symbols


def get_nasdaq_constituents() -> list[str]:
    print("[setup] fetching NASDAQ-listed symbols...")

    resp = requests.get(NASDAQ_LISTED_URL, headers=get_headers(), timeout=30)
    resp.raise_for_status()

    lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
    data_lines = [
        line for line in lines
        if not line.startswith("File Creation Time")
    ]

    df = pd.read_csv(StringIO("\n".join(data_lines)), sep="|")

    if "Symbol" not in df.columns:
        raise RuntimeError("NASDAQ symbol file does not contain a Symbol column.")

    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].astype(str).str.upper() == "N"].copy()

    symbols: list[str] = []

    for raw_sym in df["Symbol"].tolist():
        if pd.isna(raw_sym):
            continue

        sym = normalize_symbol(raw_sym)

        if not sym or sym == "NAN":
            continue

        # Remove preferreds, warrants, units, special classes, etc.
        if "." in sym or "$" in sym or "^" in sym or "/" in sym:
            continue

        symbols.append(sym)

    symbols = sorted(set(symbols))

    print(f"[setup] NASDAQ-listed symbols={len(symbols)}")
    return symbols


def get_download_universe() -> list[str]:
    sp500 = get_sp500()
    nasdaq = get_nasdaq_constituents()

    symbols = list(dict.fromkeys(CORE_SYMBOLS + sp500 + nasdaq))
    symbols = [normalize_symbol(s) for s in symbols if s and str(s).strip()]
    symbols = sorted(set(symbols))

    if MAX_SYMBOLS is not None:
        symbols = symbols[:MAX_SYMBOLS]

        for required in CORE_SYMBOLS:
            if required not in symbols:
                symbols.append(required)

        for required in REGIME_SYMBOLS:
            if required not in symbols:
                symbols.append(required)

        symbols = sorted(set(symbols))

    out = pd.DataFrame({"symbol": symbols})
    out.to_csv(OUT_SYMBOLS, index=False)

    print(f"[setup] final combined download universe={len(symbols)}")
    print(f"[saved] symbols: {OUT_SYMBOLS}")

    return symbols


def load_alpaca_client() -> StockHistoricalDataClient:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            f"Missing Alpaca credentials. Expected APCA_API_KEY_ID and "
            f"APCA_API_SECRET_KEY in {ENV_PATH}"
        )

    return StockHistoricalDataClient(api_key, secret_key)


def _download_chunk(
    client: StockHistoricalDataClient,
    chunk: list[str],
    start: datetime,
    end: datetime,
    feed: DataFeed,
) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=chunk,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=feed,
    )

    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()

    if df.empty:
        return pd.DataFrame()

    df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_alpaca_daily_bars(
    symbols: List[str],
    lookback_days: int = 520,
    chunk_size: int = 50,
    feed: DataFeed = DataFeed.IEX,
) -> pd.DataFrame:
    client = load_alpaca_client()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    symbols = list(dict.fromkeys([normalize_symbol(s) for s in symbols if s]))
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

    frames: list[pd.DataFrame] = []

    print(
        f"[download] {start.date()} -> {end.date()} "
        f"| symbols={len(symbols)} | chunks={len(chunks)}"
    )

    for i, chunk in enumerate(chunks, start=1):
        print(f"[download] chunk {i}/{len(chunks)} size={len(chunk)}")

        try:
            df = _download_chunk(
                client=client,
                chunk=chunk,
                start=start,
                end=end,
                feed=feed,
            )

            if not df.empty:
                frames.append(df)

        except Exception as e:
            print(f"[warning] failed chunk={i} error={e}")
            print("[warning] retrying symbols individually...")

            for sym in chunk:
                try:
                    df = _download_chunk(
                        client=client,
                        chunk=[sym],
                        start=start,
                        end=end,
                        feed=feed,
                    )

                    if not df.empty:
                        frames.append(df)

                except Exception as sym_exc:
                    print(f"[skip] symbol={sym} error={sym_exc}")

    if not frames:
        raise RuntimeError("No data downloaded from Alpaca.")

    out = pd.concat(frames, ignore_index=True)

    out = (
        out.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        .drop_duplicates(subset=["symbol", "timestamp"])
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    print(f"[download] rows={len(out):,} symbols={out['symbol'].nunique():,}")

    return out


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _safe_merge_context(
    features: pd.DataFrame,
    symbol: str,
    prefix: str,
    cols: list[str],
) -> pd.DataFrame:
    available = [c for c in cols if c in features.columns]
    base_cols = ["timestamp"] + available

    if not available:
        return features

    proxy = features[features["symbol"] == symbol][base_cols].copy()

    if proxy.empty:
        return features

    proxy = proxy.rename(columns={c: f"{prefix}_{c}" for c in available})
    return features.merge(proxy, on="timestamp", how="left")


def build_features(
    raw: pd.DataFrame,
    regime_symbols: list[str] | None = None,
    add_regime: bool = True,
) -> pd.DataFrame:
    """
    Build daily features.

    Signature intentionally accepts regime_symbols and add_regime so trade.py can call:

        build_features(raw, regime_symbols=REGIME_SYMBOLS, add_regime=True)

    Your previous version only accepted build_features(raw), which caused incompatibility
    with the walk-forward-emulated orchestrator.
    """
    df = raw.copy()

    if regime_symbols is None:
        regime_symbols = REGIME_SYMBOLS

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    feature_frames = []

    for symbol, g in df.groupby("symbol", sort=True):
        g = g.copy().sort_values("timestamp")

        g["return_1d"] = g["close"].pct_change()
        g["return_3d"] = g["close"].pct_change(3)
        g["return_5d"] = g["close"].pct_change(5)
        g["return_10d"] = g["close"].pct_change(10)
        g["return_20d"] = g["close"].pct_change(20)

        # Alias columns expected by some newer model files.
        g["ret_1d"] = g["return_1d"]
        g["ret_3d"] = g["return_3d"]
        g["ret_5d"] = g["return_5d"]
        g["ret_10d"] = g["return_10d"]
        g["ret_20d"] = g["return_20d"]

        g["dollar_volume"] = g["close"] * g["volume"]
        g["volume_sma_20"] = g["volume"].rolling(20).mean()
        g["relative_volume_20"] = g["volume"] / g["volume_sma_20"].replace(0, np.nan)

        g["sma_5"] = g["close"].rolling(5).mean()
        g["sma_10"] = g["close"].rolling(10).mean()
        g["sma_20"] = g["close"].rolling(20).mean()
        g["sma_50"] = g["close"].rolling(50).mean()
        g["sma_200"] = g["close"].rolling(200).mean()

        # Alias columns expected by walk-forward-emulated logic.
        g["sma5"] = g["sma_5"]
        g["sma20"] = g["sma_20"]
        g["sma50"] = g["sma_50"]
        g["sma200"] = g["sma_200"]

        g["close_vs_sma_5"] = g["close"] / g["sma_5"] - 1
        g["close_vs_sma_20"] = g["close"] / g["sma_20"] - 1
        g["close_vs_sma_50"] = g["close"] / g["sma_50"] - 1
        g["close_vs_sma_200"] = g["close"] / g["sma_200"] - 1

        g["dist_sma5"] = g["close_vs_sma_5"]
        g["dist_sma20"] = g["close_vs_sma_20"]
        g["dist_sma50"] = g["close_vs_sma_50"]
        g["dist_sma200"] = g["close_vs_sma_200"]

        g["sma_5_above_20"] = (g["sma_5"] > g["sma_20"]).astype(int)
        g["sma_50_above_200"] = (g["sma_50"] > g["sma_200"]).astype(int)

        g["rsi_14"] = rsi(g["close"], 14)
        g["rsi"] = g["rsi_14"]

        ema_12 = g["close"].ewm(span=12, adjust=False).mean()
        ema_26 = g["close"].ewm(span=26, adjust=False).mean()
        g["macd"] = ema_12 - ema_26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]

        bb_mid = g["close"].rolling(20).mean()
        bb_std = g["close"].rolling(20).std()
        g["bb_mid"] = bb_mid
        g["bb_high"] = bb_mid + 2 * bb_std
        g["bb_low"] = bb_mid - 2 * bb_std
        g["bb_width"] = (g["bb_high"] - g["bb_low"]) / bb_mid.replace(0, np.nan)
        g["bb_position"] = (g["close"] - g["bb_low"]) / (g["bb_high"] - g["bb_low"]).replace(0, np.nan)

        prev_close = g["close"].shift(1)
        tr1 = g["high"] - g["low"]
        tr2 = (g["high"] - prev_close).abs()
        tr3 = (g["low"] - prev_close).abs()
        g["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["atr_14"] = g["true_range"].rolling(14).mean()
        g["atr_pct"] = g["atr_14"] / g["close"].replace(0, np.nan)

        g["volatility_10d"] = g["return_1d"].rolling(10).std()
        g["volatility_20d"] = g["return_1d"].rolling(20).std()

        g["daily_range_pct"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)
        g["close_location"] = (g["close"] - g["low"]) / (g["high"] - g["low"]).replace(0, np.nan)

        g["prev_high"] = g["high"].shift(1)
        g["prev_low"] = g["low"].shift(1)
        g["closed_above_prev_high"] = (g["close"] > g["prev_high"]).astype(int)
        g["closed_below_prev_low"] = (g["close"] < g["prev_low"]).astype(int)

        # Alias names used by the explosive walk-forward file.
        g["breakout_above_prev_high"] = g["closed_above_prev_high"]
        g["breakdown_below_prev_low"] = g["closed_below_prev_low"]

        g["future_return_1d"] = g["close"].shift(-1) / g["close"] - 1
        g["future_return_3d"] = g["close"].shift(-3) / g["close"] - 1
        g["future_return_5d"] = g["close"].shift(-5) / g["close"] - 1

        g["symbol_id"] = symbol

        feature_frames.append(g)

    features = pd.concat(feature_frames, ignore_index=True)

    if add_regime:
        context_cols = [
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "close_vs_sma_20",
            "rsi_14",
        ]

        for sym in regime_symbols:
            sym_norm = normalize_symbol(sym)
            prefix = sym_norm.lower()
            features = _safe_merge_context(features, sym_norm, prefix, context_cols)

    # Preserve your original QQQ/SPY/VIXY names.
    if "qqq_return_5d" not in features.columns and "qqq_return_5d" not in features:
        pass

    # The generic merge above creates qqq_return_1d, spy_return_1d, etc.
    if "rel_strength_vs_qqq_5d" not in features.columns:
        if "qqq_return_5d" in features.columns:
            features["rel_strength_vs_qqq_5d"] = features["return_5d"] - features["qqq_return_5d"]
        else:
            features["rel_strength_vs_qqq_5d"] = 0.0

    if "rel_strength_vs_spy_5d" not in features.columns:
        if "spy_return_5d" in features.columns:
            features["rel_strength_vs_spy_5d"] = features["return_5d"] - features["spy_return_5d"]
        else:
            features["rel_strength_vs_spy_5d"] = 0.0

    features["momentum_score"] = (
        features["return_5d"].fillna(0)
        + features["return_10d"].fillna(0)
        + features["rel_strength_vs_qqq_5d"].fillna(0)
        + features["macd_hist"].fillna(0)
    )

    features["trend_alignment"] = (
        (features["sma_5"] > features["sma_20"])
        & (features["sma_20"] > features["sma_50"])
    ).astype(int)

    features["trend_score"] = (
        features["sma_5_above_20"].fillna(0)
        + features["sma_50_above_200"].fillna(0)
        + (features["close_vs_sma_20"] > 0).astype(int)
        + (features["close_vs_sma_50"] > 0).astype(int)
    )

    features["breakout_score"] = (
        features["closed_above_prev_high"].fillna(0)
        + (features["relative_volume_20"] > 1.5).astype(int)
        + (features["close_location"] > 0.75).astype(int)
        + (features["return_1d"] > 0).astype(int)
    )

    features["risk_score"] = (
        features["atr_pct"].fillna(0)
        + features["volatility_20d"].fillna(0)
        + features["daily_range_pct"].fillna(0)
    )

    features["high_conviction_score"] = (
        features["momentum_score"].fillna(0)
        + features["trend_score"].fillna(0)
        + features["breakout_score"].fillna(0)
        - features["risk_score"].fillna(0)
    )

    # Simple regime classification expected by newer trade scripts.
    qqq_ret = features.get("qqq_return_5d", pd.Series(0.0, index=features.index)).fillna(0)
    spy_ret = features.get("spy_return_5d", pd.Series(0.0, index=features.index)).fillna(0)
    vixy_ret = features.get("vixy_return_5d", pd.Series(0.0, index=features.index)).fillna(0)
    qqq_vol = features.get("qqq_volatility_20d", pd.Series(0.0, index=features.index)).fillna(0)

    rolling_vol_q = qqq_vol.rolling(120, min_periods=20).quantile(0.75).fillna(qqq_vol.expanding().quantile(0.75))

    features["risk_on_score"] = (
        0.35 * (qqq_ret > 0).astype(int)
        + 0.35 * (spy_ret > 0).astype(int)
        + 0.20 * (vixy_ret < 0).astype(int)
        + 0.10 * (features.get("qqq_close_vs_sma_20", pd.Series(0.0, index=features.index)).fillna(0) > 0).astype(int)
    )

    features["market_volatility_high"] = (qqq_vol > rolling_vol_q).astype(int)

    features["market_regime"] = np.select(
        [
            (features["risk_on_score"] >= 0.70) & (features["market_volatility_high"] == 0),
            (features["risk_on_score"] >= 0.55) & (features["market_volatility_high"] == 1),
            (features["risk_on_score"] <= 0.35) | (vixy_ret > 0.03),
        ],
        [
            "TREND_RISK_ON",
            "VOLATILE_RISK_ON",
            "DEFENSIVE_RISK_OFF",
        ],
        default="CHOP_NEUTRAL",
    )

    route_map = {
        "TREND_RISK_ON": "BREAKOUT_TREND_ROUTE",
        "VOLATILE_RISK_ON": "CONFIRMED_MOMENTUM_ROUTE",
        "CHOP_NEUTRAL": "SELECTIVE_WATCH_ROUTE",
        "DEFENSIVE_RISK_OFF": "DEFENSIVE_OR_CASH_ROUTE",
    }

    features["route"] = features["market_regime"].map(route_map)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    return features


if __name__ == "__main__":
    print("[run] downloading dynamic S&P 500 + NASDAQ universe from Alpaca...")

    all_symbols = get_download_universe()

    raw = download_alpaca_daily_bars(
        symbols=all_symbols,
        lookback_days=520,
        chunk_size=50,
    )

    raw.to_csv(OUT_RAW, index=False)
    print(f"[saved] raw bars: {OUT_RAW}")

    features = build_features(
        raw,
        regime_symbols=REGIME_SYMBOLS,
        add_regime=True,
    )

    features.to_csv(OUT_FEATURES, index=False)
    print(f"[saved] daily_features: {OUT_FEATURES}")

    print("\n===== DAILY FEATURES HEAD =====")
    print(features.head(20).to_string(index=False))

    print("\n===== DAILY FEATURES TAIL =====")
    print(features.tail(20).to_string(index=False))

    print(f"\n[done] rows={len(features):,} cols={len(features.columns):,}")
