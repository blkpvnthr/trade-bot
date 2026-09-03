#!/usr/bin/env python3
import os
import math
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor

load_dotenv()

AV_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0:
        return max(0, S - K) if opt_type == "call" else max(0, K - S)

    sigma = max(float(sigma), 0.0001)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if opt_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def years_to_expiration(expiration):
    exp = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max((exp - now).total_seconds() / (365 * 24 * 60 * 60), 0)


def av_get(params):
    if not AV_KEY:
        raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY in .env")

    params["apikey"] = AV_KEY
    params["entitlement"] = "realtime"

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "Note" in data:
        raise RuntimeError(data["Note"])

    if "Information" in data:
        raise RuntimeError(data["Information"])

    if "Error Message" in data:
        raise RuntimeError(data["Error Message"])

    return data


def get_global_quote(symbol):
    data = av_get({
        "function": "GLOBAL_QUOTE",
        "symbol": symbol
    })

    q = data.get("Global Quote", {})

    if not q:
        raise RuntimeError(f"No quote returned for {symbol}")

    return {
        "price": float(q["05. price"]),
        "volume": float(q["06. volume"])
    }


def get_daily(symbol):
    data = av_get({
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full"
    })

    ts = data.get("Time Series (Daily)", {})

    if not ts:
        raise RuntimeError(f"No daily data returned for {symbol}")

    rows = []

    for date, bar in ts.items():
        rows.append({
            "date": date,
            "open": float(bar["1. open"]),
            "high": float(bar["2. high"]),
            "low": float(bar["3. low"]),
            "close": float(bar["4. close"]),
            "volume": float(bar["6. volume"])
        })

    rows.sort(key=lambda x: x["date"], reverse=True)
    return rows


def get_intraday(symbol, interval):
    data = av_get({
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "full"
    })

    key = f"Time Series ({interval})"
    ts = data.get(key, {})

    if not ts:
        raise RuntimeError(f"No {interval} intraday data returned for {symbol}")

    rows = []

    for dt, bar in ts.items():
        rows.append({
            "datetime": dt,
            "open": float(bar["1. open"]),
            "high": float(bar["2. high"]),
            "low": float(bar["3. low"]),
            "close": float(bar["4. close"]),
            "volume": float(bar["5. volume"])
        })

    rows.sort(key=lambda x: x["datetime"], reverse=True)
    return rows


def range_stats(rows, label):
    high = max(x["high"] for x in rows)
    low = min(x["low"] for x in rows)
    close = rows[0]["close"]
    midpoint = (high + low) / 2
    rng = high - low

    return {
        "label": label,
        "high": high,
        "low": low,
        "midpoint": midpoint,
        "range": rng,
        "close": close
    }


def calc_key_levels(symbol, current_price, daily_rows):
    candles_5m = get_intraday(symbol, "5min")
    candles_15m = get_intraday(symbol, "15min")
    candles_60m = get_intraday(symbol, "60min")

    levels = [
        range_stats(daily_rows[:252], "52W"),
        range_stats(daily_rows[:126], "6M"),
        range_stats([daily_rows[0]], "1D"),
        range_stats(candles_60m[:4], "4H"),
        range_stats(candles_15m[:1], "15m"),
        range_stats(candles_5m[:1], "5m")
    ]

    resistance_candidates = []
    support_candidates = []

    for level in levels:
        resistance_candidates.append((level["high"], f'{level["label"]} high'))
        resistance_candidates.append((level["midpoint"], f'{level["label"]} midpoint'))
        support_candidates.append((level["low"], f'{level["label"]} low'))
        support_candidates.append((level["midpoint"], f'{level["label"]} midpoint'))

    resistances = sorted(
        [(p, name) for p, name in resistance_candidates if p > current_price],
        key=lambda x: x[0]
    )

    supports = sorted(
        [(p, name) for p, name in support_candidates if p < current_price],
        key=lambda x: x[0],
        reverse=True
    )

    return levels, resistances, supports


def compute_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ml_price_prediction(daily_rows, horizon):
    df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)

    horizon_days = {
        "5d": 5,
        "10d": 10,
        "20d": 20
    }[horizon]

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_2"] = df["close"].pct_change(2)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["volume_ratio_5"] = df["volume"] / df["volume"].rolling(5).mean()
    df["volume_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()

    df["sma_5_dist"] = df["close"] / df["close"].rolling(5).mean() - 1
    df["sma_10_dist"] = df["close"] / df["close"].rolling(10).mean() - 1
    df["sma_20_dist"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["sma_50_dist"] = df["close"] / df["close"].rolling(50).mean() - 1
    df["sma_200_dist"] = df["close"] / df["close"].rolling(200).mean() - 1

    df["high_20_dist"] = df["close"] / df["high"].rolling(20).max() - 1
    df["high_50_dist"] = df["close"] / df["high"].rolling(50).max() - 1
    df["high_126_dist"] = df["close"] / df["high"].rolling(126).max() - 1
    df["high_252_dist"] = df["close"] / df["high"].rolling(252).max() - 1

    df["low_20_dist"] = df["close"] / df["low"].rolling(20).min() - 1
    df["low_50_dist"] = df["close"] / df["low"].rolling(50).min() - 1

    df["rsi"] = compute_rsi(df["close"])

    df["future_return"] = df["close"].shift(-horizon_days) / df["close"] - 1
    df = df.dropna().reset_index(drop=True)

    features = [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_20",
        "hl_range",
        "oc_change",
        "gap",
        "vol_5",
        "vol_10",
        "vol_20",
        "volume_ratio_5",
        "volume_ratio_20",
        "sma_5_dist",
        "sma_10_dist",
        "sma_20_dist",
        "sma_50_dist",
        "sma_200_dist",
        "high_20_dist",
        "high_50_dist",
        "high_126_dist",
        "high_252_dist",
        "low_20_dist",
        "low_50_dist",
        "rsi"
    ]

    X = df[features]
    y = df["future_return"]

    if len(X) < 250:
        raise RuntimeError("Not enough historical price data for ML prediction.")

    split = int(len(X) * 0.8)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    mae_return = float(np.mean(np.abs(test_pred - y_test)))
    mape = float(mae_return * 100)

    latest = X.iloc[[-1]]
    predicted_return = float(model.predict(latest)[0])

    current_price = float(df["close"].iloc[-1])
    predicted_price = current_price * (1 + predicted_return)

    daily_df = pd.DataFrame(daily_rows)

    lifetime_ath = float(daily_df["high"].max())
    high_52w = float(daily_df.head(252)["high"].max())
    high_6m = float(daily_df.head(126)["high"].max())

    return {
        "predicted_price": predicted_price,
        "predicted_return": predicted_return,
        "current_price": current_price,
        "lifetime_ath": lifetime_ath,
        "high_52w": high_52w,
        "high_6m": high_6m,
        "predicted_move_pct": predicted_return * 100,
        "lifetime_ath_breakout_pct": ((predicted_price / lifetime_ath) - 1) * 100,
        "high_52w_breakout_pct": ((predicted_price / high_52w) - 1) * 100,
        "high_6m_breakout_pct": ((predicted_price / high_6m) - 1) * 100,
        "above_lifetime_ath": predicted_price > lifetime_ath,
        "above_52w_high": predicted_price > high_52w,
        "above_6m_high": predicted_price > high_6m,
        "mae": current_price * mae_return,
        "mape": mape
    }


def choose_bullish_target(current_price, resistances, ml_result, levels):
    normal_price, normal_name = resistances[0] if resistances else (current_price * 1.03, "fallback +3% target")

    six_month_high = next(x["high"] for x in levels if x["label"] == "6M")
    fifty_two_week_high = next(x["high"] for x in levels if x["label"] == "52W")

    ml_price = ml_result["predicted_price"]
    ml_mape = ml_result["mape"]

    if ml_mape <= 6 and ml_price > ml_result["lifetime_ath"]:
        return ml_price, f'ML breakout target above lifetime ATH ${ml_result["lifetime_ath"]:.2f}'

    if ml_mape <= 8 and ml_price > fifty_two_week_high:
        return ml_price, f'ML breakout target above 52W high ${fifty_two_week_high:.2f}'

    if ml_mape <= 10 and ml_price > six_month_high:
        return ml_price, f'ML breakout target above 6M high ${six_month_high:.2f}'

    if ml_mape <= 12 and ml_price > normal_price:
        return ml_price, "ML near-term upside target"

    return normal_price, normal_name


def choose_bearish_target(current_price, supports):
    return supports[0] if supports else (current_price * 0.97, "fallback -3% target")


def liquidity_score(volume, open_interest, bid, ask):
    mid = (bid + ask) / 2
    spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 999
    score = 0

    if volume >= 1000:
        score += 30
    elif volume >= 500:
        score += 20
    elif volume >= 100:
        score += 10

    if open_interest >= 5000:
        score += 30
    elif open_interest >= 1000:
        score += 20
    elif open_interest >= 250:
        score += 10

    if spread_pct <= 5:
        score += 40
    elif spread_pct <= 10:
        score += 30
    elif spread_pct <= 20:
        score += 15

    return score, spread_pct


def timeframe_weight_from_name(name):
    if name.startswith("52W"):
        return 28
    if name.startswith("6M"):
        return 25
    if name.startswith("1D"):
        return 15
    if name.startswith("4H"):
        return 10
    if name.startswith("15m"):
        return 6
    if name.startswith("5m"):
        return 3
    if name.startswith("ML breakout target above lifetime ATH"):
        return 40
    if name.startswith("ML breakout target above 52W"):
        return 35
    if name.startswith("ML breakout target above 6M"):
        return 30
    if name.startswith("ML near-term"):
        return 18
    return 5


def compute_target_profit_pct(distance_pct, timeframe_weight, liq_score, iv, ml_result, opt_type):
    base = 15

    if distance_pct >= 15:
        base += 60
    elif distance_pct >= 10:
        base += 45
    elif distance_pct >= 7:
        base += 35
    elif distance_pct >= 5:
        base += 25
    elif distance_pct >= 3:
        base += 15
    elif distance_pct >= 1.5:
        base += 8

    base += timeframe_weight

    if liq_score >= 85:
        base += 10
    elif liq_score >= 65:
        base += 5
    elif liq_score < 40:
        base -= 10

    if iv >= 0.90:
        base -= 15
    elif iv >= 0.70:
        base -= 10
    elif iv >= 0.55:
        base -= 5
    elif iv <= 0.30:
        base += 5

    if opt_type == "call" and ml_result["above_lifetime_ath"]:
        base += 25
    elif opt_type == "call" and ml_result["above_52w_high"]:
        base += 18
    elif opt_type == "call" and ml_result["above_6m_high"]:
        base += 12

    if ml_result["mape"] > 20:
        base -= 25
    elif ml_result["mape"] > 12:
        base -= 18
    elif ml_result["mape"] > 8:
        base -= 10
    elif ml_result["mape"] < 4:
        base += 5

    return max(10, min(base, 200))


def ask_float(prompt, default=None):
    while True:
        v = input(prompt).strip()

        if v == "" and default is not None:
            return float(default)

        try:
            return float(v)
        except ValueError:
            print("Enter a valid number.")


def ask_int(prompt, default=None):
    while True:
        v = input(prompt).strip()

        if v == "" and default is not None:
            return int(default)

        try:
            return int(v)
        except ValueError:
            print("Enter a valid whole number.")


def ask_type():
    while True:
        v = input("Option type [call/put]: ").strip().lower()

        if v in ["call", "c"]:
            return "call"

        if v in ["put", "p"]:
            return "put"

        print("Enter call or put.")


def ask_horizon():
    while True:
        v = input("ML horizon [5d/10d/20d] [10d]: ").strip().lower()

        if v == "":
            return "10d"

        if v in ["5d", "10d", "20d"]:
            return v

        print("Enter 5d, 10d, or 20d.")


def analyze_once():
    print("\n=== AV + Historical ML Options Exit Target Calculator ===\n")

    symbol = input("Ticker: ").strip().upper()
    opt_type = ask_type()

    strike = ask_float("Strike price: ")
    expiration = input("Expiration YYYY-MM-DD: ").strip()
    contracts = ask_int("Contracts [1]: ", 1)

    entry = ask_float("Entry option price: ")
    iv = ask_float("Implied volatility decimal, example 0.55: ")
    r = ask_float("Risk-free rate decimal [0.045]: ", 0.045)

    option_volume = ask_float("Option volume: ")
    open_interest = ask_float("Open interest: ")
    bid = ask_float("Bid: ")
    ask = ask_float("Ask: ")

    ml_horizon = ask_horizon()

    print("\nPulling Alpha Vantage quote/history...")
    quote = get_global_quote(symbol)
    underlying_now = quote["price"]

    daily_rows = get_daily(symbol)

    print("Calculating 52W / 6M / 1D / 4H / 15m / 5m key levels...")
    levels, resistances, supports = calc_key_levels(symbol, underlying_now, daily_rows)

    print("Training historical-price ML return model...")
    ml_result = ml_price_prediction(daily_rows, ml_horizon)

    if opt_type == "call":
        key_price, key_name = choose_bullish_target(underlying_now, resistances, ml_result, levels)
        distance_pct = ((key_price / underlying_now) - 1) * 100
    else:
        key_price, key_name = choose_bearish_target(underlying_now, supports)
        distance_pct = ((underlying_now / key_price) - 1) * 100

    T = years_to_expiration(expiration)

    liq_score, spread_pct = liquidity_score(option_volume, open_interest, bid, ask)
    weight = timeframe_weight_from_name(key_name)

    target_profit_pct = compute_target_profit_pct(
        distance_pct=distance_pct,
        timeframe_weight=weight,
        liq_score=liq_score,
        iv=iv,
        ml_result=ml_result,
        opt_type=opt_type
    )

    target_option_sell_price = entry * (1 + target_profit_pct / 100)
    target_total_pl = (target_option_sell_price - entry) * 100 * contracts

    current_bs_value = bs_price(underlying_now, strike, T, r, iv, opt_type)
    current_bs_pl = (current_bs_value - entry) * 100 * contracts
    current_bs_return = ((current_bs_value / entry) - 1) * 100

    bs_value_at_key_price = bs_price(key_price, strike, T, r, iv, opt_type)
    bs_profit_pct_at_key = ((bs_value_at_key_price / entry) - 1) * 100
    bs_total_pl_at_key = (bs_value_at_key_price - entry) * 100 * contracts

    print("\n========================================")
    print("KEY PRICE RANGES FROM ALPHA VANTAGE")
    print("========================================")

    for level in levels:
        print(f'{level["label"]}: Low ${level["low"]:.2f} | Mid ${level["midpoint"]:.2f} | High ${level["high"]:.2f} | Range ${level["range"]:.2f}')

    print("\n========================================")
    print("HISTORICAL PRICE ML PREDICTION")
    print("========================================")
    print(f"ML horizon: {ml_horizon}")
    print(f"Lifetime ATH: ${ml_result['lifetime_ath']:.2f}")
    print(f"52W high: ${ml_result['high_52w']:.2f}")
    print(f"6M high: ${ml_result['high_6m']:.2f}")
    print(f"ML predicted return: {ml_result['predicted_return'] * 100:.2f}%")
    print(f"ML predicted price: ${ml_result['predicted_price']:.2f}")
    print(f"Predicted extension beyond lifetime ATH: {ml_result['lifetime_ath_breakout_pct']:.2f}%")
    print(f"Predicted extension beyond 52W high: {ml_result['high_52w_breakout_pct']:.2f}%")
    print(f"Predicted extension beyond 6M high: {ml_result['high_6m_breakout_pct']:.2f}%")
    print(f"Above lifetime ATH: {ml_result['above_lifetime_ath']}")
    print(f"Above 52W high: {ml_result['above_52w_high']}")
    print(f"Above 6M high: {ml_result['above_6m_high']}")
    print(f"Historical model MAE: ${ml_result['mae']:.2f}")
    print(f"Historical model error: {ml_result['mape']:.2f}%")

    print("\n========================================")
    print("RESULT")
    print("========================================")
    print(f"Contract: {symbol} {expiration} {strike:g} {opt_type.upper()}")
    print(f"Contracts: {contracts}")
    print("")
    print(f"Underlying now: ${underlying_now:.2f}")
    print(f"Selected key price: ${key_price:.2f}")
    print(f"Selected key source: {key_name}")
    print(f"Distance to selected key price: {distance_pct:.2f}%")
    print("")
    print(f"Option volume: {option_volume:,.0f}")
    print(f"Open interest: {open_interest:,.0f}")
    print(f"Bid/ask spread: {spread_pct:.2f}%")
    print(f"Liquidity score: {liq_score}/100")
    print(f"IV: {iv:.2%}")
    print("")
    print(f"Entry option price: ${entry:.2f}")
    print(f"Current Black-Scholes value: ${current_bs_value:.2f}")
    print(f"Current theoretical P/L: ${current_bs_pl:.2f}")
    print(f"Current theoretical return: {current_bs_return:.2f}%")
    print("")
    print("AUTO-CALCULATED EXIT TARGET")
    print("----------------------------------------")
    print(f"Target profit: {target_profit_pct:.2f}%")
    print(f"Target option sell price: ${target_option_sell_price:.2f}")
    print(f"Target total P/L: ${target_total_pl:.2f}")
    print("")
    print("BLACK-SCHOLES VALUE AT SELECTED KEY PRICE")
    print("----------------------------------------")
    print(f"Estimated option value at key price: ${bs_value_at_key_price:.2f}")
    print(f"Estimated profit % at key price: {bs_profit_pct_at_key:.2f}%")
    print(f"Estimated total P/L at key price: ${bs_total_pl_at_key:.2f}")

    print("")
    print("INTERPRETATION")
    print("----------------------------------------")

    if liq_score < 40:
        print("Liquidity is weak. Expect poor fills and avoid aggressive target pricing.")

    if spread_pct > 20:
        print("Bid/ask spread is wide. Use limit orders only.")

    if ml_result["mape"] > 20:
        print("ML confidence is poor. Historical price behavior has not predicted this ticker cleanly.")

    if opt_type == "call" and ml_result["above_lifetime_ath"]:
        print("Historical ML supports a speculative lifetime ATH breakout target.")
    elif opt_type == "call" and ml_result["above_52w_high"]:
        print("Historical ML supports a speculative 52W breakout target.")
    elif opt_type == "call" and ml_result["above_6m_high"]:
        print("Historical ML supports a speculative 6M breakout target.")
    elif opt_type == "call":
        print("Historical ML does not justify a breakout extension. Range-derived resistance was used.")

    print("========================================\n")


def main():
    while True:
        try:
            analyze_once()
        except Exception as e:
            print(f"\nERROR: {e}\n")

        again = input("Analyze another contract? [y/n]: ").strip().lower()

        if again not in ["y", "yes"]:
            print("Done.")
            break


if __name__ == "__main__":
    main()