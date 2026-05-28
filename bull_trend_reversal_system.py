#!/usr/bin/env python3
"""
Bull Trend Reversal + Recession Month Prediction Engine

This script combines:
- market trend/reversal signals from ETFs and volatility
- credit/rates/sector leadership signals
- macro-cycle recession timing signals from FRED, when available

Outputs:
- daily_system_outputs/bull_trend_reversal_context.json
- system-data/YYYYMMDD/YYYYMMDD_bull_trend_reversal_context.json

Required:
    pip install yfinance pandas numpy

Optional, for FRED macro data:
    pip install fredapi

Environment:
    export FRED_API_KEY="YOUR_FRED_API_KEY"
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


LOOKBACK_DAYS = 420
MACRO_LOOKBACK_YEARS = 10

TICKERS = [
    "SPY", "QQQ", "IWM",
    "^VIX",
    "TLT", "HYG", "LQD",
    "XLK", "SMH", "XLF", "XLY", "XLI",
    "XLU", "XLP", "XLV",
    "KRE", "KBE",
    "UUP", "GLD", "USO",
]

MACRO_SERIES = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "DGS10": "10Y Treasury Yield",
    "PAYEMS": "Nonfarm Payrolls",
    "ICSA": "Initial Jobless Claims",
    "UMCSENT": "Consumer Sentiment",
    "T10Y3M": "10Y-3M Treasury Spread",
    "BAMLH0A0HYM2": "High Yield Credit Spread",
    "BAMLC0A0CM": "Investment Grade Credit Spread",
    "MANEMP": "Manufacturing Employment",
    "WALCL": "Fed Balance Sheet",
    "INDPRO": "Industrial Production",
    "PERMIT": "Building Permits",
}

OPTIONAL_MACRO_SERIES: Dict[str, str] = {}


@dataclass
class ReversalContext:
    timestamp: str
    regime_state: str
    reversal_risk_score: float
    recession_risk_score: float
    recession_risk_state: str
    predicted_recession_month: str
    months_until_predicted_recession: int
    recession_decision_reasons: List[str]
    confidence: float
    macro_confidence: float
    trade_permission: str
    allow_new_longs: bool
    allow_inverse_etfs: bool
    allow_hedges: bool
    max_long_exposure_multiplier: float
    recommended_action: str
    decision_reasons: List[str]
    metrics: Dict[str, float]


def warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def get_fred_client():
    """
    Creates a fredapi client only when fredapi and FRED_API_KEY are available.
    The script remains usable without FRED data.
    """
    api_key = os.getenv("FRED_API_KEY", "").strip()

    if not api_key:
        warn("FRED_API_KEY is not set. Macro recession indicators will be skipped.")
        return None

    try:
        from fredapi import Fred
    except Exception:
        warn("fredapi is not installed. Install with: pip install fredapi")
        return None

    try:
        return Fred(api_key=api_key)
    except Exception as exc:
        warn(f"Could not initialize FRED client: {exc}")
        return None


def download_prices(tickers: List[str]) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        period=f"{LOOKBACK_DAYS}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    closes: Dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.get_level_values(0):
                    closes[ticker] = data[ticker]["Close"]
            else:
                closes[ticker] = data["Close"]
        except Exception:
            continue

    df = pd.DataFrame(closes).dropna(how="all")
    return df.ffill().dropna(how="all")


def fetch_fred_series(fred, series_id: str, start: datetime, end: datetime) -> pd.Series:
    if fred is None:
        return pd.Series(dtype=float)

    try:
        data = fred.get_series(series_id, observation_start=start, observation_end=end)
        s = pd.Series(data).dropna()
        s.index = pd.to_datetime(s.index)
        s.name = series_id
        return s
    except Exception as exc:
        print(f"FRED error for {series_id}: {exc}")
        return pd.Series(dtype=float)


def download_macro_data() -> pd.DataFrame:
    """
    Downloads macro data from FRED using fredapi.

    If FRED_API_KEY or fredapi is missing, this returns an empty DataFrame
    and the script still calculates market-based reversal/recession risk.
    """
    fred = get_fred_client()
    if fred is None:
        return pd.DataFrame()

    end = datetime.now()
    start = end - pd.DateOffset(years=MACRO_LOOKBACK_YEARS)

    macro = pd.DataFrame()
    series_codes = list(MACRO_SERIES.keys()) + list(OPTIONAL_MACRO_SERIES.keys())

    for code in series_codes:
        s = fetch_fred_series(fred, code, start, end)
        if not s.empty:
            macro[code] = s

    if macro.empty:
        warn("No macro data was downloaded from FRED.")
        return macro

    return macro.sort_index().ffill()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def pct_change(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return 0.0
    value = clean.pct_change(periods).iloc[-1]
    return float(value) if pd.notna(value) else 0.0


def abs_change(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return 0.0
    value = clean.diff(periods).iloc[-1]
    return float(value) if pd.notna(value) else 0.0


def slope(series: pd.Series, window: int = 20) -> float:
    recent = series.dropna().tail(window)

    if len(recent) < window:
        return 0.0

    x = np.arange(len(recent))
    y = recent.values

    try:
        denom = recent.iloc[-1]
        if denom == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0] / denom)
    except Exception:
        return 0.0


def relative_strength(asset: pd.Series, benchmark: pd.Series, window: int = 20) -> float:
    joined = pd.concat([asset, benchmark], axis=1).dropna()
    if joined.shape[0] <= window:
        return 0.0
    ratio = joined.iloc[:, 0] / joined.iloc[:, 1]
    return pct_change(ratio, window)


def safe_last(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def add_score(score: float, points: float, reasons: List[str], reason: str) -> float:
    reasons.append(f"+{points:.0f}: {reason}")
    return score + points


def add_macro_metrics(metrics: Dict[str, float], macro: pd.DataFrame) -> Tuple[Dict[str, float], int]:
    macro_signals = 0

    if macro.empty:
        metrics["macro_data_available"] = 0.0
        return metrics, macro_signals

    metrics["macro_data_available"] = 1.0

    if "UNRATE" in macro:
        unrate = macro["UNRATE"].dropna()
        if len(unrate) >= 12:
            unrate_3m_avg = float(unrate.tail(3).mean())
            unrate_12m_low = float(unrate.tail(12).min())
            unrate_6m_change = abs_change(unrate, 6)
            sahm_gap = unrate_3m_avg - unrate_12m_low

            metrics["unemployment_rate"] = float(unrate.iloc[-1])
            metrics["unemployment_3m_avg"] = unrate_3m_avg
            metrics["unemployment_12m_low"] = unrate_12m_low
            metrics["sahm_rule_gap"] = sahm_gap
            metrics["unemployment_6m_change"] = unrate_6m_change
            macro_signals += 2

    if "T10Y2Y" in macro:
        spread = macro["T10Y2Y"].dropna()
        if len(spread) >= 24:
            latest_spread = float(spread.iloc[-1])
            prior_curve_inversion = bool((spread.tail(24) < 0).any())
            min_spread_24m = float(spread.tail(24).min())
            spread_6m_change = abs_change(spread, 6)

            metrics["ten_two_spread"] = latest_spread
            metrics["ten_two_prior_curve_inversion"] = 1.0 if prior_curve_inversion else 0.0
            metrics["ten_two_min_spread_24m"] = min_spread_24m
            metrics["ten_two_spread_6m_change"] = spread_6m_change
            macro_signals += 2

    if "T10Y3M" in macro:
        spread = macro["T10Y3M"].dropna()
        if len(spread) >= 24:
            latest_spread = float(spread.iloc[-1])
            prior_curve_inversion = bool((spread.tail(24) < 0).any())

            metrics["ten_three_month_spread"] = latest_spread
            metrics["ten_three_month_prior_inversion"] = 1.0 if prior_curve_inversion else 0.0
            macro_signals += 1

    if "FEDFUNDS" in macro:
        fed = macro["FEDFUNDS"].dropna()
        if len(fed) >= 12:
            metrics["fed_funds_rate"] = float(fed.iloc[-1])
            metrics["fed_funds_6m_change"] = abs_change(fed, 6)
            metrics["fed_funds_12m_change"] = abs_change(fed, 12)
            macro_signals += 1

    if "DGS10" in macro:
        dgs10 = macro["DGS10"].dropna()
        if len(dgs10) >= 63:
            metrics["ten_year_yield"] = float(dgs10.iloc[-1])
            metrics["ten_year_yield_3m_change"] = abs_change(dgs10, 63)
            macro_signals += 1

    if "PAYEMS" in macro:
        payrolls = macro["PAYEMS"].dropna()
        if len(payrolls) >= 12:
            payroll_3m_avg_change = float(payrolls.diff().tail(3).mean())
            payroll_6m_change = pct_change(payrolls, 6)

            metrics["payrolls_3m_avg_change"] = payroll_3m_avg_change
            metrics["payrolls_6m_change"] = payroll_6m_change
            macro_signals += 1

    if "ICSA" in macro:
        claims = macro["ICSA"].dropna()
        if len(claims) >= 52:
            claims_4w = float(claims.tail(4).mean())
            claims_52w_low = float(claims.tail(52).min())
            claims_gap = (claims_4w / claims_52w_low) - 1 if claims_52w_low else 0.0

            metrics["initial_claims_4w_avg"] = claims_4w
            metrics["initial_claims_52w_low"] = claims_52w_low
            metrics["initial_claims_gap_from_52w_low"] = claims_gap
            macro_signals += 1

    if "UMCSENT" in macro:
        sentiment = macro["UMCSENT"].dropna()
        if len(sentiment) >= 12:
            metrics["consumer_sentiment"] = float(sentiment.iloc[-1])
            metrics["consumer_sentiment_6m_change"] = abs_change(sentiment, 6)
            macro_signals += 1

    if "BAMLH0A0HYM2" in macro:
        hy_spread = macro["BAMLH0A0HYM2"].dropna()
        if len(hy_spread) >= 126:
            metrics["hy_credit_spread"] = float(hy_spread.iloc[-1])
            metrics["hy_credit_spread_3m_change"] = abs_change(hy_spread, 63)
            metrics["hy_credit_spread_6m_change"] = abs_change(hy_spread, 126)
            macro_signals += 1

    if "BAMLC0A0CM" in macro:
        ig_spread = macro["BAMLC0A0CM"].dropna()
        if len(ig_spread) >= 126:
            metrics["ig_credit_spread"] = float(ig_spread.iloc[-1])
            metrics["ig_credit_spread_3m_change"] = abs_change(ig_spread, 63)
            macro_signals += 1

    if "WALCL" in macro:
        walcl = macro["WALCL"].dropna()
        if len(walcl) >= 52:
            metrics["fed_balance_sheet_13w_change"] = pct_change(walcl, 13)
            metrics["fed_balance_sheet_26w_change"] = pct_change(walcl, 26)
            macro_signals += 1

    if "INDPRO" in macro:
        indpro = macro["INDPRO"].dropna()
        if len(indpro) >= 6:
            metrics["industrial_production_6m_change"] = pct_change(indpro, 6)
            macro_signals += 1

    if "PERMIT" in macro:
        permits = macro["PERMIT"].dropna()
        if len(permits) >= 12:
            metrics["building_permits_12m_change"] = pct_change(permits, 12)
            macro_signals += 1

    return metrics, macro_signals


def predict_recession_month(metrics: Dict[str, float], reversal_score: float) -> Dict[str, object]:
    today = datetime.now()
    risk = 0.0
    reasons: List[str] = []

    if reversal_score >= 80:
        risk = add_score(risk, 30, reasons, "market reversal risk is confirmed")
    elif reversal_score >= 65:
        risk = add_score(risk, 22, reasons, "market reversal warning is active")
    elif reversal_score >= 45:
        risk = add_score(risk, 12, reasons, "bull trend is weakening")

    sahm_gap = metrics.get("sahm_rule_gap", 0.0)

    if sahm_gap >= 0.50:
        risk = add_score(
            risk,
            22,
            reasons,
            "Sahm-rule unemployment trigger is active"
        )

    elif sahm_gap >= 0.40:
        risk = add_score(
            risk,
            16,
            reasons,
            "unemployment deterioration is accelerating"
        )

    elif sahm_gap >= 0.30:
        risk = add_score(
            risk,
            10,
            reasons,
            "unemployment is rising toward Sahm-rule threshold"
        )

    elif sahm_gap >= 0.20:
        risk = add_score(
            risk,
            5,
            reasons,
            "unemployment is rising modestly from cycle lows"
        )

    curve_confirmation = (
        metrics.get("ten_two_spread", -999) > 0
        and metrics.get("ten_two_prior_curve_inversion", 0) == 1.0
    )

    if curve_confirmation and sahm_gap >= 0.35:
        risk = add_score(
            risk,
            10,
            reasons,
            "yield-curve re-steepening confirmed by unemployment deterioration"
        )
    if metrics.get("ten_two_spread", -999) > 0 and metrics.get("ten_two_prior_curve_inversion", 0) == 1.0:
        risk = add_score(risk, 12, reasons, "10Y-2Y curve has re-steepened after prior inversion")
    elif metrics.get("ten_two_spread", 999) < 0:
        risk = add_score(risk, 6, reasons, "10Y-2Y curve is inverted")

    if metrics.get("ten_three_month_spread", -999) > 0 and metrics.get("ten_three_month_prior_inversion", 0) == 1.0:
        risk = add_score(risk, 8, reasons, "10Y-3M curve has re-steepened after prior inversion")
    elif metrics.get("ten_three_month_spread", 999) < 0:
        risk = add_score(risk, 4, reasons, "10Y-3M curve is inverted")

    if metrics.get("hy_credit_spread", 0) > 5.0:
        risk = add_score(risk, 20, reasons, "high-yield credit spread is above stress threshold")
    elif metrics.get("hy_credit_spread_3m_change", 0) > 0.75:
        risk = add_score(risk, 12, reasons, "high-yield credit spreads are widening quickly")

    if metrics.get("ig_credit_spread_3m_change", 0) > 0.25:
        risk = add_score(risk, 8, reasons, "investment-grade credit spreads are widening")

    if metrics.get("hyg_vs_spy_20d", 0) < -0.025:
        risk = add_score(risk, 10, reasons, "HYG is underperforming SPY")

    if metrics.get("lqd_vs_spy_20d", 0) < -0.03:
        risk = add_score(risk, 6, reasons, "LQD is underperforming SPY")

    if metrics.get("kre_vs_spy_60d", 0) < -0.10:
        risk = add_score(risk, 15, reasons, "regional banks are underperforming SPY by more than 10% over 60 days")
    elif metrics.get("kre_vs_spy_20d", 0) < -0.05:
        risk = add_score(risk, 8, reasons, "regional banks are weakening versus SPY")

    if metrics.get("payrolls_3m_avg_change", 1) < 50:
        risk = add_score(risk, 12, reasons, "payroll momentum is weakening")
    elif metrics.get("payrolls_6m_change", 1) < 0:
        risk = add_score(risk, 10, reasons, "payrolls are contracting over 6 months")

    if metrics.get("initial_claims_gap_from_52w_low", 0) > 0.25:
        risk = add_score(risk, 12, reasons, "initial jobless claims are rising from 52-week lows")

    if metrics.get("ism_manufacturing_pmi_3m_avg", 100) < 48:
        risk = add_score(risk, 12, reasons, "ISM manufacturing is in contraction")
    elif metrics.get("ism_manufacturing_pmi_3m_avg", 100) < 50:
        risk = add_score(risk, 8, reasons, "ISM manufacturing is below expansion threshold")

    if metrics.get("fed_funds_12m_change", 0) > 1.0:
        risk = add_score(risk, 8, reasons, "Fed policy has tightened over the last 12 months")

    if metrics.get("fed_balance_sheet_26w_change", 0) < -0.04:
        risk = add_score(risk, 8, reasons, "Fed balance sheet liquidity is contracting")

    if metrics.get("building_permits_12m_change", 0) < -0.10:
        risk = add_score(risk, 8, reasons, "building permits are contracting")

    if metrics.get("consumer_sentiment_6m_change", 0) < -10:
        risk = add_score(risk, 6, reasons, "consumer sentiment is deteriorating")

    if metrics.get("vix", 0) > 25:
        risk = add_score(risk, 15, reasons, "VIX is above 25")
    elif metrics.get("vix", 0) > 20:
        risk = add_score(risk, 8, reasons, "VIX is above 20")

    if metrics.get("defensive_avg_rs_20d", 0) > metrics.get("growth_avg_rs_20d", 0) + 0.03:
        risk = add_score(risk, 8, reasons, "defensive sectors are outperforming growth sectors")

    if metrics.get("iwm_vs_spy_20d", 0) < -0.04:
        risk = add_score(risk, 8, reasons, "small caps are underperforming SPY")

    if metrics.get("spy_60d_return", 0) < -0.08:
        risk = add_score(risk, 8, reasons, "SPY 60-day return is worse than -8%")

    risk = min(100.0, max(0.0, risk))

    if risk >= 90:
        months_ahead = 1
        state = "IMMINENT_RECESSION_RISK"
    elif risk >= 80:
        months_ahead = 3
        state = "HIGH_RECESSION_RISK"
    elif risk >= 65:
        months_ahead = 6
        state = "MODERATE_RECESSION_RISK"
    elif risk >= 45:
        months_ahead = 9
        state = "EARLY_WARNING"
    else:
        months_ahead = 12
        state = "LOW_RECESSION_RISK"

    predicted_date = today + pd.DateOffset(months=months_ahead)

    if not reasons:
        reasons.append("No major recession timing warnings detected.")

    return {
        "recession_risk_score": round(float(risk), 2),
        "recession_risk_state": state,
        "predicted_recession_month": predicted_date.strftime("%B %Y"),
        "months_until_predicted_recession": int(months_ahead),
        "recession_decision_reasons": reasons,
    }


def calculate_reversal_context(prices: pd.DataFrame, macro: Optional[pd.DataFrame] = None) -> ReversalContext:
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    score = 0.0
    available_signals = 0
    max_signals = 21

    spy = prices.get("SPY")
    qqq = prices.get("QQQ")
    iwm = prices.get("IWM")
    vix = prices.get("^VIX")
    hyg = prices.get("HYG")
    lqd = prices.get("LQD")
    tlt = prices.get("TLT")
    kre = prices.get("KRE")
    kbe = prices.get("KBE")

    if spy is None or spy.dropna().shape[0] < 200:
        raise RuntimeError("Not enough SPY history to calculate reversal risk.")

    spy_close = safe_last(spy)
    spy_sma20 = safe_last(sma(spy, 20))
    spy_sma50 = safe_last(sma(spy, 50))
    spy_sma200 = safe_last(sma(spy, 200))

    metrics["spy_close"] = spy_close or 0
    metrics["spy_sma20"] = spy_sma20 or 0
    metrics["spy_sma50"] = spy_sma50 or 0
    metrics["spy_sma200"] = spy_sma200 or 0
    metrics["spy_50sma_slope"] = slope(sma(spy, 50), 20)
    metrics["spy_200sma_slope"] = slope(sma(spy, 200), 20)

    if spy_close and spy_sma20 and spy_close < spy_sma20:
        score = add_score(score, 8, reasons, "SPY closed below 20SMA")
    available_signals += 1

    if spy_close and spy_sma50 and spy_close < spy_sma50:
        score = add_score(score, 14, reasons, "SPY closed below 50SMA")
    available_signals += 1

    if spy_sma50 and spy_sma200 and spy_sma50 < spy_sma200:
        score = add_score(score, 18, reasons, "SPY 50SMA is below 200SMA")
    available_signals += 1

    if metrics["spy_50sma_slope"] < 0:
        score = add_score(score, 8, reasons, "SPY 50SMA slope is negative")
    available_signals += 1

    if metrics["spy_200sma_slope"] < 0:
        score = add_score(score, 10, reasons, "SPY 200SMA slope is negative")
    available_signals += 1

    spy_20d_return = pct_change(spy, 20)
    spy_60d_return = pct_change(spy, 60)

    metrics["spy_20d_return"] = spy_20d_return
    metrics["spy_60d_return"] = spy_60d_return

    if spy_20d_return < -0.04:
        score = add_score(score, 8, reasons, "SPY 20-day return is worse than -4%")
    available_signals += 1

    if spy_60d_return < -0.08:
        score = add_score(score, 10, reasons, "SPY 60-day return is worse than -8%")
    available_signals += 1

    if qqq is not None:
        qqq_rs = relative_strength(qqq, spy, 20)
        metrics["qqq_vs_spy_20d"] = qqq_rs

        if qqq_rs < -0.03:
            score = add_score(score, 8, reasons, "QQQ is underperforming SPY over 20 days")

        available_signals += 1

    if iwm is not None:
        iwm_rs = relative_strength(iwm, spy, 20)
        metrics["iwm_vs_spy_20d"] = iwm_rs

        if iwm_rs < -0.04:
            score = add_score(score, 7, reasons, "IWM is underperforming SPY, showing weak risk appetite")

        available_signals += 1

    if vix is not None:
        vix_last = safe_last(vix)
        vix_5d = pct_change(vix, 5)
        vix_20d = pct_change(vix, 20)

        metrics["vix"] = vix_last or 0
        metrics["vix_5d_change"] = vix_5d
        metrics["vix_20d_change"] = vix_20d

        if vix_last and vix_last > 20:
            score = add_score(score, 10, reasons, "VIX is above 20")

        available_signals += 1

        if vix_5d > 0.20:
            score = add_score(score, 8, reasons, "VIX is up more than 20% over 5 days")

        available_signals += 1

        if vix_20d > 0.35:
            score = add_score(score, 8, reasons, "VIX is expanding sharply over 20 days")

        available_signals += 1

    if hyg is not None:
        hyg_sma50 = safe_last(sma(hyg, 50))
        hyg_last = safe_last(hyg)
        hyg_rs = relative_strength(hyg, spy, 20)

        metrics["hyg_vs_spy_20d"] = hyg_rs

        if hyg_last and hyg_sma50 and hyg_last < hyg_sma50:
            score = add_score(score, 8, reasons, "HYG is below 50SMA, showing credit weakness")

        available_signals += 1

        if hyg_rs < -0.025:
            score = add_score(score, 6, reasons, "HYG is underperforming SPY")

        available_signals += 1

    if lqd is not None:
        lqd_rs = relative_strength(lqd, spy, 20)
        metrics["lqd_vs_spy_20d"] = lqd_rs

        if lqd_rs < -0.03:
            score = add_score(score, 5, reasons, "LQD is underperforming SPY")

        available_signals += 1

    if tlt is not None:
        tlt_20d = pct_change(tlt, 20)
        metrics["tlt_20d_return"] = tlt_20d

        if tlt_20d < -0.05:
            score = add_score(score, 6, reasons, "TLT is falling sharply, suggesting rate pressure")

        available_signals += 1

    if kre is not None:
        kre_rs_20d = relative_strength(kre, spy, 20)
        kre_rs_60d = relative_strength(kre, spy, 60)

        metrics["kre_vs_spy_20d"] = kre_rs_20d
        metrics["kre_vs_spy_60d"] = kre_rs_60d

        if kre_rs_60d < -0.10:
            score = add_score(score, 10, reasons, "KRE is underperforming SPY by more than 10% over 60 days")
        elif kre_rs_20d < -0.05:
            score = add_score(score, 6, reasons, "KRE is weakening versus SPY")

        available_signals += 1

    if kbe is not None:
        kbe_rs_60d = relative_strength(kbe, spy, 60)
        metrics["kbe_vs_spy_60d"] = kbe_rs_60d

        if kbe_rs_60d < -0.10:
            score = add_score(score, 8, reasons, "KBE bank ETF is underperforming SPY by more than 10% over 60 days")

        available_signals += 1

    defensive = ["XLU", "XLP", "XLV"]
    growth = ["XLK", "SMH", "XLY"]

    defensive_rs_values = []
    growth_rs_values = []

    for ticker in defensive:
        if ticker in prices:
            defensive_rs_values.append(relative_strength(prices[ticker], spy, 20))

    for ticker in growth:
        if ticker in prices:
            growth_rs_values.append(relative_strength(prices[ticker], spy, 20))

    if defensive_rs_values and growth_rs_values:
        defensive_avg = float(np.mean(defensive_rs_values))
        growth_avg = float(np.mean(growth_rs_values))

        metrics["defensive_avg_rs_20d"] = defensive_avg
        metrics["growth_avg_rs_20d"] = growth_avg

        if defensive_avg > growth_avg + 0.03:
            score = add_score(score, 8, reasons, "Defensive sectors are outperforming growth sectors")

        available_signals += 1

    uup = prices.get("UUP")

    if uup is not None:
        uup_20d = pct_change(uup, 20)
        metrics["uup_20d_return"] = uup_20d

        if uup_20d > 0.035:
            score = add_score(score, 4, reasons, "Dollar strength is pressuring risk assets")

        available_signals += 1

    macro = macro if macro is not None else pd.DataFrame()
    metrics, macro_signal_count = add_macro_metrics(metrics, macro)

    normalized_score = min(100.0, max(0.0, score))
    confidence = min(1.0, available_signals / max_signals)
    macro_confidence = min(1.0, macro_signal_count / 15)

    recession_forecast = predict_recession_month(metrics, normalized_score)

    metrics["recession_risk_score"] = recession_forecast["recession_risk_score"]
    metrics["months_until_predicted_recession"] = float(recession_forecast["months_until_predicted_recession"])
    metrics["macro_signal_count"] = float(macro_signal_count)

    if normalized_score >= 80:
        regime_state = "REVERSAL_CONFIRMED"
        trade_permission = "DEFENSIVE_ONLY"
        allow_new_longs = False
        allow_inverse_etfs = True
        allow_hedges = True
        max_long_exposure_multiplier = 0.0
        recommended_action = "Reversal confirmed. Defensive positions only."
    elif normalized_score >= 65:
        regime_state = "REVERSAL_WARNING"
        trade_permission = "HEDGE_OR_REDUCED_RISK_ONLY"
        allow_new_longs = False
        allow_inverse_etfs = True
        allow_hedges = True
        max_long_exposure_multiplier = 0.25
        recommended_action = "Stop weak longs, reduce exposure, allow hedges."
    elif normalized_score >= 45:
        regime_state = "BULL_WEAKENING"
        trade_permission = "SELECTIVE_LONGS_ONLY"
        allow_new_longs = True
        allow_inverse_etfs = False
        allow_hedges = True
        max_long_exposure_multiplier = 0.50
        recommended_action = "Only take high-quality momentum longs with reduced size."
    else:
        regime_state = "BULL_CONFIRMED"
        trade_permission = "NORMAL_LONGS_ALLOWED"
        allow_new_longs = True
        allow_inverse_etfs = False
        allow_hedges = False
        max_long_exposure_multiplier = 1.00
        recommended_action = "Bull trend intact. Normal long execution allowed."

    recession_score = float(recession_forecast.get("recession_risk_score", 0.0))

    if normalized_score < 45 and recession_score >= 80:
        trade_permission = "MACRO_RISK_REDUCED_LONGS_ONLY"
        allow_new_longs = True
        allow_hedges = True
        max_long_exposure_multiplier = min(max_long_exposure_multiplier, 0.70)
        recommended_action = (
            "Bull trend remains intact, but macro recession risk is high. "
            "Allow only high-quality longs, reduce position sizing, avoid over-leverage, "
            "and monitor credit/breadth deterioration."
        )
    elif normalized_score < 45 and recession_score >= 65:
        trade_permission = "MACRO_RISK_SELECTIVE_LONGS"
        allow_new_longs = True
        allow_hedges = True
        max_long_exposure_multiplier = min(max_long_exposure_multiplier, 0.80)
        recommended_action = (
            "Bull trend remains intact, but macro cycle risk is elevated. "
            "Keep longs selective, reduce marginal position size, and watch credit, labor, and breadth confirmation."
        )

    if not reasons:
        reasons.append("No major market reversal warnings detected.")

    clean_metrics = {
        k: round(float(v), 6)
        for k, v in metrics.items()
        if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v)
    }

    return ReversalContext(
        timestamp=datetime.now().isoformat(),
        regime_state=regime_state,
        reversal_risk_score=round(normalized_score, 2),
        recession_risk_score=recession_forecast["recession_risk_score"],
        recession_risk_state=recession_forecast["recession_risk_state"],
        predicted_recession_month=recession_forecast["predicted_recession_month"],
        months_until_predicted_recession=recession_forecast["months_until_predicted_recession"],
        recession_decision_reasons=recession_forecast["recession_decision_reasons"],
        confidence=round(confidence, 2),
        macro_confidence=round(macro_confidence, 2),
        trade_permission=trade_permission,
        allow_new_longs=allow_new_longs,
        allow_inverse_etfs=allow_inverse_etfs,
        allow_hedges=allow_hedges,
        max_long_exposure_multiplier=max_long_exposure_multiplier,
        recommended_action=recommended_action,
        decision_reasons=reasons,
        metrics=clean_metrics,
    )


def write_outputs(context: ReversalContext) -> None:
    today = datetime.now().strftime("%Y%m%d")

    daily_dir = Path("daily_system_outputs")
    dated_dir = Path("system-data") / today

    daily_dir.mkdir(parents=True, exist_ok=True)
    dated_dir.mkdir(parents=True, exist_ok=True)

    payload = asdict(context)

    daily_path = daily_dir / "bull_trend_reversal_context.json"
    dated_path = dated_dir / f"{today}_bull_trend_reversal_context.json"

    daily_path.write_text(json.dumps(payload, indent=2))
    dated_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"\nWrote: {daily_path}")
    print(f"Wrote: {dated_path}")


def main() -> None:
    prices = download_prices(TICKERS)
    macro = download_macro_data()
    context = calculate_reversal_context(prices, macro)
    write_outputs(context)


if __name__ == "__main__":
    main()
