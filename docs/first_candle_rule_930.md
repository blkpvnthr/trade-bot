# 9:35 First Candle Rule

## Plain-English Strategy

The video describes a regular-session opening strategy built from the first 5-minute candle of the trading day. Mark the high and low of the 09:30-09:35 ET candle. That candle defines the morning range and directional bias.

For this implementation, `run.py` creates long-only bracket orders at the close of that first candle when the candle is bullish and the range is tradable. The entry is the 09:35 close, the stop is one tick below the first candle low, and the target is a fixed 3:1 reward-to-risk.

The video also discusses a 1-minute fair value gap, a pullback into that gap, and bullish engulfing confirmation. The user clarified that the system should already have actionable plans by 09:35 and execute at the 09:35 close, so those later confirmations are documented as optional research context, not required by the production rule.

## Formal Checklist

- Market/session: U.S. equity regular session.
- Universe: same stock universe already selected by `run.py`.
- Setup candle: first 5-minute candle from 09:30 through 09:34 ET, locked at 09:35 ET.
- Direction: long-only.
- Entry condition: first candle is bullish, unless `FCR_REQUIRE_BULLISH_CANDLE=0`.
- Entry price: close of the 09:30-09:35 ET candle.
- Stop: one tick below the first candle low.
- Target: `entry + 3 * (entry - stop)` by default.
- Invalid setup: missing/incomplete first candle, invalid range, bearish first candle, range too tight/wide, stop distance too tight/wide, missing price, duplicate order, or any existing `run.py` risk gate failure.
- Optional volume filter: disabled by default; enable with `FCR_REQUIRE_VOLUME_CONFIRMATION=1`.

## Pine Script Strategy

```pine
//@version=6
strategy("9:35 First Candle Rule", overlay=true, process_orders_on_close=true)

targetR = input.float(3.0, "Target R", minval=0.5, step=0.25)
minRangePct = input.float(0.001, "Min range %", minval=0.0, step=0.0005)
maxRangePct = input.float(0.08, "Max range %", minval=0.001, step=0.005)
requireBullish = input.bool(true, "Require bullish first candle")

isFirstFive = hour(time, "America/New_York") == 9 and minute(time, "America/New_York") >= 30 and minute(time, "America/New_York") < 35
is935 = hour(time, "America/New_York") == 9 and minute(time, "America/New_York") == 35
newDay = ta.change(dayofmonth(time, "America/New_York")) != 0

var float firstOpen = na
var float firstHigh = na
var float firstLow = na
var float firstClose = na
var bool tradedToday = false

if newDay
    firstOpen := na
    firstHigh := na
    firstLow := na
    firstClose := na
    tradedToday := false

if isFirstFive
    firstOpen := na(firstOpen) ? open : firstOpen
    firstHigh := na(firstHigh) ? high : math.max(firstHigh, high)
    firstLow := na(firstLow) ? low : math.min(firstLow, low)
    firstClose := close

rangePct = (firstHigh - firstLow) / math.max((firstHigh + firstLow) / 2.0, syminfo.mintick)
validRange = not na(firstClose) and firstHigh > firstLow and rangePct >= minRangePct and rangePct <= maxRangePct
bullishOk = not requireBullish or firstClose > firstOpen

if is935 and validRange and bullishOk and not tradedToday
    entry = close
    stop = firstLow - syminfo.mintick
    risk = entry - stop
    if risk > 0
        target = entry + risk * targetR
        strategy.entry("FCR Long", strategy.long)
        strategy.exit("FCR 3R", "FCR Long", stop=stop, limit=target)
        tradedToday := true

plot(firstHigh, "First Candle High", color=color.green)
plot(firstLow, "First Candle Low", color=color.red)
```

## Python/Alpaca Execution Pseudocode

```python
symbols = load_run_py_universe()
bars_1m = download_1m_bars(symbols)

for symbol in symbols:
    first = bars_1m[symbol].between("09:30", "09:35")
    if len(first) < 5:
        skip("incomplete_first_candle")
        continue

    open_ = first.open.iloc[0]
    high = first.high.max()
    low = first.low.min()
    close = first.close.iloc[-1]

    if require_bullish and close <= open_:
        skip("first_candle_not_bullish")
        continue

    stop = low - tick_size
    risk = close - stop
    target = close + 3.0 * risk

    submit_bracket_buy(
        symbol=symbol,
        qty=size_from_risk_budget(symbol, close),
        entry="market_at_935_close",
        stop_price=stop,
        target_price=target,
    )
```

## Backtest Assumptions

- Use 1-minute OHLCV bars with timestamps convertible to America/New_York.
- The first candle is exactly 09:30:00 through 09:34:59 ET.
- Entry fills at the 09:35 close or next available market print.
- Stop and target are modeled as attached bracket orders.
- Default target is fixed 3R.
- Slippage, spread, halts, and partial fills require separate modeling.

## Missing Data and Uncertainties

- The exact symbol and market shown in the video are not visible in the provided frames.
- The video emphasizes high opening volume, but does not give a precise volume threshold.
- The later FVG/retest/engulf sequence is visible and captioned, but the user clarified execution should occur at the 09:35 close.
- Short-side rules are not implemented because the repo is long-only by default and the provided example is bullish.
