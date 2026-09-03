# Vertical Spreads vs Long Calls — real chains, liquid mega-caps

Same momentum entries / 10-day hold as the long-call studies. Real AV daily chains, real bid/ask, $0.65/contract/side. bull_call=δ0.50/0.30 debit; bull_put=δ0.30/0.15 credit (return on capital-at-risk).

| Strategy | n | Mean | Median | Sharpe | Win% | MaxDD |
|---|---|---|---|---|---|---|
| **long_call** | 244 | +13.2% | **-18.0%** | +0.109 | 40% | -97% |
| **bull_call** | 239 | -11.6% | **-19.8%** | -0.164 | 42% | -100% |
| **bull_put** | 242 | -0.5% | **+6.2%** | -0.020 | 62% | -68% |

_Judge by median + win + MaxDD. Bull_call should compress the long-call −90%+ drawdown; bull_put (credit) wins when the underlying merely holds — a different risk shape entirely._