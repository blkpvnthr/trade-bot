# Alpha Momentum — ATM vs Deep-ITM Calls (head-to-head)

Universe (17): SPY, QQQ, IWM, NVDA, TSLA, AMD, AAPL, MSFT, META, AMZN, GOOGL, AVGO, NFLX, COIN, MSTR, PLTR, SMCI
Signal: breakout>20d-high & close>SMA50 & rel-vol>1.5. Call: 35DTE, IV=realised×1.1, extrinsic spread 10%. ATM δ=0.5 vs deep-ITM δ=0.8.

| Hold | n | Share mean | **ATM net** | ATM win/t | **ITM net** | ITM win/t | Winner |
|---|---|---|---|---|---|---|---|
| 3d | 382 | +0.80% | **-1.3%** | 36%/-0.4 | **+2.9%** | 48%/+1.4 | **deep-ITM** |
| 10d | 313 | +2.56% | **+18.2%** | 40%/+2.5 | **+15.2%** | 53%/+3.6 | **ATM** |

## Read
- Head-to-head on identical momentum entries → directly answers the engine's default.
- If ATM wins, the convexity (gamma) of momentum continuation pays for its theta → keep ATM default
  (OPTIONS_LEV_SELECT_WEIGHT low). If deep-ITM wins, flip the default like the insider engine.
- Share mean shows whether momentum entries actually have forward drift (vs the ~0 of the quality screen).

_NET is an optimistic ceiling; the COMPARISON across deltas is the robust part._