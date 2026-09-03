# 7:00 AM Strategy Readiness

Primary strategy for this morning: paper-run the validated vol-gated insider-buy shares copier only.

Rule: greater than $1M recent open-market Form-4 buy, current realized_vol_20 >= 0.0563, 20-day average dollar volume >= $1M, buy shares, hold 3 trading days, max 3 open positions, risk-based sizing, rolling PF kill-switch at 1.8.

Do not use options, KMeans regimes, VWAP/VPIN/BVC/ORB timing, Form-4 sells, or price-pattern candidates A/B/C. Those were falsified or rejected.

13D/G remains research-only unless it finishes and clears the full gauntlet before launch. Do not wire 13D/G into `run.py` today.

Launch path, paper only:

```bash
bash scripts/launch_0700_vol_insider_paper.sh
```

Keep `ENABLE_INSIDER_COPIER` off if not actively paper-trading. All orders must continue through `run.py` and RiskManager; exits must remain allowed.
