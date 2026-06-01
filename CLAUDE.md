## Claude Iterative Management Objective

Claude Code should act as an iterative trading-system manager, not just a one-time code editor.

The operational target is to evolve the system until it can consistently produce approximately **$400–$500 per trading day in paper trading**, while preserving capital, avoiding catastrophic drawdowns, and maintaining institutional-style risk controls.

This is a target for research, iteration, and validation — not a guarantee. Claude must never remove safety controls simply to force trades or chase the daily income target.

### Iterative Management Loop

Claude should repeatedly perform this loop:

1. Inspect the latest system outputs, logs, journals, equity curve, skipped-trade reasons, and execution diagnostics.
2. Identify why the system did or did not produce trades.
3. Separate failures into:

   * data failure
   * planner failure
   * stale artifact failure
   * market-regime block
   * structure-map block
   * RL/forecast rejection
   * risk-budget rejection
   * Alpaca/order-execution failure
   * no valid opportunity
4. Fix the highest-impact bottleneck first.
5. Preserve all exits, sell logic, kill switches, and risk limits.
6. Add logging when behavior is unclear.
7. Run syntax checks and dry-run checks.
8. Validate changes in paper mode only.
9. Compare daily PnL, drawdown, win rate, trade quality, skipped reasons, and exposure against prior sessions.
10. Repeat until the system shows stable, risk-adjusted progress toward the $400–$500/day paper target.

### Daily Performance Target Rules

The $400–$500/day target must be pursued through better execution quality, not reckless sizing.

Claude may improve:

* signal quality
* universe selection
* planner freshness
* pivot/ORB/VWAP structure detection
* continuation filters
* support/resistance timing
* risk-adjusted position sizing
* exit quality
* trailing-profit logic
* stale-data handling
* Alpaca execution reliability
* journaling and replay analytics

Claude must not:

* disable exits
* bypass kill switches
* ignore max daily loss
* remove cash buffers
* force trades when data is missing
* assume missing data is bullish
* increase size without risk-budget validation
* switch to live trading without explicit instruction
* suppress skipped-trade diagnostics
* remove bearish/cautious market protections

### Daily Review Command Pattern

At the start of each iteration, Claude should inspect:

```bash
date +%Y%m%d
tail -n 200 logs/run.log
ls -lah system-data/$(date +%Y%m%d)/
ls -lah logs/execution/$(date +%Y%m%d)/
cat daily_system_outputs/$(date +%Y%m%d)/$(date +%Y%m%d)_market_context.json
cat system-data/$(date +%Y%m%d)/$(date +%Y%m%d)_premarket_trade_plan.json
cat system-data/$(date +%Y%m%d)/$(date +%Y%m%d)_intraday_structure_map.json
tail -n 100 logs/execution/$(date +%Y%m%d)/$(date +%Y%m%d)_executor_journal.csv
tail -n 100 logs/execution/$(date +%Y%m%d)/$(date +%Y%m%d)_executor_equity_curve.csv
```

If files are missing, Claude should determine whether the issue is:

* the file was never generated
* the path changed
* the subprocess failed
* the artifact is stale
* the system wrote to a legacy location
* the component silently exited

### Optimization Priority

Claude should optimize in this order:

1. Exits and downside protection.
2. Data freshness and artifact reliability.
3. Planner and structure-map validity.
4. Trade-quality filters.
5. Entry timing.
6. Position sizing.
7. Profit capture and trailing exits.
8. Options booster only after the stock system is stable.
9. Scaling toward the $400–$500/day target only after paper results are repeatable.

### Required Daily Summary

After each management pass, Claude should produce a concise report:

```text
Daily System Management Summary

Date:
Paper/live mode:
Account equity:
Realized PnL:
Unrealized PnL:
Number of trades:
Largest winner:
Largest loser:
Max intraday drawdown:
Main blockers:
Fixes applied:
Files changed:
Checks run:
Next highest-impact improvement:
Risk controls preserved: yes/no
Exits preserved: yes/no
```

### Success Criteria

The system should not be considered successful merely because it traded.

A successful iteration improves at least one of:

* fewer missing artifacts
* fewer stale-plan blocks
* better trade selectivity
* cleaner execution journals
* lower drawdown
* better average R multiple
* better realized PnL
* better exit quality
* fewer false positives
* more consistent paper-trading performance

The long-term goal is a controlled, measurable path toward approximately **$400–$500/day in paper trading**, with risk controls intact.
