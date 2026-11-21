# Archive

## Week of 2026-11-10
Completed Obj1 infrastructure via daily-green trading engine and MomentumSetupScanner fixes (9-commit deployment), then falsified due to excessive turnover (6K%+). Reconciled Obj1/Obj2 baseline discrepancy through cost/signal rewrite; validated 7-ablation conclusions with MRNA confirmed (213 tests). Pivoted to Obj2 nested WF with risk-aware targeting and covariance-aware sizing. Scaled risk thresholds for $824 account live validation and began pre-mkt safety setup.

## Week of 2026-11-17
Closed M1 fractional orders (exact-$ sizing, exit-qty fix); expanded momentum-gated green init. Designed M2 streaming/low-latency arch w/ order state machine + latency tracker.
```