# RL Experiment Ledger — Summary

_Auto-generated from rl_experiments.jsonl — 126 record(s). Append-only; failures retained._

| id | passed | reward | universe | OOS daily ret | OOS PF | OOS maxDD | rejection / next |
|---|---|---|---|---|---|---|---|
| rl-cc44bc2d4af7 | - | n/a (representation-quality experiment, not a trading policy) | DIA,IWM,QQQ,SPY,SQQQ,TQQQ,VIXY | - | - | - | component experiment — not evaluated against the 1%/day t... |
| rl-44c24e3c1cc9 | PASS | n/a (label-quality experiment) | DIA,IWM,QQQ,SPY,SQQQ,TQQQ,VIXY | - | - | - | If a chop label is useful, wire its prediction into env o... |
| rl-f9c36a6c624a | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-f0bb4eac7882 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-19a651340e86 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-c5014ef7fa13 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-8908d3db3ea4 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-3044a188cff0 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-553510270e80 | - | n/a (trade-quality model experiment) | AAPL,ADBE,AMAT,AMD,AMZN,ASML,AVGO,BA,BAC,CAT,COP,COST,CRM,CVX,GE,GOOGL,GS,HD | - | - | - | component experiment (trade-quality), not the 1%/day trad... |
| rl-fe26344d5e9e | - | n/a (trade-quality model experiment) | AAPL,ADBE,AMAT,AMD,AMZN,ASML,AVGO,BA,BAC,CAT,COP,COST,CRM,CVX,GE,GOOGL,GS,HD | - | - | - | component experiment (trade-quality), not the 1%/day trad... |
| rl-e8e7bee8e3b6 | - | n/a (trade-quality model experiment) | AAPL,ADBE,AMAT,AMD,AMZN,ASML,AVGO,BA,BAC,CAT,COP,COST,CRM,CVX,GE,GOOGL,GS,HD | - | - | - | component experiment (trade-quality), not the 1%/day trad... |
| rl-3975fe571362 | - | n/a (trade-quality model experiment) | AAPL,ADBE,AMAT,AMD,AMZN,ASML,AVGO,BA,BAC,CAT,COP,COST,CRM,CVX,GE,GOOGL,GS,HD | - | - | - | component experiment (trade-quality), not the 1%/day trad... |
| rl-6d3ccc3a7301 | - | n/a (trade-quality model experiment) | AAPL,ADBE,AMAT,AMD,AMZN,ASML,AVGO,BA,BAC,CAT,COP,COST,CRM,CVX,GE,GOOGL,GS,HD | - | - | - | component experiment (trade-quality), not the 1%/day trad... |
| rl-ebd97667ad67 | - | n/a (trade-quality model experiment) | AAPL,ADBE,AMAT,AMD,AMZN,ASML,AVGO,BA,BAC,CAT,COP,COST,CRM,CVX,GE,GOOGL,GS,HD | - | - | - | component experiment (trade-quality), not the 1%/day trad... |
| rl-a81f1cb5b5ae | FAIL | geometric(dd/giveback/turnover/slippage/concentration/chop/gbrisk) | AVGO,BA,COST,QCOM,IWM,MU | 0.0005 | 1.1260 | -0.1671 | OOS ADR 0.050% < 1% target; beats_SPY_sharpe=False |
| rl-fc7868cf674f | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-63d60514ef5c | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-df5258c62e63 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-27f0b31ebf5a | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-8aaa338f3adf | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-c72b7f486602 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-495d3b681f4c | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-ac1612b5391d | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-9f65f3c05ad3 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-071097126002 | FAIL | None | None | - | - | - | worker_error:BrokenProcessPool:A process in the process p... |
| rl-545cc52d808f | FAIL | geometric defaults | steps=150000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 1.4130 | -0.0018 | OOS ADR 0.001%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-b33ea809bc40 | FAIL | geometric defaults | steps=300000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 1.0010 | -0.0001 | OOS ADR 0.000%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-1edadc39a7d8 | FAIL | geometric {'lam_giveback': 3.0} | steps=150000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 1.1000 | -0.0234 | OOS ADR 0.002%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-701ae7d50105 | FAIL | geometric {'lam_chop': 2.0} | steps=150000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | -0.0000 | 0.8540 | -0.0020 | OOS ADR -0.000%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-f0a96c75aa11 | FAIL | geometric {'lam_gbrisk': 2.0} | steps=150000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 1.0350 | -0.0009 | OOS ADR 0.000%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-67c1af595c83 | FAIL | geometric {'lam_turnover': 0.01} | steps=150000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | -0.0000 | 0.8310 | -0.0016 | OOS ADR -0.000%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-a60a6a206006 | FAIL | geometric {'lam_concentration': 0.5} | steps=150000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 1.4180 | -0.0006 | OOS ADR 0.001%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-71380b37efaf | FAIL | geometric defaults | steps=150000 ent=0.0 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 1.3740 | -0.0021 | OOS ADR 0.001%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-80a19ed5504a | FAIL | geometric defaults | steps=150000 ent=0.05 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 2.2020 | -0.0000 | OOS ADR 0.000%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-73e183c7b160 | FAIL | geometric defaults | steps=150000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 1.1190 | -0.0090 | OOS ADR 0.002%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-3b8d3049e585 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | LLY,MRK,PG,AVGO,JNJ,PFE | 0.0000 | 1.1100 | -0.0295 | OOS ADR 0.004%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-f56b6de757bb | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | LLY,MRK,PG,AVGO,JNJ,PFE | 0.0000 | 1.1370 | -0.0124 | OOS ADR 0.003%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-913778d150bc | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | LLY,MRK,PG,AVGO,JNJ,PFE | -0.0002 | 0.5720 | -0.0511 | OOS ADR -0.019%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-04108d201c12 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0004 | 1.2070 | -0.0396 | OOS ADR 0.036%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-25acda53004a | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0001 | 1.1760 | -0.0402 | OOS ADR 0.012%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-f927e82ba3ad | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0001 | 1.0640 | -0.0825 | OOS ADR 0.008%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-d0b608051d20 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0002 | 0.3510 | -0.0561 | OOS ADR -0.021%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-a8358c2c3d1c | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0003 | 0.4440 | -0.0797 | OOS ADR -0.029%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-f7071b5a40d3 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0003 | 0.3700 | -0.0743 | OOS ADR -0.031%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-d755d7f8be2e | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0002 | 1.2980 | -0.0312 | OOS ADR 0.019%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-bdadecf007f1 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0000 | 1.3410 | -0.0062 | OOS ADR 0.003%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-af42f2f0f0a2 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0000 | 1.4090 | -0.0094 | OOS ADR 0.003%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-74145721c61b | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | -0.0001 | 0.8730 | -0.0304 | OOS ADR -0.005%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-a23bc88889be | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0001 | 1.2660 | -0.0122 | OOS ADR 0.005%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-c86e5d606c95 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=False | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0000 | 1.3460 | -0.0080 | OOS ADR 0.004%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-919d279257c3 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | -0.0001 | 0.8260 | -0.0446 | OOS ADR -0.011%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-2b8f0c5bf259 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | 0.0000 | 1.0230 | -0.0163 | OOS ADR 0.001%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-804ef5bf83c7 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | -0.0000 | 0.9520 | -0.0308 | OOS ADR -0.003%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-221ccdbf7a87 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0001 | 1.1310 | -0.0424 | OOS ADR 0.011%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-b9d53fd0a3cd | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0002 | 1.1870 | -0.0281 | OOS ADR 0.016%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-9b53eedc6792 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0002 | 1.2910 | -0.0232 | OOS ADR 0.023%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-f3ff195bb095 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0001 | 0.4460 | -0.0375 | OOS ADR -0.015%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-1e57d5ece551 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0001 | 0.3710 | -0.0313 | OOS ADR -0.013%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-ed913046bdb9 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0002 | 0.3670 | -0.0421 | OOS ADR -0.017%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-c888c4eef4f7 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0001 | 1.3270 | -0.0133 | OOS ADR 0.012%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-c201f994c7a2 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0002 | 1.5310 | -0.0101 | OOS ADR 0.018%<1%; beats_spy=False beats_ewsel=True beats... |
| rl-9b5b4d4b6846 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0001 | 1.2270 | -0.0108 | OOS ADR 0.006%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-ccf17e160e92 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0001 | 1.3840 | -0.0174 | OOS ADR 0.014%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-b6eb951ca4ed | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0001 | 1.2840 | -0.0105 | OOS ADR 0.010%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-ba05f6b33fc4 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0002 | 1.4760 | -0.0093 | OOS ADR 0.017%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-fe6b260af0fa | FAIL | geometric defaults 100k | per-fold PIT top6 | -0.0000 | - | -0.0375 | mean OOS sharpe -0.10 vs SPY 1.17; ADR -0.001%; collapse=... |
| rl-9bff39d62135 | FAIL | geometric defaults 100k | per-fold PIT top6 | 0.0001 | - | -0.0245 | mean OOS sharpe 0.25 vs SPY 1.17; ADR 0.005%; collapse=True |
| rl-117e965680e1 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | -0.0001 | 0.8260 | -0.0446 | OOS ADR -0.011%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-24f55df77071 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | 0.0000 | 1.0230 | -0.0163 | OOS ADR 0.001%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-b8201289842e | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | -0.0000 | 0.9520 | -0.0308 | OOS ADR -0.003%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-d2f0cfa65c97 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0001 | 1.1310 | -0.0424 | OOS ADR 0.011%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-1476d181d961 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0002 | 1.1870 | -0.0281 | OOS ADR 0.016%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-657eef0719f4 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0002 | 1.2910 | -0.0232 | OOS ADR 0.023%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-2a892ce2bcfa | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0001 | 0.4460 | -0.0375 | OOS ADR -0.015%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-168187558fe8 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0001 | 0.3710 | -0.0313 | OOS ADR -0.013%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-9f2b6585cd58 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0002 | 0.3670 | -0.0421 | OOS ADR -0.017%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-1a05d3b0a186 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0001 | 1.3270 | -0.0133 | OOS ADR 0.012%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-d826b96aeede | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0002 | 1.5310 | -0.0101 | OOS ADR 0.018%<1%; beats_spy=False beats_ewsel=True beats... |
| rl-2fa07db10da5 | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0001 | 1.2270 | -0.0108 | OOS ADR 0.006%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-c166c61a1f7c | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0001 | 1.3840 | -0.0174 | OOS ADR 0.014%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-3731fcf8562d | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0001 | 1.2840 | -0.0105 | OOS ADR 0.010%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-9d8c2363bc0b | FAIL | geometric defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0002 | 1.4760 | -0.0093 | OOS ADR 0.017%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-e178d6ab6bc0 | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | -0.0001 | 0.8590 | -0.0427 | OOS ADR -0.008%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-1b288668560c | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | 0.0000 | 1.0360 | -0.0148 | OOS ADR 0.001%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-3c7556f3cd1a | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | LLY,MRK,PG,AVGO,JNJ,PFE | -0.0000 | 0.9400 | -0.0320 | OOS ADR -0.004%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-0c4684859eed | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0001 | 1.1170 | -0.0430 | OOS ADR 0.010%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-845e475af5a3 | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0001 | 1.1750 | -0.0295 | OOS ADR 0.015%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-03df1077fa9a | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | AAPL,NVDA,LRCX,JPM,BAC,UNH | 0.0002 | 1.2890 | -0.0218 | OOS ADR 0.022%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-e0e52b25191c | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0002 | 0.4860 | -0.0388 | OOS ADR -0.016%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-e8f5554c028b | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0001 | 0.3890 | -0.0333 | OOS ADR -0.014%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-8726ea7f2793 | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | IWM,AAPL,JPM,ASML,CAT,SPY | -0.0002 | 0.4050 | -0.0379 | OOS ADR -0.016%<1%; beats_spy=False beats_ewsel=False bea... |
| rl-34ed4b00c042 | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0001 | 1.3270 | -0.0125 | OOS ADR 0.011%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-6e202c3b16c0 | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0001 | 1.4590 | -0.0095 | OOS ADR 0.015%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-93406c4a86ea | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | AVGO,PFE,AAPL,COST,QCOM,HD | 0.0001 | 1.2470 | -0.0120 | OOS ADR 0.007%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-1f4925b4b642 | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0001 | 1.3430 | -0.0144 | OOS ADR 0.012%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-a1920bdcbe33 | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0001 | 1.2770 | -0.0115 | OOS ADR 0.010%<1%; beats_spy=False beats_ewsel=False beat... |
| rl-7f8594225b39 | FAIL | benchmark_relative defaults | steps=100000 ent=0.01 overlay=True | CAT,AVGO,MRK,XOM,ORCL,GE | 0.0002 | 1.4800 | -0.0081 | OOS ADR 0.017%<1%; beats_spy=True beats_ewsel=False beats... |
| rl-a6bcf657d50d | FAIL | overlay_abs | per-fold PIT top6 | 0.0001 | - | - | mean OOS sharpe 0.25 vs SPY 1.17; ADR 0.005%; collapse=True |
| rl-4c1d2c42481a | FAIL | overlay_benchrel | per-fold PIT top6 | 0.0000 | - | - | mean OOS sharpe 0.29 vs SPY 1.17; ADR 0.004%; collapse=True |
| rl-fbebb44bb38b | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0004 | 1.0632 | -0.3111 | neural trade-quality then foundation encoder (must beat t... |
| rl-bd313b1f23d0 | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0004 | 1.0632 | -0.3111 | neural trade-quality then foundation encoder (must beat t... |
| rl-556126709ec2 | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0004 | 1.0632 | -0.3111 | neural trade-quality then foundation encoder (must beat t... |
| rl-423a43843b10 | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0011 | 1.2443 | -0.1542 | neural trade-quality then foundation encoder (must beat t... |
| rl-94e1553c9e8b | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0011 | 1.2443 | -0.1542 | neural trade-quality then foundation encoder (must beat t... |
| rl-b4d2a86bc9ea | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0011 | 1.2443 | -0.1542 | neural trade-quality then foundation encoder (must beat t... |
| rl-c7f598a46cdb | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | -0.0008 | 0.8944 | -0.3464 | neural trade-quality then foundation encoder (must beat t... |
| rl-7153621712aa | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | -0.0008 | 0.8944 | -0.3464 | neural trade-quality then foundation encoder (must beat t... |
| rl-7c0645999654 | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | -0.0008 | 0.8944 | -0.3464 | neural trade-quality then foundation encoder (must beat t... |
| rl-36e2e89ef626 | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0001 | 1.0185 | -0.2044 | neural trade-quality then foundation encoder (must beat t... |
| rl-75784a03e8ac | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0006 | 1.1315 | -0.2204 | neural trade-quality then foundation encoder (must beat t... |
| rl-1ee3466dea7e | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | -0.0003 | 0.9436 | -0.2040 | neural trade-quality then foundation encoder (must beat t... |
| rl-7fad730538d1 | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0021 | 1.4697 | -0.1323 | neural trade-quality then foundation encoder (must beat t... |
| rl-c2e4a3980b91 | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0014 | 1.3078 | -0.1994 | neural trade-quality then foundation encoder (must beat t... |
| rl-c6e85bb335ea | FAIL | n/a (deterministic execution, no RL) | deterministic candidates over adjusted pool | 0.0024 | 1.5476 | -0.1283 | neural trade-quality then foundation encoder (must beat t... |
| rl-92a0dd79d34d | FAIL | deterministic | adjusted pool candidates | 0.0006 | - | -0.2349 | mean OOS sharpe 0.62 vs SPY 1.17; ADR 0.056% |
| rl-c4fc77fa7f23 | PASS | deterministic exec | adjusted pool candidates | 0.0009 | - | - | Stage 3: self-supervised market foundation encoder; keep ... |
| rl-bd7ca487b020 | PASS | deterministic exec | adjusted pool candidates | 0.0008 | - | - | Stage 3 self-supervised foundation encoder; context-aware... |
| rl-16d348f3e1fd | FAIL | deterministic exec | adjusted pool candidates | 0.0008 | - | - | embedding did not improve BOTH ordering (NDCG@5/prec@5) A... |
| rl-2b7ed31a2824 | FAIL | deterministic exec | adjusted pool candidates | 0.0010 | - | - | per-symbol daily embedding did not improve BOTH ordering ... |
| rl-e5049cb41a38 | FAIL | deterministic exec | adjusted pool candidates | - | - | - | Sharpe Δ +0.017 not significant/reproducible enough (p=0.... |
| rl-5ecf54b2787b | FAIL | deterministic exec | adjusted large-cap pool (insider-sparse: 2.8% candidates linked) | - | - | - | insider features Δ +0.182 not significant (p=0.142); note... |
| rl-eb19a2d9a882 | FAIL | deterministic exec | insider-active daily (115 symbols) | - | - | - | Δ -0.822 p=0.010 CI(-1.2799505984135742, -0.3756891489632... |
| rl-04dcdd988fcc | FAIL | deterministic exec | large-cap champion pool | - | - | - | Δ +0.181 p=0.265 CI(-0.11672124010070394, 0.4615252620449... |
| rl-56d50d7cee21 | FAIL | n/a event-study | adjusted cache 441 | - | - | - | no executable horizon with positive block-bootstrap CI re... |
| rl-e6b3c7653642 | FAIL | n/a event-study | adjusted cache 441 | - | - | - | no executable horizon with positive block-bootstrap CI re... |
| rl-7d5e9be6f231 | FAIL | n/a event-study | adjusted cache 441 | - | - | - | block-CI lower bound not >0 and/or not reproducible acros... |
