from rl_meta.confidence import strategy_confidence
from rl_meta.backtest import StrategyResult, SRC_TV, SRC_ADAPTER, SRC_NOT_EXEC
from rl_meta.ranking import composite_score
from rl_meta.types import Metrics, RollingMetrics

UNI = list("ABCDEFGH")   # 8-symbol universe


def result(src, tv=(), tested=(), missing=(), n=0, pf=0.0, wr=0.0,
           oos_n=0, oos_pf=0.0, oos_wr=0.0, years=0.0):
    r = StrategyResult("s", "S", None, src != SRC_NOT_EXEC, "1Day")
    r.evaluation_source = src
    r.tv_symbols = list(tv)
    r.tested_symbols = list(tested)
    r.missing_tv_symbols = list(missing)
    r.years_tested = years
    r.aggregate_full = Metrics(trade_count=n, profit_factor=pf, win_rate=wr)
    r.aggregate = {"full": r.aggregate_full,
                   "oos": Metrics(trade_count=oos_n, profit_factor=oos_pf,
                                  win_rate=oos_wr)}
    return r


def test_tv_broad_coverage_high_confidence():
    r = result(SRC_TV, tv=UNI, tested=UNI, n=2500, pf=2.6, wr=0.66,
               oos_n=400, oos_pf=2.5, oos_wr=0.65, years=5.0)
    assert strategy_confidence(r, UNI) >= 70.0


def test_adapter_capped_at_30():
    r = result(SRC_ADAPTER, tested=UNI, n=2500, pf=2.6, wr=0.66,
               oos_n=400, oos_pf=2.5, oos_wr=0.65, years=5.0)
    assert strategy_confidence(r, UNI) <= 30.0


def test_not_executable_zero():
    assert strategy_confidence(result(SRC_NOT_EXEC), UNI) == 0.0


def test_no_trades_zero():
    assert strategy_confidence(result(SRC_TV, tv=UNI, tested=UNI, n=0), UNI) == 0.0


def test_low_trade_count_lowers_confidence():
    big = result(SRC_TV, tv=UNI, tested=UNI, n=2500, pf=2.5, wr=0.65,
                 oos_n=300, oos_pf=2.5, oos_wr=0.65, years=5.0)
    small = result(SRC_TV, tv=UNI, tested=UNI, n=120, pf=2.5, wr=0.65,
                   oos_n=40, oos_pf=2.5, oos_wr=0.65, years=5.0)
    assert strategy_confidence(small, UNI) < strategy_confidence(big, UNI)


def test_poor_oos_stability_lowers_confidence():
    stable = result(SRC_TV, tv=UNI, tested=UNI, n=2000, pf=2.5, wr=0.65,
                    oos_n=300, oos_pf=2.4, oos_wr=0.64, years=5.0)
    unstable = result(SRC_TV, tv=UNI, tested=UNI, n=2000, pf=2.5, wr=0.65,
                      oos_n=300, oos_pf=0.6, oos_wr=0.30, years=5.0)
    assert strategy_confidence(unstable, UNI) < strategy_confidence(stable, UNI)


def test_inf_profit_factor_does_not_break():
    r = result(SRC_TV, tv=UNI, tested=UNI, n=200, pf=float("inf"), wr=0.9,
               oos_n=50, oos_pf=float("inf"), oos_wr=0.9, years=3.0)
    c = strategy_confidence(r, UNI)
    assert 0.0 <= c <= 100.0


def test_confidence_separate_from_composite_score():
    # High quality (composite) but thin evidence (1 symbol, no OOS) -> LOW trust.
    m = Metrics(trade_count=120, win_rate=0.9, profit_factor=3.0,
                max_drawdown=0.05, expectancy=100)
    comp = composite_score(m, RollingMetrics())       # quality axis
    r = result(SRC_TV, tv=["A"], tested=["A"], missing=UNI[1:], n=120,
               pf=3.0, wr=0.9, oos_n=0, years=1.0)
    conf = strategy_confidence(r, UNI)                # trust axis
    assert comp > 0.6           # looks good
    assert conf < 40.0          # but evidence is thin
    assert conf != comp * 100   # genuinely different numbers
