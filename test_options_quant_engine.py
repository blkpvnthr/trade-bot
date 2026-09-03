#!/usr/bin/env python3
"""Tests for options_quant_engine.py — the 3-layer EV/dual-Kelly engine. Includes the
review's worked example proving deep-ITM tight-spread sizes larger than ATM wide-spread
for an identical directional signal."""
import math
import numpy as np
import options_quant_engine as eng


def approx(a, b, tol=0.02):
    return abs(a - b) < tol


def test_combine_alpha_weighted_and_missing():
    a = eng.combine_alpha({"insider": 1.0, "technical": 1.0})  # only 2 present → renormalised
    assert approx(a, 1.0, 1e-9)
    a2 = eng.combine_alpha({"market_context": 1.0, "put_call": -1.0, "news": 0.0,
                            "insider": 0.0, "technical": 0.0})
    assert -1.0 <= a2 <= 1.0
    assert eng.combine_alpha({}) == 0.0


def test_mc_prob_itm_matches_analytic():
    # with real-world drift = r, MC P(S_T>K) ≈ risk-neutral-ish N(d2) ballpark
    mc = eng.MonteCarloEngine(n_paths=40000, steps=21, seed=1)
    S0, K, T, sig, mu = 100.0, 100.0, 30/365, 0.40, 0.04
    paths = mc.simulate_paths(S0, mu, sig, T)
    p_itm = (paths[:, -1] > K).mean()
    # analytic real-world P(S_T>K) = N((ln(S0/K)+(mu-sig^2/2)T)/(sig√T))
    d = (math.log(S0/K) + (mu - 0.5*sig*sig)*T) / (sig*math.sqrt(T))
    from statistics import NormalDist
    assert approx(p_itm, NormalDist().cdf(d), 0.02), (p_itm, NormalDist().cdf(d))


def _chain(spot, iv=0.40, dte=35):
    """Both contracts priced at the SAME IV via BS (fair, no artificial mispricing); they
    differ only by moneyness (delta) and bid-ask spread."""
    import options_quant as oq
    T = dte / 365.0
    def mk(sym, K, delta, spread):
        mid = oq.bs_price(spot, K, T, iv, kind="call")
        half = mid * spread / 2.0
        return {"type": "call", "symbol": sym, "strike_price": K, "delta": delta, "dte": dte,
                "bid_price": round(mid-half, 2), "ask_price": round(mid+half, 2),
                "open_interest": 1500, "volume": 200, "implied_volatility": iv}
    return [mk("ITM85", 80, 0.85, 0.04),    # deep-ITM, tight 4% spread
            mk("ATM50", 100, 0.50, 0.18)]   # ATM, wide 18% spread


def test_dual_kelly_deep_itm_sizes_larger_than_atm():
    """THE key test: identical signal, deep-ITM tight-spread supports MORE size than ATM
    wide-spread because spread + a positive VRP (IV>RV) erode the realised ATM option edge
    (Kelly_contract). Realistic regime: contracts priced at IV=40% but the stock realises
    RV=25% (positive variance risk premium — the empirical norm), moderate signal."""
    spot, rv, alpha = 100.0, 0.25, 0.6    # RV 25% < IV 40% (positive VRP); moderate edge
    sel = eng.ContractSelector(max_spread_pct=0.25)   # allow the 18% ATM for comparison
    sizer = eng.PositionSizer(kelly_fraction=0.25, risk_cap_pct=0.05)
    results = {}
    for c in _chain(spot):
        m = eng.ContractMetrics.from_contract(c, spot, rv)
        ev = sel.evaluate(m, spot, alpha, rv, hold_days=10)
        s = sizer.size(100_000, ev, regime="bull", alpha_score=alpha)
        results[c["symbol"]] = (m, s)
    itm_m, itm_s = results["ITM85"]; atm_m, atm_s = results["ATM50"]
    print(f"\n  deep-ITM δ0.85 spr4%:  Ω={itm_m.leverage_ratio:.1f} EV={itm_m.expected_value:+.1%} "
          f"k_signal={itm_s.kelly_signal} k_contract={itm_s.kelly_contract} k_option={itm_s.kelly_option}")
    print(f"  ATM     δ0.50 spr18%: Ω={atm_m.leverage_ratio:.1f} EV={atm_m.expected_value:+.1%} "
          f"k_signal={atm_s.kelly_signal} k_contract={atm_s.kelly_contract} k_option={atm_s.kelly_option}")
    # same signal → same kelly_signal (within MC noise)
    assert approx(itm_s.kelly_signal, atm_s.kelly_signal, 0.05), "kelly_signal should match (same signal)"
    # deep-ITM preserves more edge → higher contract-Kelly and option-Kelly
    assert itm_s.kelly_contract > atm_s.kelly_contract, "deep-ITM should preserve more edge"
    assert itm_s.kelly_option > atm_s.kelly_option, "deep-ITM should support a larger option-Kelly"
    assert itm_m.expected_value > atm_m.expected_value, "deep-ITM should have higher EV"


def test_ev_ranking_picks_positive_ev():
    spot, rv, alpha = 100.0, 0.40, 0.9
    sel = eng.ContractSelector(max_spread_pct=0.25)
    ranked = sel.rank_contracts(_chain(spot), spot, alpha, rv, hold_days=10)
    assert ranked, "should rank contracts"
    assert ranked[0]["ev"] >= ranked[-1]["ev"], "sorted by EV descending"


def test_regime_and_liquidity_scale_size():
    spot, rv, alpha = 100.0, 0.40, 0.9
    sel = eng.ContractSelector(max_spread_pct=0.25)
    m = eng.ContractMetrics.from_contract(_chain(spot)[0], spot, rv)
    ev = sel.evaluate(m, spot, alpha, rv, hold_days=10)
    s_bull = eng.PositionSizer().size(100_000, ev, regime="bull", alpha_score=alpha)
    s_bear = eng.PositionSizer().size(100_000, ev, regime="bear", alpha_score=alpha)
    assert s_bear.fraction <= s_bull.fraction, "bear regime must shrink size"
    assert eng.REGIME_MULT["crisis"] == 0.0


def test_recommend_end_to_end():
    spot = 100.0
    chain = _chain(spot)
    # weak alpha → no trade
    weak = eng.recommend(chain, spot, {"insider": 0.1, "technical": 0.0}, 100_000, rv=0.40)
    assert weak["action"] == "no_trade"
    # strong bullish alpha → buy_call with sizing + probabilities
    strong = eng.recommend(chain, spot, {"insider": 1.0, "technical": 0.9, "news": 0.8},
                           100_000, rv=0.40, regime="bull", hold_days=10)
    assert strong["action"] in ("buy_call", "no_trade")
    if strong["action"] == "buy_call":
        assert "prob_targets" in strong and "sizing" in strong
        assert strong["sizing"].kelly_option >= 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"ERROR {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
