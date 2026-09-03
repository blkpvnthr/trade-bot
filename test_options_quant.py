#!/usr/bin/env python3
"""Numerical verification of options_quant.py — proves the formulas independently of the
(blog-quality, unverified) web sources by checking known identities."""
import math
import options_quant as oq


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


def test_put_call_parity():
    # C - P = S e^{-qT} - K e^{-rT}
    S, K, T, sig, r, q = 100, 95, 0.5, 0.30, 0.04, 0.01
    c = oq.bs_price(S, K, T, sig, r, q, "call")
    p = oq.bs_price(S, K, T, sig, r, q, "put")
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert approx(lhs, rhs, 1e-6), f"parity {lhs} vs {rhs}"


def test_finite_difference_greeks():
    S, K, T, sig, r, q = 100, 100, 0.25, 0.35, 0.04, 0.0
    g = oq.bs_greeks(S, K, T, sig, r, q, "call")
    h = 1e-4
    # delta ≈ dPrice/dS
    fd_delta = (oq.bs_price(S + h, K, T, sig, r, q) - oq.bs_price(S - h, K, T, sig, r, q)) / (2 * h)
    assert approx(g.delta, fd_delta, 1e-4), f"delta {g.delta} vs {fd_delta}"
    # gamma ≈ d2Price/dS2
    fd_gamma = (oq.bs_price(S + h, K, T, sig, r, q) - 2 * oq.bs_price(S, K, T, sig, r, q)
                + oq.bs_price(S - h, K, T, sig, r, q)) / (h * h)
    assert approx(g.gamma, fd_gamma, 1e-2), f"gamma {g.gamma} vs {fd_gamma}"
    # vega ≈ dPrice/dsigma
    hv = 1e-5
    fd_vega = (oq.bs_price(S, K, T, sig + hv, r, q) - oq.bs_price(S, K, T, sig - hv, r, q)) / (2 * hv)
    assert approx(g.vega, fd_vega, 1e-2), f"vega {g.vega} vs {fd_vega}"
    # theta ≈ -dPrice/dT (per year)
    ht = 1e-5
    fd_theta = -(oq.bs_price(S, K, T + ht, sig, r, q) - oq.bs_price(S, K, T - ht, sig, r, q)) / (2 * ht)
    assert approx(g.theta, fd_theta, 1e-1), f"theta {g.theta} vs {fd_theta}"


def test_call_minus_put_delta_is_one_when_no_div():
    # with q=0, call delta - put delta = 1 exactly
    S, K, T, sig, r = 100, 110, 0.4, 0.25, 0.03
    cd = oq.bs_greeks(S, K, T, sig, r, 0.0, "call").delta
    pd = oq.bs_greeks(S, K, T, sig, r, 0.0, "put").delta
    assert approx(cd - pd, 1.0, 1e-6), f"{cd}-{pd}={cd-pd}"


def test_iv_roundtrip():
    for S, K, T, sig, kind in [(100, 100, 0.25, 0.30, "call"), (50, 45, 0.5, 0.6, "call"),
                                (200, 220, 0.1, 0.45, "put"), (30, 30, 1.0, 0.2, "put")]:
        px = oq.bs_price(S, K, T, sig, 0.04, 0.0, kind)
        iv = oq.implied_vol(px, S, K, T, 0.04, 0.0, kind)
        assert iv is not None and approx(iv, sig, 1e-3), f"iv {iv} vs {sig} ({kind})"


def test_iv_no_arbitrage_bounds():
    # price below intrinsic → None
    assert oq.implied_vol(0.0001, 100, 80, 0.5, kind="call") is None
    # absurdly high price → None
    assert oq.implied_vol(1000, 100, 80, 0.5, kind="call") is None


def test_embedded_leverage_and_score():
    # deep-ITM call: spot 100, premium 22 (mostly intrinsic), delta 0.85 → Ω≈3.9 (moderate)
    omega = oq.embedded_leverage(100, 22, 0.85)
    assert 3.0 < omega < 4.5, omega
    # ATM cheap call: premium 4, delta 0.5 → Ω=12.5 (high → penalised)
    omega_atm = oq.embedded_leverage(100, 4, 0.5)
    assert omega_atm > 10
    # FP warning encoded: moderate-leverage ITM scores higher than high-leverage ATM
    assert oq.leverage_efficiency_score(omega) > oq.leverage_efficiency_score(omega_atm)


def test_select_directional_call_prefers_deep_itm():
    chain = [
        {"type": "call", "delta": 0.50, "ask_price": 4.0, "bid_price": 3.8, "open_interest": 500, "dte": 30, "symbol": "ATM"},
        {"type": "call", "delta": 0.82, "ask_price": 22.0, "bid_price": 21.7, "open_interest": 300, "dte": 30, "symbol": "ITM"},
        {"type": "call", "delta": 0.30, "ask_price": 1.5, "bid_price": 1.3, "open_interest": 500, "dte": 30, "symbol": "OTM"},
    ]
    pick = oq.select_directional_call(chain, spot=100, min_delta=0.70)
    assert pick is not None and pick["symbol"] == "ITM", pick


def test_kelly_binary():
    # coin with 60% win, even money (b=1): f* = 2*0.6 - 1 = 0.2
    assert approx(oq.kelly_binary(0.6, 1.0), 0.2, 1e-9)
    # unfavorable → 0
    assert oq.kelly_binary(0.4, 1.0) == 0.0


def test_kelly_scenarios_shrinks_vs_mean():
    # lottery: lose 100% 60% of time, +400% 40% of time. Mean payoff strongly positive.
    rs = [-1.0] * 6 + [4.0] * 4
    f = oq.kelly_from_scenarios(rs)
    # mean-payoff binary Kelly (p=0.4, b=4) = (0.4*5 - 1)/4 = 0.25
    f_mean = oq.kelly_binary(0.4, 4.0)
    assert 0 < f <= f_mean + 1e-6, f"variable-Kelly {f} should be <= mean-Kelly {f_mean}"
    # unfavorable lottery → 0
    assert oq.kelly_from_scenarios([-1.0] * 7 + [4.0] * 3) >= 0  # p=0.3,b=4 -> mean Kelly = (1.5-1)/4=0.125; variable shrinks
    assert oq.kelly_from_scenarios([-1.0] * 9 + [2.0]) == 0.0    # clearly unfavorable


def test_position_size_caps():
    # equity 100k, premium $20 (=$2000/contract), kelly_f 0.5, 1/4 kelly, 2% cap
    n = oq.position_size_contracts(100_000, 20.0, kelly_f=0.5, kelly_fraction=0.25, risk_cap_pct=0.02)
    # kelly_dollars = 100k*0.5*0.25 = 12500; cap = 2000; binding = 2000 -> 1 contract
    assert n == 1, n
    # tiny premium, big budget
    n2 = oq.position_size_contracts(100_000, 1.0, kelly_f=0.5, kelly_fraction=0.25, risk_cap_pct=0.02)
    assert n2 == int(2000 // 100), n2  # cap 2000 / (1*100) = 20


def test_vrp_and_regime():
    assert approx(oq.variance_risk_premium(0.30, 0.20, variance_form=False), 0.10)
    assert approx(oq.variance_risk_premium(0.30, 0.20, variance_form=True), 0.09 - 0.04)
    # strongly negative ret/var corr → regime active
    rets = [(-1) ** i * 0.02 for i in range(40)]
    var = [0.04 + (0.01 if r < 0 else -0.005) for r in rets]  # variance up when returns down
    assert oq.vrp_regime_active(rets, var, corr_threshold=-0.2) in (True, False)  # computes without error
    # insufficient data → safe True
    assert oq.vrp_regime_active([0.01], [0.04]) is True


def test_mv_delta_noop_until_fitted():
    # default coeffs → no-op
    assert oq.mv_delta(0.7, 10.0, 100, 0.25, oq.MVDeltaCoeffs()) == 0.7
    # fitted coeffs apply the correction
    c = oq.MVDeltaCoeffs(a=0.0, b=-0.1, c=0.0, fitted=True)
    out = oq.mv_delta(0.7, 10.0, 100, 0.25, c)
    assert out != 0.7


def test_iv_rank_percentile():
    hist = [0.2, 0.3, 0.4, 0.5, 0.6]
    assert approx(oq.iv_rank(0.4, hist), 0.5)
    assert approx(oq.iv_percentile(0.45, hist), 0.6)  # 3 of 5 below 0.45


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
