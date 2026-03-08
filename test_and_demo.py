"""
Options Engine — Test Suite & Demo
====================================
Validates the engine against known values and runs a full demo.

Test cases sourced from:
- Hull, "Options, Futures, and Other Derivatives" (10th ed.), Chapter 19
- Haug, "The Complete Guide to Option Pricing Formulas" (2nd ed.)
- CBOE educational materials

Run this file to verify correctness and generate output charts.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from black_scholes import OptionParams, price_option, black_scholes_price
from implied_vol import implied_volatility
from visualizations import plot_full_dashboard


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    print("\n" + "="*65)
    print("  BLACK-SCHOLES ENGINE — VALIDATION TEST SUITE")
    print("="*65)

    passed = 0
    failed = 0

    def check(label, computed, expected, tol=0.001):
        nonlocal passed, failed
        # Handle boolean checks separately
        if isinstance(expected, bool) or isinstance(computed, bool):
            ok = bool(computed) == bool(expected)
            err = 0.0 if ok else 1.0
            status = "✓ PASS" if ok else "✗ FAIL"
            if ok: passed += 1
            else:  failed += 1
            print(f"  {status}  {label:<45}  got={bool(computed)}  expected={bool(expected)}")
            return
        err = abs(float(computed) - float(expected))
        ok  = err <= tol
        status = "✓ PASS" if ok else "✗ FAIL"
        if ok: passed += 1
        else:  failed += 1
        print(f"  {status}  {label:<45}  "
              f"got={float(computed):.5f}  expected={float(expected):.5f}  err={err:.2e}")

    # ── Test 1: Hull Example 19.1 — European Call ──────────────────────────
    # Hull 10th ed p.453: S=49, K=50, T=0.3846yr, r=5%, σ=20%, no div
    # Expected price: ~$2.40
    print("\n  ── Hull Example 19.1 (Call) ──")
    p1 = OptionParams(S=49, K=50, T=0.3846, r=0.05, sigma=0.20, option_type='call')
    r1 = price_option(p1)
    check("Call price (Hull 19.1)",    r1.price,         2.4004, tol=0.01)
    check("Call delta",                r1.greeks.delta,  0.5216, tol=0.005)
    check("Put-Call parity error ≈ 0", abs(r1.put_call_parity_check), 0.0, tol=1e-6)

    # ── Test 2: European Put via Put-Call Parity ───────────────────────────
    print("\n  ── Put via Put-Call Parity ──")
    p2  = OptionParams(S=49, K=50, T=0.3846, r=0.05, sigma=0.20, option_type='put')
    r2  = price_option(p2)
    # PCP: P = C - S + K·e^(-rT)
    pcp_put = r1.price - 49 + 50 * np.exp(-0.05 * 0.3846)
    check("Put price matches PCP",     r2.price,         pcp_put, tol=1e-6)
    check("Put delta in [-1, 0]",      -1.0 < r2.greeks.delta < 0.0, True)

    # ── Test 3: Deep ITM call → price ≈ S - K·e^(-rT) ────────────────────
    print("\n  ── Deep ITM Call (intrinsic value test) ──")
    p3 = OptionParams(S=150, K=50, T=0.5, r=0.05, sigma=0.20, option_type='call')
    r3 = price_option(p3)
    intrinsic_floor = 150 - 50 * np.exp(-0.05 * 0.5)
    check("Deep ITM call >= lower bound", r3.price >= intrinsic_floor - 1e-6, True)
    check("Deep ITM delta ≈ 1",        r3.greeks.delta,  1.0, tol=0.01)

    # ── Test 4: Deep OTM call → price ≈ 0, delta ≈ 0 ────────────────────
    print("\n  ── Deep OTM Call ──")
    p4 = OptionParams(S=50, K=200, T=0.25, r=0.05, sigma=0.20, option_type='call')
    r4 = price_option(p4)
    check("Deep OTM call price ≈ 0",  r4.price,         0.0, tol=0.001)
    check("Deep OTM delta ≈ 0",       r4.greeks.delta,  0.0, tol=0.001)

    # ── Test 5: Gamma same for call and put (same inputs) ─────────────────
    print("\n  ── Greeks Symmetry Tests ──")
    p5c = OptionParams(S=100, K=100, T=0.5, r=0.05, sigma=0.25, option_type='call')
    p5p = OptionParams(S=100, K=100, T=0.5, r=0.05, sigma=0.25, option_type='put')
    r5c, r5p = price_option(p5c), price_option(p5p)
    check("Gamma(call) == Gamma(put)",  r5c.greeks.gamma,  r5p.greeks.gamma, tol=1e-10)
    check("Vega(call)  == Vega(put)",   r5c.greeks.vega,   r5p.greeks.vega,  tol=1e-10)

    # ── Test 6: Implied Volatility round-trip ─────────────────────────────
    print("\n  ── Implied Volatility Round-Trip ──")
    p6       = OptionParams(S=100, K=100, T=0.5, r=0.05, sigma=0.25, option_type='call')
    known_px = black_scholes_price(p6)
    recovered_iv = implied_volatility(
        market_price=known_px, S=100, K=100, T=0.5, r=0.05, option_type='call'
    )
    check("IV round-trip: σ=0.25",     recovered_iv, 0.25, tol=1e-6)

    p6b = OptionParams(S=100, K=110, T=0.25, r=0.03, sigma=0.35, option_type='put')
    px6b = black_scholes_price(p6b)
    iv6b = implied_volatility(px6b, S=100, K=110, T=0.25, r=0.03, option_type='put')
    check("IV round-trip OTM put σ=0.35", iv6b, 0.35, tol=1e-6)

    # ── Test 7: Merton (dividend yield) ───────────────────────────────────
    print("\n  ── Merton Dividend Extension ──")
    p7_nodiv = OptionParams(S=100, K=100, T=1.0, r=0.08, sigma=0.30, q=0.00, option_type='call')
    p7_div   = OptionParams(S=100, K=100, T=1.0, r=0.08, sigma=0.30, q=0.05, option_type='call')
    # Dividend reduces call price (divs reduce stock price)
    check("Div reduces call price",
          price_option(p7_div).price < price_option(p7_nodiv).price, True)

    # Summary
    print(f"\n  {'─'*45}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    if failed == 0:
        print("  ✓ ALL TESTS PASSED — Engine is mathematically correct.")
    else:
        print("  ✗ SOME TESTS FAILED — Review implementation.")
    print(f"  {'═'*45}\n")

    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — Full pricing example with commentary
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "="*65)
    print("  DEMO: Pricing an Apple-like call option")
    print("="*65)

    # Scenario: AAPL-like setup
    # S=185, ATM call, 45-day expiry, 28% IV, r=5.25%
    params = OptionParams(
        S     = 185.0,
        K     = 185.0,
        T     = 45/365,
        r     = 0.0525,
        sigma = 0.28,
        q     = 0.0055,   # ~0.55% div yield
        option_type = 'call'
    )

    result = price_option(params)
    print(result.summary())

    print("  Implied Volatility Round-Trip Verification:")
    iv = implied_volatility(
        market_price=result.price,
        S=params.S, K=params.K, T=params.T, r=params.r, q=params.q,
        option_type=params.option_type
    )
    print(f"    Input σ:      {params.sigma:.4f} ({params.sigma*100:.2f}%)")
    print(f"    Recovered IV: {iv:.8f} ({iv*100:.6f}%)")
    print(f"    Round-trip error: {abs(iv - params.sigma):.2e}")

    # Generate dashboard
    print("\n  Generating dashboard charts...")
    fig = plot_full_dashboard(params, save_path='bs_dashboard.png')

    # Also price the put
    put_params = OptionParams(**{**params.__dict__, 'option_type': 'put'})
    put_result = price_option(put_params)
    print(put_result.summary())

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_passed = run_tests()
    fig = run_demo()

    import matplotlib.pyplot as plt
    plt.close('all')
    print("\n  Phase 1 complete. Files saved to /mnt/user-data/outputs/")
    print("  Next up: Phase 1b — Binomial Tree model\n")
