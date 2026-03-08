"""
Implied Volatility Solver
==========================
Phase 1, Module 2: Inverting the BS formula to find the market's implied vol.

The Problem:
------------
Black-Scholes maps: (S, K, T, r, σ, q) → Price
Implied Vol asks:   (S, K, T, r, Price, q) → σ

There is NO closed-form inverse. We must solve numerically.

We implement two approaches:
1. Newton-Raphson: Fast, quadratic convergence, uses vega (analytical gradient)
2. Brent's Method: Robust fallback via scipy — guaranteed convergence on [a,b]

Industry note: Real vol desks use Newton-Raphson (with Brent as safety net),
exactly as implemented here.
"""

import numpy as np
from scipy.optimize import brentq
from black_scholes import OptionParams, black_scholes_price, compute_greeks


# ---------------------------------------------------------------------------
# IMPLIED VOLATILITY — NEWTON-RAPHSON + BRENT FALLBACK
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = 'call',
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Solve for implied volatility given observed market price.

    Algorithm: Newton-Raphson iteration
    ------------------------------------
    We want to find σ* such that BS(σ*) = market_price.

    Define f(σ) = BS(σ) - market_price.
    Newton update: σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)

    The key insight: f'(σ) = ∂BS/∂σ = Vega (which we already compute!)

    So each iteration:
      σ_new = σ_old - (BS_price(σ_old) - market_price) / Vega(σ_old)

    Convergence is quadratic — roughly doubles correct digits per iteration.
    Typically converges in 4-8 iterations.

    Fallback: If Newton-Raphson diverges (e.g., near-zero vega for deep
    OTM options), we fall back to Brent's method on [0.001, 10.0].

    Parameters:
    -----------
    market_price : Observed option price (mid-market ideally)
    S, K, T, r, q: Standard BS inputs
    option_type  : 'call' or 'put'
    tol          : Convergence tolerance (default: 1e-8 — very tight)
    max_iter     : Max Newton iterations before fallback

    Returns:
    --------
    Implied volatility as decimal (e.g., 0.25 = 25%)

    Raises:
    -------
    ValueError if no solution found (e.g., price violates no-arbitrage bounds)
    """
    # ── No-arbitrage bounds check ──────────────────────────────────────────
    # Call: price must be in (max(S·e^(-qT) - K·e^(-rT), 0), S·e^(-qT))
    # Put:  price must be in (max(K·e^(-rT) - S·e^(-qT), 0), K·e^(-rT))
    import math
    S_adj   = S * math.exp(-q * T)
    K_disc  = K * math.exp(-r * T)

    if option_type == 'call':
        lower_bound = max(S_adj - K_disc, 0.0)
        upper_bound = S_adj
    else:
        lower_bound = max(K_disc - S_adj, 0.0)
        upper_bound = K_disc

    if market_price <= lower_bound or market_price >= upper_bound:
        raise ValueError(
            f"Market price ${market_price:.4f} violates no-arbitrage bounds "
            f"[${lower_bound:.4f}, ${upper_bound:.4f}] for {option_type}. "
            f"Cannot compute implied vol."
        )

    def _build_params(sigma):
        return OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)

    # ── Newton-Raphson ─────────────────────────────────────────────────────
    # Initial guess: Brenner-Subrahmanyam approximation (1988)
    # σ₀ ≈ √(2π/T) · (C/S) — works well for ATM options
    sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
    sigma = np.clip(sigma, 0.01, 5.0)   # Keep initial guess reasonable

    for i in range(max_iter):
        params = _build_params(sigma)
        price  = black_scholes_price(params)
        vega   = compute_greeks(params).vega * 100   # vega was per 1%, scale back

        price_error = price - market_price

        if abs(price_error) < tol:
            return float(sigma)

        if abs(vega) < 1e-10:
            # Near-zero vega: Newton diverges, switch to Brent
            break

        sigma -= price_error / vega
        sigma  = max(sigma, 1e-6)   # Ensure positivity

    # ── Brent's Method fallback ────────────────────────────────────────────
    # Guaranteed to converge if a root exists in [vol_low, vol_high]
    try:
        def objective(s):
            return black_scholes_price(_build_params(s)) - market_price

        iv = brentq(objective, a=1e-4, b=10.0, xtol=tol, maxiter=500)
        return float(iv)

    except ValueError:
        raise ValueError(
            f"Implied volatility solver failed to converge. "
            f"Check inputs: price={market_price}, S={S}, K={K}, T={T}"
        )


def iv_surface_point(market_price: float, S: float, K: float, T: float,
                     r: float, q: float = 0.0, option_type: str = 'call') -> dict:
    """
    Compute a single point on the implied volatility surface, with metadata.
    Returns dict with iv, moneyness, log_moneyness for surface construction.
    """
    try:
        iv = implied_volatility(market_price, S, K, T, r, q, option_type)
        moneyness     = K / S              # Simple moneyness: K/S
        log_moneyness = np.log(K / S)      # Log-moneyness: ln(K/S) — more symmetric
        return {
            'iv': iv,
            'moneyness': moneyness,
            'log_moneyness': log_moneyness,
            'strike': K,
            'expiry': T,
            'status': 'ok'
        }
    except ValueError as e:
        return {'iv': np.nan, 'status': str(e), 'strike': K, 'expiry': T}
