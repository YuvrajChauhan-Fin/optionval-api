"""
Black-Scholes Options Pricing Engine
=====================================
Author: Options Valuation Engine v1.0
Phase 1: Core Pricing + Greeks

Mathematical Foundation:
------------------------
The Black-Scholes model prices European options under these assumptions:
  1. Underlying follows Geometric Brownian Motion: dS = μS·dt + σS·dW
  2. No dividends (base model — we extend for continuous dividend yield q)
  3. Constant volatility σ and risk-free rate r over option life
  4. No arbitrage, continuous trading, no transaction costs
  5. Log-normal distribution of terminal stock price

The closed-form solution derives from solving the BS PDE:
  ∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S − rV = 0

Key insight: Under risk-neutral measure, μ → r, and the expectation
of the discounted payoff gives the price today.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# 1. INPUT CONTAINER
# ---------------------------------------------------------------------------

@dataclass
class OptionParams:
    """
    All parameters needed to price a European option.

    S  : Current stock price (spot price)
    K  : Strike price — the agreed purchase/sale price
    T  : Time to expiry in YEARS (e.g., 30 days = 30/365 ≈ 0.0822)
    r  : Risk-free rate, continuously compounded (e.g., 0.05 = 5%)
    sigma: Implied/historical volatility, annualised (e.g., 0.20 = 20%)
    q  : Continuous dividend yield (default 0). Merton (1973) extension.
    option_type: 'call' or 'put'
    """
    S: float       # Spot price
    K: float       # Strike price
    T: float       # Time to expiry (years)
    r: float       # Risk-free rate (annualised, continuously compounded)
    sigma: float   # Volatility (annualised)
    q: float = 0.0 # Dividend yield (continuous)
    option_type: Literal['call', 'put'] = 'call'

    def validate(self):
        """Sanity-check inputs before pricing. Raises ValueError on bad data."""
        if self.S <= 0:
            raise ValueError(f"Spot price must be positive. Got S={self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike must be positive. Got K={self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry must be positive. Got T={self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive. Got sigma={self.sigma}")
        if self.option_type not in ('call', 'put'):
            raise ValueError(f"option_type must be 'call' or 'put'. Got '{self.option_type}'")


# ---------------------------------------------------------------------------
# 2. CORE d1 / d2 COMPUTATION
# ---------------------------------------------------------------------------

def compute_d1_d2(params: OptionParams) -> tuple[float, float]:
    """
    Compute the d1 and d2 standardised normal variates used throughout BS.

    d1 = [ln(S/K) + (r - q + ½σ²)T] / (σ√T)
    d2 = d1 - σ√T

    Intuition:
    - ln(S/K): log-moneyness — how far in/out of the money we are
    - (r - q + ½σ²)T: drift adjustment. The ½σ² term is the Itô correction
      that arises because E[ln S_T] ≠ ln E[S_T] for log-normal processes.
    - σ√T: normalisation — scales by total vol over the option's life.

    d2 is the risk-neutral probability (in z-score space) that the option
    expires in-the-money. N(d2) = risk-neutral ITM probability.
    d1 relates to the delta-hedge ratio (share of stock needed to hedge).
    """
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    sqrt_T = np.sqrt(T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    return d1, d2


# ---------------------------------------------------------------------------
# 3. OPTION PRICE
# ---------------------------------------------------------------------------

def black_scholes_price(params: OptionParams) -> float:
    """
    Closed-form Black-Scholes price for a European call or put.

    Call:  C = S·e^(-qT)·N(d1) − K·e^(-rT)·N(d2)
    Put:   P = K·e^(-rT)·N(-d2) − S·e^(-qT)·N(-d1)

    Where:
    - S·e^(-qT): present value of the stock, adjusted for dividends paid out
    - K·e^(-rT): present value of the strike (discounted at risk-free rate)
    - N(d1), N(d2): cumulative standard normal CDF values

    Interpretation of the call formula:
    - N(d2): probability option expires ITM under risk-neutral measure
    - N(d1): delta — hedge ratio (shares of stock needed per option written)
    - First term: expected stock receipt if ITM, discounted
    - Second term: expected payment of strike if ITM, discounted

    Put-Call Parity check: C - P = S·e^(-qT) - K·e^(-rT)
    """
    params.validate()
    d1, d2 = compute_d1_d2(params)

    discount_stock  = params.S * np.exp(-params.q * params.T)   # S·e^(-qT)
    discount_strike = params.K * np.exp(-params.r * params.T)   # K·e^(-rT)

    if params.option_type == 'call':
        price = discount_stock * norm.cdf(d1) - discount_strike * norm.cdf(d2)
    else:  # put
        price = discount_strike * norm.cdf(-d2) - discount_stock * norm.cdf(-d1)

    return float(price)


# ---------------------------------------------------------------------------
# 4. GREEKS
# ---------------------------------------------------------------------------

@dataclass
class Greeks:
    """
    Container for option sensitivity measures (Greeks).
    Each measures the rate of change of option price w.r.t. one parameter.
    """
    delta: float    # ∂V/∂S       — sensitivity to spot price
    gamma: float    # ∂²V/∂S²     — rate of change of delta
    vega:  float    # ∂V/∂σ       — sensitivity to volatility (per 1% move)
    theta: float    # ∂V/∂t       — time decay (per calendar day)
    rho:   float    # ∂V/∂r       — sensitivity to interest rate (per 1% move)

    def __repr__(self):
        return (
            f"Greeks(\n"
            f"  delta = {self.delta:+.6f}   [option price change per $1 spot move]\n"
            f"  gamma = {self.gamma:+.6f}   [delta change per $1 spot move]\n"
            f"  vega  = {self.vega:+.6f}   [price change per 1% vol move]\n"
            f"  theta = {self.theta:+.6f}   [price change per calendar day]\n"
            f"  rho   = {self.rho:+.6f}   [price change per 1% rate move]\n"
            f")"
        )


def compute_greeks(params: OptionParams) -> Greeks:
    """
    Analytical Greeks — derived by differentiating the BS formula directly.

    Using analytical (closed-form) Greeks rather than finite-difference
    approximations ensures machine precision and is the industry standard.

    ─────────────────────────────────────────────────────────────────────
    DELTA (∂V/∂S):
      Call: Δ = e^(-qT)·N(d1)          Range: [0, 1]
      Put:  Δ = e^(-qT)·(N(d1) - 1)   Range: [-1, 0]

      Interpretation: If Δ = 0.60 for a call, the option price increases
      by ~$0.60 for each $1 rise in the stock. Also the hedge ratio —
      you need to hold 0.60 shares of stock to delta-hedge one call.

    ─────────────────────────────────────────────────────────────────────
    GAMMA (∂²V/∂S² = ∂Δ/∂S):
      Γ = [e^(-qT)·N'(d1)] / (S·σ·√T)   (same for call and put)

      Where N'(x) = φ(x) = (1/√2π)·e^(-x²/2) is the standard normal PDF.

      Interpretation: Rate of change of delta. High gamma = delta changes
      rapidly with spot — useful for scalpers but dangerous for sellers.
      Gamma is highest for near-the-money options near expiry.

    ─────────────────────────────────────────────────────────────────────
    VEGA (∂V/∂σ):
      ν = S·e^(-qT)·N'(d1)·√T           (same for call and put)

      We report vega per 1% move in vol (divide by 100).

      Interpretation: If ν = 0.15, option gains $0.15 per 1% rise in vol.
      Vega is always positive for long options (long vol position).
      Highest for ATM options with significant time remaining.

    ─────────────────────────────────────────────────────────────────────
    THETA (∂V/∂t, reported as daily decay):
      The full formula has two terms:
      Term 1: −[S·e^(-qT)·N'(d1)·σ] / (2√T)   — vol-related decay
      Term 2 (call): + q·S·e^(-qT)·N(d1) − r·K·e^(-rT)·N(d2)
      Term 2 (put):  − q·S·e^(-qT)·N(-d1) + r·K·e^(-rT)·N(-d2)

      We divide by 365 to express as daily decay.

      Interpretation: Options lose value as time passes (positive theta
      is good for sellers, negative theta for buyers). The "enemy of
      option buyers" — time erodes the option's extrinsic value.

    ─────────────────────────────────────────────────────────────────────
    RHO (∂V/∂r):
      Call: ρ = K·T·e^(-rT)·N(d2) / 100
      Put:  ρ = −K·T·e^(-rT)·N(-d2) / 100

      Divided by 100 to express per 1% interest rate move.

      Interpretation: Calls benefit from higher rates (positive rho),
      puts are hurt (negative rho). Usually the smallest Greek in practice.
      Most relevant for long-dated options (LEAPS).
    """
    params.validate()
    d1, d2 = compute_d1_d2(params)

    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    sqrt_T = np.sqrt(T)

    # Pre-compute reused quantities
    n_d1        = norm.pdf(d1)                    # N'(d1): standard normal PDF
    exp_qT      = np.exp(-q * T)                  # e^(-qT)
    exp_rT      = np.exp(-r * T)                  # e^(-rT)
    S_adj       = S * exp_qT                       # Dividend-adjusted spot
    K_disc      = K * exp_rT                       # Discounted strike

    # ── DELTA ──────────────────────────────────────────────────────────────
    if params.option_type == 'call':
        delta = exp_qT * norm.cdf(d1)
    else:
        delta = exp_qT * (norm.cdf(d1) - 1)

    # ── GAMMA (identical for call and put) ─────────────────────────────────
    gamma = (exp_qT * n_d1) / (S * sigma * sqrt_T)

    # ── VEGA (per 1% vol move, same for call and put) ──────────────────────
    vega = S_adj * n_d1 * sqrt_T / 100

    # ── THETA (per calendar day) ───────────────────────────────────────────
    theta_term1 = -(S_adj * n_d1 * sigma) / (2 * sqrt_T)

    if params.option_type == 'call':
        theta_term2 = q * S_adj * norm.cdf(d1) - r * K_disc * norm.cdf(d2)
    else:
        theta_term2 = -q * S_adj * norm.cdf(-d1) + r * K_disc * norm.cdf(-d2)

    theta = (theta_term1 + theta_term2) / 365   # Convert annual to daily

    # ── RHO (per 1% rate move) ─────────────────────────────────────────────
    if params.option_type == 'call':
        rho = K * T * exp_rT * norm.cdf(d2) / 100
    else:
        rho = -K * T * exp_rT * norm.cdf(-d2) / 100

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ---------------------------------------------------------------------------
# 5. FULL PRICING RESULT
# ---------------------------------------------------------------------------

@dataclass
class PricingResult:
    """Complete output from the pricing engine."""
    params:       OptionParams
    price:        float
    greeks:       Greeks
    d1:           float
    d2:           float
    intrinsic:    float   # max(S-K, 0) for call; max(K-S, 0) for put
    extrinsic:    float   # time value = price - intrinsic
    put_call_parity_check: float   # Should be ~0 if engine is correct

    def summary(self) -> str:
        p = self.params
        moneyness = "ATM" if abs(p.S - p.K) / p.K < 0.005 else \
                    ("ITM" if (p.option_type == 'call' and p.S > p.K) or
                               (p.option_type == 'put'  and p.S < p.K) else "OTM")
        return (
            f"\n{'='*60}\n"
            f"  BLACK-SCHOLES PRICING ENGINE — RESULTS\n"
            f"{'='*60}\n"
            f"  Inputs:\n"
            f"    Spot (S)     = ${p.S:.2f}\n"
            f"    Strike (K)   = ${p.K:.2f}  [{moneyness}]\n"
            f"    Expiry (T)   = {p.T:.4f} yr  ({p.T*365:.1f} days)\n"
            f"    Risk-free(r) = {p.r*100:.2f}%\n"
            f"    Volatility   = {p.sigma*100:.1f}%\n"
            f"    Div yield(q) = {p.q*100:.2f}%\n"
            f"    Type         = {p.option_type.upper()}\n"
            f"{'─'*60}\n"
            f"  Intermediate:\n"
            f"    d1           = {self.d1:+.6f}\n"
            f"    d2           = {self.d2:+.6f}\n"
            f"    N(d1)        = {norm.cdf(self.d1):.6f}  (delta-related)\n"
            f"    N(d2)        = {norm.cdf(self.d2):.6f}  (RN ITM probability)\n"
            f"{'─'*60}\n"
            f"  Pricing:\n"
            f"    Option Price = ${self.price:.4f}\n"
            f"    Intrinsic    = ${self.intrinsic:.4f}\n"
            f"    Extrinsic    = ${self.extrinsic:.4f}  (time value)\n"
            f"    PCP Error    = {self.put_call_parity_check:.2e}  (should ≈ 0)\n"
            f"{'─'*60}\n"
            f"  Greeks:\n"
            f"    Δ Delta      = {self.greeks.delta:+.6f}\n"
            f"    Γ Gamma      = {self.greeks.gamma:+.6f}\n"
            f"    ν Vega       = {self.greeks.vega:+.6f}  (per 1% vol)\n"
            f"    Θ Theta      = {self.greeks.theta:+.6f}  (per day)\n"
            f"    ρ Rho        = {self.greeks.rho:+.6f}  (per 1% rate)\n"
            f"{'='*60}\n"
        )


def price_option(params: OptionParams) -> PricingResult:
    """
    Master pricing function — returns full PricingResult with price + Greeks.

    Also computes:
    - Intrinsic value: the payoff if exercised right now
    - Extrinsic (time) value: premium above intrinsic — what you pay for
      the optionality and remaining time
    - Put-Call Parity check: C - P = S·e^(-qT) - K·e^(-rT)
      This should equal ~0 for a correctly implemented engine.
    """
    params.validate()
    price  = black_scholes_price(params)
    greeks = compute_greeks(params)
    d1, d2 = compute_d1_d2(params)

    # Intrinsic value
    if params.option_type == 'call':
        intrinsic = max(params.S - params.K, 0.0)
    else:
        intrinsic = max(params.K - params.S, 0.0)

    extrinsic = max(price - intrinsic, 0.0)

    # Put-Call Parity check
    call_params = OptionParams(**{**params.__dict__, 'option_type': 'call'})
    put_params  = OptionParams(**{**params.__dict__, 'option_type': 'put'})
    call_price  = black_scholes_price(call_params)
    put_price   = black_scholes_price(put_params)
    forward     = params.S * np.exp(-params.q * params.T) - params.K * np.exp(-params.r * params.T)
    pcp_error   = (call_price - put_price) - forward   # Should be ~0

    return PricingResult(
        params=params,
        price=price,
        greeks=greeks,
        d1=d1,
        d2=d2,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        put_call_parity_check=pcp_error
    )
