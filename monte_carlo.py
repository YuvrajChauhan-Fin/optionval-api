"""
Monte Carlo Options Pricing Engine
=====================================
Phase 1, Module 5: Monte Carlo Simulation under GBM

Why Monte Carlo?
----------------
1. Handles path-dependent options (Asian, Barrier, Lookback) — BS and
   Binomial can't price these easily.
2. Completely general — any payoff function, any process.
3. Parallelisable — scales with compute, not model complexity.
4. Standard error is quantifiable: error ∝ 1/√N simulations.

The Math:
---------
Under the risk-neutral measure, stock price follows GBM:
  dS = (r - q)S·dt + σS·dW

The exact discrete solution (no Euler approximation error) is:
  S_T = S · exp[(r - q - ½σ²)T + σ√T · Z]
  where Z ~ N(0,1)

For path-dependent options we simulate the full path:
  S_{t+Δt} = S_t · exp[(r - q - ½σ²)Δt + σ√Δt · Z_t]

Variance Reduction:
-------------------
Raw MC has high variance. We use two professional techniques:

1. ANTITHETIC VARIATES: For each random Z, also simulate -Z.
   The two paths are negatively correlated → variance ≈ halved.
   Effectively doubles the number of paths for free.

2. CONTROL VARIATE: Use the known BS price as a control.
   Since BS is the analytic truth for European options:
   MC_adjusted = MC_raw + (BS_price - MC_of_control)
   This dramatically reduces variance for European options.
"""

import numpy as np
from dataclasses import dataclass
from black_scholes import OptionParams, black_scholes_price


# ---------------------------------------------------------------------------
# RESULT CONTAINER
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Full output from Monte Carlo simulation."""
    price:              float
    std_error:          float    # Standard error of the estimate
    confidence_interval: tuple   # 95% CI: (lower, upper)
    n_simulations:      int
    n_steps:            int
    variance_reduction: str      # Method used
    bs_price:           float    # Analytical comparison
    convergence_error:  float    # |MC - BS|
    paths_sample:       np.ndarray  # Small sample of paths for plotting

    def summary(self) -> str:
        ci_lo, ci_hi = self.confidence_interval
        return (
            f"\n{'='*60}\n"
            f"  MONTE CARLO SIMULATION — {self.option_type.upper() if hasattr(self,'option_type') else ''}\n"
            f"{'='*60}\n"
            f"  Simulation Config:\n"
            f"    Simulations (N)  = {self.n_simulations:,}\n"
            f"    Time steps       = {self.n_steps}\n"
            f"    Variance reduc.  = {self.variance_reduction}\n"
            f"{'─'*60}\n"
            f"  Results:\n"
            f"    MC Price         = ${self.price:.4f}\n"
            f"    Std Error        = ${self.std_error:.5f}\n"
            f"    95% CI           = [${ci_lo:.4f}, ${ci_hi:.4f}]\n"
            f"    Black-Scholes    = ${self.bs_price:.4f}\n"
            f"    Convergence Err  = ${self.convergence_error:.5f}\n"
            f"{'='*60}\n"
        )


# ---------------------------------------------------------------------------
# CORE MONTE CARLO ENGINE
# ---------------------------------------------------------------------------

def monte_carlo_price(
    params: OptionParams,
    n_simulations: int = 100_000,
    n_steps: int = 1,
    variance_reduction: str = 'antithetic',
    seed: int = 42
) -> MonteCarloResult:
    """
    Price a European option via Monte Carlo simulation.

    Parameters:
    -----------
    params             : Standard OptionParams
    n_simulations      : Number of price paths (more = more accurate, slower)
                         100K gives ~2-3 decimal places; 1M gives ~3-4
    n_steps            : Steps per path. 1 = direct terminal price (exact for
                         European). >1 = needed for path-dependent payoffs.
    variance_reduction : 'antithetic' | 'control_variate' | 'both' | 'none'
    seed               : Random seed for reproducibility

    The standard error of Monte Carlo is:
      SE = σ_payoffs / √N
    where σ_payoffs is the standard deviation of discounted payoffs.

    For European call, SE ≈ 0.01 with N=100K (i.e., ±$0.02 at 95%).
    With antithetic variates, effective N doubles → SE ≈ 0.007.

    Theoretical convergence rate: O(1/√N)
    → To halve the error, you need 4× the simulations.
    This is the fundamental MC trade-off vs analytic methods.
    """
    params.validate()
    rng = np.random.default_rng(seed)

    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    dt     = T / n_steps
    disc   = np.exp(-r * T)   # Full discount factor
    drift  = (r - q - 0.5 * sigma**2) * dt
    vol    = sigma * np.sqrt(dt)

    # ── Simulate terminal prices ───────────────────────────────────────────
    if variance_reduction in ('antithetic', 'both'):
        # Generate half the paths, mirror with -Z for the other half
        half_N = n_simulations // 2
        Z      = rng.standard_normal((half_N, n_steps))
        Z_full = np.concatenate([Z, -Z], axis=0)   # Antithetic pairs
    else:
        Z_full = rng.standard_normal((n_simulations, n_steps))

    # Build price paths via log-normal steps
    log_returns  = drift + vol * Z_full          # shape: (N, steps)
    log_S_T      = np.log(S) + log_returns.sum(axis=1)   # Sum log-returns
    S_T          = np.exp(log_S_T)               # Terminal stock prices

    # ── Compute payoffs ────────────────────────────────────────────────────
    if params.option_type == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    # Discount to present value
    discounted = disc * payoffs

    # ── Control Variate adjustment ─────────────────────────────────────────
    if variance_reduction in ('control_variate', 'both'):
        # Use BS price as control — we know the exact E[BS_payoff]
        # Estimate β (optimal coefficient) via covariance
        bs_ref = black_scholes_price(params)
        # Control: same payoff computed under BS assumptions
        # For European: control variates use the same terminal prices
        cov_matrix  = np.cov(discounted, disc * np.maximum(S_T - K if params.option_type == 'call'
                                                            else K - S_T, 0.0))
        if cov_matrix[0, 0] > 0:
            beta    = cov_matrix[0, 1] / cov_matrix[1, 1]
            discounted = discounted - beta * (discounted - bs_ref)

    # ── Statistics ─────────────────────────────────────────────────────────
    price       = float(np.mean(discounted))
    std_error   = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    ci_lower    = price - 1.96 * std_error
    ci_upper    = price + 1.96 * std_error

    bs_price    = black_scholes_price(params)
    conv_error  = abs(price - bs_price)

    # ── Save sample paths for visualisation (50 paths, full steps) ─────────
    n_path_samples = min(50, n_simulations)
    if n_steps > 1:
        sample_Z = rng.standard_normal((n_path_samples, n_steps))
        sample_paths = np.zeros((n_path_samples, n_steps + 1))
        sample_paths[:, 0] = S
        for step in range(n_steps):
            sample_paths[:, step + 1] = sample_paths[:, step] * np.exp(
                drift + vol * sample_Z[:, step]
            )
    else:
        # For single-step, reconstruct 50-step paths for viz
        n_viz_steps = 50
        dt_viz  = T / n_viz_steps
        drift_v = (r - q - 0.5 * sigma**2) * dt_viz
        vol_v   = sigma * np.sqrt(dt_viz)
        Z_viz   = rng.standard_normal((n_path_samples, n_viz_steps))
        sample_paths = np.zeros((n_path_samples, n_viz_steps + 1))
        sample_paths[:, 0] = S
        for step in range(n_viz_steps):
            sample_paths[:, step + 1] = sample_paths[:, step] * np.exp(
                drift_v + vol_v * Z_viz[:, step]
            )

    result = MonteCarloResult(
        price=price,
        std_error=std_error,
        confidence_interval=(ci_lower, ci_upper),
        n_simulations=n_simulations,
        n_steps=n_steps,
        variance_reduction=variance_reduction,
        bs_price=bs_price,
        convergence_error=conv_error,
        paths_sample=sample_paths
    )
    result.option_type = params.option_type
    return result


# ---------------------------------------------------------------------------
# CONVERGENCE STUDY
# ---------------------------------------------------------------------------

def mc_convergence_study(params: OptionParams, max_sims: int = 500_000) -> dict:
    """
    Show how MC price and standard error evolve with more simulations.

    Key result: SE ∝ 1/√N — doubling accuracy requires 4× more simulations.
    This is the fundamental limitation of MC vs analytic methods.
    """
    bs_ref   = black_scholes_price(params)
    sim_counts = [100, 500, 1000, 5000, 10000, 50000, 100000,
                  min(500000, max_sims)]

    prices, errors, std_errors = [], [], []
    for n in sim_counts:
        r = monte_carlo_price(params, n_simulations=n, seed=42)
        prices.append(r.price)
        errors.append(abs(r.price - bs_ref))
        std_errors.append(r.std_error)

    return {
        'sims':       np.array(sim_counts),
        'prices':     np.array(prices),
        'errors':     np.array(errors),
        'std_errors': np.array(std_errors),
        'bs_price':   bs_ref
    }
