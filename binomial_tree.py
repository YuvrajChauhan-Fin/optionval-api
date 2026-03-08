"""
Binomial Tree Options Pricing Model
=====================================
Phase 1, Module 4: Cox-Ross-Rubinstein (CRR) Binomial Tree

Why the Binomial Tree matters:
-------------------------------
1. It prices AMERICAN options (early exercise) — Black-Scholes CANNOT do this.
2. It makes the risk-neutral pricing argument completely transparent.
3. It converges to Black-Scholes as steps N → ∞.
4. It's the foundation for more exotic lattice models (trinomial, etc.)

The Core Idea:
--------------
At each time step Δt = T/N, the stock price can move:
  UP:   S → S·u   with risk-neutral probability p
  DOWN: S → S·d   with risk-neutral probability (1-p)

CRR parameterisation (the standard):
  u = e^(σ√Δt)          — up factor
  d = 1/u = e^(-σ√Δt)   — down factor (ensures recombining tree)
  p = (e^((r-q)Δt) - d) / (u - d)   — risk-neutral probability

The recombining property is crucial: after an up-then-down move,
S·u·d = S (since d = 1/u). This means an N-step tree has N+1 terminal
nodes, not 2^N — making computation O(N²) not O(2^N).

American Option Pricing:
------------------------
At each node, we compare:
  Continuation value: discounted expected value from child nodes
  Exercise value:     immediate payoff if exercised now
  Node value:         max(continuation, exercise)

This backward induction captures the early exercise premium —
the extra value of being able to exercise before expiry.
"""

import numpy as np
from dataclasses import dataclass
from black_scholes import OptionParams, black_scholes_price


# ---------------------------------------------------------------------------
# RESULT CONTAINER
# ---------------------------------------------------------------------------

@dataclass
class BinomialResult:
    """Full output from the binomial tree model."""
    price:              float
    option_type:        str
    exercise_style:     str      # 'european' or 'american'
    steps:              int
    early_exercise_premium: float   # American price - European price (≥ 0)
    delta:              float    # Numerical delta from tree
    gamma:              float    # Numerical gamma from tree
    theta:              float    # Numerical theta from tree
    bs_price:           float    # BS price for comparison
    convergence_error:  float    # |binomial - BS| for European options
    tree_params:        dict     # u, d, p for transparency

    def summary(self) -> str:
        style = self.exercise_style.upper()
        return (
            f"\n{'='*60}\n"
            f"  BINOMIAL TREE ({self.steps} steps) — {style} {self.option_type.upper()}\n"
            f"{'='*60}\n"
            f"  Tree Parameters:\n"
            f"    u (up factor)    = {self.tree_params['u']:.6f}\n"
            f"    d (down factor)  = {self.tree_params['d']:.6f}\n"
            f"    p (risk-neutral) = {self.tree_params['p']:.6f}\n"
            f"    Δt (time step)   = {self.tree_params['dt']:.6f} yr\n"
            f"{'─'*60}\n"
            f"  Results:\n"
            f"    Binomial Price   = ${self.price:.4f}\n"
            f"    Black-Scholes    = ${self.bs_price:.4f}\n"
            f"    Convergence Err  = ${self.convergence_error:.4f}\n"
            f"{'─'*60}\n"
            f"  Greeks (numerical):\n"
            f"    Δ Delta          = {self.delta:+.6f}\n"
            f"    Γ Gamma          = {self.gamma:+.6f}\n"
            f"    Θ Theta          = {self.theta:+.6f}  (per day)\n"
        )
        if self.exercise_style == 'american':
            return (f"  Early Exercise Premium = ${self.early_exercise_premium:.4f}\n"
                    f"    (extra value vs European for being able to exercise early)\n"
                    f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CORE BINOMIAL TREE ENGINE
# ---------------------------------------------------------------------------

def binomial_tree_price(
    params: OptionParams,
    steps: int = 200,
    exercise_style: str = 'european'
) -> BinomialResult:
    """
    Price an option using the CRR binomial tree.

    Parameters:
    -----------
    params         : OptionParams — same inputs as Black-Scholes
    steps          : Number of time steps N. More steps → more accurate.
                     200 steps gives ~4 decimal places for European options.
                     Odd number of steps gives cleaner ATM convergence.
    exercise_style : 'european' or 'american'

    Algorithm (Backward Induction):
    ---------------------------------
    Step 1: Build terminal stock prices at T
      S_{N,j} = S · u^j · d^(N-j)   for j = 0, 1, ..., N
      (j = number of up-moves out of N steps)

    Step 2: Compute terminal payoffs
      Call: max(S_{N,j} - K, 0)
      Put:  max(K - S_{N,j}, 0)

    Step 3: Backward induction through the tree
      V_{i,j} = e^(-rΔt) · [p·V_{i+1,j+1} + (1-p)·V_{i+1,j}]
      For American: V_{i,j} = max(V_{i,j}, exercise_value_{i,j})

    Step 4: Extract Greeks from first two layers of tree
      Delta: (V_{1,1} - V_{1,0}) / (S·u - S·d)
      Gamma: second-order finite difference on second layer
      Theta: (V_{2,1} - V_{0,0}) / (2Δt) → per day

    Returns BinomialResult with full diagnostics.
    """
    params.validate()
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q

    # ── Step 1: CRR parameters ─────────────────────────────────────────────
    dt    = T / steps
    u     = np.exp(sigma * np.sqrt(dt))       # Up factor
    d     = 1.0 / u                            # Down factor (recombining)
    disc  = np.exp(-r * dt)                    # Discount factor per step
    p     = (np.exp((r - q) * dt) - d) / (u - d)   # Risk-neutral prob of up

    # Validate risk-neutral probability (arbitrage check)
    if not (0 < p < 1):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside (0,1). "
            f"Check inputs — possible arbitrage violation."
        )

    # ── Step 2: Terminal stock prices (vectorised) ─────────────────────────
    # j = number of up-moves, range [0, N]
    j       = np.arange(steps + 1)
    S_T     = S * (u ** j) * (d ** (steps - j))

    # ── Step 3: Terminal payoffs ───────────────────────────────────────────
    if params.option_type == 'call':
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # ── Step 4: Backward induction ─────────────────────────────────────────
    # We store the first few layers for Greek extraction
    for i in range(steps - 1, -1, -1):
        # Discount and average child nodes
        V = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])

        # American early exercise check
        if exercise_style == 'american':
            S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            if params.option_type == 'call':
                exercise = np.maximum(S_i - K, 0.0)
            else:
                exercise = np.maximum(K - S_i, 0.0)
            V = np.maximum(V, exercise)

        # Store layers needed for Greeks
        if i == 2:
            V2 = V.copy()
        elif i == 1:
            V1 = V.copy()

    price = float(V[0])

    # ── Step 5: Greeks from tree ───────────────────────────────────────────
    # Delta: central finite difference at node (1, up) and (1, down)
    S_u = S * u    # Stock price after one up-move
    S_d = S * d    # Stock price after one down-move
    delta = (V1[1] - V1[0]) / (S_u - S_d)

    # Gamma: second-order finite difference using three nodes at step 2
    S_uu = S * u**2
    S_ud = S           # u·d = 1 for CRR
    S_dd = S * d**2
    delta_u = (V2[2] - V2[1]) / (S_uu - S_ud)
    delta_d = (V2[1] - V2[0]) / (S_ud - S_dd)
    gamma   = (delta_u - delta_d) / (0.5 * (S_uu - S_dd))

    # Theta: (V at node (2,1) - V at node (0,0)) / (2·Δt)
    # This gives the price change over one full Δt step
    theta = (V2[1] - price) / (2 * dt) / 365   # Per calendar day

    # ── Step 6: Comparison with Black-Scholes ─────────────────────────────
    bs_price = black_scholes_price(params)

    # European: convergence error measures model accuracy
    # American: difference is the early exercise premium
    if exercise_style == 'european':
        conv_error  = abs(price - bs_price)
        eep         = 0.0
    else:
        # Reprice European to isolate early exercise premium
        euro_result = binomial_tree_price(params, steps=steps, exercise_style='european')
        eep         = max(price - euro_result.price, 0.0)
        conv_error  = abs(euro_result.price - bs_price)

    return BinomialResult(
        price=price,
        option_type=params.option_type,
        exercise_style=exercise_style,
        steps=steps,
        early_exercise_premium=eep,
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        bs_price=bs_price,
        convergence_error=conv_error,
        tree_params={'u': u, 'd': d, 'p': p, 'dt': dt}
    )


# ---------------------------------------------------------------------------
# CONVERGENCE ANALYSIS
# ---------------------------------------------------------------------------

def convergence_analysis(params: OptionParams, max_steps: int = 500) -> dict:
    """
    Show how binomial price converges to BS as steps increase.

    This is the most revealing diagnostic of the model — it shows:
    1. The oscillation pattern (even vs odd steps) in BS convergence
    2. How quickly the tree reaches machine precision
    3. The "odd-steps trick" (odd step count gives better ATM convergence)

    Returns arrays for plotting convergence curves.
    """
    bs_ref = black_scholes_price(params)
    step_counts = list(range(5, max_steps + 1, 5))

    prices = []
    errors = []
    for n in step_counts:
        result = binomial_tree_price(params, steps=n, exercise_style='european')
        prices.append(result.price)
        errors.append(abs(result.price - bs_ref))

    return {
        'steps':    np.array(step_counts),
        'prices':   np.array(prices),
        'errors':   np.array(errors),
        'bs_price': bs_ref
    }
