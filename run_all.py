"""
Options Valuation Engine — Phase 1 Master Runner
==================================================
Runs all three models, prints comparison table, generates dashboard.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from black_scholes   import OptionParams
from model_comparison import run_all_models, plot_model_comparison, print_comparison_table


def main():
    print("\n" + "="*65)
    print("  OPTIONS VALUATION ENGINE — PHASE 1")
    print("  Black-Scholes  |  Binomial Tree  |  Monte Carlo")
    print("="*65)

    # ── Scenario: ATM call, 45 days, 28% vol ──────────────────────────────
    params = OptionParams(
        S=185.0, K=185.0, T=45/365,
        r=0.0525, sigma=0.28, q=0.0055,
        option_type='call'
    )

    print(f"\n  Scenario: CALL  S=${params.S}  K=${params.K}  "
          f"T={params.T*365:.0f}d  σ={params.sigma*100:.0f}%  r={params.r*100:.2f}%\n")

    results = run_all_models(params, binomial_steps=200, mc_simulations=100_000)

    print_comparison_table(results)

    print("  Generating comparison dashboard...")
    fig = plot_model_comparison(results, save_path='model_comparison.png')
    plt.close('all')

    # ── Bonus: OTM Put comparison ─────────────────────────────────────────
    print("\n  Running OTM Put scenario...")
    put_params = OptionParams(
        S=185.0, K=175.0, T=45/365,
        r=0.0525, sigma=0.28, q=0.0055,
        option_type='put'
    )
    put_results = run_all_models(put_params, binomial_steps=200, mc_simulations=100_000)
    print_comparison_table(put_results)

    print("  Generating OTM put dashboard...")
    plot_model_comparison(put_results, save_path='model_comparison_put.png')
    plt.close('all')

    print("\n  ✓ Phase 1 complete — all three models operational.")
    print("  Output files:")
    print("    → model_comparison.png      (ATM Call)")
    print("    → model_comparison_put.png  (OTM Put)")
    print("\n  Next: Phase 2 — Interactive Web UI\n")


if __name__ == "__main__":
    main()
