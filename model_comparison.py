"""
Model Comparison Engine
========================
Phase 1, Module 6: Side-by-side comparison of all three pricing models.

This module produces the definitive output chart — a professional
multi-panel dashboard that clearly distinguishes each model's:
  - Price output
  - Greeks (where available)
  - Convergence behaviour
  - Theoretical assumptions and limitations

Design principle: Each model gets its own visual identity.
  Black-Scholes → Blue  (exact, analytical)
  Binomial Tree → Green (discrete, lattice)
  Monte Carlo   → Orange (stochastic, simulation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from black_scholes import OptionParams, price_option, black_scholes_price
from binomial_tree import binomial_tree_price, convergence_analysis
from monte_carlo   import monte_carlo_price, mc_convergence_study


# ── Model color palette ────────────────────────────────────────────────────
MODEL_COLORS = {
    'bs':       '#58a6ff',   # Blue  — Black-Scholes
    'binomial': '#3fb950',   # Green — Binomial
    'mc':       '#f0883e',   # Orange — Monte Carlo
    'american': '#bc8cff',   # Purple — American (special)
}

BG      = '#0d1117'
PANEL   = '#161b22'
GRID    = '#21262d'
TEXT    = '#e6edf3'
MUTED   = '#8b949e'
ZERO    = '#484f58'


def _style(ax, title='', xlabel='', ylabel='', fontsize=10):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(GRID)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if title:  ax.set_title(title, color=TEXT, fontsize=fontsize, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def dollar_fmt(x, _): return f'${x:.2f}'
def pct_fmt(x, _):    return f'{x:.1f}%'


# ---------------------------------------------------------------------------
# RUN ALL THREE MODELS
# ---------------------------------------------------------------------------

def run_all_models(
    params: OptionParams,
    binomial_steps: int = 200,
    mc_simulations: int = 100_000,
    include_american: bool = True
) -> dict:
    """
    Run BS, Binomial (European + American), and Monte Carlo on same inputs.
    Returns a structured dict of all results for comparison and plotting.
    """
    print(f"  Running Black-Scholes...", end=' ', flush=True)
    bs_result  = price_option(params)
    print(f"${bs_result.price:.4f}")

    print(f"  Running Binomial Tree ({binomial_steps} steps)...", end=' ', flush=True)
    bin_euro   = binomial_tree_price(params, steps=binomial_steps, exercise_style='european')
    print(f"${bin_euro.price:.4f}", end='')

    bin_amer = None
    if include_american:
        bin_amer = binomial_tree_price(params, steps=binomial_steps, exercise_style='american')
        print(f"  |  American: ${bin_amer.price:.4f}")
    else:
        print()

    print(f"  Running Monte Carlo ({mc_simulations:,} sims)...", end=' ', flush=True)
    mc_result  = monte_carlo_price(params, n_simulations=mc_simulations, n_steps=50,
                                   variance_reduction='antithetic', seed=42)
    print(f"${mc_result.price:.4f}  ±${mc_result.std_error:.4f}")

    return {
        'params':    params,
        'bs':        bs_result,
        'binomial':  bin_euro,
        'american':  bin_amer,
        'mc':        mc_result,
    }


# ---------------------------------------------------------------------------
# MASTER COMPARISON DASHBOARD
# ---------------------------------------------------------------------------

def plot_model_comparison(results: dict, save_path: str = None):
    """
    7-panel professional comparison dashboard.

    Layout:
    ┌──────────────┬──────────────┬──────────────┐
    │  Price Card  │  Price Card  │  Price Card  │  Row 0: Model price cards
    │  (BS)        │  (Binomial)  │  (MC)        │
    ├──────────────┴──────────────┴──────────────┤
    │         Price Comparison Bar Chart          │  Row 1
    ├────────────────────┬────────────────────────┤
    │  Binomial          │  Monte Carlo           │  Row 2
    │  Convergence       │  Convergence + Paths   │
    ├────────────────────┬────────────────────────┤
    │  Greeks Comparison │  Payoff + All models   │  Row 3
    └────────────────────┴────────────────────────┘
    """
    params = results['params']
    bs     = results['bs']
    binom  = results['binomial']
    amer   = results['american']
    mc     = results['mc']

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        4, 6,
        figure=fig,
        hspace=0.55, wspace=0.45,
        height_ratios=[0.8, 1.2, 1.4, 1.4]
    )

    # ── Row 0: Price cards ─────────────────────────────────────────────────
    _draw_price_card(fig, gs[0, 0:2], 'BLACK-SCHOLES', 'Analytical (Exact)',
                     bs.price, bs.greeks.delta, bs.greeks.gamma, bs.greeks.vega,
                     bs.greeks.theta, MODEL_COLORS['bs'], 'European only')

    _draw_price_card(fig, gs[0, 2:4], 'BINOMIAL TREE', f'{binom.steps}-step CRR',
                     binom.price, binom.delta, binom.gamma, None,
                     binom.theta, MODEL_COLORS['binomial'],
                     f'Err vs BS: ${binom.convergence_error:.4f}',
                     american_price=amer.price if amer else None)

    _draw_price_card(fig, gs[0, 4:6], 'MONTE CARLO', f'{mc.n_simulations:,} paths',
                     mc.price, None, None, None, None,
                     MODEL_COLORS['mc'],
                     f'95% CI: [${mc.confidence_interval[0]:.3f}, ${mc.confidence_interval[1]:.3f}]',
                     std_error=mc.std_error)

    # ── Row 1: Price comparison bar ────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :])
    _plot_price_comparison(ax_bar, bs, binom, mc, amer, params)

    # ── Row 2: Convergence panels ──────────────────────────────────────────
    ax_bconv = fig.add_subplot(gs[2, :3])
    ax_mc    = fig.add_subplot(gs[2, 3:])
    _plot_binomial_convergence(ax_bconv, params)
    _plot_mc_paths_and_convergence(ax_mc, mc, params)

    # ── Row 3: Greeks + Payoff ─────────────────────────────────────────────
    ax_greeks = fig.add_subplot(gs[3, :3])
    ax_payoff = fig.add_subplot(gs[3, 3:])
    _plot_greeks_comparison(ax_greeks, params)
    _plot_payoff_comparison(ax_payoff, bs, binom, mc, params)

    # ── Master title ───────────────────────────────────────────────────────
    type_str = params.option_type.upper()
    fig.suptitle(
        f"OPTIONS VALUATION ENGINE  ·  MODEL COMPARISON  ·  "
        f"{type_str}   S=${params.S}   K=${params.K}   "
        f"T={params.T*365:.0f}d   σ={params.sigma*100:.0f}%   r={params.r*100:.1f}%",
        color=TEXT, fontsize=11, fontweight='bold', y=1.002
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=BG, edgecolor='none')
        print(f"  Saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# HELPER PANELS
# ---------------------------------------------------------------------------

def _draw_price_card(fig, gs_slot, model_name, subtitle, price,
                     delta, gamma, vega, theta, color, note,
                     american_price=None, std_error=None):
    """Render a model price card with key stats."""
    ax = fig.add_subplot(gs_slot)
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # Colored top bar
    bar = FancyBboxPatch((0, 0.85), 1, 0.15,
                          boxstyle="round,pad=0.01",
                          facecolor=color, alpha=0.25, transform=ax.transAxes,
                          clip_on=False)
    ax.add_patch(bar)

    ax.text(0.5, 0.93, model_name, transform=ax.transAxes,
            ha='center', va='center', color=color,
            fontsize=11, fontweight='bold', fontfamily='monospace')
    ax.text(0.5, 0.80, subtitle, transform=ax.transAxes,
            ha='center', va='center', color=MUTED, fontsize=8)

    # Main price
    ax.text(0.5, 0.62, f'${price:.4f}', transform=ax.transAxes,
            ha='center', va='center', color=TEXT,
            fontsize=20, fontweight='bold')

    # Greeks grid
    greek_rows = []
    if delta is not None: greek_rows.append(('Δ', f'{delta:+.4f}'))
    if gamma is not None: greek_rows.append(('Γ', f'{gamma:.4f}'))
    if vega  is not None: greek_rows.append(('ν', f'{vega:.4f}'))
    if theta is not None: greek_rows.append(('Θ', f'{theta:.4f}'))

    y0 = 0.48
    for label, val in greek_rows:
        ax.text(0.28, y0, label, transform=ax.transAxes,
                ha='right', color=MUTED, fontsize=9, fontfamily='monospace')
        ax.text(0.32, y0, val, transform=ax.transAxes,
                ha='left', color=TEXT, fontsize=9, fontfamily='monospace')
        y0 -= 0.10

    # Special fields
    if american_price is not None:
        ax.text(0.5, y0 - 0.03, f'American: ${american_price:.4f}',
                transform=ax.transAxes, ha='center', color=MODEL_COLORS['american'],
                fontsize=8.5, fontweight='bold')

    if std_error is not None:
        ax.text(0.5, y0 - 0.03, f'Std Error: ±${std_error:.5f}',
                transform=ax.transAxes, ha='center', color=MODEL_COLORS['mc'],
                fontsize=8.5)

    # Note at bottom
    ax.text(0.5, 0.04, note, transform=ax.transAxes,
            ha='center', color=MUTED, fontsize=7.5, style='italic')

    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.0)
        spine.set_visible(True)


def _plot_price_comparison(ax, bs, binom, mc, amer, params):
    """Horizontal bar chart comparing all model prices."""
    models  = ['Black-Scholes\n(Analytical)', f'Binomial Tree\n({binom.steps} steps)',
               'Monte Carlo\n(Antithetic)']
    prices  = [bs.price, binom.price, mc.price]
    colors  = [MODEL_COLORS['bs'], MODEL_COLORS['binomial'], MODEL_COLORS['mc']]

    if amer is not None:
        models.append(f'Binomial Tree\n(American)')
        prices.append(amer.price)
        colors.append(MODEL_COLORS['american'])

    y_pos = np.arange(len(models))
    bars  = ax.barh(y_pos, prices, color=colors, alpha=0.85, height=0.5)

    # Add value labels on bars
    for bar, price in zip(bars, prices):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'${price:.4f}', va='center', ha='left',
                color=TEXT, fontsize=9, fontweight='bold')

    # MC confidence interval
    ci = mc.confidence_interval
    mc_y = 2 if amer is None else 2
    ax.barh(mc_y, ci[1] - ci[0], left=ci[0],
            color=MODEL_COLORS['mc'], alpha=0.2, height=0.5)
    ax.text(ci[0], mc_y - 0.35, f'95% CI: [${ci[0]:.3f}, ${ci[1]:.3f}]',
            color=MODEL_COLORS['mc'], fontsize=7.5, alpha=0.8)

    # Reference line at BS price
    ax.axvline(bs.price, color=MODEL_COLORS['bs'], lw=1.0,
               linestyle='--', alpha=0.5, label=f'BS Reference: ${bs.price:.4f}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, color=TEXT, fontsize=9)
    ax.legend(loc='lower right', framealpha=0.2, fontsize=8,
              labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)
    _style(ax, title='Price Comparison — All Models', xlabel='Option Price ($)')
    ax.xaxis.set_major_formatter(FuncFormatter(dollar_fmt))

    # Tight x range around prices
    all_p = prices + [ci[0], ci[1]]
    margin = (max(all_p) - min(all_p)) * 0.5 + 0.05
    ax.set_xlim(max(0, min(all_p) - margin), max(all_p) + margin * 3)


def _plot_binomial_convergence(ax, params):
    """Binomial price convergence to BS as N increases."""
    data = convergence_analysis(params, max_steps=300)

    ax.plot(data['steps'], data['prices'], color=MODEL_COLORS['binomial'],
            lw=1.5, alpha=0.8, label='Binomial Price')
    ax.axhline(data['bs_price'], color=MODEL_COLORS['bs'],
               lw=2.0, linestyle='--', label=f'BS Price ${data["bs_price"]:.4f}')

    # Highlight convergence band ±$0.01
    ax.fill_between(data['steps'],
                    data['bs_price'] - 0.01, data['bs_price'] + 0.01,
                    color=MODEL_COLORS['bs'], alpha=0.08, label='±$0.01 band')

    # Even/odd oscillation annotation
    even_mask = data['steps'] % 2 == 0
    odd_mask  = ~even_mask
    ax.scatter(data['steps'][even_mask], data['prices'][even_mask],
               s=6, color=MODEL_COLORS['binomial'], alpha=0.4)

    ax.legend(framealpha=0.2, fontsize=8, labelcolor=TEXT,
              facecolor=PANEL, edgecolor=GRID)
    _style(ax, title='Binomial Tree Convergence to Black-Scholes',
           xlabel='Number of Steps (N)', ylabel='Option Price ($)')
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))


def _plot_mc_paths_and_convergence(ax, mc_result, params):
    """Monte Carlo simulated price paths."""
    paths = mc_result.paths_sample
    T     = params.T
    n_steps_viz = paths.shape[1] - 1
    time  = np.linspace(0, T * 365, n_steps_viz + 1)

    # Plot sample paths
    for i, path in enumerate(paths):
        alpha = 0.15 if i < len(paths) - 1 else 0.6
        lw    = 0.7  if i < len(paths) - 1 else 1.5
        color = MODEL_COLORS['mc'] if path[-1] > params.K else MUTED
        ax.plot(time, path, color=color, lw=lw, alpha=alpha)

    # Strike line
    ax.axhline(params.K, color=MODEL_COLORS['bs'], lw=1.5,
               linestyle='--', alpha=0.8, label=f'Strike K=${params.K}')
    ax.axhline(params.S, color=TEXT, lw=1.0,
               linestyle=':', alpha=0.5, label=f'Spot S=${params.S}')

    # Shade ITM region
    ax.fill_between([0, T * 365], [params.K, params.K], [params.S * 2, params.S * 2],
                    alpha=0.04, color=MODEL_COLORS['mc'])

    # Price annotation
    ax.text(0.97, 0.97,
            f'MC Price: ${mc_result.price:.4f}\n'
            f'SE: ±${mc_result.std_error:.4f}\n'
            f'N={mc_result.n_simulations:,}',
            transform=ax.transAxes, ha='right', va='top',
            color=MODEL_COLORS['mc'], fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL,
                      edgecolor=MODEL_COLORS['mc'], alpha=0.8))

    ax.legend(framealpha=0.2, fontsize=8, labelcolor=TEXT,
              facecolor=PANEL, edgecolor=GRID)
    _style(ax, title=f'Monte Carlo — {len(paths)} Sample Price Paths',
           xlabel='Days to Expiry', ylabel='Stock Price ($)')
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))


def _plot_greeks_comparison(ax, params):
    """Delta comparison: BS (analytical) vs Binomial (numerical)."""
    S_range = np.linspace(params.S * 0.5, params.S * 1.8, 80)

    bs_deltas, bin_deltas = [], []

    from black_scholes import compute_greeks

    for s in S_range:
        p = OptionParams(S=s, K=params.K, T=params.T, r=params.r,
                         sigma=params.sigma, q=params.q, option_type=params.option_type)
        bs_deltas.append(compute_greeks(p).delta)
        b = binomial_tree_price(p, steps=50, exercise_style='european')
        bin_deltas.append(b.delta)

    ax.plot(S_range, bs_deltas,  color=MODEL_COLORS['bs'],
            lw=2.5, label='BS Delta (analytical)', zorder=3)
    ax.plot(S_range, bin_deltas, color=MODEL_COLORS['binomial'],
            lw=1.5, linestyle='--', label='Binomial Δ (numerical)', alpha=0.8, zorder=2)

    ax.axvline(params.K, color=MODEL_COLORS['mc'], lw=0.8, linestyle=':', alpha=0.6)
    ax.axvline(params.S, color=TEXT,               lw=0.8, linestyle=':', alpha=0.5)
    ax.axhline(0, color=ZERO, lw=0.5)
    ax.axhline(0.5 if params.option_type == 'call' else -0.5,
               color=ZERO, lw=0.4, linestyle=':', alpha=0.4)

    ax.legend(framealpha=0.2, fontsize=8, labelcolor=TEXT,
              facecolor=PANEL, edgecolor=GRID)
    _style(ax, title='Delta: Analytical (BS) vs Numerical (Binomial)',
           xlabel='Stock Price ($)', ylabel='Delta (Δ)')


def _plot_payoff_comparison(ax, bs, binom, mc, params):
    """Payoff diagram with all three model prices marked."""
    S_range = np.linspace(params.S * 0.4, params.S * 2.0, 400)

    if params.option_type == 'call':
        payoff = np.maximum(S_range - params.K, 0)
    else:
        payoff = np.maximum(params.K - S_range, 0)

    # Net P&L using BS price as cost basis
    net_pnl = payoff - bs.price

    ax.fill_between(S_range, net_pnl, 0,
                    where=(net_pnl >= 0), color=MODEL_COLORS['mc'], alpha=0.10)
    ax.fill_between(S_range, net_pnl, 0,
                    where=(net_pnl < 0),  color='#ff7b72', alpha=0.08)
    ax.plot(S_range, net_pnl, color=TEXT, lw=2.0, label='Net P&L at expiry')
    ax.axhline(0, color=ZERO, lw=0.8)

    # Mark each model's price as horizontal lines
    for label, price, color in [
        ('BS',       bs.price,    MODEL_COLORS['bs']),
        ('Binomial', binom.price, MODEL_COLORS['binomial']),
        ('MC',       mc.price,    MODEL_COLORS['mc']),
    ]:
        ax.axhline(-price, color=color, lw=1.2, linestyle='--',
                   alpha=0.7, label=f'{label} Cost: ${price:.4f}')

    ax.axvline(params.K, color=MODEL_COLORS['mc'], lw=0.8, linestyle=':', alpha=0.5)
    ax.axvline(params.S, color=TEXT, lw=0.8, linestyle=':', alpha=0.4)

    ax.legend(framealpha=0.2, fontsize=8, labelcolor=TEXT,
              facecolor=PANEL, edgecolor=GRID, loc='upper left'
              if params.option_type == 'put' else 'upper right')
    _style(ax, title='Payoff at Expiry — Model Cost Comparison',
           xlabel='Stock Price at Expiry ($)', ylabel='P&L ($)')
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))


# ---------------------------------------------------------------------------
# PRINTED COMPARISON TABLE
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict):
    """Console output — clean side-by-side model comparison table."""
    bs    = results['bs']
    binom = results['binomial']
    mc    = results['mc']
    amer  = results['american']
    p     = results['params']

    print(f"\n{'═'*72}")
    print(f"  MODEL COMPARISON TABLE")
    print(f"  {p.option_type.upper()} | S=${p.S} | K=${p.K} | T={p.T*365:.0f}d | σ={p.sigma*100:.0f}% | r={p.r*100:.1f}%")
    print(f"{'═'*72}")
    print(f"  {'Metric':<24} {'Black-Scholes':>14} {'Binomial':>14} {'Monte Carlo':>14}")
    print(f"  {'':─<24} {'':─>14} {'':─>14} {'':─>14}")
    print(f"  {'Price':<24} {'${:.4f}'.format(bs.price):>14} {'${:.4f}'.format(binom.price):>14} {'${:.4f}'.format(mc.price):>14}")
    print(f"  {'Method':<24} {'Closed-form':>14} {'Lattice':>14} {'Simulation':>14}")
    print(f"  {'Steps / Sims':<24} {'N/A':>14} {str(binom.steps):>14} {'{:,}'.format(mc.n_simulations):>14}")
    print(f"  {'Error vs BS':<24} {'—':>14} {'${:.5f}'.format(binom.convergence_error):>14} {'${:.5f}'.format(mc.convergence_error):>14}")
    print(f"  {'Std Error':<24} {'—':>14} {'—':>14} {'±${:.5f}'.format(mc.std_error):>14}")
    print(f"  {'Delta (Δ)':<24} {'{:+.6f}'.format(bs.greeks.delta):>14} {'{:+.6f}'.format(binom.delta):>14} {'N/A':>14}")
    print(f"  {'Gamma (Γ)':<24} {'{:.6f}'.format(bs.greeks.gamma):>14} {'{:.6f}'.format(binom.gamma):>14} {'N/A':>14}")
    print(f"  {'Theta (Θ/day)':<24} {'{:+.6f}'.format(bs.greeks.theta):>14} {'{:+.6f}'.format(binom.theta):>14} {'N/A':>14}")
    print(f"  {'Vega (ν/1%)':<24} {'{:+.6f}'.format(bs.greeks.vega):>14} {'N/A':>14} {'N/A':>14}")
    print(f"  {'American Price':<24} {'N/A (EU only)':>14} {'${:.4f}'.format(amer.price) if amer else 'N/A':>14} {'N/A':>14}")
    if amer:
        print(f"  {'Early Ex. Premium':<24} {'—':>14} {'${:.4f}'.format(amer.early_exercise_premium):>14} {'—':>14}")
    print(f"{'─'*72}")
    print(f"  {'Handles American?':<24} {'✗':>14} {'✓':>14} {'✓ (approx)':>14}")
    print(f"  {'Path-dependent?':<24} {'✗':>14} {'✗':>14} {'✓':>14}")
    print(f"  {'Speed':<24} {'Instant':>14} {'Fast':>14} {'Moderate':>14}")
    print(f"  {'Exact for European?':<24} {'✓ Yes':>14} {'Converges':>14} {'Converges':>14}")
    print(f"{'═'*72}\n")
