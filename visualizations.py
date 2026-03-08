"""
Payoff & Greeks Visualization
==============================
Phase 1, Module 3: Professional-grade charts for the options engine.

Charts produced:
1. Payoff at expiry (P&L diagram) — the classic "hockey stick"
2. Greeks vs spot — how each Greek varies across stock prices
3. Option price vs spot (theoretical value curve)
4. Theta decay — price erosion over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from black_scholes import OptionParams, price_option, black_scholes_price, compute_greeks


# ---------------------------------------------------------------------------
# STYLE CONFIGURATION — consistent professional look
# ---------------------------------------------------------------------------

STYLE = {
    'bg':          '#0d1117',    # Dark background (quant terminal aesthetic)
    'panel':       '#161b22',
    'grid':        '#21262d',
    'text':        '#e6edf3',
    'accent1':     '#58a6ff',    # Blue — calls
    'accent2':     '#ff7b72',    # Red — puts
    'accent3':     '#3fb950',    # Green — profit
    'accent4':     '#f0883e',    # Orange — warning
    'breakeven':   '#8b949e',
    'zero_line':   '#484f58',
}


def _apply_style(ax, title='', xlabel='', ylabel=''):
    """Apply consistent dark quant terminal style to an axis."""
    ax.set_facecolor(STYLE['panel'])
    ax.tick_params(colors=STYLE['text'], labelsize=9)
    ax.spines['bottom'].set_color(STYLE['grid'])
    ax.spines['left'].set_color(STYLE['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_color(STYLE['text'])
    ax.yaxis.label.set_color(STYLE['text'])
    if title:  ax.set_title(title, color=STYLE['text'], fontsize=11, fontweight='bold', pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=STYLE['text'], fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=STYLE['text'], fontsize=9)
    ax.grid(True, color=STYLE['grid'], linewidth=0.5, alpha=0.8)


def dollar_fmt(x, pos):
    return f'${x:.2f}'

def pct_fmt(x, pos):
    return f'{x:.2f}'


# ---------------------------------------------------------------------------
# 1. PAYOFF DIAGRAM
# ---------------------------------------------------------------------------

def plot_payoff_diagram(params: OptionParams, ax=None, premium_paid: float = None):
    """
    Plot the classic hockey-stick payoff diagram.

    Shows:
    - Payoff at expiry (gross): max(S_T - K, 0) for calls
    - Net P&L: payoff minus premium paid (if provided)
    - Breakeven point(s)
    - Current spot price marker

    The payoff diagram is the most fundamental options visualization:
    it shows profit/loss at EVERY possible terminal stock price.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(STYLE['bg'])

    # Spot price range: 50% to 200% of current spot
    S_range = np.linspace(params.S * 0.4, params.S * 2.0, 500)

    if params.option_type == 'call':
        payoff = np.maximum(S_range - params.K, 0)
        color  = STYLE['accent1']
        label  = f"Call Payoff at Expiry"
    else:
        payoff = np.maximum(params.K - S_range, 0)
        color  = STYLE['accent2']
        label  = f"Put Payoff at Expiry"

    # Net P&L (payoff minus cost of option)
    premium = premium_paid if premium_paid else black_scholes_price(params)
    net_pnl = payoff - premium

    # Fill profitable / loss zones
    ax.fill_between(S_range, net_pnl, 0,
                    where=(net_pnl >= 0), color=STYLE['accent3'], alpha=0.15, label='Profit zone')
    ax.fill_between(S_range, net_pnl, 0,
                    where=(net_pnl < 0),  color=STYLE['accent2'], alpha=0.10, label='Loss zone')

    # Plot lines
    ax.plot(S_range, payoff,  color=color,             lw=1.5, alpha=0.5,  linestyle='--', label='Gross payoff')
    ax.plot(S_range, net_pnl, color=color,             lw=2.5,             label=f'Net P&L (premium: ${premium:.2f})')
    ax.axhline(0,                                       color=STYLE['zero_line'], lw=1.0, linestyle='-')

    # Strike line
    ax.axvline(params.K, color=STYLE['accent4'], lw=1.0, linestyle='--', alpha=0.7, label=f'Strike K=${params.K:.0f}')

    # Current spot
    current_pnl = (max(params.S - params.K, 0) if params.option_type == 'call'
                   else max(params.K - params.S, 0)) - premium
    ax.axvline(params.S, color=STYLE['text'], lw=1.0, linestyle=':', alpha=0.6, label=f'Spot S=${params.S:.0f}')
    ax.scatter([params.S], [current_pnl], color=STYLE['text'], s=60, zorder=5)

    # Breakeven label
    if params.option_type == 'call':
        be = params.K + premium
        ax.axvline(be, color=STYLE['accent3'], lw=1.0, linestyle=':', alpha=0.7, label=f'Breakeven=${be:.2f}')
    else:
        be = params.K - premium
        ax.axvline(be, color=STYLE['accent3'], lw=1.0, linestyle=':', alpha=0.7, label=f'Breakeven=${be:.2f}')

    _apply_style(ax,
        title=f"{params.option_type.upper()} Option — Payoff at Expiry",
        xlabel="Stock Price at Expiry ($)",
        ylabel="Profit / Loss ($)"
    )
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))
    ax.legend(loc='upper left' if params.option_type == 'put' else 'upper right',
              framealpha=0.2, fontsize=8, labelcolor=STYLE['text'],
              facecolor=STYLE['panel'], edgecolor=STYLE['grid'])

    return ax


# ---------------------------------------------------------------------------
# 2. GREEKS vs SPOT
# ---------------------------------------------------------------------------

def plot_greeks_vs_spot(params: OptionParams, fig=None):
    """
    4-panel chart showing Delta, Gamma, Vega, Theta vs stock price.

    This is essential for understanding how the option's risk profile
    changes across different spot prices — used daily on trading desks.
    """
    S_range = np.linspace(params.S * 0.5, params.S * 1.8, 200)

    deltas, gammas, vegas, thetas = [], [], [], []
    for s in S_range:
        p = OptionParams(S=s, K=params.K, T=params.T, r=params.r,
                         sigma=params.sigma, q=params.q, option_type=params.option_type)
        g = compute_greeks(p)
        deltas.append(g.delta)
        gammas.append(g.gamma)
        vegas.append(g.vega)
        thetas.append(g.theta)

    if fig is None:
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor(STYLE['bg'])

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]

    greek_data = [
        (deltas, 'Δ Delta', STYLE['accent1'],  'Rate of change of price w.r.t. spot'),
        (gammas, 'Γ Gamma', STYLE['accent3'],  'Rate of change of delta w.r.t. spot'),
        (vegas,  'ν Vega',  STYLE['accent4'],  'Sensitivity to 1% vol change ($)'),
        (thetas, 'Θ Theta', STYLE['accent2'],  'Daily time decay ($)'),
    ]

    for ax, (data, name, color, subtitle) in zip(axes, greek_data):
        ax.plot(S_range, data, color=color, lw=2.0)
        ax.axvline(params.K, color=STYLE['accent4'], lw=0.8, linestyle='--', alpha=0.5, label=f'K={params.K}')
        ax.axvline(params.S, color=STYLE['text'],    lw=0.8, linestyle=':',  alpha=0.5, label=f'S={params.S}')
        ax.axhline(0, color=STYLE['zero_line'], lw=0.5)

        # Mark current value
        current_greek = data[np.argmin(np.abs(S_range - params.S))]
        ax.scatter([params.S], [current_greek], color=color, s=50, zorder=5)
        ax.annotate(f'{current_greek:.4f}',
                    xy=(params.S, current_greek),
                    xytext=(8, 0), textcoords='offset points',
                    color=color, fontsize=8)

        _apply_style(ax, title=f'{name}  [{subtitle}]',
                     xlabel='Stock Price ($)', ylabel=name)

    fig.suptitle(
        f"Greeks Profile — {params.option_type.upper()} | "
        f"K=${params.K} | T={params.T*365:.0f}d | σ={params.sigma*100:.0f}%",
        color=STYLE['text'], fontsize=13, fontweight='bold', y=0.98
    )
    return fig


# ---------------------------------------------------------------------------
# 3. THETA DECAY — price erosion over time
# ---------------------------------------------------------------------------

def plot_theta_decay(params: OptionParams, ax=None):
    """
    Show how option price decays as time to expiry decreases.

    Key insight: Theta decay is NOT linear — it accelerates near expiry.
    This is the "gamma risk / theta reward" tradeoff that options sellers exploit.

    We show multiple curves for different moneyness levels (ITM, ATM, OTM).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(STYLE['bg'])

    days_range = np.linspace(1, params.T * 365, 200)
    T_range    = days_range / 365

    # ATM (current strike)
    strikes     = [params.K * 0.85, params.K, params.K * 1.15]
    labels      = [f'OTM  K=${params.K*0.85:.0f}', f'ATM  K=${params.K:.0f}', f'ITM  K=${params.K*1.15:.0f}'] \
                  if params.option_type == 'call' else \
                  [f'ITM  K=${params.K*0.85:.0f}', f'ATM  K=${params.K:.0f}', f'OTM  K=${params.K*1.15:.0f}']
    colors      = [STYLE['breakeven'], STYLE['accent1'], STYLE['accent3']]

    for K, lbl, col in zip(strikes, labels, colors):
        prices = []
        for T in T_range:
            p = OptionParams(S=params.S, K=K, T=T, r=params.r,
                             sigma=params.sigma, q=params.q, option_type=params.option_type)
            prices.append(black_scholes_price(p))
        ax.plot(days_range, prices, color=col, lw=2.0, label=lbl)

    # Mark current T
    ax.axvline(params.T * 365, color=STYLE['text'], lw=1.0, linestyle=':', alpha=0.6,
               label=f'Current T={params.T*365:.0f}d')

    _apply_style(ax,
        title='Theta Decay — Option Price vs. Days to Expiry',
        xlabel='Days to Expiry',
        ylabel='Option Price ($)'
    )
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))
    ax.legend(framealpha=0.2, fontsize=9, labelcolor=STYLE['text'],
              facecolor=STYLE['panel'], edgecolor=STYLE['grid'])
    ax.invert_xaxis()   # Time flows left to right → expiry is on the right

    return ax


# ---------------------------------------------------------------------------
# 4. MASTER DASHBOARD
# ---------------------------------------------------------------------------

def plot_full_dashboard(params: OptionParams, save_path: str = None):
    """
    Full 5-panel quant dashboard for a single option.

    Layout:
    ┌─────────────────┬─────────────────┐
    │  Payoff Diagram │  Theta Decay    │
    ├─────┬─────┬─────┴─────────────────┤
    │ Δ   │ Γ   │     ν     │    Θ      │
    └─────┴─────┴───────────┴───────────┘
    """
    result = price_option(params)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(STYLE['bg'])
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # Row 1
    ax_payoff = fig.add_subplot(gs[0, :2])
    ax_theta  = fig.add_subplot(gs[0, 2:])

    # Row 2
    ax_delta  = fig.add_subplot(gs[1, 0])
    ax_gamma  = fig.add_subplot(gs[1, 1])
    ax_vega   = fig.add_subplot(gs[1, 2])
    ax_thetag = fig.add_subplot(gs[1, 3])

    plot_payoff_diagram(params, ax=ax_payoff)
    plot_theta_decay(params,    ax=ax_theta)

    # Greeks vs spot (manual — reuse axes)
    S_range = np.linspace(params.S * 0.5, params.S * 1.8, 200)
    greek_arrays = {'delta': [], 'gamma': [], 'vega': [], 'theta': []}
    for s in S_range:
        p = OptionParams(S=s, K=params.K, T=params.T, r=params.r,
                         sigma=params.sigma, q=params.q, option_type=params.option_type)
        g = compute_greeks(p)
        greek_arrays['delta'].append(g.delta)
        greek_arrays['gamma'].append(g.gamma)
        greek_arrays['vega'].append(g.vega)
        greek_arrays['theta'].append(g.theta)

    greek_axes = [
        (ax_delta,  greek_arrays['delta'],  'Δ Delta',  STYLE['accent1']),
        (ax_gamma,  greek_arrays['gamma'],  'Γ Gamma',  STYLE['accent3']),
        (ax_vega,   greek_arrays['vega'],   'ν Vega',   STYLE['accent4']),
        (ax_thetag, greek_arrays['theta'],  'Θ Theta',  STYLE['accent2']),
    ]
    for ax, data, name, color in greek_axes:
        ax.plot(S_range, data, color=color, lw=2.0)
        ax.axvline(params.K, color=STYLE['accent4'], lw=0.7, linestyle='--', alpha=0.5)
        ax.axvline(params.S, color=STYLE['text'],    lw=0.7, linestyle=':',  alpha=0.5)
        ax.axhline(0, color=STYLE['zero_line'], lw=0.5)
        _apply_style(ax, title=name, xlabel='S ($)', ylabel=name)

    # Title bar with pricing summary
    fig.suptitle(
        f"OPTIONS VALUATION ENGINE  |  {params.option_type.upper()}  "
        f"S=${params.S}  K=${params.K}  T={params.T*365:.0f}d  "
        f"σ={params.sigma*100:.0f}%  r={params.r*100:.1f}%  "
        f"→  Price: ${result.price:.4f}  "
        f"Δ={result.greeks.delta:+.4f}  "
        f"Γ={result.greeks.gamma:.4f}  "
        f"ν={result.greeks.vega:.4f}  "
        f"Θ={result.greeks.theta:.4f}",
        color=STYLE['text'], fontsize=10, fontweight='bold', y=1.01
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=STYLE['bg'], edgecolor='none')
        print(f"  Dashboard saved → {save_path}")

    return fig
