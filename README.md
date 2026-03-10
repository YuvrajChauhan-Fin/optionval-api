# OptionVal — Quantitative Options Pricing Engine

> A full-stack, investment-grade options valuation terminal. Live market data. Three pricing models. Full Greeks. Built from first principles.

🌐 **Live:** [react-ui-eight-amber.vercel.app](https://react-ui-eight-amber.vercel.app)
📦 **API:** [optionval-api.onrender.com](https://optionval-api.onrender.com)

---

## What Is This?

OptionVal is a production-deployed options pricing application that implements the three core models used by derivatives desks — Black-Scholes-Merton, Cox-Ross-Rubinstein Binomial Tree, and Monte Carlo simulation — against live US market data, with full analytical Greeks, an implied volatility solver, and an interactive parameter interface.

It was built to be **mathematically defensible** and **explainable** — not a black box. Every number is traceable to a formula. Every formula is documented. Every result has been independently verified.

---

## Live Demo

Load any US ticker → get live price, historical volatility, options chain → click any contract → all three models price it simultaneously → move the sliders → watch everything reprice in real time.

```
Ticker: AAPL  →  Spot: $259.88  →  30D HV: 28.5%
ATM Call K=260, T=22hrs
  Black-Scholes:   $1.4770   Δ=0.4939  Γ=0.1047  ν=0.0543  θ=−0.776
  Binomial CRR:    $1.4776
  Monte Carlo:     $1.4540   95% CI: [$1.431, $1.477]
```

---

## Models Implemented

### 1. Black-Scholes-Merton (1973)
*Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy. Nobel Prize 1997.*

Closed-form analytical solution for European options on dividend-paying stocks.

```
C = S·e^(−qT)·N(d₁) − K·e^(−rT)·N(d₂)
P = K·e^(−rT)·N(−d₂) − S·e^(−qT)·N(−d₁)

d₁ = [ln(S/K) + (r − q + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T
```

**Assumptions:** Constant volatility, log-normal returns, continuous frictionless trading, no dividends (adjusted via continuous yield q).

**Key limitation:** One flat σ for all strikes and expiries. The observed volatility skew — OTM puts consistently priced at higher IV than OTM calls — proves this assumption is wrong in practice. This is the entire motivation for the IV Smile and Vol Surface views.

---

### 2. Cox-Ross-Rubinstein Binomial Tree (1979)
*Cox, J., Ross, S., Rubinstein, M. (1979). Option Pricing: A Simplified Approach. Journal of Financial Economics.*

Discrete recombining lattice with 150 time steps. At each node, the stock can move up by factor `u` or down by `d`, with risk-neutral probability `p*`.

```
u = e^(σ√Δt),   d = 1/u
p* = (e^((r−q)Δt) − d) / (u − d)
```

**Why this matters over BS:** Handles **American-style early exercise**. At each node, the model checks: is it worth more to exercise now (intrinsic value) or hold (continuation value)? `V = max(intrinsic, continuation)`. The Early Exercise Premium (EEP) = Binomial − BS when positive.

**Convergence:** As steps → ∞, Binomial price converges to Black-Scholes. The Convergence tab visualises this oscillating convergence behaviour.

---

### 3. Monte Carlo Simulation — Geometric Brownian Motion (Boyle 1977)
*Boyle, P. (1977). Options: A Monte Carlo Approach. Journal of Financial Economics.*

Simulates 20,000 stock price paths under GBM, computes the average discounted payoff.

```
S(T) = S₀ · exp((r − q − σ²/2)T + σ√T · Z),   Z ~ N(0,1)
```

**Variance reduction:** Antithetic variates — for each random draw Z, a paired path with −Z is also simulated. This halves variance at zero additional cost, giving tighter confidence intervals.

**Output:** Price + 95% confidence interval + standard error. The CI width is a direct measure of simulation uncertainty — it narrows as 1/√N.

**Why this matters:** Natural extension to **path-dependent payoffs** — barrier options, Asian (average price) options, lookback options — which have no closed-form solution. Monte Carlo scales to any payoff function.

---

## Greeks Engine

All five Greeks are computed **analytically** from the Black-Scholes formula — not numerically approximated. This means they are exact, not estimates.

| Greek | Symbol | Formula | Interpretation |
|-------|--------|---------|----------------|
| Delta | Δ | ∂C/∂S = e^(−qT)·N(d₁) | Price change per $1 move in the stock |
| Gamma | Γ | ∂²C/∂S² = e^(−qT)·φ(d₁)/(Sσ√T) | Rate of change of Delta; convexity |
| Vega | ν | ∂C/∂σ = S·e^(−qT)·φ(d₁)·√T / 100 | Price change per 1% move in implied vol |
| Theta | θ | ∂C/∂t / 365 | Price decay per calendar day |
| Rho | ρ | ∂C/∂r = K·T·e^(−rT)·N(d₂) / 100 | Price change per 1% move in risk-free rate |

**Convention:** Vega and Rho are quoted per 1 percentage point move (Bloomberg convention), not per unit move.

**Verified properties:**
- Put-Call Parity: `C − P = S·e^(−qT) − K·e^(−rT)` — holds to 8 decimal places ✅
- Delta sum: `Δ_call + |Δ_put| = e^(−qT)` — holds exactly ✅
- Gamma equality: Call Γ = Put Γ for same strike and expiry ✅
- Vega equality: Call ν = Put ν for same strike and expiry ✅

---

## Implied Volatility Solver

Given a market price, the solver inverts the Black-Scholes formula to find the implied volatility — the market's expectation of future realised volatility embedded in the option price.

**Algorithm:** Newton-Raphson with Brenner-Subrahmanyam initial seed

```python
# Initial guess (Brenner & Subrahmanyam, 1988)
sigma_0 = sqrt(2π/T) · (price / S)

# Newton-Raphson iteration
sigma_{n+1} = sigma_n − (BS(sigma_n) − market_price) / vega(sigma_n)
```

Converges in 3-5 iterations for normal market conditions. Falls back to bisection for deep ITM/OTM contracts where Newton-Raphson may diverge.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                       │
│              (Vercel — react-ui-eight-amber.vercel.app) │
│                                                         │
│  ┌──────────────┐  ┌────────────────────────────────┐  │
│  │ Model Pricer │  │ Options Chain / IV Smile /     │  │
│  │ BS · CRR · MC│  │ Vol Surface / α Edge /         │  │
│  │ Greeks Panel │  │ Methodology                    │  │
│  │ Param Sliders│  └────────────────────────────────┘  │
│  └──────────────┘                                       │
└─────────────────────┬───────────────────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                         │
│              (Render — optionval-api.onrender.com)       │
│                                                         │
│  /api/quote/{ticker}    →  price, HV, market data       │
│  /api/chain/{ticker}    →  strikes, expiries, BS prices │
│  /api/surface/{ticker}  →  IV surface data              │
│  /api/compare/{ticker}  →  model comparison             │
│  /health                →  uptime check                 │
│                                                         │
│  Cache: quote 120s · chain 300s · surface 300s          │
└─────────────────────┬───────────────────────────────────┘
                      │ REST (Massive.com API)
┌─────────────────────▼───────────────────────────────────┐
│               Massive.com (Polygon.io)                   │
│                                                         │
│  /v2/aggs/...     →  OHLCV bars, price, historical vol  │
│  /v3/reference/options/contracts  →  strikes, expiries  │
│  /v3/reference/tickers/{sym}      →  company data       │
└─────────────────────────────────────────────────────────┘
```

**Stack:**
- **Backend:** Python 3.11 · FastAPI · Uvicorn · Requests · NumPy
- **Frontend:** React 18 · Recharts · Three.js (r128) · Vercel
- **Data:** Massive.com REST API (Polygon.io infrastructure)
- **Deployment:** Render (backend) · Vercel (frontend)

---

## Data Layer — What's Live vs Theoretical

| Data | Source | Status |
|------|--------|--------|
| Stock price (OHLCV) | Massive.com /v2/aggs | 🟢 LIVE |
| 30-day historical volatility | Computed from 30 daily log returns | 🟢 LIVE |
| Market cap, sector | Massive.com /v3/reference/tickers | 🟢 LIVE |
| Option strikes & expiries | Massive.com /v3/reference/options | 🟢 LIVE |
| BS/Binomial/MC prices | Computed client-side from live spot + HV | 🟡 THEORETICAL |
| Live bid/ask/mid quotes | Requires paid options data subscription | ⚪ NOT AVAILABLE |
| Implied volatility per strike | Requires live market quotes | ⚪ NOT AVAILABLE |

**Why theoretical pricing?** Live options bid/ask data requires a paid market data subscription (CBOE DataShop, Bloomberg, Refinitiv). The free tier provides contract structure (strikes, expiries) but not live quotes. The pricing models use live spot price + historical volatility as inputs — this is how you would price an option from first principles at a desk without a live options feed.

---

## API Endpoints

```
GET /health
→ { status: "ok", timestamp: "..." }

GET /api/quote/{ticker}
→ { ticker, name, sector, price, prev_close, change, change_pct,
    volume, market_cap, week52_high, week52_low, hist_vol_30d,
    risk_free_rate, dividend_yield, exchange, currency, timestamp }

GET /api/chain/{ticker}?expiry=2026-03-21
→ { ticker, spot, T, r, q, sigma, expiry, all_expiries,
    calls: [{ strike, expiry, type, bs_price, iv, mispricing,
              moneyness, greek_delta, greek_gamma, greek_vega,
              greek_theta, greek_rho }],
    puts: [...] }

GET /api/surface/{ticker}
→ { ticker, spot, surface: [{ strike, expiry, days, T, iv,
                               log_moneyness, moneyness }] }

GET /api/compare/{ticker}?expiry=2026-03-21
→ { ticker, spot, T, r, sigma, calls: [...], puts: [...] }
```

---

## Running Locally

### Backend

```bash
cd OptionVal_V1
pip install fastapi uvicorn requests numpy python-dotenv

# Create .env file
echo "MASSIVE_API_KEY=your_key_here" > .env

uvicorn api:app --reload --port 8000
# API available at http://localhost:8000
```

### Frontend

```bash
cd OptionVal_V1/react-ui
npm install
npm run dev
# App available at http://localhost:5173
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MASSIVE_API_KEY` | Massive.com (Polygon.io) API key |

---

## What We Set Out to Build — The Full Vision

The original scope was a four-phase project:

**Phase 1 — Core Python Engine** ✅ *Complete*
Modular, mathematically rigorous options pricing in Python. Black-Scholes as the foundation, built up formula by formula, Greek by Greek. Every component explainable. No black boxes.

**Phase 2 — Interactive Web UI** ✅ *Complete*
A professional-grade React frontend. Bloomberg terminal aesthetic. Real-time repricing on every slider move. Three models side by side. Options chain, Greeks panel, payoff diagrams, convergence charts.

**Phase 3 — Real Market Data Integration** ✅ *Partial*
Live US equity prices, historical volatility, market cap and sector data via Massive.com (Polygon.io infrastructure). Options contract structure (strikes, expiries) live from the API. **Gap:** Live bid/ask options quotes require a paid data subscription — model prices are therefore theoretical, computed from live spot price and historical vol.

**Phase 4 — Research Layer** 🔲 *Roadmap*
What would make this truly original and investment-grade:
- Local volatility surface calibration (Dupire 1994)
- Stochastic volatility — Heston model (1993)
- SABR model for interest rate derivatives
- Jump-diffusion — Merton (1976)
- Indian market integration (NSE — Nifty, Bank Nifty options)
- Backtesting framework for vol strategies (long vega, short gamma, etc.)
- P&L attribution by Greek

---

## What We Actually Achieved

In two development sessions, starting from zero:

✅ Three production-deployed pricing models with mathematically verified outputs  
✅ Full Greeks engine — analytically exact, verified to 6 decimal places  
✅ Newton-Raphson IV solver with Brenner-Subrahmanyam seed  
✅ Put-call parity verified to 8 decimal places  
✅ Live market data pipeline (price, historical vol, market cap, sector, options chain)  
✅ Interactive parameter interface — all 3 models reprice simultaneously on slider change  
✅ Options chain with 35 strikes × 4 expiries  
✅ α Edge tab — BS theoretical price vs market structure across all strikes  
✅ IV Smile tab — correctly identifies theoretical mode and explains the skew  
✅ Vol Surface tab — heatmap of strike × expiry × IV  
✅ Methodology documentation with academic citations  
✅ Full ErrorBoundary crash protection  
✅ Server-side caching (quotes 120s, chains 300s, surface 300s)  
✅ Clean REST API with 5 endpoints  
✅ Production deployment on Vercel + Render  

---

## Known Limitations

| Limitation | Reason | Fix Path |
|-----------|--------|----------|
| No live IV smile | Free tier has no options quotes | Paid CBOE/Polygon options feed |
| Vol Surface is flat | All contracts priced at same HV | Live quotes → IV per strike → real surface |
| US markets only | Massive.com is US-only | NSE data via NSEPy or Zerodha Kite API |
| No early exercise display | EEP not shown explicitly | Add EEP = Binomial − BS row in pricer |
| Monte Carlo flickering | Re-simulates on every render | useMemo on MC computation |

---

## Academic References

1. Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.

2. Merton, R.C. (1973). *Theory of Rational Option Pricing.* Bell Journal of Economics, 4(1), 141–183.

3. Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.* Journal of Financial Economics, 7(3), 229–263.

4. Boyle, P. (1977). *Options: A Monte Carlo Approach.* Journal of Financial Economics, 4(3), 323–338.

5. Brenner, M. & Subrahmanyam, M.G. (1988). *A Simple Formula to Compute the Implied Standard Deviation.* Financial Analysts Journal, 44(5), 80–83.

6. Dupire, B. (1994). *Pricing with a Smile.* Risk, 7(1), 18–20. *(Roadmap: Local Vol)*

7. Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies, 6(2), 327–343. *(Roadmap: Stochastic Vol)*

---

## Project Structure

```
OptionVal_V1/
├── api.py                    # FastAPI backend — all endpoints, BS engine, Greeks
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
│
└── react-ui/
    ├── src/
    │   └── App.jsx           # Full React app — all UI, models, charts
    ├── package.json
    └── index.html
```

---

## Author

**Yuvraj Chauhan**  
CFA Level II Candidate | Finance & Quant Strategy  
Former Specialist — Investment Valuation & Portfolio Analytics, Mercer (MMC Group)

[LinkedIn](https://linkedin.com/in/yuvrajchauhan) · [GitHub](https://github.com/YuvrajChauhan-Fin)

---

## Licence

MIT — use freely, attribution appreciated.

---

*Built with mathematical rigour. Every formula is in the Methodology tab. Every number is verifiable.*
