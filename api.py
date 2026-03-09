"""
Options Valuation Engine — FastAPI Backend
==========================================
Phase 3: Live Market Data Layer

Endpoints:
  GET /api/quote/{ticker}          — Live stock price + company info
  GET /api/chain/{ticker}          — Full options chain with IV + Greeks
  GET /api/surface/{ticker}        — Vol surface data across strikes/expiries
  GET /api/compare/{ticker}        — Model vs market mispricing analysis
  GET /api/search?q=               — Ticker search
  GET /health                      — Server health check

Market support:
  US  — Full options chain (AAPL, SPY, TSLA, etc.)
  NSE — Stock prices + limited options (RELIANCE.NS, INFY.NS)
  BSE — Stock prices (RELIANCE.BO)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import yfinance as yf
import numpy as np
import math
import time
from datetime import datetime, date
import traceback

app = FastAPI(
    title="Options Valuation Engine API",
    description="Phase 3: Live market data + pricing engine",
    version="3.0"
)

# Allow React dev server + Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# IN-MEMORY CACHE  (5-minute TTL)
# ---------------------------------------------------------------------------

_cache: dict = {}
CACHE_TTL = 300  # seconds

def cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry['ts']) < CACHE_TTL:
        return entry['data']
    return None

def cache_set(key: str, data):
    _cache[key] = {'data': data, 'ts': time.time()}

# ---------------------------------------------------------------------------
# BLACK-SCHOLES ENGINE (self-contained, no import dependency)
# ---------------------------------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_d1d2(S, K, T, r, sigma, q=0):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    return float(d1), float(d2)

def bs_price(S, K, T, r, sigma, q=0, option_type='call'):
    if T <= 0:
        return max(S-K, 0) if option_type=='call' else max(K-S, 0)
    d1, d2 = bs_d1d2(S, K, T, r, sigma, q)
    exp_q, exp_r = np.exp(-q*T), np.exp(-r*T)
    if option_type == 'call':
        return float(S*exp_q*norm_cdf(d1) - K*exp_r*norm_cdf(d2))
    return float(K*exp_r*norm_cdf(-d2) - S*exp_q*norm_cdf(-d1))

def bs_greeks(S, K, T, r, sigma, q=0, option_type='call'):
    if T <= 0.0001:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    d1, d2 = bs_d1d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    exp_q, exp_r = np.exp(-q*T), np.exp(-r*T)
    nd1 = norm_pdf(d1)
    delta = exp_q*norm_cdf(d1) if option_type=='call' else exp_q*(norm_cdf(d1)-1)
    gamma = exp_q*nd1 / (S*sigma*sqrt_T)
    vega  = S*exp_q*nd1*sqrt_T / 100
    t1    = -(S*exp_q*nd1*sigma) / (2*sqrt_T)
    t2    = (q*S*exp_q*norm_cdf(d1) - r*K*exp_r*norm_cdf(d2)) if option_type=='call' \
            else (-q*S*exp_q*norm_cdf(-d1) + r*K*exp_r*norm_cdf(-d2))
    theta = (t1+t2)/365
    rho   = K*T*exp_r*norm_cdf(d2)/100 if option_type=='call' \
            else -K*T*exp_r*norm_cdf(-d2)/100
    return {'delta':round(delta,6),'gamma':round(gamma,6),
            'vega':round(vega,6),'theta':round(theta,6),'rho':round(rho,6)}

def implied_vol(market_price, S, K, T, r, q=0, option_type='call'):
    """Newton-Raphson IV solver with Brent fallback."""
    if T <= 0 or market_price <= 0:
        return None
    # Initial guess: Brenner-Subrahmanyam
    sigma = np.sqrt(2*np.pi/T) * (market_price/S)
    sigma = np.clip(sigma, 0.01, 5.0)
    for _ in range(100):
        price = bs_price(S, K, T, r, sigma, q, option_type)
        vega  = bs_greeks(S, K, T, r, sigma, q, option_type)['vega'] * 100
        err   = price - market_price
        if abs(err) < 1e-7:
            return round(float(sigma), 6)
        if abs(vega) < 1e-10:
            break
        sigma -= err/vega
        sigma = max(sigma, 0.001)
    return None

# ---------------------------------------------------------------------------
# MARKET DETECTION
# ---------------------------------------------------------------------------

def detect_market(ticker: str) -> dict:
    t = ticker.upper()
    if t.endswith('.NS'):
        return {'market': 'NSE', 'currency': 'INR', 'flag': '🇮🇳', 'exchange': 'NSE'}
    elif t.endswith('.BO'):
        return {'market': 'BSE', 'currency': 'INR', 'flag': '🇮🇳', 'exchange': 'BSE'}
    else:
        return {'market': 'US', 'currency': 'USD', 'flag': '🇺🇸', 'exchange': 'NASDAQ/NYSE'}

def get_risk_free_rate(market: str) -> float:
    """Approximate risk-free rates by market."""
    rates = {'US': 0.0525, 'NSE': 0.067, 'BSE': 0.067}
    return rates.get(market, 0.05)

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "online", "timestamp": datetime.utcnow().isoformat(), "version": "3.0"}


@app.get("/api/quote/{ticker}")
def get_quote(ticker: str):
    """
    Live stock quote + company metadata.
    Returns everything needed to populate the UI header.
    """
    cache_key = f"quote:{ticker.upper()}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        time.sleep(2)
        t    = yf.Ticker(ticker.upper())
        info = t.info
        hist = t.history(period="2d")

        if hist.empty:
            raise HTTPException(404, f"No data found for ticker '{ticker}'")

        price      = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else price
        change     = price - prev_close
        change_pct = (change / prev_close) * 100

        market_info = detect_market(ticker)
        r           = get_risk_free_rate(market_info['market'])

        # 30-day historical volatility
        if len(hist) >= 20:
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            hist_vol    = float(log_returns.std() * np.sqrt(252))
        else:
            hist_vol = 0.25

        result = {
            "ticker":       ticker.upper(),
            "name":         info.get('longName', ticker.upper()),
            "sector":       info.get('sector', 'N/A'),
            "price":        round(price, 2),
            "prev_close":   round(prev_close, 2),
            "change":       round(change, 2),
            "change_pct":   round(change_pct, 3),
            "volume":       info.get('volume', 0),
            "market_cap":   info.get('marketCap', 0),
            "dividend_yield": round(min(max(float(info.get('dividendYield') or 0), 0), 0.20), 4),
            "hist_vol_30d": round(hist_vol, 4),
            "risk_free_rate": r,
            "currency":     market_info['currency'],
            "market":       market_info['market'],
            "exchange":     market_info['exchange'],
            "flag":         market_info['flag'],
            "timestamp":    datetime.utcnow().isoformat(),
        }
        cache_set(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR [quote/{ticker}]: {e}\n{tb}", flush=True)
        raise HTTPException(500, detail={"error": str(e), "traceback": tb})


@app.get("/api/chain/{ticker}")
def get_options_chain(ticker: str, expiry: str = None):
    """
    Full options chain with market prices, BS IV, and Greeks.

    For each contract:
    - Market price (last/mid)
    - Implied volatility (solved from market price)
    - BS theoretical price
    - Full Greeks
    - Moneyness classification
    """
    cache_key = f"chain:{ticker.upper()}:{expiry or ''}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        quote       = get_quote(ticker)
        time.sleep(2)
        t           = yf.Ticker(ticker.upper())
        S           = quote['price']
        r           = quote['risk_free_rate']
        q           = quote['dividend_yield']
        expiries    = t.options

        if not expiries:
            raise HTTPException(404, f"No options data available for {ticker}. "
                              f"Note: Indian market options may have limited coverage.")

        # Use requested expiry or nearest one
        target_exp  = expiry if expiry in expiries else expiries[0]
        chain       = t.option_chain(target_exp)

        # Days to expiry
        exp_date    = datetime.strptime(target_exp, '%Y-%m-%d')
        T           = max((exp_date - datetime.utcnow()).days / 365, 0.001)

        def process_chain(df, option_type):
            results = []
            for _, row in df.iterrows():
                K           = float(row['strike'])
                market_px   = float(row.get('lastPrice') or 0)
                bid         = float(row.get('bid') or 0)
                ask         = float(row.get('ask') or 0)
                mid         = (bid+ask)/2 if bid>0 and ask>0 else market_px
                volume      = int(float(str(row.get('volume') or 0).replace('nan', '0')))
                oi          = int(float(str(row.get('openInterest') or 0).replace('nan', '0')))

                if mid <= 0 and market_px <= 0:
                    continue

                price_for_iv = mid if mid > 0 else market_px
                iv           = implied_vol(price_for_iv, S, K, T, r, q, option_type)
                sigma        = iv if iv else quote['hist_vol_30d']

                bs_px        = bs_price(S, K, T, r, sigma, q, option_type)
                greeks       = bs_greeks(S, K, T, r, sigma, q, option_type)

                # Moneyness
                moneyness_ratio = S/K if option_type=='call' else K/S
                if moneyness_ratio > 1.005:   moneyness = 'ITM'
                elif moneyness_ratio < 0.995: moneyness = 'OTM'
                else:                          moneyness = 'ATM'

                mispricing = round(bs_px - price_for_iv, 4)

                results.append({
                    'strike':     round(K, 2),
                    'expiry':     target_exp,
                    'type':       option_type,
                    'bid':        round(bid, 2),
                    'ask':        round(ask, 2),
                    'mid':        round(mid, 2),
                    'last':       round(market_px, 2),
                    'volume':     volume,
                    'open_interest': oi,
                    'iv':         round(iv*100, 2) if iv else None,
                    'bs_price':   round(bs_px, 4),
                    'mispricing': mispricing,
                    'moneyness':  moneyness,
                    **{f'greek_{k}': v for k,v in greeks.items()},
                })
            return results

        calls = process_chain(chain.calls, 'call')
        puts  = process_chain(chain.puts,  'put')

        result = {
            'ticker':        ticker.upper(),
            'spot':          S,
            'expiry':        target_exp,
            'T':             round(T, 4),
            'r':             r,
            'q':             q,
            'all_expiries':  list(expiries),
            'calls':         calls,
            'puts':          puts,
            'total_contracts': len(calls)+len(puts),
            'timestamp':     datetime.utcnow().isoformat(),
        }
        cache_set(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR [chain/{ticker}]: {e}\n{tb}", flush=True)
        raise HTTPException(500, detail={"error": str(e), "traceback": tb})


@app.get("/api/surface/{ticker}")
def get_vol_surface(ticker: str):
    """
    Volatility surface: IV across all strikes and all expiries.
    Returns structured data for 3D surface plotting.
    """
    cache_key = f"surface:{ticker.upper()}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        quote    = get_quote(ticker)
        time.sleep(2)
        t        = yf.Ticker(ticker.upper())
        S        = quote['price']
        r        = quote['risk_free_rate']
        q        = quote['dividend_yield']
        expiries = t.options

        if not expiries:
            raise HTTPException(404, "No options data for surface construction.")

        surface_data = []
        # Use up to 6 expiries for performance
        for exp in expiries[:6]:
            try:
                chain    = t.option_chain(exp)
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                T        = max((exp_date - datetime.utcnow()).days / 365, 0.001)
                days     = round(T * 365)

                for _, row in chain.calls.iterrows():
                    K        = float(row['strike'])
                    bid      = float(row.get('bid') or 0)
                    ask      = float(row.get('ask') or 0)
                    mid      = (bid+ask)/2 if bid>0 and ask>0 else float(row.get('lastPrice') or 0)
                    if mid <= 0: continue
                    iv = implied_vol(mid, S, K, T, r, q, 'call')
                    if iv and 0.01 < iv < 3.0:
                        surface_data.append({
                            'strike':        round(K, 2),
                            'expiry':        exp,
                            'days':          days,
                            'T':             round(T, 4),
                            'iv':            round(iv*100, 2),
                            'log_moneyness': round(float(np.log(K/S)), 4),
                            'moneyness':     round(K/S, 4),
                        })
            except:
                continue

        result = {
            'ticker':       ticker.upper(),
            'spot':         S,
            'surface':      surface_data,
            'expiries_used': list(expiries[:6]),
            'points':       len(surface_data),
            'timestamp':    datetime.utcnow().isoformat(),
        }
        cache_set(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Surface error: {str(e)}")


@app.get("/api/compare/{ticker}")
def get_model_comparison(ticker: str, expiry: str = None):
    """
    Model vs Market: Where is BS mispricing relative to market IV?
    The core research output — identifies systematic over/under-pricing.
    """
    cache_key = f"compare:{ticker.upper()}:{expiry or ''}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        chain_data = get_options_chain(ticker, expiry)
        S          = chain_data['spot']

        comparison = []
        for contract in chain_data['calls'] + chain_data['puts']:
            if contract['iv'] is None: continue
            comparison.append({
                'strike':        contract['strike'],
                'type':          contract['type'],
                'moneyness':     contract['moneyness'],
                'log_moneyness': round(float(np.log(contract['strike']/S)), 4),
                'market_iv':     contract['iv'],
                'market_price':  contract['mid'],
                'bs_price':      contract['bs_price'],
                'mispricing':    contract['mispricing'],
                'mispricing_pct': round((contract['mispricing'] / contract['mid'])*100, 2)
                                  if contract['mid'] > 0 else 0,
                'delta':         contract['greek_delta'],
                'gamma':         contract['greek_gamma'],
            })

        # Summary stats
        mispricings = [c['mispricing'] for c in comparison]
        avg_misprice = round(float(np.mean(mispricings)), 4) if mispricings else 0
        max_over     = max(comparison, key=lambda x: x['mispricing']) if comparison else {}
        max_under    = min(comparison, key=lambda x: x['mispricing']) if comparison else {}

        result = {
            'ticker':          ticker.upper(),
            'spot':            S,
            'expiry':          chain_data['expiry'],
            'contracts':       comparison,
            'summary': {
                'avg_mispricing':   avg_misprice,
                'max_overpriced':   max_over.get('strike'),
                'max_underpriced':  max_under.get('strike'),
                'total_contracts':  len(comparison),
            },
            'timestamp': datetime.utcnow().isoformat(),
        }
        cache_set(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR [compare/{ticker}]: {e}\n{tb}", flush=True)
        raise HTTPException(500, detail={"error": str(e), "traceback": tb})


@app.get("/api/search")
def search_tickers(q: str = Query(..., min_length=1)):
    """Quick ticker suggestions including Indian market."""
    suggestions = {
        # US
        'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'TSLA': 'Tesla', 'NVDA': 'NVIDIA',
        'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq ETF', 'META': 'Meta',
        'JPM': 'JPMorgan', 'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley',
        # India NSE
        'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy',
        'INFY.NS': 'Infosys', 'HDFCBANK.NS': 'HDFC Bank',
        'ICICIBANK.NS': 'ICICI Bank', 'WIPRO.NS': 'Wipro',
        'BAJFINANCE.NS': 'Bajaj Finance', 'ADANIENT.NS': 'Adani Enterprises',
        'NIFTY50.NS': 'Nifty 50 Index', 'SENSEX.BO': 'BSE Sensex',
    }
    q_upper = q.upper()
    results = [
        {'ticker': k, 'name': v,
         'market': '🇮🇳 NSE' if '.NS' in k else ('🇮🇳 BSE' if '.BO' in k else '🇺🇸 US')}
        for k, v in suggestions.items()
        if q_upper in k or q_upper in v.upper()
    ]
    return {'results': results[:8]}


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

