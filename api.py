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
import requests
from datetime import datetime, date
import traceback

app = FastAPI(
    title="Options Valuation Engine API",
    description="Phase 3: Live market data + pricing engine",
    version="3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SHARED SESSION — custom headers bypass Yahoo rate limiting
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
})

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
    """Newton-Raphson IV solver."""
    if T <= 0 or market_price <= 0:
        return None
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
# HELPERS
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
    rates = {'US': 0.0525, 'NSE': 0.067, 'BSE': 0.067}
    return rates.get(market, 0.05)

def safe_fast(fi, attr, default=None):
    """Safely read a fast_info attribute, returning default on any failure."""
    try:
        val = getattr(fi, attr, default)
        return val if val is not None else default
    except Exception:
        return default

def flatten_download(hist: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-level columns from yf.download() to single level."""
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "online", "timestamp": datetime.utcnow().isoformat(), "version": "3.1"}


@app.get("/api/quote/{ticker}")
def get_quote(ticker: str):
    """
    Robust quote endpoint with layered fallbacks:
      1. fast_info  — price, market cap, 52-week range (fast, low rate-limit)
      2. t.history  — 30d OHLCV for historical vol (separate endpoint)
      3. t.info     — name, sector, dividend yield (slower; isolated try/except)
      4. yf.download — last-resort price if fast_info fails entirely
    """
    cache_key = f"quote:{ticker.upper()}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        sym          = ticker.upper()
        market_info  = detect_market(ticker)
        r            = get_risk_free_rate(market_info['market'])
        t            = yf.Ticker(sym, session=_session)

        # ── 1. fast_info: price + market metadata ──────────────────────────
        fi           = t.fast_info
        price        = safe_fast(fi, 'last_price')
        prev_close   = safe_fast(fi, 'previous_close')
        market_cap   = safe_fast(fi, 'market_cap') or 0
        volume       = safe_fast(fi, 'three_month_average_volume') \
                       or safe_fast(fi, 'regular_market_volume') or 0
        week52_high  = safe_fast(fi, 'fifty_two_week_high')
        week52_low   = safe_fast(fi, 'fifty_two_week_low')
        currency     = safe_fast(fi, 'currency') or market_info['currency']
        exchange     = safe_fast(fi, 'exchange') or market_info['exchange']

        # ── 2. t.history: 30-day OHLCV for historical vol ─────────────────
        hist         = t.history(period='30d')
        hist_vol     = 0.25  # default
        if not hist.empty:
            close    = hist['Close'].dropna()
            if len(close) >= 5:
                log_returns = np.log(close / close.shift(1)).dropna()
                hist_vol    = float(log_returns.std() * np.sqrt(252))
            # fill price from history if fast_info gave nothing
            if not price and len(close):
                price = float(close.iloc[-1])
            if not prev_close and len(close) > 1:
                prev_close = float(close.iloc[-2])

        # ── 3. yf.download fallback if price still missing ─────────────────
        if not price:
            dl   = yf.download(sym, period='1d', progress=False, session=_session)
            dl   = flatten_download(dl)
            if not dl.empty:
                price = float(dl['Close'].dropna().iloc[-1])

        if not price:
            raise HTTPException(404, f"No price data found for '{ticker}'")

        if not prev_close:
            prev_close = price
        change     = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0

        # ── 4. t.info: name, sector, dividend yield (best-effort) ──────────
        name           = sym
        sector         = 'N/A'
        dividend_yield = 0.0
        try:
            info           = t.info
            name           = info.get('longName') or info.get('shortName') or sym
            sector         = info.get('sector') or 'N/A'
            dividend_yield = round(min(max(float(info.get('dividendYield') or 0), 0), 0.20), 4)
        except Exception as info_err:
            print(f"WARN [quote/{sym}] t.info failed (non-fatal): {info_err}", flush=True)

        result = {
            "ticker":         sym,
            "name":           name,
            "sector":         sector,
            "price":          round(price, 2),
            "prev_close":     round(prev_close, 2),
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 3),
            "volume":         int(volume or 0),
            "market_cap":     int(market_cap or 0),
            "week52_high":    round(week52_high, 2) if week52_high else None,
            "week52_low":     round(week52_low, 2)  if week52_low  else None,
            "dividend_yield": dividend_yield,
            "hist_vol_30d":   round(hist_vol, 4),
            "risk_free_rate": r,
            "currency":       currency,
            "market":         market_info['market'],
            "exchange":       exchange,
            "flag":           market_info['flag'],
            "timestamp":      datetime.utcnow().isoformat(),
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
    Uses fast_info for spot price, option_chain() for contract data.
    """
    cache_key = f"chain:{ticker.upper()}:{expiry or ''}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        quote    = get_quote(ticker)
        S        = quote['price']
        r        = quote['risk_free_rate']
        q        = quote['dividend_yield']

        t        = yf.Ticker(ticker.upper(), session=_session)
        expiries = t.options

        if not expiries:
            raise HTTPException(404, f"No options data available for {ticker}. "
                                     f"Note: Indian market options may have limited coverage.")

        target_exp = expiry if expiry in expiries else expiries[0]
        chain      = t.option_chain(target_exp)

        exp_date   = datetime.strptime(target_exp, '%Y-%m-%d')
        T          = max((exp_date - datetime.utcnow()).days / 365, 0.001)

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

                moneyness_ratio = S/K if option_type=='call' else K/S
                if moneyness_ratio > 1.005:   moneyness = 'ITM'
                elif moneyness_ratio < 0.995: moneyness = 'OTM'
                else:                          moneyness = 'ATM'

                mispricing = round(bs_px - price_for_iv, 4)

                results.append({
                    'strike':        round(K, 2),
                    'expiry':        target_exp,
                    'type':          option_type,
                    'bid':           round(bid, 2),
                    'ask':           round(ask, 2),
                    'mid':           round(mid, 2),
                    'last':          round(market_px, 2),
                    'volume':        volume,
                    'open_interest': oi,
                    'iv':            round(iv*100, 2) if iv else None,
                    'bs_price':      round(bs_px, 4),
                    'mispricing':    mispricing,
                    'moneyness':     moneyness,
                    **{f'greek_{k}': v for k,v in greeks.items()},
                })
            return results

        calls = process_chain(chain.calls, 'call')
        puts  = process_chain(chain.puts,  'put')

        result = {
            'ticker':          ticker.upper(),
            'spot':            S,
            'expiry':          target_exp,
            'T':               round(T, 4),
            'r':               r,
            'q':               q,
            'all_expiries':    list(expiries),
            'calls':           calls,
            'puts':            puts,
            'total_contracts': len(calls)+len(puts),
            'timestamp':       datetime.utcnow().isoformat(),
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
    Volatility surface: IV across all strikes and up to 6 expiries.
    Uses fast_info for spot, option_chain() per expiry for IV data.
    """
    cache_key = f"surface:{ticker.upper()}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    try:
        quote    = get_quote(ticker)
        S        = quote['price']
        r        = quote['risk_free_rate']
        q        = quote['dividend_yield']

        t        = yf.Ticker(ticker.upper(), session=_session)
        expiries = t.options

        if not expiries:
            raise HTTPException(404, "No options data for surface construction.")

        surface_data = []
        for exp in expiries[:6]:
            try:
                chain    = t.option_chain(exp)
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                T        = max((exp_date - datetime.utcnow()).days / 365, 0.001)
                days     = round(T * 365)

                for _, row in chain.calls.iterrows():
                    K   = float(row['strike'])
                    bid = float(row.get('bid') or 0)
                    ask = float(row.get('ask') or 0)
                    mid = (bid+ask)/2 if bid>0 and ask>0 else float(row.get('lastPrice') or 0)
                    if mid <= 0:
                        continue
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
            except Exception:
                continue

        result = {
            'ticker':        ticker.upper(),
            'spot':          S,
            'surface':       surface_data,
            'expiries_used': list(expiries[:6]),
            'points':        len(surface_data),
            'timestamp':     datetime.utcnow().isoformat(),
        }
        cache_set(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR [surface/{ticker}]: {e}\n{tb}", flush=True)
        raise HTTPException(500, detail={"error": str(e), "traceback": tb})


@app.get("/api/compare/{ticker}")
def get_model_comparison(ticker: str, expiry: str = None):
    """
    Model vs Market: BS mispricing analysis across all strikes.
    Reuses cached chain data — no additional yfinance calls.
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
            if contract['iv'] is None:
                continue
            comparison.append({
                'strike':         contract['strike'],
                'type':           contract['type'],
                'moneyness':      contract['moneyness'],
                'log_moneyness':  round(float(np.log(contract['strike']/S)), 4),
                'market_iv':      contract['iv'],
                'market_price':   contract['mid'],
                'bs_price':       contract['bs_price'],
                'mispricing':     contract['mispricing'],
                'mispricing_pct': round((contract['mispricing'] / contract['mid'])*100, 2)
                                  if contract['mid'] > 0 else 0,
                'delta':          contract['greek_delta'],
                'gamma':          contract['greek_gamma'],
            })

        mispricings  = [c['mispricing'] for c in comparison]
        avg_misprice = round(float(np.mean(mispricings)), 4) if mispricings else 0
        max_over     = max(comparison, key=lambda x: x['mispricing']) if comparison else {}
        max_under    = min(comparison, key=lambda x: x['mispricing']) if comparison else {}

        result = {
            'ticker':   ticker.upper(),
            'spot':     S,
            'expiry':   chain_data['expiry'],
            'contracts': comparison,
            'summary': {
                'avg_mispricing':  avg_misprice,
                'max_overpriced':  max_over.get('strike'),
                'max_underpriced': max_under.get('strike'),
                'total_contracts': len(comparison),
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
