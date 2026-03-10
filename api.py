"""
Options Valuation Engine — FastAPI Backend
==========================================
Phase 4.2: Confirmed free-tier endpoints only

Confirmed working on Massive.com free tier:
  ✅ /v2/aggs/ticker/{sym}/range/1/day/{from}/{to}  ← price + hist vol
  ✅ /v3/reference/options/contracts                ← strikes + expiries
  ✅ /v3/reference/tickers/{sym}                    ← name (best-effort)
  ❌ /v2/snapshot/...                               ← 403 NOT_AUTHORIZED

Cold /api/chain/{ticker} call sequence (max 2 HTTP calls total):
  Call 1: /v2/aggs/ticker/{sym}/range/1/day/{from}/{to}  → price + hist vol
  Call 2: /v3/reference/options/contracts                → strikes + expiries

  Optional Call 3 (best-effort, non-blocking):
          /v3/reference/tickers/{sym}                    → company name

Everything downstream reuses caches — zero additional calls.

Endpoints:
  GET /api/quote/{ticker}          — Live stock price + company info
  GET /api/chain/{ticker}          — Full options chain with BS prices + Greeks
  GET /api/surface/{ticker}        — Vol surface data across strikes/expiries
  GET /api/compare/{ticker}        — Model vs market mispricing analysis
  GET /api/search?q=               — Ticker search
  GET /health                      — Server health check
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import math
import time
import requests
from datetime import datetime, date, timedelta, timezone
import traceback

app = FastAPI(
    title="Options Valuation Engine API",
    description="Phase 4.2: Massive.com confirmed free-tier endpoints",
    version="4.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# MASSIVE.COM API CONFIG
# ---------------------------------------------------------------------------

MASSIVE_API_KEY = "SV4Mf7QHtAb7tsTjYBZNlH58p8Zeq4Su"
MASSIVE_BASE    = "https://api.massive.com"

# ---------------------------------------------------------------------------
# IN-MEMORY CACHE — aggressive TTLs to stay inside free-tier rate limits
# ---------------------------------------------------------------------------

_cache: dict = {}
CACHE_TTL = {
    "quote":     120,   # 2 min
    "chain":     300,   # 5 min
    "surface":   300,   # 5 min
    "compare":   300,   # 5 min
    "hist":      600,   # 10 min
    "contracts": 300,   # 5 min — raw contracts list shared by chain + surface
}

def cache_get(key: str, ttl: int) -> object:
    entry = _cache.get(key)
    if entry and (time.time() - entry['ts']) < ttl:
        return entry['data']
    return None

def cache_set(key: str, data) -> None:
    _cache[key] = {'data': data, 'ts': time.time()}

# ---------------------------------------------------------------------------
# BLACK-SCHOLES ENGINE (self-contained, no scipy)
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
        return max(S-K, 0) if option_type == 'call' else max(K-S, 0)
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
    delta = exp_q*norm_cdf(d1) if option_type == 'call' else exp_q*(norm_cdf(d1)-1)
    gamma = exp_q*nd1 / (S*sigma*sqrt_T)
    vega  = S*exp_q*nd1*sqrt_T / 100
    t1    = -(S*exp_q*nd1*sigma) / (2*sqrt_T)
    t2    = (q*S*exp_q*norm_cdf(d1) - r*K*exp_r*norm_cdf(d2)) if option_type == 'call' \
            else (-q*S*exp_q*norm_cdf(-d1) + r*K*exp_r*norm_cdf(-d2))
    theta = (t1+t2)/365
    rho   = K*T*exp_r*norm_cdf(d2)/100 if option_type == 'call' \
            else -K*T*exp_r*norm_cdf(-d2)/100
    return {'delta': round(delta,6), 'gamma': round(gamma,6),
            'vega':  round(vega,6),  'theta': round(theta,6), 'rho': round(rho,6)}

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
    return {'market': 'US', 'currency': 'USD', 'flag': '🇺🇸', 'exchange': 'NASDAQ/NYSE'}

def get_risk_free_rate(market: str) -> float:
    return 0.0525

def moneyness_label(S: float, K: float) -> str:
    ratio = S / K
    if ratio > 1.005: return 'ITM'
    if ratio < 0.995: return 'OTM'
    return 'ATM'

def time_to_expiry(expiration_date_str: str) -> float:
    """Returns T in years from now (minimum 0.001)."""
    exp = datetime.strptime(expiration_date_str, '%Y-%m-%d')
    T = (exp - datetime.utcnow()).days / 365
    return max(T, 0.001)

def mget(path: str, params: dict = None, timeout: int = 10) -> dict:
    """
    Unified Massive.com REST helper — apiKey query-param auth.
    Used for ALL Massive.com calls throughout this file.
    """
    if params is None:
        params = {}
    params["apiKey"] = MASSIVE_API_KEY
    url = f"{MASSIVE_BASE}{path}"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.Timeout:
        raise HTTPException(504, f"Timed out: {path}")
    except requests.ConnectionError as e:
        raise HTTPException(503, f"Connection error: {e}")
    if resp.status_code == 403:
        raise HTTPException(403,
            f"Massive.com free tier does not include this endpoint: {path}")
    if resp.status_code == 429:
        raise HTTPException(429,
            "Rate limit reached. Please wait 30 seconds and try again.")
    if resp.status_code == 404:
        return {}
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"API error: {resp.text[:200]}")
    return resp.json()

def fetch_option_contracts(ticker: str, spot: float) -> list:
    """
    ONE call to /v3/reference/options/contracts. No pagination.
    Cached for 5 min — shared by get_options_chain() and get_vol_surface().
    250 contracts at ±20% of spot covers all liquid strikes.
    """
    sym       = ticker.upper()
    cache_key = f"contracts:{sym}"
    cached    = cache_get(cache_key, ttl=CACHE_TTL["contracts"])
    if cached is not None:
        return cached

    today = datetime.now(tz=timezone.utc).date().isoformat()
    resp  = mget("/v3/reference/options/contracts", params={
        "underlying_ticker":   sym,
        "expiration_date.gte": today,
        "expired":             "false",
        "strike_price.gte":    round(spot * 0.80),
        "strike_price.lte":    round(spot * 1.20),
        "limit":               250,
        "order":               "asc",
        "sort":                "expiration_date",
    })
    results = resp.get("results", [])
    cache_set(cache_key, results)
    return results

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "online", "timestamp": datetime.utcnow().isoformat(), "version": "4.2"}


@app.get("/api/quote/{ticker}")
def get_quote(ticker: str):
    """
    Stock quote — 1 confirmed call + 1 optional best-effort call:

      Call 1 (required): /v2/aggs/ticker/{sym}/range/1/day/{from}/{to}
        → last bar close = price, second-to-last = prev_close
        → hist_vol computed from all 30 closes in the same response
        → week52 high/low from the same bar array

      Call 2 (optional, non-blocking): /v3/reference/tickers/{sym}
        → company name only; falls back to sym on any failure
    """
    cache_key = f"quote:{ticker.upper()}"
    cached = cache_get(cache_key, ttl=CACHE_TTL["quote"])
    if cached:
        return cached
    try:
        sym         = ticker.upper()
        market_info = detect_market(ticker)
        r           = get_risk_free_rate(market_info['market'])

        # ── Call 1: 45-day daily bars → price + hist vol (one request) ────
        from_date = (date.today() - timedelta(days=45)).isoformat()
        to_date   = date.today().isoformat()
        bars_resp = mget(
            f"/v2/aggs/ticker/{sym}/range/1/day/{from_date}/{to_date}",
            params={'adjusted': 'true', 'sort': 'asc', 'limit': 50}
        )
        bars = bars_resp.get('results', [])
        if len(bars) < 2:
            raise HTTPException(404, f"No price data found for '{ticker}'")

        last     = bars[-1]
        prev     = bars[-2]
        price      = float(last['c'])
        prev_close = float(prev['c'])
        volume     = int(last['v'])

        closes      = [float(b['c']) for b in bars]
        log_ret     = [np.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
        hist_vol    = float(np.std(log_ret) * np.sqrt(252))
        week52_high = max(float(b['h']) for b in bars)
        week52_low  = min(float(b['l']) for b in bars)

        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        # ── Call 2: reference ticker → name, sector, market_cap (best-effort) ──
        name       = sym
        market_cap = 0
        sector     = 'N/A'
        try:
            ref      = mget(f"/v3/reference/tickers/{sym}")
            ref_data = ref.get('results', {})
            name       = ref_data.get('name') or sym
            market_cap = int(ref_data.get('market_cap') or 0)
            sector     = ref_data.get('sic_description') or 'N/A'
        except Exception as ref_err:
            print(f"WARN [quote/{sym}] reference/tickers failed (non-fatal): {ref_err}",
                  flush=True)

        result = {
            "ticker":         sym,
            "name":           name,
            "sector":         sector,
            "price":          round(price, 2),
            "prev_close":     round(prev_close, 2),
            "change":         round(change, 2),
            "change_pct":     round(change_pct, 3),
            "volume":         volume,
            "market_cap":     market_cap,
            "week52_high":    round(week52_high, 2),
            "week52_low":     round(week52_low, 2),
            "dividend_yield": 0.0,
            "hist_vol_30d":   round(hist_vol, 4),
            "risk_free_rate": r,
            "currency":       market_info['currency'],
            "market":         market_info['market'],
            "exchange":       market_info['exchange'],
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
        raise HTTPException(500, detail=f"{e}\n\nTraceback:\n{tb}")


@app.get("/api/chain/{ticker}")
def get_options_chain(ticker: str, expiry: str = None):
    """
    Options chain — zero new calls if quote is cached.
    fetch_option_contracts() makes the only new HTTP call (Call 2),
    and its result is shared with /api/surface/ at zero cost.
    """
    cache_key = f"chain:{ticker.upper()}:{expiry or ''}"
    cached = cache_get(cache_key, ttl=CACHE_TTL["chain"])
    if cached:
        return cached
    try:
        quote = get_quote(ticker)   # cache hit after first load
        S     = quote['price']
        r     = quote['risk_free_rate']
        q     = quote['dividend_yield']
        sigma = quote['hist_vol_30d']
        sym   = ticker.upper()

        # Contracts call — cached for 5 min, reused by /api/surface/
        results = fetch_option_contracts(sym, S)
        if not results:
            raise HTTPException(404, f"No options contracts found for {sym}.")

        expiries   = sorted(set(c['expiration_date'] for c in results))
        target_exp = expiry if expiry in expiries else expiries[0]
        T          = time_to_expiry(target_exp)

        calls, puts = [], []
        for c in results:
            if c['expiration_date'] != target_exp:
                continue
            K     = float(c['strike_price'])
            ctype = c['contract_type']   # 'call' or 'put'

            bs_px  = bs_price(S, K, T, r, sigma, q, ctype)
            greeks = bs_greeks(S, K, T, r, sigma, q, ctype)
            mono   = moneyness_label(S, K)

            contract = {
                'strike':        round(K, 2),
                'expiry':        target_exp,
                'type':          ctype,
                'bid':           None,
                'ask':           None,
                'mid':           None,
                'last':          None,
                'volume':        None,
                'open_interest': None,
                'iv':            None,
                'bs_price':      round(bs_px, 4),
                'mispricing':    0,
                'moneyness':     mono,
                'log_moneyness': round(float(np.log(K/S)), 4),
                **{f'greek_{k}': v for k, v in greeks.items()},
            }
            if ctype == 'call':
                calls.append(contract)
            else:
                puts.append(contract)

        result = {
            'ticker':          sym,
            'spot':            S,
            'expiry':          target_exp,
            'T':               round(T, 4),
            'r':               r,
            'q':               q,
            'all_expiries':    expiries,
            'calls':           calls,
            'puts':            puts,
            'total_contracts': len(calls) + len(puts),
            'timestamp':       datetime.utcnow().isoformat(),
        }
        cache_set(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR [chain/{ticker}]: {e}\n{tb}", flush=True)
        raise HTTPException(500, detail=f"{e}\n\nTraceback:\n{tb}")


@app.get("/api/surface/{ticker}")
def get_vol_surface(ticker: str):
    """
    Vol surface — ZERO new calls if chain was already loaded.
    Reuses fetch_option_contracts() cache (same 5-min TTL as chain).
    Uses hist_vol_30d as uniform sigma (flat surface — honest representation).
    """
    cache_key = f"surface:{ticker.upper()}"
    cached = cache_get(cache_key, ttl=CACHE_TTL["surface"])
    if cached:
        return cached
    try:
        quote = get_quote(ticker)   # cache hit
        S     = quote['price']
        r     = quote['risk_free_rate']
        q     = quote['dividend_yield']
        sigma = quote['hist_vol_30d']
        sym   = ticker.upper()

        # Reuses contracts cache — zero new HTTP calls if chain was loaded
        results = fetch_option_contracts(sym, S)
        if not results:
            raise HTTPException(404, "No options data for surface construction.")

        call_contracts = [c for c in results if c['contract_type'] == 'call']

        all_exp_sorted = sorted(set(c['expiration_date'] for c in call_contracts))
        expiries_used  = all_exp_sorted[:6]

        raw_surface = []
        for c in call_contracts:
            if c['expiration_date'] not in expiries_used:
                continue
            K    = float(c['strike_price'])
            T    = time_to_expiry(c['expiration_date'])
            days = max(round(T * 365), 1)
            raw_surface.append({
                'strike':        round(K, 2),
                'expiry':        c['expiration_date'],
                'days':          days,
                'T':             round(T, 4),
                'iv':            round(sigma * 100, 2),
                'log_moneyness': round(float(np.log(K/S)), 4),
                'moneyness':     round(K/S, 4),
            })

        # Filter out contracts expiring today (T < 0.003 ≈ 1 day) — avoids days=0 chart crash
        surface_data = [p for p in raw_surface if p['T'] >= 0.003]

        if not surface_data:
            result = {
                'ticker':        sym,
                'spot':          S,
                'surface':       [],
                'expiries':      [],
                'strikes':       [],
                'iv_grid':       [],
                'expiries_used': expiries_used,
                'points':        0,
                'message':       'Insufficient data — all contracts near expiry',
                'timestamp':     datetime.utcnow().isoformat(),
            }
            cache_set(cache_key, result)
            return result

        all_strikes  = sorted(set(p['strike'] for p in surface_data))
        all_expiries = sorted(set(p['expiry'] for p in surface_data))
        iv_lookup    = {(p['expiry'], p['strike']): p['iv'] for p in surface_data}
        iv_grid      = [
            [iv_lookup.get((exp, s)) for s in all_strikes]
            for exp in all_expiries
        ]

        result = {
            'ticker':        sym,
            'spot':          S,
            'surface':       surface_data,
            'expiries':      all_expiries,
            'strikes':       all_strikes,
            'iv_grid':       iv_grid,
            'expiries_used': expiries_used,
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
        raise HTTPException(500, detail=f"{e}\n\nTraceback:\n{tb}")


@app.get("/api/compare/{ticker}")
def get_model_comparison(ticker: str, expiry: str = None):
    """
    Model comparison — zero new calls. Derives from chain cache.
    Since market prices are unavailable from reference endpoint,
    market_price and market_iv are None; mispricing is 0.
    """
    cache_key = f"compare:{ticker.upper()}:{expiry or ''}"
    cached = cache_get(cache_key, ttl=CACHE_TTL["compare"])
    if cached:
        return cached
    try:
        chain_data = get_options_chain(ticker, expiry)   # cache hit
        S          = chain_data['spot']

        comparison = []
        for contract in chain_data['calls'] + chain_data['puts']:
            mid = contract.get('mid')
            comparison.append({
                'strike':         contract['strike'],
                'type':           contract['type'],
                'moneyness':      contract['moneyness'],
                'log_moneyness':  round(float(np.log(contract['strike']/S)), 4),
                'market_iv':      contract['iv'],
                'market_price':   mid,
                'bs_price':       contract['bs_price'],
                'mispricing':     contract['mispricing'],
                'mispricing_pct': (round((contract['mispricing'] / mid)*100, 2)
                                   if mid and mid > 0 else 0),
                'delta':          contract['greek_delta'],
                'gamma':          contract['greek_gamma'],
            })

        mispricings  = [c['mispricing'] for c in comparison]
        avg_misprice = round(float(np.mean(mispricings)), 4) if mispricings else 0
        max_over     = max(comparison, key=lambda x: x['mispricing']) if comparison else {}
        max_under    = min(comparison, key=lambda x: x['mispricing']) if comparison else {}

        result = {
            'ticker':    ticker.upper(),
            'spot':      S,
            'expiry':    chain_data['expiry'],
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
        raise HTTPException(500, detail=f"{e}\n\nTraceback:\n{tb}")


@app.get("/api/search")
def search_tickers(q: str = Query(..., min_length=1)):
    """Quick ticker suggestions — US markets only. No API calls."""
    suggestions = {
        'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'TSLA': 'Tesla', 'NVDA': 'NVIDIA',
        'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq ETF', 'META': 'Meta',
        'JPM': 'JPMorgan', 'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley',
        'ORCL': 'Oracle', 'AMD': 'AMD', 'INTC': 'Intel', 'NFLX': 'Netflix',
        'DIS': 'Disney', 'BA': 'Boeing', 'WMT': 'Walmart', 'V': 'Visa',
    }
    q_upper = q.upper()
    results = [
        {'ticker': k, 'name': v, 'market': '🇺🇸 US'}
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
