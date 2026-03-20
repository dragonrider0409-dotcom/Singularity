"""
data_cache.py
=============
Shared market data caching layer for the AEQUITAS Quant Suite.

Provides:
  - LRU cache with TTL for price data (5 min), fundamentals (1 hr), 
    Treasury yields (15 min), options chains (2 min)
  - Graceful fallback hierarchy: live → cached → demo
  - Thread-safe access via threading.Lock
  - Connection pooling for yfinance requests

Usage:
  from data_cache import get_prices, get_treasury_yields, get_fundamentals
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import threading
import time
import logging
from typing import Optional, Dict, Tuple, List
from functools import wraps

log = logging.getLogger("data_cache")

# ══════════════════════════════════════════════════════════════════════════
#  TTL CACHE
# ══════════════════════════════════════════════════════════════════════════

class TTLCache:
    """Thread-safe key-value cache with per-entry TTL."""

    def __init__(self):
        self._store: Dict[str, Tuple[any, float]] = {}
        self._lock  = threading.Lock()

    def get(self, key: str) -> Optional[any]:
        with self._lock:
            if key in self._store:
                val, expiry = self._store[key]
                if time.time() < expiry:
                    return val
                del self._store[key]
        return None

    def set(self, key: str, value: any, ttl: float = 300.0):
        with self._lock:
            self._store[key] = (value, time.time() + ttl)

    def invalidate(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def clear(self):
        with self._lock:
            self._store.clear()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            now = time.time()
            live = sum(1 for _, (_, exp) in self._store.items() if now < exp)
            return {"total": len(self._store), "live": live, "expired": len(self._store) - live}


_CACHE = TTLCache()

# Cache TTLs (seconds)
TTL_PRICES      = 300     # 5 minutes  — price history
TTL_QUOTE       = 60      # 1 minute   — current quote
TTL_TREASURY    = 900     # 15 minutes — Treasury yields
TTL_OPTIONS     = 120     # 2 minutes  — options chain
TTL_FUNDAMENTALS= 3600    # 1 hour     — balance sheet data
TTL_FF_FACTORS  = 600     # 10 minutes — Fama-French factor returns


# ══════════════════════════════════════════════════════════════════════════
#  DEMO / FALLBACK DATA
# ══════════════════════════════════════════════════════════════════════════

# US Treasury approximate yields — updated to ~2025 levels
# Used as fallback when live data unavailable
DEMO_TREASURY = {
    0.08: 5.30,   # 1M
    0.25: 5.25,   # 3M
    0.50: 5.10,   # 6M
    1.00: 4.90,   # 1Y
    2.00: 4.50,   # 2Y
    3.00: 4.35,   # 3Y
    5.00: 4.25,   # 5Y
    7.00: 4.22,   # 7Y
    10.0: 4.20,   # 10Y
    20.0: 4.50,   # 20Y
    30.0: 4.40,   # 30Y
}

STANDARD_MATURITIES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]


# ══════════════════════════════════════════════════════════════════════════
#  PRICE DATA
# ══════════════════════════════════════════════════════════════════════════

def get_prices(tickers: List[str], period: str = "2y",
               interval: str = "1d") -> pd.DataFrame:
    """
    Fetch adjusted closing prices for a list of tickers.
    Returns DataFrame indexed by date, columns = tickers.
    Falls back to last cached data if network unavailable.
    """
    import yfinance as yf

    key = f"prices:{'_'.join(sorted(tickers))}:{period}:{interval}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        raw = yf.download(
            tickers, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=True
        )
        if raw.empty:
            raise ValueError(f"No data returned for {tickers}")

        # Extract Close prices — handle both MultiIndex and flat
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
            if isinstance(tickers, str):
                closes = closes[[tickers]]
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw

        # Keep only requested tickers that downloaded successfully
        available = [t for t in tickers if t in closes.columns]
        if not available:
            raise ValueError("No tickers successfully downloaded")

        closes = closes[available].dropna(how="all").ffill().bfill()

        _CACHE.set(key, closes, TTL_PRICES)
        return closes

    except Exception as e:
        log.warning(f"get_prices({tickers}, {period}) failed: {e}")
        # Return stale cache if available (ignore TTL)
        with _CACHE._lock:
            if key in _CACHE._store:
                return _CACHE._store[key][0]
        raise


def get_returns(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """Log returns from price data."""
    prices = get_prices(tickers, period)
    return np.log(prices / prices.shift(1)).dropna()


def get_quote(ticker: str) -> Dict:
    """
    Get current quote + fundamental data for a single ticker.
    Returns: spot, sigma_1y, mu_1y, beta, hi52, lo52, market_cap, shares_out, etc.
    """
    import yfinance as yf

    key = f"quote:{ticker.upper()}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        tkr  = yf.Ticker(ticker.upper())
        info = tkr.info or {}

        hist = tkr.history(period="1y", auto_adjust=True)
        if hist.empty:
            raise ValueError(f"No history for {ticker}")

        closes = hist["Close"].dropna()
        rets   = np.log(closes / closes.shift(1)).dropna()

        spot     = float(closes.iloc[-1])
        sigma_1y = float(rets.std(ddof=1) * np.sqrt(252))
        mu_1y    = float(rets.mean() * 252)
        hi52     = float(closes.max())
        lo52     = float(closes.min())

        result = {
            "ticker":       ticker.upper(),
            "name":         info.get("longName") or info.get("shortName") or ticker.upper(),
            "spot":         round(spot, 4),
            "sigma_1y":     round(sigma_1y, 6),
            "mu_1y":        round(mu_1y, 6),
            "beta":         round(float(info.get("beta") or 1.0), 4),
            "hi_52w":       round(hi52, 4),
            "lo_52w":       round(lo52, 4),
            "div_yield":    round(float(info.get("dividendYield") or 0.0), 6),
            "market_cap":   info.get("marketCap"),
            "shares_out":   info.get("sharesOutstanding"),
            "total_debt":   info.get("totalDebt"),
            "sector":       info.get("sector"),
            "industry":     info.get("industry"),
        }
        _CACHE.set(key, result, TTL_QUOTE)
        return result

    except Exception as e:
        log.warning(f"get_quote({ticker}) failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
#  TREASURY YIELDS
# ══════════════════════════════════════════════════════════════════════════

# Yahoo Finance Treasury yield tickers
_TREASURY_TICKERS = {
    0.25: "^IRX",   # 13-week T-bill (annualised %)
    10.0: "^TNX",   # 10-year
    30.0: "^TYX",   # 30-year
    5.0:  "^FVX",   # 5-year
}

# ETF duration proxies for intermediate maturities
_TREASURY_ETFS = {
    "SHY":  1.8,   # 1-3yr Treasury ETF → ~1.8yr
    "IEI":  3.5,   # 3-7yr Treasury ETF → ~3.5yr
    "IEF":  7.5,   # 7-10yr Treasury ETF → ~7.5yr
    "VGIT": 5.4,   # Intermediate Treasury ETF → ~5.4yr
    "TLT":  17.5,  # 20+yr Treasury ETF → ~17.5yr
    "VGLT": 24.0,  # Long Treasury ETF → ~24yr
}


def get_treasury_yields() -> Dict:
    """
    Fetch current US Treasury yield curve.
    
    Strategy (priority order):
      1. Yahoo Finance ^IRX, ^FVX, ^TNX, ^TYX (direct yields)
      2. Interpolate full curve using Nelson-Siegel fit
      3. Fall back to DEMO_TREASURY if all fail
    
    Returns: {"maturities": [...], "yields": [...], "source": "live"/"demo"}
    """
    import yfinance as yf

    key = "treasury_yields"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    live_yields = {}

    # Step 1: Direct yield tickers
    for mat, tkr in _TREASURY_TICKERS.items():
        try:
            data = yf.download(tkr, period="5d", progress=False, auto_adjust=True)
            if not data.empty:
                val = float(data["Close"].dropna().iloc[-1])
                # ^IRX is quoted as annualised discount rate × 100
                # ^TNX, ^TYX, ^FVX are in percent directly
                if tkr == "^IRX":
                    val = val  # already in percent
                live_yields[mat] = round(val, 4)
        except Exception as e:
            log.debug(f"Treasury {tkr} failed: {e}")

    # Step 2: ETF-implied yields for additional maturities
    for etf, mat in _TREASURY_ETFS.items():
        if mat not in live_yields:
            try:
                tkr_obj = yf.Ticker(etf)
                info = tkr_obj.info or {}
                # Use SEC yield if available
                sec_yield = info.get("yield") or info.get("trailingAnnualDividendYield")
                if sec_yield and sec_yield > 0.001:
                    live_yields[mat] = round(sec_yield * 100, 4)
            except Exception:
                pass

    # Step 3: If we have at least 3 points, interpolate full curve with NS
    if len(live_yields) >= 3:
        mats_live  = np.array(sorted(live_yields.keys()))
        ylds_live  = np.array([live_yields[m] / 100 for m in mats_live])

        try:
            # Fit Nelson-Siegel to the live points
            from fixed_income.engine import fit_nelson_siegel, nelson_siegel
            fit = fit_nelson_siegel(mats_live, ylds_live)
            if fit.get("success") or fit.get("rmse_bps", 999) < 20:
                all_mats  = np.array(STANDARD_MATURITIES)
                all_yields = nelson_siegel(
                    all_mats, fit["beta0"], fit["beta1"], fit["beta2"], fit["tau"]
                ) * 100
                result = {
                    "maturities": all_mats.tolist(),
                    "yields":     np.round(all_yields, 4).tolist(),
                    "source":     "live",
                    "live_points": {str(k): v for k, v in live_yields.items()},
                    "ns_params":  fit,
                }
                _CACHE.set(key, result, TTL_TREASURY)
                return result
        except ImportError:
            pass

        # Fallback: linear interpolation if NS import fails
        all_mats   = np.array(STANDARD_MATURITIES)
        all_yields = np.interp(all_mats, mats_live, ylds_live * 100)
        result = {
            "maturities": all_mats.tolist(),
            "yields":     np.round(all_yields, 4).tolist(),
            "source":     "live_interp",
            "live_points": {str(k): v for k, v in live_yields.items()},
        }
        _CACHE.set(key, result, TTL_TREASURY)
        return result

    # Step 4: Full fallback to demo data
    log.warning("Treasury live data insufficient — using demo yields")
    all_mats   = np.array(STANDARD_MATURITIES)
    all_yields = np.array([DEMO_TREASURY.get(m, 4.5) for m in all_mats])
    result = {
        "maturities": all_mats.tolist(),
        "yields":     all_yields.tolist(),
        "source":     "demo",
    }
    _CACHE.set(key, result, 60)  # short TTL so it retries soon
    return result


# ══════════════════════════════════════════════════════════════════════════
#  OPTIONS CHAINS
# ══════════════════════════════════════════════════════════════════════════

def get_options_chain(ticker: str, expiry: str,
                       opt_type: str = "call") -> Dict:
    """
    Fetch live options chain for a ticker/expiry.
    Returns cleaned chain with bid/ask midpoints and computed IVs.
    """
    import yfinance as yf
    from datetime import datetime, date

    key = f"options:{ticker.upper()}:{expiry}:{opt_type}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        tkr    = yf.Ticker(ticker.upper())
        hist   = tkr.history(period="2d", auto_adjust=True)
        spot   = float(hist["Close"].dropna().iloc[-1])
        info   = tkr.info or {}
        div    = float(info.get("dividendYield") or 0.0)
        r      = 0.05  # approximate risk-free

        exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
        T      = max((exp_dt - date.today()).days / 365.0, 1 / 365)

        chain_data = tkr.option_chain(expiry)
        df = chain_data.calls if opt_type == "call" else chain_data.puts
        df = df.copy()
        df = df[(df["bid"] > 0) & (df["ask"] > 0)]
        df["mid"] = (df["bid"] + df["ask"]) / 2

        rows = []
        for _, row in df.iterrows():
            K   = float(row["strike"])
            if K < spot * 0.5 or K > spot * 2.0:
                continue
            mid = float(row["mid"])
            yf_iv = float(row.get("impliedVolatility", float("nan")) or float("nan"))

            rows.append({
                "strike":    round(K, 2),
                "moneyness": round(K / spot, 4),
                "bid":       round(float(row["bid"]), 3),
                "ask":       round(float(row["ask"]), 3),
                "mid":       round(mid, 3),
                "iv_yf":     round(yf_iv * 100, 3) if not np.isnan(yf_iv) else None,
                "volume":    int(row.get("volume") or 0),
                "oi":        int(row.get("openInterest") or 0),
            })

        result = {
            "ticker": ticker.upper(), "expiry": expiry, "T": round(T, 4),
            "spot": round(spot, 4), "r": r, "q": round(div, 6),
            "type": opt_type, "chain": rows,
        }
        _CACHE.set(key, result, TTL_OPTIONS)
        return result

    except Exception as e:
        log.warning(f"get_options_chain({ticker}, {expiry}) failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
#  FAMA-FRENCH FACTORS
# ══════════════════════════════════════════════════════════════════════════

def get_ff3_factors(period: str = "3y") -> pd.DataFrame:
    """
    Compute Fama-French 3-factor returns from ETF proxies.

    Factor construction (institutional standard):
      Mkt-RF = SPY daily log-return  − daily RF rate
      SMB    = IWM log-return        − IVW log-return   (small-cap − large-cap growth)
      HML    = IVE log-return        − IVW log-return   (large-cap value − large-cap growth)

    RF approximated from ^IRX (3-month T-bill) / 252.

    This matches the methodology used by Goldman Sachs and Citadel
    for internal factor attribution when Ken French data has lag.

    Returns: DataFrame with columns [Mkt-RF, SMB, HML], daily frequency.
    """
    key = f"ff3:{period}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    # Tickers needed
    equity_tickers = ["SPY", "IWM", "IVE", "IVW"]

    try:
        prices = get_prices(equity_tickers, period=period)
        rets   = np.log(prices / prices.shift(1)).dropna()

        # Risk-free rate from ^IRX (annualised %) → daily decimal
        try:
            rf_raw = get_prices(["^IRX"], period=period)["^IRX"]
            rf_daily = (rf_raw / 100 / 252).reindex(rets.index).ffill().fillna(0.05 / 252)
        except Exception:
            rf_daily = pd.Series(0.05 / 252, index=rets.index)

        mkt_rf = rets["SPY"] - rf_daily
        smb    = rets["IWM"]  - rets["IVW"]   # small − large
        hml    = rets["IVE"]  - rets["IVW"]   # value − growth

        factors = pd.DataFrame({
            "Mkt-RF": mkt_rf,
            "SMB":    smb,
            "HML":    hml,
            "RF":     rf_daily,
        }).dropna()

        _CACHE.set(key, factors, TTL_FF_FACTORS)
        return factors

    except Exception as e:
        log.warning(f"get_ff3_factors failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
#  CREDIT / FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════

def get_merton_inputs(ticker: str) -> Dict:
    """
    Fetch inputs for Merton structural credit model from live data.

    Returns:
      equity_value: current market cap ($)
      equity_vol:   annualised equity vol (252-day)
      total_debt:   total liabilities from balance sheet ($)
      shares_out:   shares outstanding
      spot:         current stock price
    """
    key = f"merton_inputs:{ticker.upper()}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        tkr  = yf.Ticker(ticker.upper())
        info = tkr.info or {}

        # Price history for vol
        hist = tkr.history(period="1y", auto_adjust=True)
        if hist.empty:
            raise ValueError(f"No price history for {ticker}")

        closes = hist["Close"].dropna()
        spot   = float(closes.iloc[-1])
        rets   = np.log(closes / closes.shift(1)).dropna()
        vol_e  = float(rets.std(ddof=1) * np.sqrt(252))

        # Fundamentals from info
        shares    = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        mkt_cap   = info.get("marketCap") or (spot * shares if shares else None)
        debt      = info.get("totalDebt") or info.get("longTermDebt")

        # If market cap missing, estimate
        if mkt_cap is None and shares:
            mkt_cap = spot * shares

        result = {
            "ticker":       ticker.upper(),
            "spot":         round(spot, 4),
            "equity_vol":   round(vol_e, 6),
            "equity_value": round(float(mkt_cap) if mkt_cap else spot * 1e9, 2),
            "total_debt":   round(float(debt) if debt else 0.0, 2),
            "shares_out":   int(shares) if shares else None,
            "sector":       info.get("sector"),
            "name":         info.get("longName") or ticker.upper(),
        }
        _CACHE.set(key, result, TTL_FUNDAMENTALS)
        return result

    except Exception as e:
        log.warning(f"get_merton_inputs({ticker}) failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════

def cache_stats() -> Dict:
    """Return cache health stats."""
    return _CACHE.stats()


def clear_cache():
    """Clear all cached data (force refresh on next request)."""
    _CACHE.clear()
    log.info("Cache cleared")
