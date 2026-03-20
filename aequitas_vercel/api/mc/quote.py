import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))

from helpers import ok, err, qs

def handler(request, response):
    if request.method == "OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np
        ticker = (qs(request, "ticker") or "AAPL").upper()
        tkr  = yf.Ticker(ticker)
        hist = tkr.history(period="1y", auto_adjust=True)
        info = tkr.info or {}
        closes = hist["Close"].dropna()
        spot   = float(closes.iloc[-1])
        rets   = np.log(closes / closes.shift(1)).dropna().values
        return ok({
            "ticker": ticker,
            "spot":   round(spot, 4),
            "sigma":  round(float(rets.std(ddof=1) * (252**0.5)), 6),
            "mu":     round(float(rets.mean() * 252), 6),
            "beta":   round(float(info.get("beta") or 1.0), 4),
            "hi52":   round(float(closes.max()), 4),
            "lo52":   round(float(closes.min()), 4),
            "q":      round(float(info.get("dividendYield") or 0.0), 6),
            "name":   info.get("longName") or ticker,
        })
    except Exception as e: return err(str(e))
