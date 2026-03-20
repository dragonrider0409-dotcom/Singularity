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
        spot = float(closes.iloc[-1])
        rets = np.log(closes / closes.shift(1)).dropna().values
        exps = list(tkr.options or [])[:8]
        return ok({
            "ticker": ticker, "spot": round(spot, 4),
            "sigma":  round(float(rets.std(ddof=1)*252**0.5), 6),
            "q":      round(float(info.get("dividendYield") or 0.0), 6),
            "beta":   round(float(info.get("beta") or 1.0), 4),
            "expiries": exps,
            "name": info.get("longName") or ticker,
        })
    except Exception as e: return err(str(e))
