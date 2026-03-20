import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, qs
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np, pandas as pd
        raw     = qs(req,"tickers") or "SPY,QQQ,AAPL"
        period  = qs(req,"period") or "2y"
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        data    = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
        closes  = data["Close"] if hasattr(data.columns,"levels") else data
        avail   = [t for t in tickers if t in closes.columns]
        closes  = closes[avail].dropna(how="all").ffill()
        return ok({"tickers":avail,
                   "dates":[str(d.date()) for d in closes.index],
                   "prices":{t:closes[t].round(4).tolist() for t in avail}})
    except Exception as e: return err(str(e))
