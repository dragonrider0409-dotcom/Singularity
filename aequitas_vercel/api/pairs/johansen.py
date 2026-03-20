import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np
        from pairs_engine import johansen_trace
        b       = gb(req)
        tickers = [t.upper() for t in b.get("tickers",[])]
        period  = b.get("period","2y")
        if len(tickers) < 2: return err("Need at least 2 tickers",400)
        data    = yf.download(tickers,period=period,auto_adjust=True,progress=False,threads=True)
        closes  = data["Close"] if hasattr(data.columns,"levels") else data
        avail   = [t for t in tickers if t in closes.columns]
        result  = johansen_trace(closes[avail].dropna().values)
        result["tickers"] = avail
        return ok(result)
    except Exception as e: return err(str(e))
