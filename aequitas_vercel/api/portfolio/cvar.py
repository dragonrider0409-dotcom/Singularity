import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np
        from portfolio_engine import portfolio_cvar, optimize_min_cvar
        b        = gb(req)
        tickers  = b.get("tickers",["SPY","QQQ","TLT","GLD"])
        period   = b.get("period","2y")
        alpha    = float(b.get("alpha",0.05))
        optimize = bool(b.get("optimize",False))
        weights  = b.get("weights")

        data  = yf.download(tickers,period=period,auto_adjust=True,progress=False,threads=True)
        closes= data["Close"] if hasattr(data.columns,"levels") else data
        avail = [t for t in tickers if t in closes.columns]
        rets  = np.log(closes[avail]/closes[avail].shift(1)).dropna().values
        n     = rets.shape[1]
        w     = np.array(weights,float)/sum(weights) if weights else np.ones(n)/n

        cv     = portfolio_cvar(w, rets, alpha)
        result = {"current": cv}
        if optimize:
            result["optimized"] = optimize_min_cvar(rets, alpha)
        return ok(result)
    except Exception as e: return err(str(e))
