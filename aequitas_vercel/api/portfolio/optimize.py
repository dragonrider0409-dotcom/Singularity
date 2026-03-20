import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np, pandas as pd
        from portfolio_engine import (
            mean_variance_frontier, black_litterman,
            risk_parity, ledoit_wolf_cov, rolling_backtest, factor_decomposition
        )
        b        = gb(req)
        tickers  = b.get("tickers", ["SPY","QQQ","TLT","GLD","IWM"])
        period   = b.get("period","2y")
        method   = b.get("method","markowitz")
        rf       = float(b.get("rf",0.05))

        data = yf.download(tickers,period=period,auto_adjust=True,progress=False,threads=True)
        closes = data["Close"] if hasattr(data.columns,"levels") else data
        avail  = [t for t in tickers if t in closes.columns]
        rets   = np.log(closes[avail]/closes[avail].shift(1)).dropna()
        R      = rets.values
        dates  = [str(d.date()) for d in rets.index]

        cov = ledoit_wolf_cov(R)
        mu  = R.mean(axis=0)*252

        if method == "black_litterman":
            views = b.get("views",[])
            w = black_litterman(R, rf, views)["weights"]
        elif method == "risk_parity":
            w = risk_parity(cov)["weights"]
        else:
            frontier = mean_variance_frontier(mu, cov, rf, n_points=60)
            w = frontier["tangency_weights"]

        port_r = R @ w
        ann_r  = float(port_r.mean()*252)
        ann_v  = float(port_r.std(ddof=1)*(252**0.5))
        sharpe = ann_r/max(ann_v,1e-8)

        bt = rolling_backtest(R, dates, w, method=method)
        fd = factor_decomposition(R, w, avail)

        return ok({"tickers":avail,"weights":dict(zip(avail,[round(float(x),6) for x in w])),
                   "ann_return":round(ann_r,4),"ann_vol":round(ann_v,4),"sharpe":round(sharpe,4),
                   "backtest":bt,"factor":fd,"dates":dates,
                   "returns":np.round(port_r*100,4).tolist()})
    except Exception as e:
        import traceback; return err(str(e))
