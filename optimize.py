import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import yfinance as yf, numpy as np
            from engine_portfolio import (mean_variance_frontier, black_litterman,
                risk_parity, ledoit_wolf_cov, rolling_backtest, factor_decomposition)
            b=read_body(self)
            tickers=b.get("tickers",["SPY","QQQ","TLT","GLD","IWM"])
            period=b.get("period","2y"); method=b.get("method","markowitz")
            rf=float(b.get("rf",0.05))
            data=yf.download(tickers,period=period,auto_adjust=True,progress=False,threads=True)
            closes=data["Close"] if hasattr(data.columns,"levels") else data
            avail=[t for t in tickers if t in closes.columns]
            rets=np.log(closes[avail]/closes[avail].shift(1)).dropna()
            R=rets.values; dates=[str(d.date()) for d in rets.index]
            cov=ledoit_wolf_cov(R); mu=R.mean(axis=0)*252
            if method=="black_litterman":
                w=black_litterman(R,rf,b.get("views",[]))["weights"]
            elif method=="risk_parity":
                w=risk_parity(cov)["weights"]
            else:
                w=mean_variance_frontier(mu,cov,rf,n_points=60)["tangency_weights"]
            port_r=R@w; ann_r=float(port_r.mean()*252); ann_v=float(port_r.std(ddof=1)*252**.5)
            bt=rolling_backtest(R,dates,w,method=method)
            fd=factor_decomposition(R,w,avail)
            send_json(self,{"tickers":avail,"weights":dict(zip(avail,[round(float(x),6) for x in w])),
                "ann_return":round(ann_r,4),"ann_vol":round(ann_v,4),
                "sharpe":round(ann_r/max(ann_v,1e-8),4),
                "backtest":bt,"factor":fd,"dates":dates,
                "returns":np.round(port_r*100,4).tolist()})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
