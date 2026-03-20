import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import yfinance as yf, numpy as np
            from engine_portfolio import portfolio_cvar, optimize_min_cvar
            b=read_body(self)
            tickers=b.get("tickers",["SPY","QQQ","TLT","GLD"])
            alpha=float(b.get("alpha",0.05)); period=b.get("period","2y")
            data=yf.download(tickers,period=period,auto_adjust=True,progress=False,threads=True)
            closes=data["Close"] if hasattr(data.columns,"levels") else data
            avail=[t for t in tickers if t in closes.columns]
            rets=np.log(closes[avail]/closes[avail].shift(1)).dropna().values
            weights=b.get("weights"); n=rets.shape[1]
            w=np.array(weights,float)/sum(weights) if weights else np.ones(n)/n
            cv=portfolio_cvar(w,rets,alpha)
            result={"current":cv}
            if b.get("optimize"): result["optimized"]=optimize_min_cvar(rets,alpha)
            send_json(self, result)
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
