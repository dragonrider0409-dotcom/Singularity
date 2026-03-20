import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import yfinance as yf, numpy as np, time
            from engine_portfolio import rolling_backtest
            b=read_body(self)
            tickers=b.get("tickers",["SPY","QQQ","TLT","GLD"])
            period=b.get("period","5y")
            if period in ("1y","2y"): period="5y"
            method=b.get("method","max_sharpe"); lookback=int(b.get("lookback",252))
            rebal=int(b.get("rebalance_every",21)); rf=float(b.get("rf",0.05))
            t0=time.perf_counter()
            data=yf.download(tickers,period=period,auto_adjust=True,progress=False,threads=True)
            closes=data["Close"] if hasattr(data.columns,"levels") else data
            avail=[t for t in tickers if t in closes.columns]
            rets=np.log(closes[avail]/closes[avail].shift(1)).dropna()
            bt=rolling_backtest(rets,method=method,lookback=lookback,rebalance_every=rebal,rf=rf)
            elapsed=round(time.perf_counter()-t0,2)
            bt["elapsed_s"]=elapsed; bt["method"]=method; bt["tickers"]=avail
            send_json(self,{"status":"done","progress":f"Done in {elapsed}s","result":bt,"job_id":"sync"})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
