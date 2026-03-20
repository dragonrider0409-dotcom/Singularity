import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import yfinance as yf
            from pairs_engine import johansen_trace
            b=read_body(self); tickers=[t.upper() for t in b.get("tickers",[])]
            if len(tickers)<2: return send_err(self,"Need at least 2 tickers",400)
            data=yf.download(tickers,period=b.get("period","2y"),auto_adjust=True,progress=False,threads=True)
            closes=data["Close"] if hasattr(data.columns,"levels") else data
            avail=[t for t in tickers if t in closes.columns]
            result=johansen_trace(closes[avail].dropna().values)
            result["tickers"]=avail
            send_json(self, result)
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
