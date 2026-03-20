import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self):
        try:
            import yfinance as yf, numpy as np
            raw = get_qs(self,"tickers") or "SPY,QQQ,AAPL"
            period = get_qs(self,"period") or "2y"
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            data = yf.download(tickers,period=period,auto_adjust=True,progress=False,threads=True)
            closes = data["Close"] if hasattr(data.columns,"levels") else data
            avail = [t for t in tickers if t in closes.columns]
            closes = closes[avail].dropna(how="all").ffill()
            send_json(self, {"tickers":avail,
                "dates":[str(d.date()) for d in closes.index],
                "prices":{t:closes[t].round(4).tolist() for t in avail}})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
