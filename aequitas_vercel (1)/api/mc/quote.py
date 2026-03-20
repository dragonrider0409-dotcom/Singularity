import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self):
        try:
            import yfinance as yf, numpy as np
            ticker = (get_qs(self,"ticker") or "AAPL").upper()
            tkr  = yf.Ticker(ticker)
            hist = tkr.history(period="1y", auto_adjust=True)
            info = tkr.info or {}
            closes = hist["Close"].dropna()
            spot   = float(closes.iloc[-1])
            rets   = np.log(closes/closes.shift(1)).dropna().values
            send_json(self, {
                "ticker": ticker, "spot": round(spot,4),
                "sigma":  round(float(rets.std(ddof=1)*252**.5),6),
                "mu":     round(float(rets.mean()*252),6),
                "beta":   round(float(info.get("beta") or 1.0),4),
                "hi52":   round(float(closes.max()),4),
                "lo52":   round(float(closes.min()),4),
                "q":      round(float(info.get("dividendYield") or 0.0),6),
                "name":   info.get("longName") or ticker,
            })
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
