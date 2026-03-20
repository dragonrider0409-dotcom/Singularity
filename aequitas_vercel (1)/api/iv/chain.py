import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self):
        try:
            import yfinance as yf, numpy as np
            from datetime import datetime, date
            ticker  = (get_qs(self,"ticker") or "AAPL").upper()
            expiry  = get_qs(self,"expiry") or ""
            opttype = get_qs(self,"type") or "call"
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period="2d",auto_adjust=True)
            spot = float(hist["Close"].dropna().iloc[-1])
            info = tkr.info or {}
            q = float(info.get("dividendYield") or 0.0)
            if not expiry or expiry not in (tkr.options or []):
                expiry = (tkr.options or [expiry])[0]
            T = max((datetime.strptime(expiry,"%Y-%m-%d").date()-date.today()).days/365.0,1/365)
            df = (tkr.option_chain(expiry).calls if opttype=="call" else tkr.option_chain(expiry).puts).copy()
            df = df[(df["bid"]>0)&(df["ask"]>0)]
            df["mid"] = (df["bid"]+df["ask"])/2
            rows = []
            for _,row in df.iterrows():
                K=float(row["strike"])
                if K<spot*0.5 or K>spot*2: continue
                rows.append({"strike":round(K,2),"moneyness":round(K/spot,4),
                    "bid":round(float(row["bid"]),3),"ask":round(float(row["ask"]),3),
                    "mid":round(float(row["mid"]),3),
                    "iv_yf":round(float(row.get("impliedVolatility") or 0)*100,3),
                    "volume":int(row.get("volume") or 0),"oi":int(row.get("openInterest") or 0)})
            send_json(self, {"ticker":ticker,"expiry":expiry,"T":round(T,4),
                "spot":round(spot,4),"r":0.05,"q":round(q,6),"type":opttype,"chain":rows})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
