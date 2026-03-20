import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self):
        try:
            import yfinance as yf, numpy as np
            from engine_iv import build_surface
            ticker=( get_qs(self,"ticker") or "AAPL").upper()
            model = get_qs(self,"model") or "sabr"
            tkr   = yf.Ticker(ticker)
            hist  = tkr.history(period="2d",auto_adjust=True)
            S     = float(hist["Close"].dropna().iloc[-1])
            info  = tkr.info or {}
            q     = float(info.get("dividendYield") or 0.0)
            alpha = float(get_qs(self,"alpha") or 0.25)
            beta  = float(get_qs(self,"beta") or 0.5)
            rho   = float(get_qs(self,"rho") or -0.3)
            nu    = float(get_qs(self,"nu") or 0.4)
            v0    = float(get_qs(self,"v0") or 0.04)
            exps  = [0.08,0.25,0.5,0.75,1.0,1.5,2.0]
            mono  = [0.75,0.80,0.85,0.90,0.95,1.0,1.05,1.10,1.15,1.20,1.25]
            if model=="sabr":
                pm=[{"alpha":alpha,"beta":beta,"rho":rho,"nu":nu}]*len(exps)
            else:
                pm=[{"v0":v0,"kappa":float(get_qs(self,"kap") or 2.0),
                     "theta":float(get_qs(self,"theta") or 0.04),
                     "xi":float(get_qs(self,"xi") or 0.3),
                     "rho":float(get_qs(self,"hrho") or -0.7)}]*len(exps)
            surf=build_surface(S,0.05,q,exps,mono,exps,mono,model=model,params_per_expiry=pm,beta=beta)
            surf["model"]=model
            send_json(self, surf)
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
