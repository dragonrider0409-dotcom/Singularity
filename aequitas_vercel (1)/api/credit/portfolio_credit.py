import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import numpy as np
            from credit_engine import portfolio_credit_loss
            b=read_body(self); n=int(b.get("n_names",10))
            notional=float(b.get("notional",1e6)); pd_avg=float(b.get("pd",0.02))
            lgd_avg=float(b.get("lgd",0.60)); corr=float(b.get("correlation",0.20))
            rng=np.random.default_rng(42)
            pds=np.clip(rng.lognormal(np.log(pd_avg),0.5,n),0.001,0.50)
            pds=(pds/pds.mean()*pd_avg).tolist()
            result=portfolio_credit_loss([notional]*n,pds,[lgd_avg]*n,corr)
            result["pds"]=[round(p,6) for p in pds]
            send_json(self, result)
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
