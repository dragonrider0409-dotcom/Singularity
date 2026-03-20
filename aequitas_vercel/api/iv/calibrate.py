import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import numpy as np
            from engine_iv import calibrate_sabr, calibrate_heston, build_surface
            b=read_body(self); chain=b.get("chain",[])
            S=float(b.get("S",175)); r=float(b.get("r",0.05))
            q=float(b.get("q",0.0)); T=float(b.get("T",0.5))
            model=b.get("model","sabr"); beta=float(b.get("beta",0.5))
            strikes=np.array([row["strike"] for row in chain],float)
            iv_mkt =np.array([row.get("iv_yf",row.get("iv_calc",0))/100 for row in chain],float)
            mids   =np.array([row.get("mid",0) for row in chain],float)
            valid  =(iv_mkt>0.01)&(iv_mkt<2.0)
            strikes,iv_mkt,mids=strikes[valid],iv_mkt[valid],mids[valid]
            if model=="sabr": res=calibrate_sabr(strikes,iv_mkt,S,T,beta=beta)
            else:             res=calibrate_heston(strikes,mids,S,r,q,T)
            exps=[0.08,0.25,0.5,0.75,1.0,1.5,2.0]
            mono=[0.75,0.80,0.85,0.90,0.95,1.0,1.05,1.10,1.15,1.20,1.25]
            pm=[res.params]*len(exps) if hasattr(res,"params") else [{}]*len(exps)
            surf=build_surface(S,r,q,exps,mono,exps,mono,model=model,params_per_expiry=pm,beta=beta)
            p=res.params if hasattr(res,"params") else {}
            pd=p.__dict__ if hasattr(p,"__dict__") else {}
            send_json(self,{"model":model,"params":pd,
                "rmse_bps":getattr(res,"rmse_bps",None),"surface":surf})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
