import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import numpy as np
            from credit_engine import merton_model, merton_calibrate, merton_term_structure
            b=read_body(self); V=float(b.get("asset_value",100)); sig_V=float(b.get("asset_vol",0.25))
            D=float(b.get("debt",80)); r=float(b.get("risk_free",0.05)); T=float(b.get("maturity",1.0))
            if b.get("calibrate"):
                result=merton_calibrate(float(b.get("equity_obs",V*0.3)),float(b.get("equity_vol",sig_V*1.5)),D,r,T)
            else:
                result=merton_model(V,sig_V,D,r,T)
            ts=merton_term_structure(V,sig_V,D,r,[0.25,0.5,1,2,3,5,7,10])
            grid=np.linspace(D*0.5,D*2.5,60)
            sens=[merton_model(v,sig_V,D,r,T) for v in grid]
            send_json(self,{"result":result,"term_structure":ts,"risk_free":round(r,6),
                "sensitivity":{"asset_values":np.round(grid,4).tolist(),
                    "pd_rn":[s["pd_risk_neutral"] for s in sens],
                    "equity":[s["equity_value"] for s in sens],
                    "spread_bps":[s["credit_spread_bps"] for s in sens],
                    "dd":[s["distance_to_default"] for s in sens]}})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
