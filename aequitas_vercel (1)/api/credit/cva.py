import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import numpy as np
            from credit_engine import cva_calculation
            from data_cache import get_treasury_yields
            b=read_body(self); T=float(b.get("maturity",5)); notional=float(b.get("notional",1e6))
            hazard=float(b.get("hazard",0.02)); recovery=float(b.get("recovery",0.40))
            profile=b.get("profile","declining")
            try: tsy=get_treasury_yields(); r=float(np.interp(T,tsy["maturities"],np.array(tsy["yields"])/100))
            except: r=0.04
            times=list(np.round(np.arange(0.25,T+0.25,0.25),4)); t_arr=np.array(times); n=len(times)
            if profile=="flat": exp=[notional*0.5]*n
            elif profile=="hump": exp=(notional*0.8*t_arr/T*np.exp(-t_arr/T*2)).tolist()
            else: exp=(notional*np.exp(-0.3*t_arr)).tolist()
            result=cva_calculation(exp,times,hazard,recovery,r); result["risk_free"]=round(r,6)
            send_json(self, result)
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
