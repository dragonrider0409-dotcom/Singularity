import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import numpy as np
            from engine_fi import par_swap_rate, swap_cashflows, swap_dv01
            from data_cache import get_treasury_yields
            b=read_body(self); notional=float(b.get("notional",1e6))
            mat=float(b.get("maturity",5)); freq=int(b.get("freq",4))
            if "maturities" in b:
                mats=np.array(b["maturities"],float); yields=np.array(b["yields"],float)/100
            else:
                tsy=get_treasury_yields(); mats=np.array(tsy["maturities"]); yields=np.array(tsy["yields"])/100
            par=par_swap_rate(mats,yields,mat,freq)
            fr=float(b.get("fixed_rate",par*100))/100 or par
            cfs=swap_cashflows(notional,fr,mat,mats,yields,freq)
            send_json(self,{"par_swap_rate":round(par*100,6),"fixed_rate":round(fr*100,6),
                "notional":notional,"maturity":mat,"mtm":round(float(cfs["net_pv"].sum()),4),
                "dv01":round(swap_dv01(notional,fr,mat,mats,yields,freq),4),
                "cashflows":cfs.to_dict("records")})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
