import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import numpy as np
            from engine_credit import credit_curve, cds_par_spread, cds_mtm
            from data_cache import get_treasury_yields
            b=read_body(self)
            spreads=b.get("spreads",[50,80,110,140,160]); maturities=b.get("maturities",[1,2,3,5,7])
            recovery=float(b.get("recovery",0.40)); notional=float(b.get("notional",1e6))
            try: tsy=get_treasury_yields(); r=float(np.interp(1.0,tsy["maturities"],np.array(tsy["yields"])/100))
            except: r=0.04
            cc=credit_curve(spreads,maturities,recovery,r)
            hz=np.interp(maturities,maturities,cc["hazard_rates"])
            ps=[round(cds_par_spread(h/100,recovery,r,T),2) for h,T in zip(hz,maturities)]
            hm=float(np.mean(cc["hazard_rates"]))/100; sp=spreads[min(3,len(spreads)-1)]
            send_json(self,{"credit_curve":cc,"par_spreads":ps,"recovery":recovery,"risk_free":round(r,6),
                "mtm_base":cds_mtm(hm,hm,recovery,r,5.0,sp,notional),
                "mtm_wide50":cds_mtm(hm,hm*1.5,recovery,r,5.0,sp,notional)})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
