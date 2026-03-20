import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            from engine_fi import Bond, bond_ytm, full_analytics, price_change_approx
            b=read_body(self)
            bnd=Bond(face=float(b.get("face",1000)),coupon_rate=float(b.get("coupon_rate",0.05)),
                     maturity=float(b.get("maturity",10)),freq=int(b.get("freq",2)))
            ytm=float(b["ytm"]) if "ytm" in b else (bond_ytm(bnd,float(b["price"])) if "price" in b else 0.045)
            an=full_analytics(bnd,ytm)
            sc={f"{'+'if bps>0 else''}{bps}bps":
                {"price":price_change_approx(bnd,ytm,bps/10000)["price_exact"],
                 "pct":price_change_approx(bnd,ytm,bps/10000)["pct_change"]}
                for bps in [-300,-200,-100,-50,50,100,200,300]}
            send_json(self, {**an,"scenarios":sc})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
