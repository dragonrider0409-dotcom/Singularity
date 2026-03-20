import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self):
        try:
            from iv_engine import synthetic_market
            send_json(self, synthetic_market(
                float(get_qs(self,"S") or 175), float(get_qs(self,"r") or 0.05),
                float(get_qs(self,"q") or 0.0), float(get_qs(self,"T") or 0.5),
                sigma_atm=float(get_qs(self,"sigma_atm") or 0.25),
                skew=float(get_qs(self,"skew") or -0.15),
                smile=float(get_qs(self,"smile") or 0.08)))
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
