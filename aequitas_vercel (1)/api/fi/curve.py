import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import numpy as np
            from fi_engine import full_curve_output
            b=read_body(self)
            send_json(self, full_curve_output(
                np.array(b["maturities"],float),
                np.array(b["yields"],float),
                b.get("model","nelson_siegel")))
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
