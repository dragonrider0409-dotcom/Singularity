import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self):
        try:
            from data_cache import get_treasury_yields
            send_json(self, get_treasury_yields())
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
