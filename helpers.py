"""
api/_lib/helpers.py — Vercel Python BaseHTTPRequestHandler utilities
"""
import json, math, urllib.parse
from http.server import BaseHTTPRequestHandler

CORS = [
    ("Access-Control-Allow-Origin",  "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "Content-Type"),
    ("Content-Type", "application/json"),
]

def clean(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return clean(obj.tolist())
    except ImportError: pass
    return obj

def send_json(h, data, status=200):
    body = json.dumps(clean(data)).encode()
    h.send_response(status)
    for k,v in CORS: h.send_header(k,v)
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)

def send_err(h, msg, status=500):
    send_json(h, {"error": msg}, status)

def send_cors(h):
    h.send_response(204)
    for k,v in CORS: h.send_header(k,v)
    h.end_headers()

def read_body(h):
    try:
        n = int(h.headers.get("Content-Length", 0))
        return json.loads(h.rfile.read(n).decode()) if n else {}
    except: return {}

def get_qs(h, key, default=None):
    try:
        p = urllib.parse.parse_qs(urllib.parse.urlparse(h.path).query)
        vals = p.get(key)
        return vals[0] if vals else default
    except: return default
