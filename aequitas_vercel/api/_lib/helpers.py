"""
api/_lib/helpers.py
===================
Shared utilities for all Vercel serverless functions.
Provides: json_response, error_response, get_body, cors_headers
"""
import json
import math
from http.server import BaseHTTPRequestHandler


CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
}


def clean(obj):
    """Recursively replace NaN/Inf floats with None for valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    # numpy scalars
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
    except ImportError:
        pass
    return obj


def ok(data: dict, status: int = 200):
    body = json.dumps(clean(data))
    return Response(body, status, CORS)


def err(message: str, status: int = 500):
    body = json.dumps({"error": message})
    return Response(body, status, CORS)


def body(request) -> dict:
    """Parse JSON body from a Vercel request."""
    try:
        if hasattr(request, 'body'):
            raw = request.body
            if isinstance(raw, bytes):
                return json.loads(raw.decode())
            if isinstance(raw, str):
                return json.loads(raw)
        return {}
    except Exception:
        return {}


def qs(request, key: str, default=None):
    """Get a query-string parameter."""
    try:
        return request.args.get(key, default)
    except Exception:
        return default


class Response:
    """Minimal HTTP response for Vercel Python functions."""
    def __init__(self, body: str, status: int = 200, headers: dict = None):
        self.body    = body
        self.status_code = status
        self.headers = headers or CORS
