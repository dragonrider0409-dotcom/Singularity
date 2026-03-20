import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))
from helpers import ok, err

def handler(request, response):
    """
    Status endpoint for Vercel - since simulate runs synchronously,
    any status check returns 'done'. The result was already in the
    initial simulate response. This endpoint exists for JS compatibility.
    """
    if request.method == "OPTIONS":
        return ok({})
    return ok({"status": "done", "progress": "Complete"})
