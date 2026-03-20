import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))
from helpers import ok, err

def handler(req, res):
    """
    Vercel doesn't support background jobs - return 'done' immediately.
    The backtest endpoint returns synchronously, so the frontend
    should handle both sync and async response patterns.
    """
    if req.method == "OPTIONS":
        return ok({})
    # If job_id is 'sync', the result was already returned by /api/portfolio/backtest
    return ok({"status": "done", "progress": "Complete"})
