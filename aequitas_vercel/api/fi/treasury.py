import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        from data_cache import get_treasury_yields
        return ok(get_treasury_yields())
    except Exception as e: return err(str(e))
