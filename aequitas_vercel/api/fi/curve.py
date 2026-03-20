import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import numpy as np
        from fi_engine import full_curve_output
        b      = gb(req)
        mats   = np.array(b["maturities"],float)
        yields = np.array(b["yields"],float)
        model  = b.get("model","nelson_siegel")
        return ok(full_curve_output(mats, yields, model))
    except Exception as e: return err(str(e))
