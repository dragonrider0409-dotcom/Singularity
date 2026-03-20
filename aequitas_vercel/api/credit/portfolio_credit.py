import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import numpy as np
        from credit_engine import portfolio_credit_loss
        b        = gb(req)
        n        = int(b.get("n_names",10))
        notional = float(b.get("notional",1_000_000))
        pd_avg   = float(b.get("pd",0.02))
        lgd_avg  = float(b.get("lgd",0.60))
        corr     = float(b.get("correlation",0.20))
        rng      = np.random.default_rng(42)
        pds      = np.clip(rng.lognormal(np.log(pd_avg),0.5,n),0.001,0.50)
        pds      = (pds/pds.mean()*pd_avg).tolist()
        result   = portfolio_credit_loss([notional]*n,pds,[lgd_avg]*n,corr)
        result["pds"] = [round(p,6) for p in pds]
        return ok(result)
    except Exception as e: return err(str(e))
