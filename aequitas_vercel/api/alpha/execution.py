import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import numpy as np
        from alpha_engine import (almgren_chriss, twap_schedule, vwap_schedule,
                                   market_impact_model)
        b    = gb(req)
        X    = float(b.get("shares",100_000))
        T    = int(b.get("horizon",10))
        sig  = float(b.get("sigma",0.02))
        eta  = float(b.get("eta",2e-7))
        gamma= float(b.get("gamma",1e-7))
        lam  = float(b.get("lambda",1e-6))
        adv  = float(b.get("adv",5_000_000))
        price= float(b.get("price",150.0))
        ac   = almgren_chriss(X,T,sig,eta,gamma,lam)
        twap = twap_schedule(X,T)
        vwap = vwap_schedule(X,T)
        mi_ac  = market_impact_model(np.array(ac["trades"]),adv,sig,price)
        mi_twap= market_impact_model(np.array(twap["trades"]),adv,sig,price)
        return ok({"almgren_chriss":ac,"twap":twap,"vwap":vwap,
                   "impact_ac":mi_ac,"impact_twap":mi_twap,
                   "params":{"shares":X,"horizon":T,"sigma":sig}})
    except Exception as e: return err(str(e))
