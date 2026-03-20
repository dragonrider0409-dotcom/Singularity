import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))

from helpers import ok, err, qs

def handler(request, response):
    if request.method == "OPTIONS": return ok({})
    try:
        from iv_engine import synthetic_market
        S        = float(qs(request,"S") or 175)
        r        = float(qs(request,"r") or 0.05)
        q        = float(qs(request,"q") or 0.0)
        T        = float(qs(request,"T") or 0.5)
        sigma_atm= float(qs(request,"sigma_atm") or 0.25)
        skew     = float(qs(request,"skew") or -0.15)
        smile    = float(qs(request,"smile") or 0.08)
        result   = synthetic_market(S,r,q,T,sigma_atm=sigma_atm,skew=skew,smile=smile)
        return ok(result)
    except Exception as e: return err(str(e))
