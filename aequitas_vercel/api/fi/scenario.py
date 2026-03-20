import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import numpy as np
        from fi_engine import Bond, bond_price, modified_duration, convexity
        b   = gb(req)
        bnd = Bond(face=float(b.get("face",1000)),coupon_rate=float(b.get("coupon_rate",0.05)),
                   maturity=float(b.get("maturity",10)),freq=int(b.get("freq",2)))
        ytm = float(b.get("ytm",0.045))
        p0  = bond_price(bnd,ytm)
        d   = modified_duration(bnd,ytm)
        cv  = convexity(bnd,ytm)
        shifts = list(range(-500,525,25))
        return ok({"shifts_bps":shifts,
                   "exact_pnl":[round(bond_price(bnd,ytm+s/10000)-p0,4) for s in shifts],
                   "duration_pnl":[round(-d*(s/10000)*p0,4) for s in shifts],
                   "convexity_pnl":[round(0.5*cv*(s/10000)**2*p0,4) for s in shifts],
                   "price_initial":round(p0,4),"ytm":round(ytm,6),
                   "modified_duration":round(d,6),"convexity":round(cv,6)})
    except Exception as e: return err(str(e))
