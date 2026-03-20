import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import numpy as np
        from fi_engine import (Bond, bond_price, bond_ytm, full_analytics,
                                price_change_approx)
        b   = gb(req)
        bnd = Bond(face=float(b.get("face",1000)),coupon_rate=float(b.get("coupon_rate",0.05)),
                   maturity=float(b.get("maturity",10)),freq=int(b.get("freq",2)))
        if "ytm" in b:   ytm = float(b["ytm"])
        elif "price" in b: ytm = bond_ytm(bnd, float(b["price"]))
        else: ytm = 0.045
        an  = full_analytics(bnd, ytm)
        sc  = {f"{'+'if bps>0 else''}{bps}bps":
               {"price":price_change_approx(bnd,ytm,bps/10000)["price_exact"],
                "pct":  price_change_approx(bnd,ytm,bps/10000)["pct_change"]}
               for bps in [-300,-200,-100,-50,50,100,200,300]}
        return ok({**an, "scenarios":sc})
    except Exception as e: return err(str(e))
