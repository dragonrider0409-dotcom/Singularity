import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import numpy as np
        from fi_engine import par_swap_rate, swap_cashflows, swap_dv01
        from data_cache import get_treasury_yields
        b        = gb(req)
        notional = float(b.get("notional",1_000_000))
        mat      = float(b.get("maturity",5))
        freq     = int(b.get("freq",4))
        if "maturities" in b:
            mats   = np.array(b["maturities"],float)
            yields = np.array(b["yields"],float)/100
        else:
            tsy    = get_treasury_yields()
            mats   = np.array(tsy["maturities"])
            yields = np.array(tsy["yields"])/100
        par  = par_swap_rate(mats,yields,mat,freq)
        fr   = float(b.get("fixed_rate",par*100))/100 or par
        cfs  = swap_cashflows(notional,fr,mat,mats,yields,freq)
        dv   = swap_dv01(notional,fr,mat,mats,yields,freq)
        return ok({"par_swap_rate":round(par*100,6),"fixed_rate":round(fr*100,6),
                   "notional":notional,"maturity":mat,"mtm":round(float(cfs["net_pv"].sum()),4),
                   "dv01":round(dv,4),"cashflows":cfs.to_dict("records")})
    except Exception as e: return err(str(e))
