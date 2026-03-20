import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        from data_cache import get_merton_inputs, get_treasury_yields
        from credit_engine import merton_calibrate, merton_model, merton_term_structure
        import numpy as np
        b      = gb(req)
        ticker = b.get("ticker","AAPL").upper()
        T      = float(b.get("maturity",1.0))
        try:
            tsy = get_treasury_yields()
            r   = float(np.interp(T,tsy["maturities"],np.array(tsy["yields"])/100))
        except: r = 0.05
        inp    = get_merton_inputs(ticker)
        spot   = inp["spot"]; sig_E = inp["equity_vol"]
        E_cap  = inp["equity_value"]; D = inp["total_debt"]
        shares = inp.get("shares_out") or E_cap/max(spot,1e-8)
        D_sh   = D/max(shares,1)
        if D <= 0:
            result = merton_model(spot,sig_E,max(D_sh,0.001),r,T)
            result["note"] = "Minimal debt"
        else:
            result = merton_calibrate(spot,sig_E,D_sh,r,T)
        V_imp  = result.get("equity_value",spot) + D_sh
        sig_imp= result.get("equity_vol",sig_E)*spot/max(V_imp,1e-8)
        ts     = merton_term_structure(V_imp,sig_imp,D_sh,r,[0.25,0.5,1,2,3,5,7,10])
        return ok({"ticker":ticker,"name":inp.get("name",ticker),"sector":inp.get("sector"),
                   "spot":round(spot,4),"equity_value_total":round(E_cap,0),
                   "total_debt":round(D,0),"equity_vol":round(sig_E,6),
                   "risk_free":round(r,6),"result":result,"term_structure":ts})
    except Exception as e: return err(str(e))
