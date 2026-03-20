import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))

from helpers import ok, err, body as get_body

def handler(request, response):
    if request.method == "OPTIONS": return ok({})
    try:
        import numpy as np
        from iv_engine import calibrate_sabr, calibrate_heston, build_surface
        b      = get_body(request)
        chain  = b.get("chain", [])
        S      = float(b.get("S", 175))
        r      = float(b.get("r", 0.05))
        q      = float(b.get("q", 0.0))
        T      = float(b.get("T", 0.5))
        model  = b.get("model", "sabr")
        beta   = float(b.get("beta", 0.5))

        strikes = np.array([row["strike"] for row in chain], float)
        mids    = np.array([row["mid"]    for row in chain], float)
        iv_mkt  = np.array([row.get("iv_yf", row.get("iv_calc", 0))/100 for row in chain], float)
        valid   = (iv_mkt > 0.01) & (iv_mkt < 2.0)
        strikes, mids, iv_mkt = strikes[valid], mids[valid], iv_mkt[valid]

        if model == "sabr":
            res = calibrate_sabr(strikes, iv_mkt, S, T, beta=beta)
        else:
            res = calibrate_heston(strikes, mids, S, r, q, T)

        expiries  = [0.08, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        moneyness = [0.75,0.80,0.85,0.90,0.95,1.0,1.05,1.10,1.15,1.20,1.25]
        if model == "sabr":
            pm = [res.params] * len(expiries)
            surf = build_surface(S,r,q,expiries,moneyness,expiries,moneyness,
                                  model="sabr",params_per_expiry=pm,beta=beta)
        else:
            pm = [res.params] * len(expiries)
            surf = build_surface(S,r,q,expiries,moneyness,expiries,moneyness,
                                  model="heston",params_per_expiry=pm)

        return ok({"model": model, "params": res.params.__dict__ if hasattr(res.params,'__dict__') else vars(res.params) if hasattr(res.params,'__class__') else {},
                   "rmse_bps": getattr(res,"rmse_bps",None),
                   "surface": surf})
    except Exception as e:
        import traceback; return err(f"{e}")
