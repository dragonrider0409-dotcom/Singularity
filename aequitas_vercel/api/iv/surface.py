import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))

from helpers import ok, err, qs

def handler(request, response):
    if request.method == "OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np
        from iv_engine import (build_surface, sabr_vol, heston_price_cf,
                                synthetic_market)
        ticker = (qs(request,"ticker") or "AAPL").upper()
        model  = qs(request,"model") or "sabr"
        src    = qs(request,"src") or "synth"

        tkr    = yf.Ticker(ticker)
        hist   = tkr.history(period="2d", auto_adjust=True)
        S      = float(hist["Close"].dropna().iloc[-1])
        info   = tkr.info or {}
        q      = float(info.get("dividendYield") or 0.0)
        r      = 0.05

        alpha  = float(qs(request,"alpha") or 0.25)
        beta   = float(qs(request,"beta")  or 0.5)
        rho    = float(qs(request,"rho")   or -0.3)
        nu     = float(qs(request,"nu")    or 0.4)
        v0     = float(qs(request,"v0")    or 0.04)

        expiries  = [0.08, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        moneyness = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25]

        if model == "sabr":
            params_list = [{"alpha":alpha,"beta":beta,"rho":rho,"nu":nu}] * len(expiries)
        else:
            params_list = [{"v0":v0,"kappa":float(qs(request,"kap") or 2.0),
                            "theta":float(qs(request,"theta") or 0.04),
                            "xi":float(qs(request,"xi") or 0.3),
                            "rho":float(qs(request,"hrho") or -0.7)}] * len(expiries)

        surf = build_surface(S, r, q, expiries, moneyness,
                             expiries, moneyness, model=model,
                             params_per_expiry=params_list, beta=beta)
        surf["model"] = model
        return ok(surf)
    except Exception as e: return err(str(e))
