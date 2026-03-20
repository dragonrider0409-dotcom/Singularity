import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np
        from pairs_engine import (engle_granger, compute_spread, zscore,
                                   PairsConfig, backtest_pair)
        b      = gb(req)
        tk1    = b.get("ticker_y","GLD").upper()
        tk2    = b.get("ticker_x","SLV").upper()
        period = b.get("period","2y")
        cfg    = PairsConfig(entry_z=float(b.get("entry_z",2)),
                              exit_z=float(b.get("exit_z",0.5)),
                              stop_z=float(b.get("stop_z",4)),
                              z_window=int(b.get("z_window",60)),
                              notional=float(b.get("notional",100_000)),
                              tc_bps=float(b.get("tc_bps",5)))
        data   = yf.download([tk1,tk2],period=period,auto_adjust=True,progress=False)
        closes = data["Close"] if hasattr(data.columns,"levels") else data
        if tk1 not in closes or tk2 not in closes:
            return err(f"Could not fetch {tk1} or {tk2}")
        p1,p2  = closes[tk1].values, closes[tk2].values
        dates  = [str(d.date()) for d in closes.index]
        eg     = engle_granger(p1,p2)
        bt     = backtest_pair(p1,p2,eg["beta"],eg["alpha"],cfg)
        sp     = compute_spread(p1,p2,eg["beta"],eg["alpha"])
        zs     = zscore(sp,cfg.z_window)
        return ok({"ticker_y":tk1,"ticker_x":tk2,"dates":dates,
                   "price_y":np.round(p1,4).tolist(),"price_x":np.round(p2,4).tolist(),
                   "spread":np.round(sp,6).tolist(),
                   "zscore":[round(float(v),4) if v==v else None for v in zs],
                   "eg":eg,"backtest":bt,"config":cfg.__dict__})
    except Exception as e: return err(str(e))
