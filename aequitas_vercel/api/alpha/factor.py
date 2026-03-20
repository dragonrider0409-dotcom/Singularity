import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np, pandas as pd
        from alpha_engine import (factor_regression, rolling_factor_regression,
                                   pca_factors, alpha_decay)
        from data_cache import get_ff3_factors, get_prices
        b      = gb(req)
        asset  = b.get("asset","AAPL").upper()
        period = b.get("period","3y")
        window = int(b.get("window",126))

        # Fetch asset returns
        data   = yf.download([asset,"SPY","IWM","IVE","IVW"],period=period,
                              auto_adjust=True,progress=False,threads=True)
        closes = data["Close"] if hasattr(data.columns,"levels") else data
        rets   = np.log(closes/closes.shift(1)).dropna()
        r      = rets[asset].values
        dates  = [str(d.date()) for d in rets.index]

        # Build FF3 factors
        from data_cache import TTLCache
        mkt = (rets["SPY"] - 0.05/252).values
        smb = (rets["IWM"] - rets["IVW"]).values
        hml = (rets["IVE"] - rets["IVW"]).values
        F   = np.column_stack([mkt,smb,hml])
        names = ["Mkt-RF","SMB","HML"]

        reg   = factor_regression(r,F,names)
        roll  = rolling_factor_regression(r,F,names,window=window)
        decay = alpha_decay(mkt,r,max_lag=15)

        # PCA on available assets
        avail_cols = [c for c in ["SPY","IWM","IVE","IVW"] if c in rets.columns]
        pca_mat = rets[avail_cols].values if len(avail_cols)>=3 else np.column_stack([mkt,smb,hml])
        pca = pca_factors(pca_mat,n_components=min(3,pca_mat.shape[1]))

        return ok({"asset":asset,"dates":dates,
                   "returns":np.round(r*100,4).tolist(),
                   "factor_returns":{"Mkt-RF":np.round(mkt*100,4).tolist(),
                                     "SMB":np.round(smb*100,4).tolist(),
                                     "HML":np.round(hml*100,4).tolist()},
                   "regression":reg,"rolling":roll,"decay":decay,"pca":pca})
    except Exception as e:
        import traceback; return err(str(e))
