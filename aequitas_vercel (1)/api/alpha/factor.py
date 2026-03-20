import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import yfinance as yf, numpy as np
            from alpha_engine import (factor_regression,rolling_factor_regression,
                pca_factors,alpha_decay)
            b=read_body(self); asset=b.get("asset","AAPL").upper()
            period=b.get("period","3y"); window=int(b.get("window",126))
            data=yf.download([asset,"SPY","IWM","IVE","IVW"],period=period,
                auto_adjust=True,progress=False,threads=True)
            closes=data["Close"] if hasattr(data.columns,"levels") else data
            rets=np.log(closes/closes.shift(1)).dropna()
            r=rets[asset].values; dates=[str(d.date()) for d in rets.index]
            mkt=(rets["SPY"]-0.05/252).values
            smb=(rets["IWM"]-rets["IVW"]).values
            hml=(rets["IVE"]-rets["IVW"]).values
            F=np.column_stack([mkt,smb,hml]); names=["Mkt-RF","SMB","HML"]
            reg=factor_regression(r,F,names)
            roll=rolling_factor_regression(r,F,names,window=window)
            decay=alpha_decay(mkt,r,max_lag=15)
            avail=[c for c in ["SPY","IWM","IVE","IVW"] if c in rets.columns]
            pca=pca_factors(rets[avail].values if len(avail)>=3 else F,n_components=min(3,len(avail) if avail else 3))
            send_json(self,{"asset":asset,"dates":dates,
                "returns":np.round(r*100,4).tolist(),
                "factor_returns":{"Mkt-RF":np.round(mkt*100,4).tolist(),
                    "SMB":np.round(smb*100,4).tolist(),"HML":np.round(hml*100,4).tolist()},
                "regression":reg,"rolling":roll,"decay":decay,"pca":pca})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
