import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '_lib'))
from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_POST(self):
        p = self.path
        try:
            import yfinance as yf, numpy as np
            b = read_body(self)

            if '/factor' in p:
                from engine_alpha import (factor_regression, rolling_factor_regression,
                    pca_factors, alpha_decay)
                asset  = b.get('asset','AAPL').upper()
                period = b.get('period','3y')
                window = int(b.get('window',126))
                data   = yf.download([asset,'SPY','IWM','IVE','IVW'], period=period,
                                      auto_adjust=True, progress=False, threads=True)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                rets   = np.log(closes/closes.shift(1)).dropna()
                r      = rets[asset].values
                dates  = [str(d.date()) for d in rets.index]
                mkt    = (rets['SPY'] - 0.05/252).values
                smb    = (rets['IWM'] - rets['IVW']).values
                hml    = (rets['IVE'] - rets['IVW']).values
                F      = np.column_stack([mkt,smb,hml])
                names  = ['Mkt-RF','SMB','HML']
                avail  = [c for c in ['SPY','IWM','IVE','IVW'] if c in rets.columns]
                pca    = pca_factors(rets[avail].values if len(avail)>=3 else F,
                                     n_components=min(3, len(avail) if avail else 3))
                return send_json(self, {'asset':asset,'dates':dates,
                    'returns':np.round(r*100,4).tolist(),
                    'factor_returns':{'Mkt-RF':np.round(mkt*100,4).tolist(),
                        'SMB':np.round(smb*100,4).tolist(),'HML':np.round(hml*100,4).tolist()},
                    'regression':factor_regression(r,F,names),
                    'rolling':rolling_factor_regression(r,F,names,window=window),
                    'decay':alpha_decay(mkt,r,max_lag=15),'pca':pca})

            if '/execution' in p:
                from engine_alpha import almgren_chriss, twap_schedule, vwap_schedule, market_impact_model
                X     = float(b.get('shares',100000)); T = int(b.get('horizon',10))
                sig   = float(b.get('sigma',0.02));   eta = float(b.get('eta',2e-7))
                gamma = float(b.get('gamma',1e-7));   lam = float(b.get('lambda',1e-6))
                adv   = float(b.get('adv',5e6));      price = float(b.get('price',150.0))
                ac    = almgren_chriss(X,T,sig,eta,gamma,lam)
                twap  = twap_schedule(X,T); vwap = vwap_schedule(X,T)
                return send_json(self, {'almgren_chriss':ac,'twap':twap,'vwap':vwap,
                    'impact_ac':  market_impact_model(np.array(ac['trades']),adv,sig,price),
                    'impact_twap':market_impact_model(np.array(twap['trades']),adv,sig,price),
                    'params':{'shares':X,'horizon':T,'sigma':sig}})

            send_err(self, 'Unknown alpha endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def log_message(self, *a): pass
