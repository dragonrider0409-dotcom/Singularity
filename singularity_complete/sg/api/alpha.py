import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self): send_err(self, 'Use POST', 405)

    def do_POST(self):
        p = self.path
        try:
            import numpy as np
            import yfinance as yf
            b = read_body(self)

            if '/factor' in p:
                from engine_alpha import (factor_regression, rolling_factor_regression,
                    pca_factors, alpha_decay)

                asset  = b.get('asset', 'AAPL').upper()
                period = b.get('period', '3y')
                window = int(b.get('window', 126))

                # Fetch asset + FF3 proxy tickers
                tickers_dl = [asset, 'SPY', 'IWM', 'IVW', 'IVE']
                raw = yf.download(tickers_dl, period=period, auto_adjust=True,
                                   progress=False, threads=True)
                closes = raw['Close'] if hasattr(raw.columns, 'levels') else raw
                closes = closes.dropna(how='all').ffill()

                if asset not in closes.columns:
                    return send_err(self, f'Could not fetch {asset}')

                rets   = np.log(closes / closes.shift(1)).dropna()
                dates  = [str(d.date()) for d in rets.index]
                r      = rets[asset].values
                mkt    = (rets['SPY'] - 0.05/252).values if 'SPY' in rets.columns else r*0
                smb    = (rets['IWM'] - rets['IVW']).values if all(t in rets.columns for t in ['IWM','IVW']) else r*0
                hml    = (rets['IVE'] - rets['IVW']).values if all(t in rets.columns for t in ['IVE','IVW']) else r*0

                F     = np.column_stack([mkt, smb, hml])
                names = ['Mkt-RF', 'SMB', 'HML']

                reg   = factor_regression(r, F, names)
                roll  = rolling_factor_regression(r, F, names, window=window)
                decay = alpha_decay(mkt, r, max_lag=15)

                # PCA on available assets
                avail = [c for c in rets.columns if c != asset][:8]
                try:
                    R_mat = rets[[asset] + avail].values
                    pca   = pca_factors(R_mat, n_components=min(4, R_mat.shape[1]))
                except Exception: pca = {}

                return send_json(self, {
                    'asset':   asset, 'dates': dates,
                    'returns': np.round(r*100, 4).tolist(),
                    'factor_returns': {
                        'Mkt': np.round(mkt*100, 4).tolist(),
                        'SMB': np.round(smb*100, 4).tolist(),
                        'HML': np.round(hml*100, 4).tolist(),
                    },
                    'regression': reg,
                    'rolling':    roll,
                    'decay':      decay,
                    'pca':        pca,
                })

            elif '/execution' in p:
                from engine_alpha import (almgren_chriss, twap_schedule,
                    vwap_schedule, market_impact_model)

                X     = float(b.get('shares',  100000))
                T     = int(b.get('horizon',   10))
                sigma = float(b.get('sigma',   0.02))
                eta   = float(b.get('eta',     2e-7))
                gamma = float(b.get('gamma',   1e-7))
                lam   = float(b.get('lambda',  1e-6))
                adv   = float(b.get('adv',     1000000))
                price = float(b.get('price',   100.0))

                ac   = almgren_chriss(X, T, sigma, eta, gamma, lam)
                twap = twap_schedule(X, T)
                vwap = vwap_schedule(X, T)
                mi_ac   = market_impact_model(np.array(ac['trades']),   adv, sigma, price)
                mi_twap = market_impact_model(np.array(twap['trades']), adv, sigma, price)

                return send_json(self, {
                    'almgren_chriss': ac,
                    'twap':  twap,
                    'vwap':  vwap,
                    'impact_ac':   mi_ac,
                    'impact_twap': mi_twap,
                    'params': {'shares': X, 'horizon': T, 'sigma': sigma,
                               'eta': eta, 'gamma': gamma, 'lambda': lam},
                })

            send_err(self, f'Unknown alpha endpoint: {p}', 404)
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
