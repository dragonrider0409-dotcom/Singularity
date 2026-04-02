import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self): send_err(self, 'Use POST', 405)

    def do_POST(self):
        try:
            import numpy as np, pandas as pd
            from engine_vol import (garch11, gjr_garch, har_rv,
                vol_forecast_with_bands, realized_vol, hmm_em,
                kalman_vol, regime_conditional_stats)
            import yfinance as yf

            b        = read_body(self)
            ticker   = b.get('ticker',   'SPY').upper()
            period   = b.get('period',   '5y')
            n_states = int(b.get('n_states', 2))

            # Fetch returns
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            closes = raw['Close'].dropna()
            if isinstance(closes, pd.DataFrame): closes = closes.iloc[:,0]
            closes = closes.astype(float)
            returns = np.log(closes / closes.shift(1)).dropna().values.flatten().astype(float)
            dates   = [str(d.date()) for d in closes.index[1:]]
            prices  = closes.values[1:].astype(float).flatten()

            # Models
            g11   = garch11(returns)
            gjr   = gjr_garch(returns)
            fcast = vol_forecast_with_bands(g11, h_steps=60, confidence=0.90)

            rv    = realized_vol(returns)
            rv21_clean = [v for v in rv.get('rv_21d',[]) if not (isinstance(v,float) and math.isnan(v))]
            har   = har_rv(np.array(rv21_clean)) if len(rv21_clean) >= 30 else {}

            kv    = kalman_vol(returns).tolist()

            obs   = np.abs(returns) * 100
            hmm   = hmm_em(obs, n_states=n_states)
            stats = regime_conditional_stats(returns, hmm['states'], hmm['state_names'])

            ann_vol     = float(returns.std(ddof=1) * np.sqrt(252) * 100)
            current_vol = g11['cond_vol_ann'][-1]

            # Clean NaN
            def clean(obj):
                if isinstance(obj, float): return None if (math.isnan(obj) or math.isinf(obj)) else obj
                if isinstance(obj, dict):  return {k: clean(v) for k, v in obj.items()}
                if isinstance(obj, list):  return [clean(v) for v in obj]
                return obj

            send_json(self, clean({
                'ticker':       ticker,
                'period':       period,
                'n_obs':        len(returns),
                'dates':        dates,
                'prices':       np.round(prices, 4).tolist(),
                'returns':      np.round(returns * 100, 4).tolist(),
                'ann_ret':      round(float(returns.mean() * 252 * 100), 4),
                'ann_vol':      round(ann_vol, 4),
                'current_vol':  round(float(current_vol), 4),
                'garch11':      g11,
                'gjr_garch':    gjr,
                'forecast':     fcast.get('forecast', fcast) if isinstance(fcast, dict) else fcast,
                'forecast_bands': fcast if isinstance(fcast, dict) else {'forecast': fcast},
                'realized_vol': rv,
                'har':          har,
                'kalman_vol':   kv,
                'hmm':          hmm,
                'regime_stats': stats.to_dict('records'),
            }))
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
