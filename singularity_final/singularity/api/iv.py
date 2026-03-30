import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body, get_qs
from http.server import BaseHTTPRequestHandler

def _clean(obj):
    if isinstance(obj, float): return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_clean(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj); return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray): return _clean(obj.tolist())
    except ImportError: pass
    return obj

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_GET(self):
        p = self.path
        try:
            import yfinance as yf, numpy as np
            from datetime import datetime, date

            if '/quote' in p:
                ticker = (get_qs(self,'ticker') or 'AAPL').upper()
                tkr  = yf.Ticker(ticker)
                info = tkr.info or {}
                hist = tkr.history(period='1y', auto_adjust=True)
                if hist.empty: return send_err(self, f'No data for {ticker}', 404)
                closes   = hist['Close'].dropna().values
                log_rets = np.diff(np.log(closes))
                spot  = float(closes[-1])
                sigma = float(log_rets.std(ddof=1) * np.sqrt(252))
                div   = float(info.get('dividendYield') or 0.0)
                exps  = list(tkr.options or [])[:8]
                return send_json(self, {
                    'ticker':    ticker,
                    'name':      info.get('longName') or info.get('shortName') or ticker,
                    'spot':      round(spot, 4),
                    'sigma_1y':  round(sigma, 6),
                    'div_yield': round(div, 6),
                    'q':         round(div, 6),
                    'beta':      round(float(info.get('beta') or 1.0), 4),
                    'expiries':  exps,
                })

            if '/chain' in p:
                from engine_iv import bs_iv
                ticker   = (get_qs(self,'ticker') or 'AAPL').upper()
                exp_str  = get_qs(self,'exp') or get_qs(self,'expiry') or ''
                opt_type = (get_qs(self,'type') or 'call').lower()

                tkr  = yf.Ticker(ticker)
                info = tkr.info or {}
                hist = tkr.history(period='2d', auto_adjust=True)
                spot = float(hist['Close'].dropna().iloc[-1])
                r    = 0.05
                div  = float(info.get('dividendYield') or 0.0)

                opts = tkr.options or []
                if not exp_str or exp_str not in opts:
                    exp_str = opts[1] if len(opts) > 1 else opts[0] if opts else ''
                if not exp_str: return send_err(self, 'No expiry dates available', 404)

                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                T = max((exp_date - date.today()).days / 365.0, 1/365)

                chain_data = tkr.option_chain(exp_str)
                df = (chain_data.calls if opt_type == 'call' else chain_data.puts).copy()
                df = df[(df['bid'] > 0) & (df['ask'] > 0)]
                df['mid'] = (df['bid'] + df['ask']) / 2

                rows = []
                for _, row in df.iterrows():
                    K = float(row['strike'])
                    if K < spot * 0.6 or K > spot * 1.6: continue
                    mid  = float(row['mid'])
                    iv   = bs_iv(mid, spot, K, T, r, div, opt_type)
                    yf_iv= float(row.get('impliedVolatility', float('nan')) or float('nan'))
                    rows.append({
                        'strike':    round(K, 2),
                        'moneyness': round(K/spot, 4),
                        'bid':       round(float(row['bid']), 3),
                        'ask':       round(float(row['ask']), 3),
                        'mid':       round(mid, 3),
                        'iv_calc':   round(iv*100, 3) if not (isinstance(iv,float) and math.isnan(iv)) else None,
                        'iv_yf':     round(yf_iv*100, 3) if not (isinstance(yf_iv,float) and math.isnan(yf_iv)) else None,
                        'volume':    int(row.get('volume') or 0),
                        'oi':        int(row.get('openInterest') or 0),
                    })
                return send_json(self, _clean({
                    'ticker': ticker, 'expiry': exp_str, 'T': round(T,4),
                    'spot': round(spot,4), 'r': r, 'q': round(div,6),
                    'type': opt_type, 'chain': rows,
                }))

            if '/surface' in p:
                from engine_iv import build_surface
                S     = float(get_qs(self,'S') or 175)
                r     = float(get_qs(self,'r') or 0.05)
                q     = float(get_qs(self,'q') or 0.0)
                model = get_qs(self,'model') or 'sabr'
                alpha = float(get_qs(self,'alpha') or 0.25)
                beta  = float(get_qs(self,'beta')  or 0.50)
                rho   = float(get_qs(self,'rho')   or -0.30)
                nu    = float(get_qs(self,'nu')    or 0.40)
                v0    = float(get_qs(self,'v0')    or 0.04)
                kappa = float(get_qs(self,'kappa') or 2.0)
                theta = float(get_qs(self,'theta') or 0.04)
                xi    = float(get_qs(self,'xi')    or 0.30)
                hrho  = float(get_qs(self,'hrho')  or -0.70)

                expiries  = [0.08, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
                moneyness = [0.70, 0.75, 0.80, 0.85, 0.875, 0.90, 0.925,
                             0.95, 0.975, 1.0, 1.025, 1.05, 1.075,
                             1.10, 1.125, 1.15, 1.20, 1.25, 1.30]

                if model == 'heston':
                    params_list = [{'v0':v0,'kappa':kappa,'theta':theta,'xi':xi,'rho':hrho}]*len(expiries)
                else:
                    params_list = [{'alpha':alpha,'rho':rho,'nu':nu}]*len(expiries)

                surf = build_surface(S, r, q, expiries, moneyness,
                                      model=model, params_per_expiry=params_list, beta=beta)
                surf['model']  = model
                surf['params'] = {'alpha':alpha,'beta':beta,'rho':rho,'nu':nu,
                                   'v0':v0,'kappa':kappa,'theta':theta,'xi':xi,'rho_h':hrho}
                return send_json(self, _clean(surf))

            if '/synthetic' in p:
                from engine_iv import synthetic_market
                S         = float(get_qs(self,'S') or 175)
                r         = float(get_qs(self,'r') or 0.05)
                q         = float(get_qs(self,'q') or 0.0)
                T         = float(get_qs(self,'T') or 0.5)
                sigma_atm = float(get_qs(self,'sigma_atm') or 0.28)
                skew      = float(get_qs(self,'skew') or -0.15)
                smile     = float(get_qs(self,'smile') or 0.08)
                df = synthetic_market(S, r, q, T, sigma_atm=sigma_atm, skew=skew, smile=smile)
                return send_json(self, _clean({'S':S,'r':r,'q':q,'T':T,'chain':df.to_dict('records')}))

            send_err(self, f'Unknown IV GET endpoint: {p}', 404)
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def do_POST(self):
        p = self.path
        try:
            import numpy as np
            from engine_iv import calibrate_sabr, calibrate_heston, build_surface

            if '/calibrate' in p:
                b      = read_body(self)
                chain  = b.get('chain', [])
                S      = float(b.get('S', 175)); r = float(b.get('r', 0.05))
                q      = float(b.get('q', 0.0)); T = float(b.get('T', 0.5))
                model  = b.get('model', 'sabr');  beta = float(b.get('beta', 0.5))

                strikes = np.array([row['strike'] for row in chain], float)
                mids    = np.array([row.get('mid', 0) for row in chain], float)
                iv_mkt  = np.array([row.get('iv_calc', row.get('iv_yf', 0)) or 0 for row in chain], float) / 100
                valid   = (iv_mkt > 0.01) & (iv_mkt < 2.0)
                strikes, mids, iv_mkt = strikes[valid], mids[valid], iv_mkt[valid]

                res = calibrate_sabr(strikes, iv_mkt, S, T, r, q, beta=beta) if model=='sabr' \
                      else calibrate_heston(strikes, mids, S, r, q, T)

                expiries  = [0.08, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
                moneyness = [0.75,0.80,0.85,0.90,0.95,1.0,1.05,1.10,1.15,1.20,1.25]
                pm = [res.params if hasattr(res,'params') else {}] * len(expiries)
                surf = build_surface(S,r,q,expiries,moneyness,model=model,
                                      params_per_expiry=pm,beta=beta)
                pd = res.params.__dict__ if hasattr(getattr(res,'params',None),'__dict__') else {}
                return send_json(self, _clean({
                    'model': model, 'params': pd,
                    'rmse_bps': getattr(res,'rmse_bps',None),
                    'surface': surf,
                }))

            send_err(self, f'Unknown IV POST endpoint: {p}', 404)
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
