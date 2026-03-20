import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_GET(self):
        p = self.path
        try:
            import yfinance as yf, numpy as np
            if '/quote' in p:
                ticker = (get_qs(self, 'ticker') or 'AAPL').upper()
                tkr    = yf.Ticker(ticker)
                hist   = tkr.history(period='1y', auto_adjust=True)
                info   = tkr.info or {}
                closes = hist['Close'].dropna()
                spot   = float(closes.iloc[-1])
                rets   = np.log(closes / closes.shift(1)).dropna().values
                return send_json(self, {
                    'ticker': ticker, 'spot': round(spot, 4),
                    'sigma':  round(float(rets.std(ddof=1)*252**0.5), 6),
                    'q':      round(float(info.get('dividendYield') or 0.0), 6),
                    'beta':   round(float(info.get('beta') or 1.0), 4),
                    'expiries': list(tkr.options or [])[:8],
                    'name':   info.get('longName') or ticker,
                })
            if '/chain' in p:
                from datetime import datetime, date
                ticker  = (get_qs(self, 'ticker') or 'AAPL').upper()
                expiry  = get_qs(self, 'expiry') or ''
                opttype = get_qs(self, 'type') or 'call'
                tkr     = yf.Ticker(ticker)
                hist    = tkr.history(period='2d', auto_adjust=True)
                spot    = float(hist['Close'].dropna().iloc[-1])
                info    = tkr.info or {}
                q       = float(info.get('dividendYield') or 0.0)
                opts    = tkr.options or []
                if not expiry or expiry not in opts:
                    expiry = opts[0] if opts else expiry
                T   = max((datetime.strptime(expiry, '%Y-%m-%d').date() - date.today()).days / 365.0, 1/365)
                raw = tkr.option_chain(expiry)
                df  = (raw.calls if opttype == 'call' else raw.puts).copy()
                df  = df[(df['bid'] > 0) & (df['ask'] > 0)]
                df['mid'] = (df['bid'] + df['ask']) / 2
                rows = []
                for _, row in df.iterrows():
                    K = float(row['strike'])
                    if K < spot * 0.5 or K > spot * 2: continue
                    rows.append({'strike': round(K, 2), 'moneyness': round(K/spot, 4),
                        'bid': round(float(row['bid']), 3), 'ask': round(float(row['ask']), 3),
                        'mid': round(float(row['mid']), 3),
                        'iv_yf': round(float(row.get('impliedVolatility') or 0) * 100, 3),
                        'volume': int(row.get('volume') or 0), 'oi': int(row.get('openInterest') or 0)})
                return send_json(self, {'ticker': ticker, 'expiry': expiry, 'T': round(T, 4),
                    'spot': round(spot, 4), 'r': 0.05, 'q': round(q, 6), 'type': opttype, 'chain': rows})
            if '/synthetic' in p:
                from engine_iv import synthetic_market
                return send_json(self, synthetic_market(
                    float(get_qs(self,'S') or 175), float(get_qs(self,'r') or 0.05),
                    float(get_qs(self,'q') or 0.0), float(get_qs(self,'T') or 0.5),
                    sigma_atm=float(get_qs(self,'sigma_atm') or 0.25),
                    skew=float(get_qs(self,'skew') or -0.15),
                    smile=float(get_qs(self,'smile') or 0.08)))
            if '/surface' in p:
                from engine_iv import build_surface
                ticker = (get_qs(self,'ticker') or 'AAPL').upper()
                model  = get_qs(self,'model') or 'sabr'
                tkr    = yf.Ticker(ticker)
                hist   = tkr.history(period='2d', auto_adjust=True)
                S      = float(hist['Close'].dropna().iloc[-1])
                q      = float((tkr.info or {}).get('dividendYield') or 0.0)
                alpha  = float(get_qs(self,'alpha') or 0.25)
                beta   = float(get_qs(self,'beta') or 0.5)
                rho    = float(get_qs(self,'rho') or -0.3)
                nu     = float(get_qs(self,'nu') or 0.4)
                v0     = float(get_qs(self,'v0') or 0.04)
                exps   = [0.08,0.25,0.5,0.75,1.0,1.5,2.0]
                mono   = [0.75,0.80,0.85,0.90,0.95,1.0,1.05,1.10,1.15,1.20,1.25]
                pm     = ([{'alpha':alpha,'beta':beta,'rho':rho,'nu':nu}] * len(exps) if model=='sabr'
                          else [{'v0':v0,'kappa':float(get_qs(self,'kap') or 2.0),
                                 'theta':float(get_qs(self,'theta') or 0.04),
                                 'xi':float(get_qs(self,'xi') or 0.3),
                                 'rho':float(get_qs(self,'hrho') or -0.7)}] * len(exps))
                surf   = build_surface(S,0.05,q,exps,mono,exps,mono,model=model,params_per_expiry=pm,beta=beta)
                surf['model'] = model
                return send_json(self, surf)
            send_err(self, 'Unknown IV endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def do_POST(self):
        p = self.path
        try:
            if '/calibrate' in p:
                import numpy as np
                from engine_iv import calibrate_sabr, calibrate_heston, build_surface
                b      = read_body(self)
                chain  = b.get('chain', [])
                S      = float(b.get('S', 175)); r = float(b.get('r', 0.05))
                q      = float(b.get('q', 0.0)); T = float(b.get('T', 0.5))
                model  = b.get('model', 'sabr'); beta = float(b.get('beta', 0.5))
                strikes = np.array([row['strike'] for row in chain], float)
                iv_mkt  = np.array([row.get('iv_yf', row.get('iv_calc', 0))/100 for row in chain], float)
                mids    = np.array([row.get('mid', 0) for row in chain], float)
                valid   = (iv_mkt > 0.01) & (iv_mkt < 2.0)
                strikes, iv_mkt, mids = strikes[valid], iv_mkt[valid], mids[valid]
                res = calibrate_sabr(strikes, iv_mkt, S, T, beta=beta) if model == 'sabr' else calibrate_heston(strikes, mids, S, r, q, T)
                exps = [0.08,0.25,0.5,0.75,1.0,1.5,2.0]
                mono = [0.75,0.80,0.85,0.90,0.95,1.0,1.05,1.10,1.15,1.20,1.25]
                pm   = [res.params] * len(exps) if hasattr(res, 'params') else [{}] * len(exps)
                surf = build_surface(S,r,q,exps,mono,exps,mono,model=model,params_per_expiry=pm,beta=beta)
                pd   = res.params.__dict__ if hasattr(getattr(res,'params',None), '__dict__') else {}
                return send_json(self, {'model': model, 'params': pd,
                    'rmse_bps': getattr(res, 'rmse_bps', None), 'surface': surf})
            send_err(self, 'Unknown IV POST endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def log_message(self, *a): pass
