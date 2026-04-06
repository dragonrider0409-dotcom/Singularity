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
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray): return _clean(obj.tolist())
    except ImportError: pass
    return obj

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_GET(self):
        try:
            if '/prices' in self.path:
                import yfinance as yf, numpy as np
                raw     = get_qs(self, 'tickers') or 'SPY,QQQ,AAPL'
                period  = get_qs(self, 'period') or '2y'
                tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]
                data    = yf.download(tickers, period=period, auto_adjust=True,
                                       progress=False, threads=True)
                closes  = data['Close'] if hasattr(data.columns,'levels') else data
                avail   = [t for t in tickers if t in closes.columns]
                closes  = closes[avail].dropna(how='all').ffill()
                return send_json(self, {
                    'tickers': avail,
                    'dates':   [str(d.date()) for d in closes.index],
                    'prices':  {t: closes[t].round(4).tolist() for t in avail},
                })
            if '/job' in self.path:
                return send_json(self, {'status': 'done', 'progress': 'Complete'})
            send_err(self, 'Unknown GET endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def do_POST(self):
        p = self.path
        try:
            import yfinance as yf, numpy as np
            b       = read_body(self)
            tickers = b.get('tickers', ['SPY','QQQ','TLT','GLD','IWM'])
            period  = b.get('period', '2y')
            rf      = float(b.get('rf', 0.05))

            if '/optimize' in p:
                from engine_portfolio import (
                    efficient_frontier, risk_parity, black_litterman,
                    market_implied_returns, max_sharpe_portfolio,
                    min_variance_portfolio, ledoit_wolf_cov, sample_cov,
                    nearest_pd, factor_decomposition, risk_attribution,
                )
                import time

                cov_meth = b.get('cov_method', 'ledoit_wolf')
                allow_sh = bool(b.get('allow_short', False))
                views    = b.get('views', [])
                t0       = time.perf_counter()

                data   = yf.download(tickers, period=period, auto_adjust=True,
                                      progress=False, threads=True)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                avail  = [t for t in tickers if t in closes.columns]
                rets   = np.log(closes[avail]/closes[avail].shift(1)).dropna()
                n      = len(avail)
                if n < 2:
                    return send_err(self, f'Only {n} tickers downloaded. Need at least 2.')

                mu  = rets.mean().values * 252
                try:
                    cov = ledoit_wolf_cov(rets)
                    np.linalg.cholesky(cov)
                except Exception:
                    cov = nearest_pd(sample_cov(rets))

                ef  = efficient_frontier(mu, cov, avail, n_points=60,
                                          allow_short=allow_sh, rf=rf)
                rp_ = risk_parity(cov, avail, mu, rf=rf)

                mkt_w  = np.ones(n)/n
                pi     = market_implied_returns(cov, mkt_w, rf=rf)
                bl_mu, bl_cov = black_litterman(pi, cov, avail, views, rf=rf)
                try: np.linalg.cholesky(bl_cov)
                except np.linalg.LinAlgError: bl_cov = nearest_pd(bl_cov)
                bl_ms  = max_sharpe_portfolio(bl_mu, bl_cov, avail, allow_sh, rf)
                bl_mv  = min_variance_portfolio(bl_mu, bl_cov, avail, allow_sh, rf)

                # Fetch mkt returns for factor decomp
                mkt_raw = yf.download('SPY', period=period, auto_adjust=True, progress=False)
                mkt_c   = mkt_raw['Close'].dropna()
                mkt_ret = np.log(mkt_c/mkt_c.shift(1)).dropna()
                common  = rets.index.intersection(mkt_ret.index)
                rets_a  = rets.loc[common]
                mkt_a   = mkt_ret.loc[common]

                fd    = factor_decomposition(rets_a, mkt_a)
                ra_ms = risk_attribution(ef.max_sharpe.weights, cov, avail)
                ra_rp = risk_attribution(rp_.weights, cov, avail)

                eq_cum  = (rets.mean(axis=1)+1).cumprod()-1
                ms_rets = rets.values @ ef.max_sharpe.weights
                ms_cum  = np.cumprod(1+ms_rets)-1
                mv_rets = rets.values @ ef.min_var.weights
                mv_cum  = np.cumprod(1+mv_rets)-1
                rp_rets = rets.values @ rp_.weights
                rp_cum  = np.cumprod(1+rp_rets)-1

                def pr(p_):
                    return {'weights':dict(zip(avail,[round(float(w),6) for w in p_.weights])),
                            'ret':round(float(p_.exp_return),6),'vol':round(float(p_.volatility),6),
                            'sharpe':round(float(p_.sharpe),6)}

                elapsed = round(time.perf_counter()-t0, 2)
                result = {
                    'tickers': avail, 'n_assets': n, 'n_obs': len(rets),
                    'elapsed_s': elapsed, 'rf': rf, 'cov_method': cov_meth,
                    'mu':  {t: round(float(v),6) for t,v in zip(avail,mu)},
                    'vol': {t: round(float(v),6) for t,v in zip(avail,rets.std().values*np.sqrt(252))},
                    'corr': rets.corr().round(4).to_dict(),
                    'efficient_frontier': {
                        'returns':  [round(float(v),6) for v in ef.returns],
                        'vols':     [round(float(v),6) for v in ef.vols],
                        'sharpes':  [round(float(v),6) for v in ef.sharpes],
                        'weights':  [[round(float(w),6) for w in row] for row in ef.weights],
                        'max_sharpe': pr(ef.max_sharpe),
                        'min_var':    pr(ef.min_var),
                        'tickers':  avail,
                    },
                    'risk_parity':  pr(rp_),
                    'bl_max_sharpe': pr(bl_ms),
                    'factor': fd.reset_index().to_dict('records') if not fd.empty else [],
                    'risk_attr_ms': ra_ms.reset_index().rename(columns={'index':'ticker'}).to_dict('records'),
                    'risk_attr_rp': ra_rp.reset_index().rename(columns={'index':'ticker'}).to_dict('records'),
                    'dates':    [str(d.date()) for d in rets.index],
                    'eq_cum':   np.round(eq_cum.values,6).tolist(),
                    'ms_cum':   np.round(ms_cum,6).tolist(),
                    'mv_cum':   np.round(mv_cum,6).tolist(),
                    'rp_cum':   np.round(rp_cum,6).tolist(),
                    'asset_cum':{t: np.round(((rets[t]+1).cumprod()-1).values,6).tolist() for t in avail},
                }
                return send_json(self, _clean({'status':'done','result':result,'job_id':'sync'}))

            elif '/backtest' in p:
                import time

                if period in ('1y','2y'): period = '5y'
                method   = b.get('method', 'max_sharpe')
                lookback = int(b.get('lookback', 252))
                rebal    = int(b.get('rebalance_every', 21))
                t0       = time.perf_counter()

                data   = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                avail  = [t for t in tickers if t in closes.columns]
                rets   = np.log(closes[avail]/closes[avail].shift(1)).dropna()

                bt      = rolling_backtest(rets, method=method, lookback=lookback,
                                            rebalance_every=rebal, rf=float(b.get('rf',0.05)))
                elapsed = round(time.perf_counter()-t0, 2)
                bt['elapsed_s'] = elapsed
                bt['tickers']   = avail
                return send_json(self, _clean({'status':'done','result':bt,'job_id':'sync'}))

            elif '/cvar' in p:
                from engine_portfolio import portfolio_cvar, optimize_min_cvar
                data   = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                avail  = [t for t in tickers if t in closes.columns]
                rets   = np.log(closes[avail]/closes[avail].shift(1)).dropna().values
                n      = rets.shape[1]
                wts    = b.get('weights')
                w      = np.array(wts,float)/sum(wts) if wts else np.ones(n)/n
                alpha  = float(b.get('alpha',0.05))
                result = {'current': portfolio_cvar(w, rets, alpha)}
                if b.get('optimize'): result['optimized'] = optimize_min_cvar(rets, alpha)
                return send_json(self, _clean(result))

            send_err(self, f'Unknown portfolio POST: {p}', 404)
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
