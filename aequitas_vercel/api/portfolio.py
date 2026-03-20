import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '_lib'))
from helpers import send_json, send_err, send_cors, read_body, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_GET(self):
        try:
            if '/prices' in self.path:
                import yfinance as yf, numpy as np
                raw     = get_qs(self, 'tickers') or 'SPY,QQQ,AAPL'
                period  = get_qs(self, 'period') or '2y'
                tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]
                data    = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
                closes  = data['Close'] if hasattr(data.columns, 'levels') else data
                avail   = [t for t in tickers if t in closes.columns]
                closes  = closes[avail].dropna(how='all').ffill()
                return send_json(self, {'tickers': avail,
                    'dates':  [str(d.date()) for d in closes.index],
                    'prices': {t: closes[t].round(4).tolist() for t in avail}})
            if '/job' in self.path:
                return send_json(self, {'status': 'done', 'progress': 'Complete'})
            send_err(self, 'Unknown portfolio GET endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def do_POST(self):
        p = self.path
        try:
            import yfinance as yf, numpy as np
            b       = read_body(self)
            tickers = b.get('tickers', ['SPY','QQQ','TLT','GLD','IWM'])
            period  = b.get('period', '2y')

            if '/optimize' in p:
                from engine_portfolio import (mean_variance_frontier, black_litterman,
                    risk_parity, ledoit_wolf_cov, rolling_backtest, factor_decomposition)
                data    = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
                closes  = data['Close'] if hasattr(data.columns,'levels') else data
                avail   = [t for t in tickers if t in closes.columns]
                rets    = np.log(closes[avail]/closes[avail].shift(1)).dropna()
                R       = rets.values
                dates   = [str(d.date()) for d in rets.index]
                cov     = ledoit_wolf_cov(R)
                mu      = R.mean(axis=0) * 252
                method  = b.get('method', 'markowitz')
                rf      = float(b.get('rf', 0.05))
                if method == 'black_litterman':
                    w = black_litterman(R, rf, b.get('views', []))['weights']
                elif method == 'risk_parity':
                    w = risk_parity(cov)['weights']
                else:
                    w = mean_variance_frontier(mu, cov, rf, n_points=60)['tangency_weights']
                port_r  = R @ w
                ann_r   = float(port_r.mean() * 252)
                ann_v   = float(port_r.std(ddof=1) * 252**0.5)
                return send_json(self, {'tickers': avail,
                    'weights':    dict(zip(avail, [round(float(x),6) for x in w])),
                    'ann_return': round(ann_r, 4), 'ann_vol': round(ann_v, 4),
                    'sharpe':     round(ann_r / max(ann_v, 1e-8), 4),
                    'backtest':   rolling_backtest(R, dates, w, method=method),
                    'factor':     factor_decomposition(R, w, avail),
                    'dates':      dates,
                    'returns':    np.round(port_r*100, 4).tolist()})

            if '/backtest' in p:
                from engine_portfolio import rolling_backtest
                import time
                if period in ('1y','2y'): period = '5y'
                method   = b.get('method', 'max_sharpe')
                lookback = int(b.get('lookback', 252))
                rebal    = int(b.get('rebalance_every', 21))
                rf       = float(b.get('rf', 0.05))
                t0       = time.perf_counter()
                data     = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
                closes   = data['Close'] if hasattr(data.columns,'levels') else data
                avail    = [t for t in tickers if t in closes.columns]
                rets     = np.log(closes[avail]/closes[avail].shift(1)).dropna()
                bt       = rolling_backtest(rets, method=method, lookback=lookback,
                                            rebalance_every=rebal, rf=rf)
                elapsed  = round(time.perf_counter()-t0, 2)
                bt['elapsed_s'] = elapsed; bt['method'] = method; bt['tickers'] = avail
                return send_json(self, {'status':'done','progress':f'Done in {elapsed}s','result':bt,'job_id':'sync'})

            if '/cvar' in p:
                from engine_portfolio import portfolio_cvar, optimize_min_cvar
                data   = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                avail  = [t for t in tickers if t in closes.columns]
                rets   = np.log(closes[avail]/closes[avail].shift(1)).dropna().values
                n      = rets.shape[1]
                wts    = b.get('weights')
                w      = np.array(wts,float)/sum(wts) if wts else np.ones(n)/n
                alpha  = float(b.get('alpha', 0.05))
                result = {'current': portfolio_cvar(w, rets, alpha)}
                if b.get('optimize'): result['optimized'] = optimize_min_cvar(rets, alpha)
                return send_json(self, result)

            send_err(self, 'Unknown portfolio POST endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def log_message(self, *a): pass
