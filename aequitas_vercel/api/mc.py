import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '_lib'))
from helpers import send_json, send_err, send_cors, read_body, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_GET(self):
        p = self.path
        try:
            if '/mc_quote' in p or '/quote' in p:
                import yfinance as yf, numpy as np
                ticker = (get_qs(self, 'ticker') or 'AAPL').upper()
                tkr    = yf.Ticker(ticker)
                hist   = tkr.history(period='1y', auto_adjust=True)
                info   = tkr.info or {}
                closes = hist['Close'].dropna()
                spot   = float(closes.iloc[-1])
                rets   = np.log(closes / closes.shift(1)).dropna().values
                return send_json(self, {
                    'ticker': ticker,
                    'spot':   round(spot, 4),
                    'sigma':  round(float(rets.std(ddof=1) * 252**0.5), 6),
                    'mu':     round(float(rets.mean() * 252), 6),
                    'beta':   round(float(info.get('beta') or 1.0), 4),
                    'hi52':   round(float(closes.max()), 4),
                    'lo52':   round(float(closes.min()), 4),
                    'q':      round(float(info.get('dividendYield') or 0.0), 6),
                    'name':   info.get('longName') or ticker,
                })
            # status endpoint — always done (sync compute)
            send_json(self, {'status': 'done', 'progress': 'Complete'})
        except Exception as e:
            send_err(self, str(e))

    def do_POST(self):
        try:
            from engine_mc import SimConfig, run_full_suite
            p = read_body(self)
            cfg = SimConfig(
                S0=float(p.get('S0', 175)), mu=float(p.get('mu', 0.12)),
                sigma=float(p.get('sigma', 0.25)), r=float(p.get('r', 0.05)),
                q=float(p.get('q', 0.0)), T=float(p.get('T', 1.0)),
                n_sims=min(int(float(p.get('n_sims', 50000))), 50000),
                K=float(p.get('K', 180)), barrier=float(p.get('barrier', 140)),
                option_type=str(p.get('option_type', 'call')),
                investment=float(p.get('investment', 10000)),
                v0=float(p.get('v0', 0.04)), theta=float(p.get('theta', 0.04)),
                kappa=float(p.get('kappa', 2.0)), xi=float(p.get('xi', 0.3)),
                rho=float(p.get('rho', -0.7)), lam=float(p.get('lam', 0.75)),
                mu_j=float(p.get('mu_j', -0.05)), sig_j=float(p.get('sig_j', 0.10)),
                alpha=float(p.get('alpha', 0.25)), beta=float(p.get('beta', 0.5)),
                nu=float(p.get('nu', 0.4)), rho_s=float(p.get('rho_s', -0.3)),
            )
            result = run_full_suite(cfg)
            send_json(self, {'job_id': 'sync', 'status': 'done', 'result': result})
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
