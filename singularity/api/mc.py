import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body, get_qs
from http.server import BaseHTTPRequestHandler

def _s(v):
    """safe float — replace nan/inf with None"""
    if v is None: return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    return v

def _price(x):
    return float(x['price']) if isinstance(x, dict) else float(x)

def _se(x):
    return float(x.get('se', 0)) if isinstance(x, dict) else 0.0

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_GET(self):
        try:
            import yfinance as yf, numpy as np
            p = self.path
            if '/quote' in p or '/mc_quote' in p:
                ticker = (get_qs(self, 'ticker') or 'AAPL').upper()
                tkr  = yf.Ticker(ticker)
                hist = tkr.history(period='1y', auto_adjust=True)
                info = tkr.info or {}
                closes = hist['Close'].dropna()
                spot   = float(closes.iloc[-1])
                rets   = np.log(closes/closes.shift(1)).dropna().values
                ann_sig = float(rets.std(ddof=1)*252**0.5)
                ann_mu  = float(rets.mean()*252)
                return send_json(self, {
                    'ticker':    ticker,
                    'spot':      round(spot, 4),
                    'ann_sigma': round(ann_sig, 6),
                    'ann_mu':    round(ann_mu, 6),
                    'sigma':     round(ann_sig, 6),
                    'mu':        round(ann_mu, 6),
                    'beta':      round(float(info.get('beta') or 1.0), 4),
                    'wk52_hi':   round(float(closes.max()), 4),
                    'wk52_lo':   round(float(closes.min()), 4),
                    'hi52':      round(float(closes.max()), 4),
                    'lo52':      round(float(closes.min()), 4),
                    'div_yield': round(float(info.get('dividendYield') or 0.0), 6),
                    'q':         round(float(info.get('dividendYield') or 0.0), 6),
                    'name':      info.get('longName') or ticker,
                })
            # status endpoint
            return send_json(self, {'status': 'done', 'progress': 'Complete'})
        except Exception as e:
            send_err(self, str(e))

    def do_POST(self):
        try:
            import numpy as np
            from engine_mc import SimConfig, run_full_suite

            p   = read_body(self)
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

            res = run_full_suite(cfg)

            # ── Build paths arrays (downsample for JSON) ─────────────────
            def sample_paths(arr, n_paths=40, n_steps=80):
                if arr is None: return []
                a = np.array(arr)
                if a.ndim < 2: return []
                # Engine returns (n_steps, n_sims) — transpose to (n_sims, n_steps)
                if a.shape[0] > a.shape[1]:
                    a = a.T
                step = max(1, a.shape[1]//n_steps)
                idx  = np.random.default_rng(0).choice(a.shape[0], min(n_paths, a.shape[0]), replace=False)
                return np.round(a[idx, ::step], 4).tolist()

            def path_pcts(arr, n_steps=80):
                if arr is None: return {}
                a = np.array(arr)
                if a.ndim < 2: return {}
                # Engine returns (n_steps, n_sims) — transpose to (n_sims, n_steps)
                if a.shape[0] > a.shape[1]:
                    a = a.T
                step = max(1, a.shape[1]//n_steps)
                a = a[:, ::step]
                return {
                    'p10': np.round(np.percentile(a, 10, axis=0), 4).tolist(),
                    'p25': np.round(np.percentile(a, 25, axis=0), 4).tolist(),
                    'p50': np.round(np.percentile(a, 50, axis=0), 4).tolist(),
                    'p75': np.round(np.percentile(a, 75, axis=0), 4).tolist(),
                    'p90': np.round(np.percentile(a, 90, axis=0), 4).tolist(),
                }

            gbm_rw = res.get('gbm_rw')
            heston = res.get('heston_S')
            jd     = res.get('jd_paths')
            sabr   = res.get('sabr_paths')
            n_steps_real = gbm_rw.shape[1] if gbm_rw is not None and hasattr(gbm_rw,'shape') else 252
            t_ax   = np.round(np.linspace(0, cfg.T, min(81, n_steps_real+1)), 6).tolist()

            opts   = res.get('options', {})
            greeks_bs = opts.get('greeks_bs', {})
            greeks_mc = opts.get('greeks_mc', {})
            barrier_raw = opts.get('barrier', {})
            risk   = res.get('risk', {})

            def flatten_risk(r_):
                if not r_: return {}
                return {k: _s(float(v)) if isinstance(v,(int,float)) else v
                        for k,v in r_.items()}

            # Terminal distribution histogram
            def term_hist(arr, bins=50):
                if arr is None: return {}
                a = np.array(arr)[:, -1]
                h, edges = np.histogram(a, bins=bins)
                return {'counts': h.tolist(), 'edges': np.round(edges,4).tolist()}

            # P&L histogram
            gbm_rn = res.get('gbm_rn')
            def pnl_hist_fn(arr, invest=10000, bins=60):
                if arr is None: return {}
                a    = np.array(arr)[:, -1]
                init = np.array(arr)[:,0]
                pnl  = (a / init - 1) * invest
                h, edges = np.histogram(pnl, bins=bins)
                return {'counts': h.tolist(), 'edges': np.round(edges,2).tolist()}

            # Convergence
            conv_df = res.get('convergence')
            conv_data = []
            if conv_df is not None and hasattr(conv_df,'to_dict'):
                conv_data = conv_df.to_dict('records')

            # Stress
            stress_df = res.get('stress')
            stress_data = []
            if stress_df is not None and hasattr(stress_df,'to_dict'):
                stress_data = stress_df.to_dict('records')

            # Portfolio
            port = res.get('portfolio', {})
            port_vals = port.get('values')
            port_risk = port.get('risk', {})

            shaped = {
                'ticker':   p.get('ticker', cfg.ticker),
                'n_sims':   cfg.n_sims,
                'elapsed_s': res.get('elapsed_s', 0),
                'K': cfg.K, 'S0': cfg.S0, 'T': cfg.T, 'r': cfg.r,
                't_ax': t_ax,
                'paths': {
                    'gbm':    sample_paths(gbm_rw),
                    'heston': sample_paths(heston),
                    'jd':     sample_paths(jd),
                    'sabr':   sample_paths(sabr),
                },
                'pcts': {
                    'gbm':    path_pcts(gbm_rw),
                    'heston': path_pcts(heston),
                    'jd':     path_pcts(jd),
                    'sabr':   path_pcts(sabr),
                },
                'term_hist': term_hist(gbm_rw),
                'pnl_hist':  pnl_hist_fn(gbm_rn, cfg.investment),
                'options': {
                    'bs':         _s(_price(opts.get('bs', 0))),
                    'eur_gbm':    _s(_price(opts.get('eur_gbm', 0))),
                    'eur_gbm_se': _s(_se(opts.get('eur_gbm', 0))),
                    'eur_heston': _s(_price(opts.get('eur_heston', 0))),
                    'eur_jd':     _s(_price(opts.get('eur_jd', 0))),
                    'eur_sabr':   _s(_price(opts.get('eur_sabr', 0))),
                    'asian_arith':_s(_price(opts.get('asian_arith', 0))),
                    'asian_geo':  _s(_price(opts.get('asian_geo', 0))),
                    'barrier':    _s(_price(barrier_raw) if barrier_raw else 0),
                    'ko_pct':     _s(float(barrier_raw.get('knock_out_pct',0)) if isinstance(barrier_raw,dict) else 0),
                    'lookback':   _s(_price(opts.get('lookback', 0))),
                    'digital':    _s(_price(opts.get('digital', 0))),
                },
                'greeks_bs': {k: _s(float(v)) for k,v in greeks_bs.items() if isinstance(v,(int,float))},
                'greeks_mc': {k: _s(float(v)) for k,v in greeks_mc.items() if isinstance(v,(int,float))},
                'risk': {
                    'gbm':    flatten_risk(risk.get('gbm',{})),
                    'heston': flatten_risk(risk.get('heston',{})),
                    'jd':     flatten_risk(risk.get('jd',{})),
                    'sabr':   flatten_risk(risk.get('sabr',{})),
                },
                'portfolio': {
                    'values': sample_paths(port_vals, n_paths=20),
                    'pcts':   path_pcts(port_vals),
                    'risk':   flatten_risk(port_risk),
                    'pnl_hist': pnl_hist_fn(port_vals, cfg.investment),
                },
                'stress': stress_data,
                'conv':   conv_data,
                # Convenience aliases
                'bs_price':  _s(_price(opts.get('bs', 0))),
                'mc_mean':   _s(_price(opts.get('eur_gbm', 0))),
                'var_95':    _s(float(risk.get('gbm',{}).get('VaR_95',0))),
                'cvar_99':   _s(float(risk.get('gbm',{}).get('CVaR_99',0))),
                'sharpe':    _s(float(risk.get('gbm',{}).get('sharpe',0))),
                'p_loss':    _s(float(risk.get('gbm',{}).get('prob_loss',0))),
            }

            send_json(self, {'job_id': 'sync', 'status': 'done', 'result': shaped})
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
