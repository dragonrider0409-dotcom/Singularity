import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

def _clean(obj):
    if isinstance(obj, float): return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_clean(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj); return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray): return _clean(obj.tolist())
    except ImportError: pass
    return obj

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self): send_err(self, 'Use POST', 405)

    def do_POST(self):
        p = self.path
        try:
            import numpy as np, yfinance as yf, time
            b = read_body(self)

            if '/scan' in p:
                from engine_pairs import scan_universe, scan_single
                DEFAULT_UNIVERSE = ['SPY','QQQ','GLD','SLV','XOM','CVX',
                                     'KO','PEP','JPM','BAC','WMT','TGT',
                                     'MSFT','AAPL','AMZN','META']
                raw    = b.get('tickers', DEFAULT_UNIVERSE)
                tickers= [t.strip().upper() for t in raw] if isinstance(raw,list) \
                         else [t.strip().upper() for t in raw.split(',')]
                period = b.get('period', '2y')
                min_hl = float(b.get('min_half_life', 1.0))
                max_hl = float(b.get('max_half_life', 60.0))
                t0     = time.perf_counter()

                data   = yf.download(tickers, period=period, auto_adjust=True,
                                      progress=False, threads=True)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                avail  = [t for t in tickers if t in closes.columns]
                closes = closes[avail].dropna(how='all').ffill()

                df  = scan_universe(closes, min_hl, max_hl)
                sdf = scan_single(closes)
                elapsed = round(time.perf_counter()-t0, 2)
                n_pairs = len(avail)*(len(avail)-1)//2

                return send_json(self, _clean({
                    'pairs':    df.to_dict('records') if not df.empty else [],
                    'singles':  sdf.to_dict('records') if not sdf.empty else [],
                    'n_tested': n_pairs,
                    'n_found':  len(df),
                    'tickers':  avail,
                    'elapsed_s':elapsed,
                    'status':   'done',
                    'job_id':   'sync',
                    'dates':    [str(d.date()) for d in closes.index],
                    'prices':   {t: closes[t].round(4).tolist() for t in avail},
                }))

            elif '/pair' in p:
                from engine_pairs import (engle_granger, compute_spread,
                    zscore, PairsConfig, backtest_pair)
                tk1      = b.get('ticker_y','GLD').upper()
                tk2      = b.get('ticker_x','SLV').upper()
                period   = b.get('period', '2y')
                entry_z  = float(b.get('entry_z', 2.0))
                exit_z   = float(b.get('exit_z',  0.5))
                stop_z   = float(b.get('stop_z',  4.0))
                z_win    = int(b.get('z_window',  60))
                notional = float(b.get('notional', 100_000))
                tc_bps   = float(b.get('tc_bps',  5.0))

                data   = yf.download([tk1,tk2], period=period, auto_adjust=True, progress=False)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                if tk1 not in closes.columns or tk2 not in closes.columns:
                    return send_err(self, f'Could not fetch {tk1} or {tk2}', 422)

                p1 = closes[tk1].values.astype(float)
                p2 = closes[tk2].values.astype(float)
                dates = [str(d.date()) for d in closes.index]

                eg  = engle_granger(p1, p2)
                cfg = PairsConfig(entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                                   z_window=z_win, notional=notional, tc_bps=tc_bps)
                bt  = backtest_pair(p1, p2, eg['beta'], eg['alpha'], cfg)
                sp  = compute_spread(p1, p2, eg['beta'], eg['alpha'])
                zs  = zscore(sp, z_win)
                return send_json(self, _clean({
                    'ticker_y': tk1, 'ticker_x': tk2, 'dates': dates,
                    'price_y':  np.round(p1,4).tolist(),
                    'price_x':  np.round(p2,4).tolist(),
                    'spread':   np.round(sp,6).tolist(),
                    'zscore':   [round(float(v),4) if not (isinstance(v,float) and math.isnan(v)) else None for v in zs],
                    'eg': eg, 'backtest': bt,
                    'config': {'entry_z':entry_z,'exit_z':exit_z,'stop_z':stop_z,
                               'z_window':z_win,'notional':notional,'tc_bps':tc_bps},
                }))

            elif '/johansen' in p:
                from engine_pairs import johansen_trace
                tickers = [t.upper() for t in b.get('tickers',[])]
                if len(tickers) < 2: return send_err(self, 'Need at least 2 tickers', 400)
                data   = yf.download(tickers, period=b.get('period','2y'),
                                      auto_adjust=True, progress=False, threads=True)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                avail  = [t for t in tickers if t in closes.columns]
                result = johansen_trace(closes[avail].dropna().values)
                result['tickers'] = avail
                return send_json(self, _clean(result))

            send_err(self, f'Unknown pairs POST endpoint: {p}', 404)
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
