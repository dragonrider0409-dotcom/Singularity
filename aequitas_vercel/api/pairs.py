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

            if '/scan' in p:
                from engine_pairs import scan_universe, scan_single
                import time
                tickers = b.get('tickers', ['GLD','SLV','XOM','CVX','KO','PEP','JPM','BAC'])
                period  = b.get('period', '2y')
                min_hl  = float(b.get('min_half_life', 1))
                max_hl  = float(b.get('max_half_life', 60))
                t0      = time.perf_counter()
                data    = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
                closes  = data['Close'] if hasattr(data.columns,'levels') else data
                avail   = [t for t in tickers if t in closes.columns]
                closes  = closes[avail].dropna(how='all').ffill()
                pairs_df  = scan_universe(closes, min_hl, max_hl)
                single_df = scan_single(closes)
                elapsed   = round(time.perf_counter()-t0, 2)
                return send_json(self, {
                    'pairs':   pairs_df.to_dict('records') if not pairs_df.empty else [],
                    'singles': single_df.to_dict('records') if not single_df.empty else [],
                    'n_tested': len(avail)*(len(avail)-1)//2, 'n_found': len(pairs_df),
                    'tickers': avail, 'elapsed_s': elapsed,
                    'dates':   [str(d.date()) for d in closes.index],
                    'prices':  {t: closes[t].round(4).tolist() for t in avail}})

            if '/pair' in p:
                from engine_pairs import engle_granger, compute_spread, zscore, PairsConfig, backtest_pair
                tk1    = b.get('ticker_y','GLD').upper()
                tk2    = b.get('ticker_x','SLV').upper()
                period = b.get('period','2y')
                cfg    = PairsConfig(entry_z=float(b.get('entry_z',2)), exit_z=float(b.get('exit_z',0.5)),
                            stop_z=float(b.get('stop_z',4)), z_window=int(b.get('z_window',60)),
                            notional=float(b.get('notional',100000)), tc_bps=float(b.get('tc_bps',5)))
                data   = yf.download([tk1,tk2], period=period, auto_adjust=True, progress=False)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                if tk1 not in closes.columns or tk2 not in closes.columns:
                    return send_err(self, f'Could not fetch {tk1} or {tk2}')
                p1, p2 = closes[tk1].values, closes[tk2].values
                dates  = [str(d.date()) for d in closes.index]
                eg     = engle_granger(p1, p2)
                bt     = backtest_pair(p1, p2, eg['beta'], eg['alpha'], cfg)
                sp     = compute_spread(p1, p2, eg['beta'], eg['alpha'])
                zs     = zscore(sp, cfg.z_window)
                return send_json(self, {'ticker_y':tk1,'ticker_x':tk2,'dates':dates,
                    'price_y':np.round(p1,4).tolist(),'price_x':np.round(p2,4).tolist(),
                    'spread':np.round(sp,6).tolist(),
                    'zscore':[round(float(v),4) if v==v else None for v in zs],
                    'eg':eg,'backtest':bt,'config':cfg.__dict__})

            if '/johansen' in p:
                from engine_pairs import johansen_trace
                tickers = [t.upper() for t in b.get('tickers',[])]
                if len(tickers) < 2: return send_err(self, 'Need at least 2 tickers', 400)
                data   = yf.download(tickers, period=b.get('period','2y'), auto_adjust=True, progress=False, threads=True)
                closes = data['Close'] if hasattr(data.columns,'levels') else data
                avail  = [t for t in tickers if t in closes.columns]
                result = johansen_trace(closes[avail].dropna().values)
                result['tickers'] = avail
                return send_json(self, result)

            send_err(self, 'Unknown pairs endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def log_message(self, *a): pass
