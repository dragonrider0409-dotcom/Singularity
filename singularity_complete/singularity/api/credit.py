import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

def _live_rf(T=1.0, fallback=0.05):
    try:
        from data_cache import get_treasury_yields
        import numpy as np
        tsy = get_treasury_yields()
        return float(np.interp(T, tsy['maturities'], np.array(tsy['yields'])/100))
    except: return fallback

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self): send_err(self, 'Use POST', 405)

    def do_POST(self):
        p = self.path
        try:
            import numpy as np
            from engine_credit import (merton_model, merton_calibrate,
                merton_term_structure, credit_curve, cds_par_spread, cds_mtm,
                cva_calculation, portfolio_credit_loss)
            b = read_body(self)

            if '/ticker_merton' in p:  # must come before the /merton check below
                from data_cache import get_merton_inputs
                ticker = b.get('ticker','AAPL').upper()
                T      = float(b.get('maturity', 1.0))
                r      = _live_rf(T)
                try:
                    inputs   = get_merton_inputs(ticker)
                    spot     = inputs['spot']
                    sig_E    = inputs['equity_vol']
                    E_mktcap = inputs['equity_value']
                    D        = inputs['total_debt']
                    shares   = inputs.get('shares_out') or E_mktcap/max(spot,1e-8)
                    if D <= 0:
                        result = merton_model(E_mktcap/max(shares,1), sig_E,
                                              max(1.0, E_mktcap/max(shares,1)*0.001), r, T)
                        result['note'] = 'No significant debt — credit risk negligible'
                    else:
                        D_sh   = D/max(shares,1)
                        result = merton_calibrate(spot, sig_E, D_sh, r, T)
                    mats  = [0.25,0.5,1,2,3,5,7,10]
                    V_imp = result.get('equity_value',spot) + D/max(shares,1)
                    s_imp = result.get('equity_vol',sig_E)*spot/max(V_imp,1e-8)
                    ts    = merton_term_structure(V_imp, s_imp, D/max(shares,1), r, mats)
                    return send_json(self, {
                        'ticker': ticker, 'name': inputs.get('name',ticker),
                        'sector': inputs.get('sector'),
                        'spot': round(spot,4),
                        'equity_value_total': round(E_mktcap,0),
                        'total_debt': round(D,0),
                        'equity_vol': round(sig_E,6),
                        'risk_free': round(r,6),
                        'result': result, 'term_structure': ts,
                    })
                except Exception as e2:
                    return send_err(self, f'ticker_merton error: {e2}')

            elif '/merton' in p:
                V      = float(b.get('asset_value', 100))
                sig_V  = float(b.get('asset_vol',   0.25))
                D      = float(b.get('debt',        80))
                T      = float(b.get('maturity',    1.0))
                r      = _live_rf(T, float(b.get('risk_free',0.05)))
                use_cal= bool(b.get('calibrate', False))
                if use_cal:
                    E_obs = float(b.get('equity_obs', V*0.3))
                    sig_E = float(b.get('equity_vol', sig_V*1.5))
                    result= merton_calibrate(E_obs, sig_E, D, r, T)
                else:
                    result= merton_model(V, sig_V, D, r, T)
                mats = [0.25,0.5,1,2,3,5,7,10]
                ts   = merton_term_structure(V, sig_V, D, r, mats)
                grid = np.linspace(D*0.5, D*2.5, 60)
                sens = [merton_model(v, sig_V, D, r, T) for v in grid]
                return send_json(self, {
                    'result': result, 'term_structure': ts,
                    'risk_free': round(r,6),
                    'sensitivity': {
                        'asset_values': np.round(grid,4).tolist(),
                        'pd_rn':      [s['pd_risk_neutral']     for s in sens],
                        'equity':     [s['equity_value']        for s in sens],
                        'spread_bps': [s['credit_spread_bps']   for s in sens],
                        'dd':         [s['distance_to_default']  for s in sens],
                    },
                })

            elif '/cds' in p:
                spreads_bps = b.get('spreads',    [50,80,110,140,160])
                maturities  = b.get('maturities', [1,2,3,5,7])
                recovery    = float(b.get('recovery', 0.40))
                notional    = float(b.get('notional', 1_000_000))
                r           = _live_rf(1.0, float(b.get('risk_free',0.04)))
                cc          = credit_curve(spreads_bps, maturities, recovery, r)
                hz          = np.interp(maturities, maturities, cc['hazard_rates'])
                par_spreads = [round(cds_par_spread(h/100,recovery,r,T),2)
                               for h,T in zip(hz,maturities)]
                hm  = float(np.mean(cc['hazard_rates']))/100
                sp  = spreads_bps[min(3,len(spreads_bps)-1)]
                return send_json(self, {
                    'credit_curve': cc, 'par_spreads': par_spreads,
                    'mtm_base':   cds_mtm(hm,hm,    recovery,r,5.0,sp,notional),
                    'mtm_wide50': cds_mtm(hm,hm*1.5,recovery,r,5.0,sp,notional),
                    'recovery': recovery, 'risk_free': round(r,6),
                })

            elif '/cva' in p:
                T        = float(b.get('maturity',   5.0))
                notional = float(b.get('notional',   1_000_000))
                hazard   = float(b.get('hazard',     0.02))
                recovery = float(b.get('recovery',   0.40))
                profile  = b.get('profile', 'declining')
                r        = _live_rf(T, float(b.get('risk_free',0.04)))
                times    = list(np.round(np.arange(0.25,T+0.25,0.25),4))
                t_arr    = np.array(times)
                n        = len(times)
                if profile=='flat':   exposure=[notional*0.5]*n
                elif profile=='hump': exposure=(notional*0.8*t_arr/T*np.exp(-t_arr/T*2)).tolist()
                else:                 exposure=(notional*np.exp(-0.3*t_arr)).tolist()
                result = cva_calculation(exposure, times, hazard, recovery, r)
                result['risk_free'] = round(r,6)
                return send_json(self, result)

            elif '/portfolio_credit' in p:
                n        = int(b.get('n_names',       10))
                notional = float(b.get('notional',    1_000_000))
                pd_avg   = float(b.get('pd',          0.02))
                lgd_avg  = float(b.get('lgd',         0.60))
                corr     = float(b.get('correlation', 0.20))
                pd_disp  = float(b.get('pd_dispersion',0.5))
                rng  = np.random.default_rng(42)
                pds  = np.clip(rng.lognormal(np.log(pd_avg),pd_disp,n),0.001,0.50)
                pds  = (pds/pds.mean()*pd_avg).tolist()
                result = portfolio_credit_loss([notional]*n, pds, [lgd_avg]*n, corr)
                result['pds'] = [round(x,6) for x in pds]
                return send_json(self, result)

            send_err(self, f'Unknown credit POST endpoint: {p}', 404)
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
