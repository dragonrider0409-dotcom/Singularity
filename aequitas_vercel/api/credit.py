import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_POST(self):
        p = self.path
        try:
            import numpy as np
            from engine_credit import (merton_model, merton_calibrate, merton_term_structure,
                credit_curve, cds_par_spread, cds_mtm, cva_calculation, portfolio_credit_loss)
            b = read_body(self)

            def live_rf(T=1.0):
                try:
                    from data_cache import get_treasury_yields
                    tsy = get_treasury_yields()
                    return float(np.interp(T, tsy['maturities'], np.array(tsy['yields'])/100))
                except: return 0.05

            if '/ticker_merton' in p:
                from data_cache import get_merton_inputs
                ticker = b.get('ticker','AAPL').upper()
                T      = float(b.get('maturity',1.0))
                r      = live_rf(T)
                inp    = get_merton_inputs(ticker)
                spot   = inp['spot']; sig_E = inp['equity_vol']
                E_cap  = inp['equity_value']; D = inp['total_debt']
                shares = inp.get('shares_out') or E_cap/max(spot,1e-8)
                D_sh   = D/max(shares,1)
                result = (merton_model(spot,sig_E,max(D_sh,0.001),r,T) if D<=0
                          else merton_calibrate(spot,sig_E,D_sh,r,T))
                if D<=0: result['note']='Minimal debt'
                V_imp  = result.get('equity_value',spot)+D_sh
                sig_imp= result.get('equity_vol',sig_E)*spot/max(V_imp,1e-8)
                ts     = merton_term_structure(V_imp,sig_imp,D_sh,r,[0.25,0.5,1,2,3,5,7,10])
                return send_json(self, {'ticker':ticker,'name':inp.get('name',ticker),
                    'sector':inp.get('sector'),'spot':round(spot,4),
                    'equity_value_total':round(E_cap,0),'total_debt':round(D,0),
                    'equity_vol':round(sig_E,6),'risk_free':round(r,6),
                    'result':result,'term_structure':ts})

            if '/merton' in p:
                V      = float(b.get('asset_value',100)); sig_V = float(b.get('asset_vol',0.25))
                D      = float(b.get('debt',80)); T = float(b.get('maturity',1.0))
                r      = live_rf(T)
                result = (merton_calibrate(float(b.get('equity_obs',V*0.3)),
                              float(b.get('equity_vol',sig_V*1.5)),D,r,T)
                          if b.get('calibrate') else merton_model(V,sig_V,D,r,T))
                ts     = merton_term_structure(V,sig_V,D,r,[0.25,0.5,1,2,3,5,7,10])
                grid   = np.linspace(D*0.5,D*2.5,60)
                sens   = [merton_model(v,sig_V,D,r,T) for v in grid]
                return send_json(self, {'result':result,'term_structure':ts,'risk_free':round(r,6),
                    'sensitivity':{'asset_values':np.round(grid,4).tolist(),
                        'pd_rn':[s['pd_risk_neutral'] for s in sens],
                        'equity':[s['equity_value'] for s in sens],
                        'spread_bps':[s['credit_spread_bps'] for s in sens],
                        'dd':[s['distance_to_default'] for s in sens]}})

            if '/cds' in p:
                spreads    = b.get('spreads',[50,80,110,140,160])
                maturities = b.get('maturities',[1,2,3,5,7])
                recovery   = float(b.get('recovery',0.40))
                notional   = float(b.get('notional',1e6))
                r          = live_rf(1.0)
                cc         = credit_curve(spreads,maturities,recovery,r)
                hz         = np.interp(maturities,maturities,cc['hazard_rates'])
                ps         = [round(cds_par_spread(h/100,recovery,r,T),2) for h,T in zip(hz,maturities)]
                hm         = float(np.mean(cc['hazard_rates']))/100
                sp         = spreads[min(3,len(spreads)-1)]
                return send_json(self, {'credit_curve':cc,'par_spreads':ps,'recovery':recovery,
                    'risk_free':round(r,6),
                    'mtm_base':   cds_mtm(hm,hm,     recovery,r,5.0,sp,notional),
                    'mtm_wide50': cds_mtm(hm,hm*1.5,recovery,r,5.0,sp,notional)})

            if '/cva' in p:
                T        = float(b.get('maturity',5)); notional = float(b.get('notional',1e6))
                hazard   = float(b.get('hazard',0.02)); recovery = float(b.get('recovery',0.40))
                profile  = b.get('profile','declining'); r = live_rf(T)
                times    = list(np.round(np.arange(0.25,T+0.25,0.25),4))
                t_arr    = np.array(times); n = len(times)
                exp      = ([notional*0.5]*n if profile=='flat'
                            else (notional*0.8*t_arr/T*np.exp(-t_arr/T*2)).tolist() if profile=='hump'
                            else (notional*np.exp(-0.3*t_arr)).tolist())
                result   = cva_calculation(exp,times,hazard,recovery,r)
                result['risk_free'] = round(r,6)
                return send_json(self, result)

            if '/portfolio_credit' in p:
                n        = int(b.get('n_names',10)); notional = float(b.get('notional',1e6))
                pd_avg   = float(b.get('pd',0.02));  lgd_avg  = float(b.get('lgd',0.60))
                corr     = float(b.get('correlation',0.20))
                rng      = np.random.default_rng(42)
                pds      = np.clip(rng.lognormal(np.log(pd_avg),0.5,n),0.001,0.50)
                pds      = (pds/pds.mean()*pd_avg).tolist()
                result   = portfolio_credit_loss([notional]*n,pds,[lgd_avg]*n,corr)
                result['pds'] = [round(x,6) for x in pds]
                return send_json(self, result)

            send_err(self, 'Unknown credit endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def log_message(self, *a): pass
