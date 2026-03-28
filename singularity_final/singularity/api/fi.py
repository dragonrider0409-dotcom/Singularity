import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_GET(self):
        try:
            if '/treasury' in self.path:
                from data_cache import get_treasury_yields
                import numpy as np
                STD_MATS   = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
                DEMO_YIELDS= {0.25:5.27,0.5:5.22,1:5.01,2:4.70,3:4.55,5:4.40,
                               7:4.38,10:4.35,20:4.55,30:4.48}
                try:
                    return send_json(self, get_treasury_yields())
                except Exception as e:
                    mats = STD_MATS
                    ylds = [DEMO_YIELDS.get(m, 4.5) for m in mats]
                    return send_json(self, {'maturities':mats,'yields':ylds,'source':'demo','error':str(e)})
            send_err(self, 'Unknown FI GET endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def do_POST(self):
        p = self.path
        try:
            import numpy as np
            b = read_body(self)

            elif '/curve' in p:
                from engine_fi import full_curve_output
                mats   = np.array(b['maturities'], float)
                yields = np.array(b['yields'], float)
                if len(mats) != len(yields):
                    return send_err(self, 'maturities and yields must have same length', 422)
                if len(mats) < 3:
                    return send_err(self, 'Need at least 3 points', 422)
                return send_json(self, full_curve_output(mats, yields, b.get('model','nelson_siegel')))

            elif '/bond' in p:
                from engine_fi import (Bond, bond_ytm, full_analytics,
                    price_change_approx, bond_price)
                bnd = Bond(
                    face        = float(b.get('face', 1000)),
                    coupon_rate = float(b.get('coupon_rate', 0.05)),
                    maturity    = float(b.get('maturity', 10.0)),
                    freq        = int(b.get('freq', 2)),
                )
                if 'ytm' in b:    ytm = float(b['ytm'])
                elif 'price' in b: ytm = bond_ytm(bnd, float(b['price']))
                else:              ytm = float(b.get('ytm', 0.045))
                if isinstance(ytm, float) and math.isnan(ytm):
                    return send_err(self, 'Could not solve for YTM', 422)

                an = full_analytics(bnd, ytm)
                scenarios = {}
                for bps in [-300,-200,-100,-50,50,100,200,300]:
                    sc = price_change_approx(bnd, ytm, bps/10000)
                    scenarios[f"{'+'if bps>0 else''}{bps}bps"] = {
                        'price':       sc['price_exact'],
                        'pct_change':  sc['pct_change'],
                        'approx_error':sc['approx_error'],
                    }
                # Cash flow schedule
                c   = bnd.face * bnd.coupon_rate / bnd.freq
                n   = int(round(bnd.maturity * bnd.freq))
                r_  = ytm / bnd.freq
                ts  = [(i+1)/bnd.freq for i in range(n)]
                cfs = [c if i < n-1 else c + bnd.face for i in range(n)]
                pvs = [cf/(1+r_)**(i+1) for i,cf in enumerate(cfs)]
                cashflows = [{'t':round(t,4),'cf':round(cf,4),'pv':round(pv,4)}
                              for t,cf,pv in zip(ts,cfs,pvs)]
                return send_json(self, {**an, 'scenarios':scenarios, 'cashflows':cashflows,
                    'bond':{'face':bnd.face,'coupon_rate':bnd.coupon_rate,
                            'maturity':bnd.maturity,'freq':bnd.freq}})

            elif '/swap' in p:
                from engine_fi import par_swap_rate, swap_cashflows, swap_dv01
                from data_cache import get_treasury_yields
                notional = float(b.get('notional', 1_000_000))
                mat      = float(b.get('maturity', 5.0))
                freq     = int(b.get('freq', 4))
                if 'maturities' in b and 'yields' in b:
                    mats   = np.array(b['maturities'], float)
                    yields = np.array(b['yields'], float) / 100
                else:
                    tsy    = get_treasury_yields()
                    mats   = np.array(tsy['maturities'])
                    yields = np.array(tsy['yields']) / 100
                par_sr  = par_swap_rate(mats, yields, mat, freq)
                fixed_r = float(b.get('fixed_rate', par_sr*100)) / 100 or par_sr
                cfs = swap_cashflows(notional, fixed_r, mat, mats, yields, freq)
                d1  = swap_dv01(notional, fixed_r, mat, mats, yields, freq)
                return send_json(self, {
                    'par_swap_rate': round(par_sr*100, 6),
                    'fixed_rate':    round(fixed_r*100, 6),
                    'notional': notional, 'maturity': mat,
                    'mtm':      round(float(cfs['net_pv'].sum()), 4),
                    'dv01':     round(d1, 4),
                    'cashflows':cfs.to_dict('records'),
                    'n_payments':len(cfs),
                })

            elif '/scenario' in p:
                from engine_fi import Bond, bond_price, modified_duration, convexity
                bnd = Bond(face=float(b.get('face',1000)), coupon_rate=float(b.get('coupon_rate',0.05)),
                           maturity=float(b.get('maturity',10.0)), freq=int(b.get('freq',2)))
                ytm   = float(b.get('ytm', 0.045))
                shifts= list(range(-500, 525, 25))
                p0    = bond_price(bnd, ytm)
                d     = modified_duration(bnd, ytm)
                cv    = convexity(bnd, ytm)
                dur_c   = [-d*(bps/10000)*p0  for bps in shifts]
                conv_c  = [0.5*cv*(bps/10000)**2*p0 for bps in shifts]
                exact   = [bond_price(bnd, ytm+bps/10000)-p0 for bps in shifts]
                return send_json(self, {
                    'shifts_bps':        shifts,
                    'exact_pnl':         [round(v,4) for v in exact],
                    'duration_pnl':      [round(v,4) for v in dur_c],
                    'convexity_pnl':     [round(v,4) for v in conv_c],
                    'approx_pnl':        [round(d+c,4) for d,c in zip(dur_c,conv_c)],
                    'price_initial':     round(p0,4),
                    'ytm':               round(ytm,6),
                    'modified_duration': round(d,6),
                    'convexity':         round(cv,6),
                })

            send_err(self, f'Unknown FI POST endpoint: {p}', 404)
        except Exception as e:
            import traceback
            send_err(self, str(e) + ' | ' + traceback.format_exc().splitlines()[-1])

    def log_message(self, *a): pass
