import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from helpers import send_json, send_err, send_cors, read_body, get_qs
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)

    def do_GET(self):
        try:
            if '/treasury' in self.path:
                from data_cache import get_treasury_yields
                return send_json(self, get_treasury_yields())
            send_err(self, 'Unknown FI GET endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def do_POST(self):
        p = self.path
        try:
            import numpy as np
            from engine_fi import (full_curve_output, Bond, bond_ytm, full_analytics,
                price_change_approx, par_swap_rate, swap_cashflows, swap_dv01,
                bond_price, modified_duration, convexity)
            b = read_body(self)

            if '/curve' in p:
                return send_json(self, full_curve_output(
                    np.array(b['maturities'], float), np.array(b['yields'], float),
                    b.get('model', 'nelson_siegel')))

            if '/bond' in p:
                bnd = Bond(face=float(b.get('face',1000)), coupon_rate=float(b.get('coupon_rate',0.05)),
                           maturity=float(b.get('maturity',10)), freq=int(b.get('freq',2)))
                ytm = (float(b['ytm']) if 'ytm' in b else
                       bond_ytm(bnd, float(b['price'])) if 'price' in b else 0.045)
                an  = full_analytics(bnd, ytm)
                sc  = {f"{'+'if bps>0 else''}{bps}bps":
                       {'price': price_change_approx(bnd,ytm,bps/10000)['price_exact'],
                        'pct':   price_change_approx(bnd,ytm,bps/10000)['pct_change']}
                       for bps in [-300,-200,-100,-50,50,100,200,300]}
                return send_json(self, {**an, 'scenarios': sc})

            if '/swap' in p:
                from data_cache import get_treasury_yields
                notional = float(b.get('notional', 1e6))
                mat      = float(b.get('maturity', 5))
                freq     = int(b.get('freq', 4))
                if 'maturities' in b:
                    mats   = np.array(b['maturities'], float)
                    yields = np.array(b['yields'], float) / 100
                else:
                    tsy    = get_treasury_yields()
                    mats   = np.array(tsy['maturities'])
                    yields = np.array(tsy['yields']) / 100
                par  = par_swap_rate(mats, yields, mat, freq)
                fr   = float(b.get('fixed_rate', par*100)) / 100 or par
                cfs  = swap_cashflows(notional, fr, mat, mats, yields, freq)
                return send_json(self, {'par_swap_rate': round(par*100,6), 'fixed_rate': round(fr*100,6),
                    'notional': notional, 'maturity': mat,
                    'mtm': round(float(cfs['net_pv'].sum()), 4),
                    'dv01': round(swap_dv01(notional,fr,mat,mats,yields,freq), 4),
                    'cashflows': cfs.to_dict('records')})

            if '/scenario' in p:
                bnd = Bond(face=float(b.get('face',1000)), coupon_rate=float(b.get('coupon_rate',0.05)),
                           maturity=float(b.get('maturity',10)), freq=int(b.get('freq',2)))
                ytm  = float(b.get('ytm', 0.045))
                p0   = bond_price(bnd, ytm)
                d    = modified_duration(bnd, ytm)
                cv   = convexity(bnd, ytm)
                shifts = list(range(-500, 525, 25))
                return send_json(self, {'shifts_bps': shifts,
                    'exact_pnl':     [round(bond_price(bnd,ytm+s/10000)-p0, 4) for s in shifts],
                    'duration_pnl':  [round(-d*(s/10000)*p0, 4) for s in shifts],
                    'convexity_pnl': [round(0.5*cv*(s/10000)**2*p0, 4) for s in shifts],
                    'price_initial': round(p0,4), 'ytm': round(ytm,6),
                    'modified_duration': round(d,6), 'convexity': round(cv,6)})

            send_err(self, 'Unknown FI POST endpoint', 404)
        except Exception as e:
            send_err(self, str(e))

    def log_message(self, *a): pass
