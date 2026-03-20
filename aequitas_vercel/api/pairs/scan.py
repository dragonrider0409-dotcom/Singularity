import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import send_json, send_err, send_cors, read_body
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): send_cors(self)
    def do_POST(self):
        try:
            import yfinance as yf, numpy as np, time
            from engine_pairs import scan_universe, scan_single
            b=read_body(self)
            tickers=b.get("tickers",["GLD","SLV","XOM","CVX","KO","PEP","JPM","BAC"])
            period=b.get("period","2y")
            min_hl=float(b.get("min_half_life",1)); max_hl=float(b.get("max_half_life",60))
            t0=time.perf_counter()
            data=yf.download(tickers,period=period,auto_adjust=True,progress=False,threads=True)
            closes=data["Close"] if hasattr(data.columns,"levels") else data
            avail=[t for t in tickers if t in closes.columns]
            closes=closes[avail].dropna(how="all").ffill()
            pairs_df=scan_universe(closes,min_hl,max_hl)
            single_df=scan_single(closes)
            elapsed=round(time.perf_counter()-t0,2)
            send_json(self,{"pairs":pairs_df.to_dict("records") if not pairs_df.empty else [],
                "singles":single_df.to_dict("records") if not single_df.empty else [],
                "n_tested":len(avail)*(len(avail)-1)//2,"n_found":len(pairs_df),
                "tickers":avail,"elapsed_s":elapsed,
                "dates":[str(d.date()) for d in closes.index],
                "prices":{t:closes[t].round(4).tolist() for t in avail}})
        except Exception as e: send_err(self, str(e))
    def log_message(self, *a): pass
