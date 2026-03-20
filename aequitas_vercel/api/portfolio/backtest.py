import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))
from helpers import ok, err, body as gb

def handler(req, res):
    """
    Rolling backtest — synchronous for Vercel (no job queue needed).
    Vercel functions can run up to 30s so we compute inline.
    """
    if req.method == "OPTIONS":
        return ok({})
    try:
        import yfinance as yf, numpy as np, pandas as pd
        from portfolio_engine import rolling_backtest, ledoit_wolf_cov
        import time

        b        = gb(req)
        tickers  = b.get("tickers", ["SPY","QQQ","TLT","GLD"])
        period   = b.get("period", "5y")
        if period in ("1y","2y"):
            period = "5y"
        method   = b.get("method", "max_sharpe")
        lookback = int(b.get("lookback", 252))
        rebal    = int(b.get("rebalance_every", 21))
        rf       = float(b.get("rf", 0.05))
        cov_meth = b.get("cov_method", "ledoit_wolf")

        t0   = time.perf_counter()
        data = yf.download(tickers, period=period, auto_adjust=True,
                           progress=False, threads=True)
        closes = data["Close"] if hasattr(data.columns, "levels") else data
        avail  = [t for t in tickers if t in closes.columns]
        rets   = np.log(closes[avail] / closes[avail].shift(1)).dropna()

        bt = rolling_backtest(rets, method=method, lookback=lookback,
                              rebalance_every=rebal, rf=rf, cov_method=cov_meth)
        elapsed = round(time.perf_counter() - t0, 2)
        bt["elapsed_s"] = elapsed
        bt["method"]    = method
        bt["tickers"]   = avail

        # Return as a direct result (no job_id needed)
        return ok({"status": "done", "progress": f"Done in {elapsed}s",
                   "result": bt, "job_id": "sync"})
    except Exception as e:
        import traceback
        return err(str(e))
