import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))

from helpers import ok, err, qs

def handler(request, response):
    if request.method == "OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np
        from datetime import datetime, date
        ticker  = (qs(request, "ticker") or "AAPL").upper()
        expiry  = qs(request, "expiry") or ""
        opttype = qs(request, "type") or "call"

        tkr  = yf.Ticker(ticker)
        hist = tkr.history(period="2d", auto_adjust=True)
        spot = float(hist["Close"].dropna().iloc[-1])
        info = tkr.info or {}
        q    = float(info.get("dividendYield") or 0.0)
        r    = 0.05

        if not expiry or expiry not in (tkr.options or []):
            expiry = (tkr.options or [expiry])[0]

        exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = max((exp_dt - date.today()).days / 365.0, 1/365)

        chain_data = tkr.option_chain(expiry)
        df = chain_data.calls if opttype == "call" else chain_data.puts
        df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2

        rows = []
        for _, row in df.iterrows():
            K = float(row["strike"])
            if K < spot * 0.5 or K > spot * 2.0: continue
            rows.append({
                "strike":    round(K, 2),
                "moneyness": round(K/spot, 4),
                "bid":       round(float(row["bid"]), 3),
                "ask":       round(float(row["ask"]), 3),
                "mid":       round(float(row["mid"]), 3),
                "iv_yf":     round(float(row.get("impliedVolatility") or 0)*100, 3),
                "volume":    int(row.get("volume") or 0),
                "oi":        int(row.get("openInterest") or 0),
            })
        return ok({"ticker": ticker, "expiry": expiry, "T": round(T,4),
                   "spot": round(spot,4), "r": r, "q": round(q,6),
                   "type": opttype, "chain": rows})
    except Exception as e: return err(str(e))
