import sys,os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','_lib'))

from helpers import ok, err, body as gb
import math

def handler(req, res):
    if req.method=="OPTIONS": return ok({})
    try:
        import yfinance as yf, numpy as np
        from vol_engine import (garch11, gjr_garch, har_rv, vol_forecast_with_bands,
                                realized_vol, hmm_em, kalman_vol, regime_conditional_stats)
        b        = gb(req)
        ticker   = b.get("ticker","SPY").upper()
        period   = b.get("period","5y")
        n_states = int(b.get("n_states",2))

        data   = yf.download(ticker,period=period,auto_adjust=True,progress=False)
        closes = data["Close"].dropna()
        if isinstance(closes,__import__("pandas").DataFrame): closes=closes.iloc[:,0]
        rets   = np.log(closes/closes.shift(1)).dropna().values.flatten().astype(float)
        dates  = [str(d.date()) for d in closes.index[1:]]
        prices = closes.values[1:].astype(float).flatten()

        g11    = garch11(rets)
        gjr    = gjr_garch(rets)
        fcast  = vol_forecast_with_bands(g11,60,0.90)
        rv     = realized_vol(rets,[5,21,63])
        rv21c  = [v for v in rv["rv_21d"] if not(isinstance(v,float) and math.isnan(v))]
        har    = har_rv(np.array(rv21c)) if len(rv21c)>=30 else {}
        kv     = kalman_vol(rets).tolist()
        obs    = np.abs(rets)*100
        hmm    = hmm_em(obs,n_states=n_states)
        stats  = regime_conditional_stats(rets,hmm["states"],hmm["state_names"])

        # Clean NaN from rv dict
        rv_clean = {k:[None if(isinstance(x,float) and math.isnan(x)) else x for x in v]
                    for k,v in rv.items()}

        return ok({"ticker":ticker,"period":period,"n_obs":len(rets),
                   "dates":dates,"prices":np.round(prices,4).tolist(),
                   "returns":np.round(rets*100,4).tolist(),
                   "ann_ret":round(float(rets.mean()*252*100),4),
                   "ann_vol":round(float(rets.std(ddof=1)*(252**0.5)*100),4),
                   "current_vol":round(g11["cond_vol_ann"][-1],4),
                   "garch11":g11,"gjr_garch":gjr,"forecast_bands":fcast,
                   "forecast":fcast.get("forecast",fcast),
                   "realized_vol":rv_clean,"har":har,"kalman_vol":kv,
                   "hmm":hmm,"regime_stats":stats.to_dict("records")})
    except Exception as e:
        import traceback; return err(f"{e}")
