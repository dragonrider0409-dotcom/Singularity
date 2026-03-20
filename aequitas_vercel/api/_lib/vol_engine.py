"""
vol_regime/engine.py
=====================
Volatility & Regime Detection Dashboard

Merges:
  1. GARCH Volatility Forecaster  — GARCH(1,1), EGARCH, GJR-GARCH, HAR-RV
  2. Regime Detection             — 2/3-state HMM, Kalman filter vol, regime stats

Run:
  python vol_regime/app.py  ->  http://127.0.0.1:5005
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import List, Dict, Optional, Tuple
import warnings, logging

warnings.filterwarnings("ignore")
log = logging.getLogger("vol_regime")


# ══════════════════════════════════════════════════════════════════════════
#  GARCH FAMILY
# ══════════════════════════════════════════════════════════════════════════

def garch11(returns: np.ndarray) -> Dict:
    """
    Standard GARCH(1,1): h_t = omega + alpha*r_{t-1}^2 + beta*h_{t-1}
    MLE via Nelder-Mead. Returns params + full conditional variance series.
    """
    r = np.asarray(returns, float)
    n = len(r)

    def neg_ll(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        h = np.empty(n)
        h[0] = np.var(r)
        for t in range(1, n):
            h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
            if h[t] <= 1e-12:
                return 1e10
        return float(0.5 * np.sum(np.log(h[1:]) + r[1:]**2 / h[1:]))

    best, best_x = 1e18, [1e-6, 0.08, 0.88]
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.08, 0.88), (0.15, 0.80)]:
        v0 = np.var(r) * (1 - a0 - b0)
        res = minimize(neg_ll, [max(v0, 1e-8), a0, b0], method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10})
        if res.fun < best:
            best, best_x = res.fun, res.x

    omega, alpha, beta = best_x
    h = np.empty(n); h[0] = np.var(r)
    for t in range(1, n):
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        h[t] = max(h[t], 1e-12)

    lr_var = omega / max(1 - alpha - beta, 1e-8)
    return {
        "model":          "GARCH(1,1)",
        "omega":          round(float(omega), 9),
        "alpha":          round(float(alpha), 6),
        "beta":           round(float(beta), 6),
        "persistence":    round(float(alpha + beta), 6),
        "long_run_vol":   round(float(np.sqrt(lr_var * 252) * 100), 4),
        "cond_var":       np.round(h, 10).tolist(),
        "cond_vol_ann":   np.round(np.sqrt(h * 252) * 100, 4).tolist(),
    }


def gjr_garch(returns: np.ndarray) -> Dict:
    """
    GJR-GARCH(1,1): h_t = omega + (alpha + gamma*I_{r<0})*r_{t-1}^2 + beta*h_{t-1}
    Captures the leverage effect (negative shocks increase vol more than positive).
    """
    r = np.asarray(returns, float)
    n = len(r)

    def neg_ll(p):
        omega, alpha, gamma, beta = p
        if omega <= 0 or alpha < 0 or gamma < -alpha or beta < 0 or alpha + 0.5*gamma + beta >= 0.9999:
            return 1e10
        h = np.empty(n); h[0] = np.var(r)
        for t in range(1, n):
            ind = 1.0 if r[t-1] < 0 else 0.0
            h[t] = omega + (alpha + gamma * ind) * r[t-1]**2 + beta * h[t-1]
            if h[t] <= 1e-12: return 1e10
        return float(0.5 * np.sum(np.log(h[1:]) + r[1:]**2 / h[1:]))

    best, best_x = 1e18, [1e-6, 0.05, 0.06, 0.88]
    for a0, g0, b0 in [(0.05, 0.05, 0.90), (0.08, 0.08, 0.85)]:
        res = minimize(neg_ll, [np.var(r)*0.05, a0, g0, b0], method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10})
        if res.fun < best:
            best, best_x = res.fun, res.x

    omega, alpha, gamma, beta = best_x
    h = np.empty(n); h[0] = np.var(r)
    for t in range(1, n):
        ind = 1.0 if r[t-1] < 0 else 0.0
        h[t] = max(omega + (alpha + gamma*ind)*r[t-1]**2 + beta*h[t-1], 1e-12)

    return {
        "model":        "GJR-GARCH(1,1)",
        "omega":        round(float(omega), 9),
        "alpha":        round(float(alpha), 6),
        "gamma":        round(float(gamma), 6),
        "beta":         round(float(beta), 6),
        "leverage":     round(float(gamma), 6),
        "cond_vol_ann": np.round(np.sqrt(h * 252) * 100, 4).tolist(),
    }


def har_rv(realized_vols: np.ndarray, horizon: int = 1) -> Dict:
    """
    HAR-RV (Heterogeneous Autoregressive Realized Volatility).
    RV_t = c + beta_d*RV_{t-1} + beta_w*RV_{t-5:t} + beta_m*RV_{t-22:t} + eps
    Corsi (2009). Models vol at daily/weekly/monthly frequencies.
    """
    rv = np.asarray(realized_vols, float)
    rv = rv[~np.isnan(rv)]   # drop leading NaNs from rolling windows
    n  = len(rv)
    if n < 30:
        return {"error": "Need at least 30 observations for HAR-RV"}

    # Build regressors
    Y  = rv[22:]
    Xd = rv[21:-1]
    Xw = np.array([rv[max(0,t-5):t].mean() for t in range(21, n-1)])
    Xm = np.array([rv[max(0,t-22):t].mean() for t in range(21, n-1)])
    X  = np.column_stack([np.ones(len(Y)), Xd, Xw, Xm])

    b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    fitted     = X @ b
    resid      = Y - fitted
    r2         = 1 - np.var(resid) / np.var(Y)

    # Forecast h steps ahead using last observation
    last_d = rv[-1]
    last_w = rv[-5:].mean()
    last_m = rv[-22:].mean()
    forecast = float(b[0] + b[1]*last_d + b[2]*last_w + b[3]*last_m)

    return {
        "model":       "HAR-RV",
        "c":           round(float(b[0]), 8),
        "beta_d":      round(float(b[1]), 6),
        "beta_w":      round(float(b[2]), 6),
        "beta_m":      round(float(b[3]), 6),
        "r_squared":   round(float(r2), 4),
        "fitted_rv":   np.round(fitted, 6).tolist(),
        "forecast_rv": round(forecast, 8),
        "forecast_vol_ann": round(float(np.sqrt(forecast * 252) * 100), 4),
    }


def vol_forecast(garch_params: Dict, h_steps: int = 30) -> List[float]:
    """
    Multi-step GARCH(1,1) volatility forecast.
    E[h_{t+k}] = lr_var + (alpha+beta)^k * (h_t - lr_var)
    """
    omega  = garch_params["omega"]
    alpha  = garch_params["alpha"]
    beta   = garch_params["beta"]
    h_last = garch_params["cond_var"][-1]
    lr_var = omega / max(1 - alpha - beta, 1e-8)
    pers   = alpha + beta

    fcast = []
    for k in range(1, h_steps + 1):
        h_k = lr_var + pers**k * (h_last - lr_var)
        fcast.append(round(float(np.sqrt(max(h_k, 0) * 252) * 100), 4))
    return fcast

def vol_forecast_with_bands(garch_params: Dict, h_steps: int = 30,
                              confidence: float = 0.90) -> Dict:
    """
    Multi-step GARCH(1,1) volatility forecast with confidence bands.

    Point forecast: E[h_{t+k}] = lr_var + (α+β)^k * (h_t - lr_var)
    Bands: analytical approximation based on forecast error variance
    growing with horizon (Baillie & Bollerslev 1992).

    Returns: forecast, lower, upper (all annualised %, same length as h_steps)
    """
    from scipy.stats import norm as _norm
    omega  = garch_params["omega"]
    alpha  = garch_params["alpha"]
    beta   = garch_params["beta"]
    h_last = float(garch_params["cond_var"][-1]) if garch_params.get("cond_var") else omega / max(1-alpha-beta,1e-8)
    lr_var = omega / max(1 - alpha - beta, 1e-8)
    pers   = alpha + beta

    steps  = np.arange(1, h_steps + 1)
    fcast_h = lr_var + pers**steps * (h_last - lr_var)
    fcast_h = np.maximum(fcast_h, 1e-12)

    # Forecast error variance grows ∝ sqrt(k) scaled by current vol and persistence
    z      = float(_norm.ppf((1 + confidence) / 2))
    se_h   = np.sqrt(np.maximum(steps, 1)) * float(h_last) * float(alpha) * 2.0
    lo_h   = np.maximum(fcast_h - z * se_h, 1e-14)
    hi_h   = fcast_h + z * se_h

    return {
        "forecast":   np.round(np.sqrt(fcast_h * 252) * 100, 4).tolist(),
        "lower":      np.round(np.sqrt(lo_h   * 252) * 100, 4).tolist(),
        "upper":      np.round(np.sqrt(hi_h   * 252) * 100, 4).tolist(),
        "horizon":    h_steps,
        "confidence": confidence,
    }



# ══════════════════════════════════════════════════════════════════════════
#  REGIME DETECTION — 2-state and 3-state HMM
# ══════════════════════════════════════════════════════════════════════════

def hmm_em(obs: np.ndarray, n_states: int = 2,
            n_iter: int = 50, n_restarts: int = 3) -> Dict:
    """
    Gaussian HMM fitted via Baum-Welch EM.
    n_states = 2: bull/bear  or  3: bull/neutral/bear
    """
    obs = np.asarray(obs, float)
    T   = len(obs)

    best_ll, best_result = -np.inf, None

    for _ in range(n_restarts):
        # Initialise parameters
        rng  = np.random.default_rng()
        idx  = np.argsort(obs)
        chunk = T // n_states
        mu   = np.array([obs[idx[k*chunk:(k+1)*chunk]].mean() for k in range(n_states)])
        sig  = np.array([obs[idx[k*chunk:(k+1)*chunk]].std() + 1e-6 for k in range(n_states)])
        # Random transition matrix with high self-persistence
        # Build transition matrix row by row
        A    = np.zeros((n_states, n_states))
        for _k in range(n_states):
            alpha_k = np.ones(n_states) * 0.5
            alpha_k[_k] += 9   # strong self-transition prior
            A[_k] = rng.dirichlet(alpha_k)
        pi   = np.ones(n_states) / n_states

        for iteration in range(n_iter):
            # E-step: emission probabilities
            B = np.column_stack([norm.pdf(obs, mu[k], sig[k]) for k in range(n_states)])
            B = np.maximum(B, 1e-300)

            # Forward pass (scaled)
            alpha_fwd = np.zeros((T, n_states))
            scale     = np.zeros(T)
            alpha_fwd[0] = pi * B[0]
            scale[0]     = alpha_fwd[0].sum() + 1e-300
            alpha_fwd[0] /= scale[0]
            for t in range(1, T):
                alpha_fwd[t] = (alpha_fwd[t-1] @ A) * B[t]
                scale[t]     = alpha_fwd[t].sum() + 1e-300
                alpha_fwd[t] /= scale[t]

            # Backward pass
            beta_bwd = np.ones((T, n_states))
            for t in range(T - 2, -1, -1):
                beta_bwd[t] = A @ (B[t+1] * beta_bwd[t+1]) / scale[t+1]
                beta_bwd[t] = np.clip(beta_bwd[t], 0, 1e10)

            # Smoothed posteriors
            gamma = alpha_fwd * beta_bwd
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Xi (joint transition posteriors)
            xi = np.zeros((T-1, n_states, n_states))
            for t in range(T-1):
                xi[t] = (alpha_fwd[t, :, None] * A *
                          B[t+1, None, :] * beta_bwd[t+1, None, :])
                xi[t] /= xi[t].sum() + 1e-300

            # M-step
            pi_new = gamma[0]
            A_new  = xi.sum(axis=0) / (gamma[:-1].sum(axis=0)[:, None] + 1e-300)
            A_new /= A_new.sum(axis=1, keepdims=True) + 1e-300
            for k in range(n_states):
                g = gamma[:, k]
                gs = g.sum() + 1e-300
                mu[k]  = (g @ obs) / gs
                sig[k] = np.sqrt((g @ (obs - mu[k])**2) / gs) + 1e-6

            pi, A = pi_new, A_new

        ll = float(np.sum(np.log(scale + 1e-300)))
        if ll > best_ll:
            best_ll     = ll
            best_result = (gamma.copy(), mu.copy(), sig.copy(), A.copy(), pi.copy())

    gamma, mu, sig, A, pi = best_result

    # Sort states by mean (state 0 = lowest obs = most bearish / highest vol)
    order = np.argsort(mu)
    mu    = mu[order]
    sig   = sig[order]
    A     = A[np.ix_(order, order)]
    gamma = gamma[:, order]
    states = gamma.argmax(axis=1)

    # Label names
    if n_states == 2:
        names = ["Low-Vol / Bull", "High-Vol / Bear"]
    else:
        names = ["Low-Vol / Bull", "Neutral", "High-Vol / Bear"]

    # Regime durations
    durations = []
    cur, start = int(states[0]), 0
    for t in range(1, T):
        if int(states[t]) != cur:
            durations.append({"state": cur, "start": start, "end": t-1, "length": t-start})
            cur, start = int(states[t]), t
    durations.append({"state": cur, "start": start, "end": T-1, "length": T-start})

    # Stats per regime
    regime_stats = []
    for k in range(n_states):
        mask = states == k
        if mask.sum() == 0:
            continue
        regime_stats.append({
            "state":       k,
            "name":        names[k],
            "mean_obs":    round(float(mu[k]), 6),
            "std_obs":     round(float(sig[k]), 6),
            "freq":        round(float(mask.mean()), 4),
            "n_days":      int(mask.sum()),
            "mean_duration": round(float(np.mean([d["length"] for d in durations if d["state"]==k])), 1),
        })

    return {
        "n_states":         n_states,
        "states":           states.tolist(),
        "state_probs":      np.round(gamma, 4).tolist(),
        "mu":               mu.tolist(),
        "sigma":            sig.tolist(),
        "transition_matrix":A.tolist(),
        "regime_stats":     regime_stats,
        "durations":        durations[-100:],
        "log_likelihood":   round(best_ll, 4),
        "state_names":      names,
    }


def kalman_vol(returns: np.ndarray, process_noise: float = 1e-5,
                obs_noise: float = 1e-3) -> np.ndarray:
    """
    Kalman filter for latent volatility tracking.
    State: h_t = log-variance.  Observation: r_t^2 ≈ exp(h_t).
    Simple linear Kalman in log-squared-return space.
    """
    r  = np.asarray(returns, float)
    n  = len(r)
    y  = np.log(r**2 + 1e-10)  # observation: log(r^2)

    # State estimate and variance
    x_filt = np.zeros(n)
    P_filt = np.zeros(n)
    x_filt[0] = y[0]
    P_filt[0] = 1.0

    for t in range(1, n):
        # Predict
        x_pred = x_filt[t-1]
        P_pred = P_filt[t-1] + process_noise

        # Update
        K           = P_pred / (P_pred + obs_noise)
        x_filt[t]   = x_pred + K * (y[t] - x_pred)
        P_filt[t]   = (1 - K) * P_pred

    # Convert back to annualised vol
    vol_ann = np.exp(x_filt / 2) * np.sqrt(252) * 100
    return np.round(np.clip(vol_ann, 0.5, 300), 4)


# ══════════════════════════════════════════════════════════════════════════
#  REALISED VOLATILITY
# ══════════════════════════════════════════════════════════════════════════

def realized_vol(returns: np.ndarray, windows: List[int] = [5, 21, 63]) -> Dict:
    """Rolling realised volatility at multiple windows."""
    r   = np.asarray(returns, float)
    r2  = r ** 2
    out = {}
    for w in windows:
        rv   = pd.Series(r2).rolling(w).mean().values
        vol  = np.sqrt(rv * 252) * 100
        out[f"rv_{w}d"] = np.round(vol, 4).tolist()
    out["rv_inst"]  = np.round(np.sqrt(r2 * 252) * 100, 4).tolist()
    return out


# ══════════════════════════════════════════════════════════════════════════
#  REGIME-CONDITIONAL STATISTICS
# ══════════════════════════════════════════════════════════════════════════

def regime_conditional_stats(returns: np.ndarray,
                               states: List[int],
                               state_names: List[str]) -> pd.DataFrame:
    """
    Per-regime statistics: mean return, vol, Sharpe, worst day, best day.
    """
    r = np.asarray(returns, float)
    rows = []
    for k, name in enumerate(state_names):
        mask = np.array(states) == k
        if mask.sum() < 5:
            continue
        r_k = r[mask]
        ann_ret = float(r_k.mean() * 252)
        ann_vol = float(r_k.std(ddof=1) * np.sqrt(252))
        sharpe  = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0
        rows.append({
            "State":    k,
            "Name":     name,
            "Freq %":   round(float(mask.mean() * 100), 2),
            "Ann Ret %":round(ann_ret * 100, 3),
            "Ann Vol %":round(ann_vol * 100, 3),
            "Sharpe":   round(sharpe, 4),
            "Worst Day%":round(float(r_k.min() * 100), 3),
            "Best Day%": round(float(r_k.max() * 100), 3),
        })
    return pd.DataFrame(rows)
