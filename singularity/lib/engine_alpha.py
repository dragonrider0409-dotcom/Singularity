"""
alpha_execution/engine.py
==========================
Alpha & Execution Dashboard

Merges:
  1. Factor Model     — Fama-French 3/5-factor, PCA factors, IC/IR, alpha decay
  2. Optimal Execution — Almgren-Chriss, TWAP/VWAP schedules, market impact model

Run:
  python alpha_execution/app.py  ->  http://127.0.0.1:5006
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from typing import List, Dict, Tuple, Optional
import warnings, logging

warnings.filterwarnings("ignore")
log = logging.getLogger("ae_engine")


# ══════════════════════════════════════════════════════════════════════════
#  FACTOR MODEL
# ══════════════════════════════════════════════════════════════════════════

def factor_regression(asset_returns: np.ndarray,
                       factor_returns: np.ndarray,
                       factor_names: List[str]) -> Dict:
    """
    OLS factor regression: r_i = alpha + sum_k beta_k * F_k + epsilon
    Returns alpha, betas, R², tracking error, information ratio.
    """
    r  = np.asarray(asset_returns, float)
    F  = np.asarray(factor_returns, float)
    n  = len(r)
    X  = np.column_stack([np.ones(n), F])

    b, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
    fitted     = X @ b
    resid      = r - fitted
    ss_tot     = float(np.sum((r - r.mean())**2))
    ss_res     = float(np.sum(resid**2))
    r2         = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    te_ann     = float(resid.std(ddof=1) * np.sqrt(252))
    alpha_ann  = float(b[0] * 252)
    ir         = alpha_ann / te_ann if te_ann > 1e-8 else 0.0

    # t-stats
    sigma2  = ss_res / max(n - X.shape[1], 1)
    XtXi    = np.linalg.pinv(X.T @ X)
    se      = np.sqrt(np.maximum(sigma2 * np.diag(XtXi), 0))
    t_stats = b / np.maximum(se, 1e-15)

    return {
        "alpha_daily":   round(float(b[0]), 8),
        "alpha_ann":     round(alpha_ann * 100, 4),   # in percent
        "betas":         {f: round(float(b[i+1]), 6) for i, f in enumerate(factor_names)},
        "r_squared":     round(float(r2), 4),
        "te_ann":        round(te_ann * 100, 4),       # in percent
        "info_ratio":    round(ir, 4),
        "t_alpha":       round(float(t_stats[0]), 4),
        "t_betas":       {f: round(float(t_stats[i+1]), 4) for i, f in enumerate(factor_names)},
        "residuals":     np.round(resid, 6).tolist(),
    }


def rolling_factor_regression(asset_returns: np.ndarray,
                                factor_returns: np.ndarray,
                                factor_names: List[str],
                                window: int = 126) -> Dict:
    """Rolling factor regression to show time-varying betas."""
    r  = np.asarray(asset_returns, float)
    F  = np.asarray(factor_returns, float)
    n  = len(r)

    roll_alpha = np.full(n, np.nan)
    roll_betas = {f: np.full(n, np.nan) for f in factor_names}
    roll_r2    = np.full(n, np.nan)

    for t in range(window, n + 1):
        r_w = r[t-window:t]
        F_w = F[t-window:t]
        X   = np.column_stack([np.ones(window), F_w])
        try:
            b, _, _, _ = np.linalg.lstsq(X, r_w, rcond=None)
            fitted = X @ b
            ss_tot = np.sum((r_w - r_w.mean())**2)
            ss_res = np.sum((r_w - fitted)**2)
            roll_alpha[t-1] = b[0] * 252 * 100
            for i, f in enumerate(factor_names):
                roll_betas[f][t-1] = b[i+1]
            roll_r2[t-1] = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        except Exception:
            pass

    return {
        "alpha_roll": [round(float(v), 4) if not np.isnan(v) else None for v in roll_alpha],
        "betas_roll": {f: [round(float(v), 4) if not np.isnan(v) else None for v in roll_betas[f]]
                       for f in factor_names},
        "r2_roll": [round(float(v), 4) if not np.isnan(v) else None for v in roll_r2],
        "window":  window,
    }


def pca_factors(returns_matrix: np.ndarray, n_components: int = 5) -> Dict:
    """
    PCA on a returns matrix to extract statistical risk factors.
    Returns factor loadings, explained variance, and factor returns.
    """
    R = np.asarray(returns_matrix, float)
    R = R - R.mean(axis=0)
    T, N = R.shape

    # Covariance matrix and eigendecomposition
    cov    = R.T @ R / (T - 1)
    eigval, eigvec = np.linalg.eigh(cov)
    # Sort descending
    order  = np.argsort(eigval)[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    nc     = min(n_components, N, T)
    loadings = eigvec[:, :nc]                    # (N, nc)
    factors  = R @ loadings                      # (T, nc) factor returns
    var_exp  = eigval[:nc] / eigval.sum()

    return {
        "n_components":    nc,
        "loadings":        np.round(loadings, 6).tolist(),
        "factor_returns":  np.round(factors, 6).tolist(),
        "var_explained":   np.round(var_exp * 100, 4).tolist(),
        "cum_var":         np.round(np.cumsum(var_exp) * 100, 4).tolist(),
        "eigenvalues":     np.round(eigval[:nc], 6).tolist(),
    }


def alpha_decay(signal: np.ndarray, returns: np.ndarray,
                 max_lag: int = 20) -> Dict:
    """
    Alpha decay: IC (Information Coefficient) at each forward lag.
    IC_k = Spearman corr(signal_t, r_{t+k})
    Shows how quickly the predictive power of a signal decays.
    """
    s = np.asarray(signal, float)
    r = np.asarray(returns, float)
    n = len(s)

    ics     = []
    lags    = list(range(1, min(max_lag + 1, n // 4)))
    for lag in lags:
        s_lag = s[:n-lag]
        r_lag = r[lag:]
        corr, _ = spearmanr(s_lag, r_lag)
        ics.append(round(float(corr), 4) if not np.isnan(corr) else 0.0)

    ic_mean = round(float(np.mean(ics[:5])), 4)   # short-horizon IC
    ic_ir   = round(float(np.mean(ics) / (np.std(ics) + 1e-8)), 4)

    return {
        "lags":    lags,
        "ics":     ics,
        "ic_mean": ic_mean,
        "ic_ir":   ic_ir,
        "half_life": next((lags[i] for i, v in enumerate(ics) if abs(v) < abs(ic_mean)/2), max_lag),
    }


# ══════════════════════════════════════════════════════════════════════════
#  OPTIMAL EXECUTION — Almgren-Chriss
# ══════════════════════════════════════════════════════════════════════════

def almgren_chriss(X: float, T: int, sigma: float,
                    eta: float = 0.1, gamma_perm: float = 0.01,
                    lam: float = 1e-6) -> Dict:
    """
    Almgren-Chriss (2001) optimal liquidation.
    X:      total shares to liquidate
    T:      time horizon (integer steps, e.g. 10 trading days)
    sigma:  daily return volatility (e.g. 0.02 for 2%)
    eta:    temporary impact coefficient (cost per unit of trading rate)
    gamma_perm: permanent impact coefficient
    lam:    risk aversion (higher = trade faster)

    Returns optimal trajectory, expected shortfall, and efficient frontier.
    """
    N = max(int(T), 2)
    kappa = np.sqrt(lam * sigma**2 / max(eta, 1e-12))
    kT    = kappa * N

    # Optimal holding trajectory x_j (inventory at each step)
    steps = np.arange(0, N + 1)
    if kT > 1e-6:
        traj = X * np.sinh(kappa * (N - steps)) / np.sinh(kT)
    else:
        traj = X * (1 - steps / N)        # kappa→0: linear (TWAP)
    traj = np.maximum(traj, 0.0)

    trades   = -np.diff(traj)              # shares sold each period
    trade_rates = trades / 1.0             # assuming dt=1

    # Expected shortfall (IS = implementation shortfall)
    # ES = 0.5 * gamma * X^2 + eta * sum(n_k^2/dt)
    perm_cost  = 0.5 * gamma_perm * X**2
    temp_cost  = float(eta * np.sum(trade_rates**2))
    total_cost = perm_cost + temp_cost

    # Variance of shortfall
    var_sf  = float(sigma**2 * np.sum(traj[:-1]**2))

    # Efficient frontier: ES vs variance across different risk aversions
    lams   = np.logspace(-8, -4, 40)
    ef_es  = []
    ef_var = []
    for l in lams:
        k2 = np.sqrt(l * sigma**2 / max(eta, 1e-12)) * N
        if k2 > 1e-6:
            tr = X * np.sinh(np.sqrt(l*sigma**2/max(eta,1e-12))*(N-steps)) / np.sinh(k2)
        else:
            tr = X * (1 - steps/N)
        tr = np.maximum(tr, 0)
        t2 = -np.diff(tr)
        ef_es.append(float(perm_cost + eta * np.sum(t2**2)))
        ef_var.append(float(sigma**2 * np.sum(tr[:-1]**2)))

    return {
        "trajectory":    np.round(traj, 4).tolist(),
        "trades":        np.round(trades, 4).tolist(),
        "trade_pct":     np.round(trades / max(X, 1) * 100, 4).tolist(),
        "perm_cost":     round(perm_cost, 4),
        "temp_cost":     round(temp_cost, 4),
        "total_cost":    round(total_cost, 4),
        "cost_bps":      round(total_cost / max(X, 1) * 10000, 4),
        "variance":      round(var_sf, 4),
        "kappa":         round(float(kappa), 6),
        "ef_es":         [round(v, 4) for v in ef_es],
        "ef_var":        [round(v, 4) for v in ef_var],
    }


def twap_schedule(X: float, T: int) -> Dict:
    """TWAP: equal-sized trades over T periods."""
    n    = int(T)
    trad = np.full(n, X / n)
    traj = X - np.cumsum(np.concatenate([[0], trad]))
    return {
        "trajectory": np.round(traj, 4).tolist(),
        "trades":     np.round(trad, 4).tolist(),
        "name":       "TWAP",
    }


def vwap_schedule(X: float, T: int, volume_profile: Optional[np.ndarray] = None) -> Dict:
    """
    VWAP: trade proportional to expected intraday volume.
    If volume_profile is None, uses a U-shaped intraday volume curve.
    """
    n = int(T)
    if volume_profile is None:
        # U-shaped profile: more volume at open and close
        t = np.linspace(0, 1, n)
        vp = 2 - np.abs(2*t - 1)**2  # simple bowl shape
        vp = vp / vp.sum()
    else:
        vp = np.asarray(volume_profile, float)
        vp = vp[:n] / vp[:n].sum()

    trad = X * vp
    traj = X - np.cumsum(np.concatenate([[0], trad]))
    return {
        "trajectory":      np.round(traj, 4).tolist(),
        "trades":          np.round(trad, 4).tolist(),
        "volume_profile":  np.round(vp, 6).tolist(),
        "name":            "VWAP",
    }


def market_impact_model(trades: np.ndarray, adv: float,
                          sigma: float, price: float = 100.0) -> Dict:
    """
    Estimate market impact costs using the Almgren et al. (2005) empirical model.
    MI = 0.5 * sigma * (|x|/adv)^0.6  (permanent)
       + eta * |n|/dt  (temporary)
    ADV: average daily volume in shares.
    """
    t = np.abs(trades) / max(adv, 1)
    # Permanent impact (square-root law approximation)
    perm_impact_bps = 0.5 * sigma * 10000 * t**0.6
    # Temporary (linear, simplified)
    temp_impact_bps = 0.5 * t * 10000 * 0.1

    total_impact_bps = perm_impact_bps + temp_impact_bps
    total_cost = float(np.sum(np.abs(trades) * price * total_impact_bps / 10000))

    return {
        "perm_impact_bps": np.round(perm_impact_bps, 4).tolist(),
        "temp_impact_bps": np.round(temp_impact_bps, 4).tolist(),
        "total_impact_bps":np.round(total_impact_bps, 4).tolist(),
        "total_cost":      round(total_cost, 4),
    }


def implementation_shortfall(prices_at_decision: float,
                               prices_executed: np.ndarray,
                               shares_executed: np.ndarray,
                               benchmark_price: Optional[float] = None) -> Dict:
    """
    Implementation Shortfall = Paper Portfolio Return − Live Portfolio Return
    Decomposes into: delay cost, market impact, timing cost, opportunity cost.
    """
    p0    = prices_at_decision
    bench = benchmark_price or p0
    pe    = np.asarray(prices_executed, float)
    ne    = np.asarray(shares_executed, float)

    total_shares = float(ne.sum())
    avg_price    = float((pe * ne).sum() / max(total_shares, 1))

    # IS components (in bps)
    explicit_bps  = 0.0   # commissions – not modelled
    delay_bps     = (bench - p0) / p0 * 10000
    impact_bps    = (avg_price - bench) / bench * 10000
    total_is_bps  = (avg_price - p0) / p0 * 10000

    return {
        "decision_price":   round(p0, 4),
        "avg_exec_price":   round(avg_price, 4),
        "benchmark_price":  round(bench, 4),
        "delay_cost_bps":   round(float(delay_bps), 4),
        "impact_cost_bps":  round(float(impact_bps), 4),
        "total_is_bps":     round(float(total_is_bps), 4),
        "total_shares":     round(total_shares, 0),
    }
