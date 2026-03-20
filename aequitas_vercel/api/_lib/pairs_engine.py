"""
pairs_trading/engine.py
========================
Pairs Trading & Mean Reversion Scanner

Combines two related projects into one dashboard:
  SCANNER  — screen a universe for cointegrated pairs and mean-reverting series
  TRADING  — OU fitting, Z-score signals, backtest, performance attribution

Run:
  python pairs_trading/app.py  ->  http://127.0.0.1:5004
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings, logging

warnings.filterwarnings("ignore")
log = logging.getLogger("pairs_engine")


# ══════════════════════════════════════════════════════════════════════════
#  STATIONARITY & COINTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════

def adf_test(series: np.ndarray, max_lag: int = 10) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    H₀: unit root (non-stationary).  Small p-value → reject → stationary.
    Returns tau statistic, approximate p-value, lag used, and critical values.
    """
    y  = np.asarray(series, float)
    n  = len(y)
    dy = np.diff(y)

    # Select lag by AIC
    best_aic, best_lag = np.inf, 1
    for lag in range(1, min(max_lag + 1, n // 4)):
        Y  = dy[lag:]
        Xl = [y[lag:-1]] + [dy[lag - k - 1: n - k - 2] for k in range(lag)] + [np.ones(len(Y))]
        X  = np.column_stack(Xl)
        if X.shape[0] < X.shape[1] + 2:
            continue
        try:
            b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            s2  = float(np.sum((Y - X @ b) ** 2)) / max(len(Y) - X.shape[1], 1)
            aic = len(Y) * np.log(max(s2, 1e-15)) + 2 * X.shape[1]
            if aic < best_aic:
                best_aic, best_lag = aic, lag
        except Exception:
            pass

    lag = best_lag
    Y   = dy[lag:]
    X   = np.column_stack([y[lag:-1]] + [dy[lag - k - 1: n - k - 2] for k in range(lag)] + [np.ones(len(Y))])
    try:
        b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ b
        s2    = float(np.sum(resid ** 2)) / max(len(Y) - X.shape[1], 1)
        XtXi  = np.linalg.pinv(X.T @ X)
        se    = np.sqrt(max(s2 * XtXi[0, 0], 1e-15))
        tau   = float(b[0]) / se
    except Exception:
        tau = 0.0

    # MacKinnon (1994) approximate critical values
    cv   = {1: -3.43, 5: -2.86, 10: -2.57}
    # Map tau to approximate p-value (logistic approximation)
    p    = float(np.clip(1.0 / (1.0 + np.exp(-1.5 * (tau + 2.5))), 0.001, 0.999))

    return {
        "tau":            round(tau, 4),
        "p_value":        round(p, 4),
        "critical_values":cv,
        "lag":            lag,
        "is_stationary":  bool(tau < cv[5]),
    }


def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    Hurst exponent via rescaled range analysis.
    H < 0.5 → mean-reverting,  H = 0.5 → random walk,  H > 0.5 → trending.
    """
    y   = np.asarray(series, float)
    n   = len(y)
    lags= np.unique(np.logspace(1, np.log10(min(max_lag, n // 2)), 20).astype(int))
    tau = []
    for lag in lags:
        if lag < 2 or lag >= n:
            continue
        pts = [np.std(y[i: i + lag]) for i in range(0, n - lag, max(1, lag // 2))]
        if pts:
            tau.append(np.mean(pts))
    if len(tau) < 3:
        return 0.5
    try:
        from numpy.polynomial.polynomial import polyfit
        h = float(polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)[1])
        return round(float(np.clip(h, 0.01, 0.99)), 4)
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════
#  ORNSTEIN-UHLENBECK FITTING
# ══════════════════════════════════════════════════════════════════════════

def fit_ou(spread: np.ndarray, dt: float = 1 / 252) -> Dict:
    """
    Fit Ornstein-Uhlenbeck process to a spread series.
    dX = κ(μ − X)dt + σ dW

    Returns κ (mean-reversion speed), μ (long-run mean),
    σ (vol), and half-life in days.
    """
    x  = np.asarray(spread, float)
    x0 = x[:-1];  x1 = x[1:]
    n  = len(x0)

    # OLS: x_{t+1} = a + b·x_t  ⟹  κ = -ln(b)/dt, μ = a/(1-b)
    sx  = x0.sum();   sy  = x1.sum()
    sxx = (x0**2).sum(); sxy = (x0*x1).sum()
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return {"kappa": 0.0, "mu": float(x.mean()), "sigma": float(x.std()),
                "half_life_days": float("inf"), "half_life_yr": float("inf")}

    b     = (n * sxy - sx * sy) / denom
    a     = (sy - b * sx) / n
    resid = x1 - a - b * x0
    se    = float(np.std(resid, ddof=2))

    b     = min(max(b, 1e-6), 1 - 1e-6)
    kappa = -np.log(b) / dt
    mu    = a / (1 - b)
    sigma = se * np.sqrt(2 * kappa / max(1 - b ** 2, 1e-8))
    hl_yr = np.log(2) / kappa if kappa > 1e-6 else float("inf")

    return {
        "kappa":          round(float(kappa), 4),
        "mu":             round(float(mu), 6),
        "sigma":          round(float(sigma), 6),
        "half_life_days": round(float(hl_yr * 252), 2),
        "half_life_yr":   round(float(hl_yr), 4),
        "r_squared":      round(float(1 - np.var(resid) / np.var(x1)), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
#  PAIR CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════

def engle_granger(price_y: np.ndarray, price_x: np.ndarray) -> Dict:
    """
    Engle-Granger two-step cointegration test.
    Step 1: Regress Y on X to find hedge ratio β.
    Step 2: ADF test on residual spread.
    """
    y = np.log(np.asarray(price_y, float))
    x = np.log(np.asarray(price_x, float))
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    try:
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return {"cointegrated": False, "error": "lstsq failed"}

    alpha, beta = float(b[0]), float(b[1])
    spread      = y - beta * x - alpha
    adf         = adf_test(spread)
    ou          = fit_ou(spread)
    h           = hurst_exponent(spread)

    return {
        "alpha":        round(alpha, 6),
        "beta":         round(beta, 6),
        "cointegrated": bool(adf["tau"] < -2.86),
        "adf_tau":      adf["tau"],
        "adf_p":        adf["p_value"],
        "half_life_days": ou["half_life_days"],
        "hurst":        round(h, 4),
        "spread_mean":  round(float(spread.mean()), 6),
        "spread_std":   round(float(spread.std()), 6),
        "ou":           ou,
    }


def compute_spread(price_y: np.ndarray, price_x: np.ndarray,
                    beta: float, alpha: float) -> np.ndarray:
    """Compute log-price spread: log(Y) - β·log(X) - α."""
    return np.log(price_y) - beta * np.log(price_x) - alpha


def zscore(spread: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling Z-score of spread: (spread - μ_roll) / σ_roll."""
    s   = pd.Series(spread)
    mu  = s.rolling(window, min_periods=window // 2).mean()
    sig = s.rolling(window, min_periods=window // 2).std(ddof=1)
    return ((s - mu) / sig.clip(lower=1e-8)).values


# ══════════════════════════════════════════════════════════════════════════
#  MEAN REVERSION SCANNER
# ══════════════════════════════════════════════════════════════════════════

def scan_universe(prices: pd.DataFrame, min_half_life: float = 1.0,
                   max_half_life: float = 60.0) -> pd.DataFrame:
    """
    Scan every pair in `prices` for cointegration.
    Returns sorted DataFrame of candidate pairs.

    prices: DataFrame where each column is one asset's price series.
    Filters: ADF τ < -2.86, half-life in [min_hl, max_hl] days.
    """
    tickers = list(prices.columns)
    n       = len(tickers)
    results = []

    for i in range(n):
        for j in range(i + 1, n):
            tk1, tk2 = tickers[i], tickers[j]
            p1 = prices[tk1].dropna().values
            p2 = prices[tk2].dropna().values
            min_len = min(len(p1), len(p2))
            if min_len < 60:
                continue
            p1 = p1[-min_len:]
            p2 = p2[-min_len:]

            try:
                eg = engle_granger(p1, p2)
            except Exception:
                continue

            if not eg["cointegrated"]:
                continue
            hl = eg["half_life_days"]
            if not (min_half_life <= hl <= max_half_life):
                continue

            spread = compute_spread(p1, p2, eg["beta"], eg["alpha"])
            corr   = float(np.corrcoef(np.log(p1), np.log(p2))[0, 1])

            results.append({
                "pair":            f"{tk1}/{tk2}",
                "ticker_y":        tk1,
                "ticker_x":        tk2,
                "beta":            eg["beta"],
                "alpha":           eg["alpha"],
                "adf_tau":         eg["adf_tau"],
                "adf_p":           eg["adf_p"],
                "half_life_days":  round(hl, 1),
                "hurst":           eg["hurst"],
                "correlation":     round(corr, 4),
                "spread_std":      eg["spread_std"],
                "ou_kappa":        eg["ou"]["kappa"],
                "ou_sigma":        eg["ou"]["sigma"],
            })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    # Sort by ADF tau (most negative = most stationary)
    df = df.sort_values("adf_tau").reset_index(drop=True)
    return df


def scan_single(prices: pd.DataFrame) -> pd.DataFrame:
    """Scan individual series for mean reversion (ADF + Hurst + OU)."""
    rows = []
    for col in prices.columns:
        y   = prices[col].dropna().values
        if len(y) < 60:
            continue
        ret = np.diff(np.log(y))
        adf = adf_test(y)
        h   = hurst_exponent(y)
        ou  = fit_ou(y - y.mean())
        rows.append({
            "ticker":          col,
            "adf_tau":         adf["tau"],
            "adf_p":           adf["p_value"],
            "is_stationary":   bool(adf["is_stationary"]),
            "hurst":           h,
            "half_life_days":  ou["half_life_days"],
            "ann_vol":         round(float(ret.std() * np.sqrt(252)), 4),
            "mean_rev_score":  round(float(max(0.5 - h, 0) * 2 * min(1, 20 / max(ou["half_life_days"], 0.1))), 4),
        })
    df = pd.DataFrame(rows).sort_values("adf_tau").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION & BACKTEST
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PairsConfig:
    entry_z:   float = 2.0    # open position when |Z| > entry_z
    exit_z:    float = 0.5    # close position when |Z| < exit_z
    stop_z:    float = 4.0    # stop-loss when |Z| > stop_z
    z_window:  int   = 60     # rolling window for Z-score
    notional:  float = 100_000.0
    tc_bps:    float = 5.0    # transaction cost per side in bps


def generate_signals(spread: np.ndarray, cfg: PairsConfig) -> np.ndarray:
    """
    Generate position signals from Z-score.
    +1 = long spread (long Y, short X)
    -1 = short spread (short Y, long X)
     0 = flat
    """
    z    = zscore(spread, cfg.z_window)
    pos  = np.zeros(len(z))
    current = 0

    for i in range(1, len(z)):
        if np.isnan(z[i]):
            pos[i] = 0
            current = 0
            continue
        # Stop-loss
        if current != 0 and abs(z[i]) > cfg.stop_z:
            current = 0
        # Exit
        elif current != 0 and abs(z[i]) < cfg.exit_z:
            current = 0
        # Entry
        elif current == 0:
            if z[i] > cfg.entry_z:
                current = -1   # spread too high → short spread
            elif z[i] < -cfg.entry_z:
                current = 1    # spread too low → long spread
        pos[i] = current

    return pos


def backtest_pair(price_y: np.ndarray, price_x: np.ndarray,
                   beta: float, alpha: float,
                   cfg: PairsConfig) -> Dict:
    """
    Backtest a pairs trade on historical prices.
    Returns full P&L series, metrics, and trade log.
    """
    n      = min(len(price_y), len(price_x))
    py     = np.asarray(price_y, float)[-n:]
    px     = np.asarray(price_x, float)[-n:]
    spread = compute_spread(py, px, beta, alpha)
    z      = zscore(spread, cfg.z_window)
    pos    = generate_signals(spread, cfg)

    # Daily P&L: position change in spread
    spread_ret = np.diff(spread)
    pos_lag    = pos[:-1]
    raw_pnl    = pos_lag * spread_ret * cfg.notional

    # Transaction costs: charged on position changes
    trades     = np.diff(pos)
    tc         = np.abs(trades) * cfg.notional * cfg.tc_bps / 10000
    net_pnl    = raw_pnl - tc

    # Cumulative
    cum_pnl    = np.cumsum(net_pnl)
    cum_ret    = cum_pnl / cfg.notional

    # Metrics
    ann_ret    = float(net_pnl.mean() * 252)
    ann_vol    = float(net_pnl.std(ddof=1) * np.sqrt(252))
    sharpe     = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0
    mdd        = float(np.min(np.minimum.accumulate(cum_ret + 1) /
                               np.maximum.accumulate(cum_ret + 1) - 1))
    calmar     = ann_ret / abs(mdd) if abs(mdd) > 1e-8 else 0.0
    n_trades   = int(np.sum(np.abs(np.diff(pos)) > 0))
    win_rate   = float(np.mean(net_pnl[net_pnl != 0] > 0)) if np.any(net_pnl != 0) else 0.0

    # Trade log
    trade_log  = []
    in_trade   = None
    for i in range(1, len(pos)):
        if pos[i-1] == 0 and pos[i] != 0:
            in_trade = {"entry": i, "side": int(pos[i]), "entry_z": float(z[i])}
        elif pos[i-1] != 0 and (pos[i] == 0 or i == len(pos)-1) and in_trade:
            in_trade["exit"] = i
            in_trade["exit_z"] = float(z[i])
            pnl = float(np.sum(net_pnl[in_trade["entry"]-1:i]))
            in_trade["pnl"] = round(pnl, 2)
            trade_log.append(in_trade)
            in_trade = None

    return {
        "pnl":          np.round(net_pnl, 4).tolist(),
        "cum_pnl":      np.round(cum_pnl, 4).tolist(),
        "cum_ret":      np.round(cum_ret, 6).tolist(),
        "spread":       np.round(spread, 6).tolist(),
        "zscore":       [round(float(v), 4) if not np.isnan(v) else None for v in z],
        "position":     pos.tolist(),
        "metrics": {
            "ann_return":  round(ann_ret, 4),
            "ann_vol":     round(ann_vol, 4),
            "sharpe":      round(sharpe, 4),
            "max_dd":      round(mdd, 6),
            "calmar":      round(calmar, 4),
            "n_trades":    n_trades,
            "win_rate":    round(win_rate, 4),
            "total_pnl":   round(float(cum_pnl[-1]), 2),
            "tc_paid":     round(float(tc.sum()), 2),
        },
        "trade_log": trade_log[-50:],   # last 50 trades
    }


def johansen_trace(prices: np.ndarray) -> Dict:
    """
    Johansen trace test for multivariate cointegration rank.
    Johansen (1991): tests H0: rank <= r vs H1: rank > r.

    More powerful than Engle-Granger for systems with >2 variables.
    Used by fixed-income relative value desks for basket cointegration.

    prices: (T, k) array of log prices
    Returns: cointegration_rank, trace statistics, critical values
    """
    from scipy.linalg import cholesky, solve_triangular

    Y = np.log(np.asarray(prices, float))
    n, k = Y.shape

    # First differences and lagged levels
    dY    = np.diff(Y, axis=0)
    Y_lag = Y[:-1]

    T  = len(dY)
    S00 = dY.T @ dY / T + np.eye(k) * 1e-10
    S11 = Y_lag.T @ Y_lag / T + np.eye(k) * 1e-10
    S01 = dY.T @ Y_lag / T

    # Solve generalised eigenvalue problem via Cholesky
    try:
        L11 = cholesky(S11, lower=True)
        S00_inv = np.linalg.pinv(S00)
        M    = solve_triangular(L11, S01 @ S00_inv @ S01.T, lower=True)
        M2   = solve_triangular(L11, M.T, lower=True).T
        eigvals_raw = np.real(np.linalg.eigvals(M2))
        eigvals = np.sort(np.clip(eigvals_raw, 0, 1 - 1e-10))[::-1]
    except Exception:
        eigvals = np.zeros(k)

    # Johansen 1991 trace critical values (5%, no trend, up to k=5)
    # rows = k-r (number of eigenvalues in sum), from Osterwald-Lenum (1992)
    cv_table = {
        1: 3.76, 2: 15.41, 3: 29.68, 4: 47.21, 5: 68.52,
        6: 94.15, 7: 124.24, 8: 157.75,
    }

    stats = []
    for r in range(k):
        stat = float(-T * np.sum(np.log(1 - eigvals[r:])))
        cv   = cv_table.get(k - r, float(3.76 + (k - r) * 20))
        stats.append({
            "rank_h0":   r,
            "trace_stat": round(stat, 4),
            "cv_5pct":    round(cv, 2),
            "reject_h0":  bool(stat > cv),
        })

    rank = sum(1 for s in stats if s["reject_h0"])
    return {
        "cointegration_rank": rank,
        "trace_stats":        stats,
        "eigenvalues":        np.round(eigvals[:k], 6).tolist(),
        "n_assets":           k,
        "n_obs":              T,
    }
