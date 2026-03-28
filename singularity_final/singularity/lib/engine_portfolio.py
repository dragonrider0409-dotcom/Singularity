"""
portfolio_optimizer/engine.py
==============================
Portfolio Optimization Engine

Models:
  1. Mean-Variance (Markowitz) — efficient frontier, min-variance, max-Sharpe
  2. Black-Litterman — blend market equilibrium with analyst views
  3. Risk Parity — equal risk contribution weights
  4. Factor Model — Fama-French beta decomposition, alpha estimation

Run:
  cd <project root>
  pip install flask flask-cors yfinance numpy scipy pandas
  python portfolio_optimizer/app.py  ->  http://127.0.0.1:5002
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings, logging

warnings.filterwarnings("ignore")
log = logging.getLogger("port_engine")


# ══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioResult:
    weights:       np.ndarray       # optimal weights (n_assets,)
    tickers:       List[str]
    exp_return:    float            # annualised expected return
    volatility:    float            # annualised volatility
    sharpe:        float            # Sharpe ratio (rf subtracted)
    weights_dict:  Dict[str, float] # {ticker: weight}


@dataclass
class EfficientFrontier:
    returns:    List[float]         # portfolio returns along frontier
    vols:       List[float]         # portfolio vols along frontier
    sharpes:    List[float]
    weights:    List[List[float]]   # weight vectors
    tickers:    List[str]
    min_var:    PortfolioResult
    max_sharpe: PortfolioResult


# ══════════════════════════════════════════════════════════════════════════
#  COVARIANCE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════

def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    """Standard sample covariance (annualised)."""
    return returns.cov().values * 252


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage estimator — reduces estimation error in
    high-dimensional covariance matrices. Standard at quant desks.
    Shrinks toward the identity (constant-correlation target).
    """
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_ * 252


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """
    EWMA (RiskMetrics) covariance — gives more weight to recent observations.
    lambda=0.94 is the RiskMetrics standard for daily data.
    """
    n = len(returns)
    weights = np.array([(1 - lam) * lam**i for i in range(n-1, -1, -1)])
    weights /= weights.sum()
    R = returns.values
    mu = (weights[:, None] * R).sum(axis=0)
    R_c = R - mu
    cov = (weights[:, None] * R_c).T @ R_c
    return cov * 252


def estimate_covariance(returns: pd.DataFrame,
                         method: str = "ledoit_wolf") -> np.ndarray:
    """Dispatch to covariance estimator."""
    try:
        if method == "ledoit_wolf":
            return ledoit_wolf_cov(returns)
        elif method == "ewma":
            return ewma_cov(returns)
        else:
            return sample_cov(returns)
    except ImportError:
        log.warning("sklearn not available — using sample covariance")
        return sample_cov(returns)


def nearest_pd(A: np.ndarray) -> np.ndarray:
    """Higham (1988) nearest positive-definite matrix."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    k = 1
    while True:
        try:
            np.linalg.cholesky(A3)
            return A3
        except np.linalg.LinAlgError:
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += (-mineig * k**2 + np.finfo(float).eps) * np.eye(A.shape[0])
            k += 1


# ══════════════════════════════════════════════════════════════════════════
#  MARKOWITZ MEAN-VARIANCE
# ══════════════════════════════════════════════════════════════════════════

def _portfolio_stats(w: np.ndarray, mu: np.ndarray,
                     cov: np.ndarray, rf: float = 0.05
                     ) -> Tuple[float, float, float]:
    """Returns (expected_return, volatility, sharpe)."""
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


def min_variance_portfolio(mu: np.ndarray, cov: np.ndarray,
                            tickers: List[str],
                            allow_short: bool = False,
                            rf: float = 0.05) -> PortfolioResult:
    """Global minimum variance portfolio."""
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    w0 = np.ones(n) / n

    res = minimize(
        lambda w: float(w @ cov @ w),
        w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-10}
    )
    w = res.x
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf)
    return PortfolioResult(
        weights=w, tickers=tickers,
        exp_return=round(ret, 6), volatility=round(vol, 6), sharpe=round(sharpe, 6),
        weights_dict={t: round(float(wi), 6) for t, wi in zip(tickers, w)}
    )


def max_sharpe_portfolio(mu: np.ndarray, cov: np.ndarray,
                          tickers: List[str],
                          allow_short: bool = False,
                          rf: float = 0.05) -> PortfolioResult:
    """Maximum Sharpe ratio portfolio."""
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        r, v, _ = _portfolio_stats(w, mu, cov, rf)
        return -(r - rf) / v if v > 1e-10 else 1e6

    # Try multiple starts for robustness
    best, best_w = 1e9, w0
    for _ in range(5):
        w_init = np.random.dirichlet(np.ones(n))
        res = minimize(neg_sharpe, w_init, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 5000, "ftol": 1e-10})
        if res.fun < best:
            best, best_w = res.fun, res.x

    w = best_w
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf)
    return PortfolioResult(
        weights=w, tickers=tickers,
        exp_return=round(ret, 6), volatility=round(vol, 6), sharpe=round(sharpe, 6),
        weights_dict={t: round(float(wi), 6) for t, wi in zip(tickers, w)}
    )


def efficient_frontier(mu: np.ndarray, cov: np.ndarray,
                        tickers: List[str],
                        n_points: int = 60,
                        allow_short: bool = False,
                        rf: float = 0.05) -> EfficientFrontier:
    """
    Trace the efficient frontier by minimising variance for each
    target return level from min-variance to max achievable return.
    """
    n = len(mu)
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    w0 = np.ones(n) / n

    min_var = min_variance_portfolio(mu, cov, tickers, allow_short, rf)
    max_sharpe = max_sharpe_portfolio(mu, cov, tickers, allow_short, rf)

    r_min = min_var.exp_return
    r_max = float(mu.max()) if not allow_short else float(mu.max() * 1.5)
    target_returns = np.linspace(r_min, r_max, n_points)

    frontier_rets, frontier_vols, frontier_sh, frontier_ws = [], [], [], []

    for target_r in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, tr=target_r: float(w @ mu) - tr},
        ]
        res = minimize(
            lambda w: float(w @ cov @ w), w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 3000, "ftol": 1e-9}
        )
        if res.success or res.fun < 0.1:
            w = res.x
            r, v, s = _portfolio_stats(w, mu, cov, rf)
            frontier_rets.append(round(r, 6))
            frontier_vols.append(round(v, 6))
            frontier_sh.append(round(s, 6))
            frontier_ws.append(w.tolist())

    return EfficientFrontier(
        returns=frontier_rets, vols=frontier_vols,
        sharpes=frontier_sh, weights=frontier_ws,
        tickers=tickers,
        min_var=min_var, max_sharpe=max_sharpe,
    )


# ══════════════════════════════════════════════════════════════════════════
#  BLACK-LITTERMAN
# ══════════════════════════════════════════════════════════════════════════

def black_litterman(mu_mkt: np.ndarray, cov: np.ndarray,
                     tickers: List[str],
                     views: List[Dict],
                     tau: float = 0.05,
                     rf: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Black-Litterman model.

    views: list of dicts, each with:
      {"assets": ["AAPL", "MSFT"], "weights": [1, -1], "return": 0.05, "confidence": 0.8}
      — "AAPL outperforms MSFT by 5% with 80% confidence"

    Returns: (bl_mu, bl_cov) — posterior expected returns and covariance.
    """
    n = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    if not views:
        return mu_mkt.copy(), cov.copy()

    k = len(views)
    P = np.zeros((k, n))   # view picking matrix
    q = np.zeros(k)        # view returns
    Omega_diag = np.zeros(k)  # view uncertainty

    for i, v in enumerate(views):
        assets  = v.get("assets", [])
        weights = v.get("weights", [1.0] * len(assets))
        for asset, w in zip(assets, weights):
            if asset in ticker_idx:
                P[i, ticker_idx[asset]] = w
        q[i] = v.get("return", 0.0)
        conf = v.get("confidence", 0.5)
        # Omega: inversely proportional to confidence
        Omega_diag[i] = (1 - conf) / conf * (tau * float(P[i] @ cov @ P[i]))

    Omega = np.diag(Omega_diag + 1e-8)
    Pi = mu_mkt                        # market equilibrium returns
    Sigma_pi = tau * cov               # prior uncertainty on Pi

    # BL posterior
    inv_Omega   = np.linalg.inv(Omega)
    inv_Sigma   = np.linalg.inv(Sigma_pi)
    M1          = inv_Sigma + P.T @ inv_Omega @ P
    M2          = inv_Sigma @ Pi + P.T @ inv_Omega @ q
    bl_mu       = np.linalg.solve(M1, M2)
    bl_cov      = cov + np.linalg.inv(M1)

    return bl_mu, bl_cov


def market_implied_returns(cov: np.ndarray, mkt_weights: np.ndarray,
                            lam: float = 2.5, rf: float = 0.05) -> np.ndarray:
    """
    Reverse-engineer market implied returns (Pi) from market cap weights.
    Pi = lambda * Sigma * w_mkt
    lambda (risk aversion) ≈ 2.5 for equity markets.
    """
    return rf + lam * cov @ mkt_weights


# ══════════════════════════════════════════════════════════════════════════
#  RISK PARITY
# ══════════════════════════════════════════════════════════════════════════

def risk_parity(cov: np.ndarray, tickers: List[str],
                mu: Optional[np.ndarray] = None,
                rf: float = 0.05) -> PortfolioResult:
    """
    Equal Risk Contribution (ERC) portfolio.
    Each asset contributes equally to total portfolio variance.
    Used by Bridgewater (All Weather), AQR, and most risk-parity funds.
    """
    n = len(tickers)
    mu_ = mu if mu is not None else np.zeros(n)

    def risk_contributions(w):
        port_var = w @ cov @ w
        mrc = cov @ w              # marginal risk contribution
        rc  = w * mrc              # risk contribution per asset
        return rc / port_var if port_var > 1e-10 else rc

    def objective(w):
        rc = risk_contributions(w)
        target = 1.0 / n
        return float(np.sum((rc - target)**2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = ((1e-6, 1.0),) * n   # long-only (risk parity is usually long-only)
    w0 = np.ones(n) / n

    # Use differential evolution for global optimum (ERC has local minima)
    de_res = differential_evolution(
        objective, bounds=[(1e-6, 1.0)] * n, seed=42,
        maxiter=500, tol=1e-10, workers=1
    )
    w = de_res.x / de_res.x.sum()

    ret, vol, sharpe = _portfolio_stats(w, mu_, cov, rf)
    rc = risk_contributions(w)
    return PortfolioResult(
        weights=w, tickers=tickers,
        exp_return=round(ret, 6), volatility=round(vol, 6), sharpe=round(sharpe, 6),
        weights_dict={t: round(float(wi), 6) for t, wi in zip(tickers, w)}
    )


# ══════════════════════════════════════════════════════════════════════════
#  FACTOR MODEL  (Fama-French style)
# ══════════════════════════════════════════════════════════════════════════

def factor_decomposition(asset_returns: pd.DataFrame,
                          market_returns: pd.Series) -> pd.DataFrame:
    """
    Single-factor CAPM decomposition: r_i = alpha_i + beta_i * r_mkt + epsilon_i
    Returns DataFrame with alpha, beta, R², tracking error, info ratio per asset.
    """
    results = []
    for ticker in asset_returns.columns:
        r = asset_returns[ticker].dropna()
        mkt = market_returns.reindex(r.index).dropna()
        common = r.index.intersection(mkt.index)
        r, mkt = r[common], mkt[common]
        if len(r) < 20:
            continue

        # OLS regression: r = alpha + beta * mkt
        X = np.column_stack([np.ones(len(mkt)), mkt.values])
        try:
            coeffs, resid, _, _ = np.linalg.lstsq(X, r.values, rcond=None)
        except Exception:
            continue
        alpha, beta = coeffs
        fitted = X @ coeffs
        ss_res = float(np.sum((r.values - fitted)**2))
        ss_tot = float(np.sum((r.values - r.values.mean())**2))
        r_sq   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        epsilon = r.values - fitted
        te_annual = float(epsilon.std(ddof=1) * np.sqrt(252))
        ir = float(alpha * 252 / te_annual) if te_annual > 1e-8 else 0.0

        results.append({
            "ticker":          ticker,
            "alpha_annual":    round(float(alpha * 252), 6),
            "beta":            round(float(beta), 4),
            "r_squared":       round(float(r_sq), 4),
            "tracking_error":  round(te_annual, 6),
            "info_ratio":      round(ir, 4),
            "vol_annual":      round(float(r.std(ddof=1) * np.sqrt(252)), 6),
            "corr_mkt":        round(float(r.corr(mkt)), 4),
        })

    return pd.DataFrame(results).set_index("ticker") if results else pd.DataFrame()


def risk_attribution(weights: np.ndarray, cov: np.ndarray,
                      tickers: List[str]) -> pd.DataFrame:
    """
    Break down portfolio variance into per-asset risk contributions.
    Essential for risk budgeting and regulatory reporting.
    """
    port_var = float(weights @ cov @ weights)
    port_vol = np.sqrt(port_var)
    mrc = cov @ weights                         # marginal risk contribution
    rc  = weights * mrc                          # risk contribution (dollar)
    pct = rc / port_var if port_var > 0 else rc  # % of total variance

    return pd.DataFrame({
        "weight":            np.round(weights, 4),
        "marginal_rc":       np.round(mrc, 6),
        "risk_contribution": np.round(rc, 6),
        "pct_risk":          np.round(pct * 100, 3),
        "vol_contribution":  np.round(np.sqrt(np.abs(rc)) * np.sign(rc), 6),
    }, index=tickers)


# ══════════════════════════════════════════════════════════════════════════
#  ROLLING BACKTEST
# ══════════════════════════════════════════════════════════════════════════

def rolling_backtest(returns: pd.DataFrame,
                      method: str = "max_sharpe",
                      lookback: int = 252,
                      rebalance_every: int = 21,
                      rf: float = 0.05,
                      cov_method: str = "ledoit_wolf") -> Dict:
    """
    Walk-forward backtest of a portfolio optimization strategy.
    Lookback: estimation window in trading days (252 = 1 year).
    Rebalances every N days.

    Returns cumulative returns, weights over time, and summary stats.
    """
    n = len(returns)
    tickers = list(returns.columns)
    portfolio_rets = []
    all_dates = []
    weight_history = []

    current_weights = np.ones(len(tickers)) / len(tickers)

    for i in range(lookback, n):
        hist = returns.iloc[i-lookback:i]

        # Rebalance
        if (i - lookback) % rebalance_every == 0:
            mu  = hist.mean().values * 252
            cov = estimate_covariance(hist, cov_method)
            # Ensure PD
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                cov = nearest_pd(cov)

            try:
                if method == "max_sharpe":
                    res = max_sharpe_portfolio(mu, cov, tickers, rf=rf)
                elif method == "min_variance":
                    res = min_variance_portfolio(mu, cov, tickers, rf=rf)
                elif method == "risk_parity":
                    res = risk_parity(cov, tickers, mu, rf=rf)
                else:
                    res = max_sharpe_portfolio(mu, cov, tickers, rf=rf)
                current_weights = res.weights
            except Exception as e:
                log.warning(f"Backtest rebalance failed at step {i}: {e}")

        weight_history.append(current_weights.tolist())
        day_ret = float(returns.iloc[i].values @ current_weights)
        portfolio_rets.append(day_ret)
        all_dates.append(returns.index[i])

    rets = np.array(portfolio_rets)
    cum  = np.cumprod(1 + rets) - 1

    # Summary stats
    ann_ret  = float(rets.mean() * 252)
    ann_vol  = float(rets.std(ddof=1) * np.sqrt(252))
    sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    mdd      = float(np.min(np.minimum.accumulate(cum + 1) / np.maximum.accumulate(cum + 1) - 1))
    calmar   = ann_ret / abs(mdd) if abs(mdd) > 1e-8 else 0.0
    sortino_d = float(rets[rets < 0].std(ddof=1) * np.sqrt(252)) if (rets < 0).any() else 1e-10
    sortino  = ann_ret / sortino_d

    # Equal-weight benchmark
    eq_rets  = returns.iloc[lookback:].mean(axis=1).values
    eq_cum   = np.cumprod(1 + eq_rets) - 1
    eq_ar    = float(eq_rets.mean() * 252)
    eq_vol   = float(eq_rets.std(ddof=1) * np.sqrt(252))
    eq_sh    = (eq_ar - rf) / eq_vol if eq_vol > 0 else 0.0

    return {
        "dates":       [str(d.date()) if hasattr(d, "date") else str(d) for d in all_dates],
        "port_rets":   np.round(rets, 6).tolist(),
        "port_cum":    np.round(cum, 6).tolist(),
        "eq_cum":      np.round(eq_cum, 6).tolist(),
        "weight_hist": weight_history,
        "tickers":     tickers,
        "stats": {
            "ann_return":  round(ann_ret, 6),
            "ann_vol":     round(ann_vol, 6),
            "sharpe":      round(sharpe, 4),
            "sortino":     round(sortino, 4),
            "max_dd":      round(mdd, 6),
            "calmar":      round(calmar, 4),
            "eq_return":   round(eq_ar, 6),
            "eq_vol":      round(eq_vol, 6),
            "eq_sharpe":   round(eq_sh, 4),
        },
    }


def portfolio_cvar(weights: np.ndarray, returns: np.ndarray,
                    alpha: float = 0.05) -> Dict:
    """
    Historical Expected Shortfall (CVaR) at confidence level alpha.

    CVaR = E[ -r  |  r <= VaR_alpha ]

    This is the primary risk metric used by institutional desks since the
    Basel III accord mandated CVaR over VaR for internal capital models.

    Returns: var, cvar, worst_5_pct daily returns
    """
    w      = np.asarray(weights, float)
    R      = np.asarray(returns, float)
    port_r = R @ w  # daily portfolio returns
    var    = float(np.percentile(port_r, alpha * 100))
    tail   = port_r[port_r <= var]
    cvar   = float(tail.mean()) if len(tail) > 0 else var
    return {
        "var":         round(var, 6),
        "cvar":        round(cvar, 6),
        "var_ann":     round(var * np.sqrt(252), 6),
        "cvar_ann":    round(cvar * np.sqrt(252), 6),
        "var_pct":     round(var * 100, 4),
        "cvar_pct":    round(cvar * 100, 4),
        "tail_obs":    int(len(tail)),
        "confidence":  round(1 - alpha, 4),
    }


def optimize_min_cvar(returns: np.ndarray, alpha: float = 0.05,
                       constraints: Optional[Dict] = None) -> Dict:
    """
    Minimum CVaR portfolio optimisation (Rockafellar & Uryasev 2000).

    Solves the linear programming formulation:
      min_w CVaR_alpha(w' r)
      s.t.  sum(w) = 1, w >= 0

    This is more robust than minimum variance for fat-tailed returns
    (as is characteristic of equity distributions).
    """
    from scipy.optimize import minimize

    R = np.asarray(returns, float)
    T, n = R.shape

    def neg_cvar(w):
        w = w / (w.sum() + 1e-10)
        port_r = R @ w
        var    = np.percentile(port_r, alpha * 100)
        tail   = port_r[port_r <= var]
        return float(-tail.mean()) if len(tail) > 0 else float(-var)

    # Initial guess: equal weight
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    if constraints:
        if "max_weight" in constraints:
            mx = constraints["max_weight"]
            bounds = [(0, mx)] * n

    res = minimize(neg_cvar, w0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 2000, "ftol": 1e-10})

    if not res.success:
        return {"error": "CVaR optimisation did not converge", "weights": w0.tolist()}

    w_opt = res.x / res.x.sum()
    cv    = portfolio_cvar(w_opt, R, alpha)
    mu    = float(R @ w_opt)
    vol   = float((R @ w_opt).std(ddof=1))

    return {
        "weights":  np.round(w_opt, 6).tolist(),
        "cvar":     cv,
        "ann_ret":  round(mu * 252 * 100, 4),
        "ann_vol":  round(vol * np.sqrt(252) * 100, 4),
        "sharpe":   round(mu * 252 / max(vol * np.sqrt(252), 1e-8), 4),
    }
