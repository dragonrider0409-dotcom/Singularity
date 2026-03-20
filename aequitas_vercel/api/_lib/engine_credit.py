"""
credit_risk/engine.py
======================
Credit Risk Dashboard

Covers:
  1. Merton Structural Model  — equity as call option on assets, PD, DD
  2. CDS Pricing              — par spread, upfront payment, hazard rate
  3. CVA                      — Credit Valuation Adjustment for derivatives
  4. Credit Curves            — survival probabilities, hazard rates, term structure
  5. Portfolio Credit Risk    — correlated defaults, concentration, expected loss

Run:
  python credit_risk/app.py  ->  http://127.0.0.1:5007
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from typing import List, Dict, Tuple, Optional
import warnings, logging

warnings.filterwarnings("ignore")
log = logging.getLogger("cr_engine")


# ══════════════════════════════════════════════════════════════════════════
#  MERTON STRUCTURAL MODEL
# ══════════════════════════════════════════════════════════════════════════

def merton_model(V: float, sigma_V: float, D: float,
                  r: float, T: float) -> Dict:
    """
    Merton (1974): firm equity = call option on asset value.
    E = V*N(d1) - D*e^{-rT}*N(d2)

    V:       firm asset value
    sigma_V: asset volatility (annual)
    D:       face value of debt
    r:       risk-free rate
    T:       debt maturity (years)
    """
    if V <= 0 or sigma_V <= 0 or D <= 0 or T <= 0:
        return {"error": "All parameters must be positive"}

    sT   = sigma_V * np.sqrt(T)
    d1   = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / sT
    d2   = d1 - sT

    E    = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    E    = max(E, 1e-8)

    # Equity vol from Ito's lemma
    sigma_E = sigma_V * V * norm.cdf(d1) / E

    # Risk-neutral PD
    pd_rn    = norm.cdf(-d2)

    # Real-world PD (Bharath-Shumway approximation, Sharpe ratio = 0.5)
    mu_adjust = 0.5 * sigma_V   # approx: drift = r + 0.5*sigma adjustment
    d2_rw     = (np.log(V / D) + (mu_adjust - 0.5 * sigma_V**2) * T) / sT
    pd_rw     = norm.cdf(-d2_rw)

    # Debt value
    debt_val  = V - E

    # Credit spread (yield above risk-free)
    if debt_val > 1e-6 and D > 0 and T > 0:
        y_risky = -np.log(debt_val / D) / T
        cs_bps  = max((y_risky - r) * 10000, 0)
    else:
        cs_bps  = 0.0

    return {
        "equity_value":          round(float(E), 4),
        "debt_value":            round(float(debt_val), 4),
        "equity_vol":            round(float(sigma_E), 6),
        "distance_to_default":   round(float(d2), 4),
        "d1":                    round(float(d1), 4),
        "d2":                    round(float(d2), 4),
        "pd_risk_neutral":       round(float(pd_rn), 6),
        "pd_real_world":         round(float(pd_rw), 6),
        "credit_spread_bps":     round(float(cs_bps), 4),
        "leverage":              round(float(D / max(V, 1e-8)), 4),
        "ltv":                   round(float(D / max(E + debt_val, 1e-8)), 4),
    }


def merton_calibrate(equity: float, sigma_equity: float,
                      D: float, r: float, T: float) -> Dict:
    """
    Calibrate Merton model from observable equity price and equity vol.
    Solves system: E = f(V, sigma_V) and sigma_E * E = sigma_V * V * N(d1)
    """
    def equations(x):
        V_try, sig_try = x
        if V_try <= 0 or sig_try <= 0:
            return [1e10, 1e10]
        sT   = sig_try * np.sqrt(T)
        d1   = (np.log(V_try/D) + (r + 0.5*sig_try**2)*T) / sT
        d2   = d1 - sT
        E_implied     = V_try*norm.cdf(d1) - D*np.exp(-r*T)*norm.cdf(d2)
        sigE_implied  = sig_try * V_try * norm.cdf(d1) / max(E_implied, 1e-8)
        return [E_implied - equity, sigE_implied - sigma_equity]

    # Solve via minimisation
    def obj(x): return float(np.sum(np.array(equations(x))**2))
    V0 = equity + D * np.exp(-r * T)
    res = minimize(obj, [V0, sigma_equity * 0.7], method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-12})

    V_cal, sig_cal = res.x
    if V_cal <= 0 or sig_cal <= 0:
        return {"error": "Calibration failed"}

    return merton_model(V_cal, sig_cal, D, r, T)


def merton_term_structure(V: float, sigma_V: float, D: float,
                           r: float, maturities: List[float]) -> Dict:
    """Merton credit spread term structure across maturities."""
    spreads = []
    pds     = []
    dds     = []
    for T in maturities:
        m = merton_model(V, sigma_V, D, r, T)
        spreads.append(m.get("credit_spread_bps", 0))
        pds.append(m.get("pd_risk_neutral", 0))
        dds.append(m.get("distance_to_default", 0))
    return {
        "maturities": maturities,
        "spreads_bps": spreads,
        "pd_rn":       pds,
        "distance_to_default": dds,
    }


# ══════════════════════════════════════════════════════════════════════════
#  CDS PRICING
# ══════════════════════════════════════════════════════════════════════════

def hazard_from_spread(spread_bps: float, recovery: float = 0.40) -> float:
    """Bootstrap constant hazard rate from par CDS spread."""
    lgd = 1 - recovery
    return spread_bps / (lgd * 10000) if lgd > 0 else 0.0


def survival_probs(hazard_rates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Q(t) = exp(-integral_0^t h(s) ds). Constant hazard per period."""
    h  = np.asarray(hazard_rates, float)
    t  = np.asarray(times, float)
    dt = np.diff(np.concatenate([[0], t]))
    return np.exp(-np.cumsum(h * dt))


def cds_par_spread(hazard: float, recovery: float,
                    r: float, maturity: float = 5.0,
                    freq: int = 4) -> float:
    """
    Par CDS spread for constant hazard rate.
    Par spread = LGD * sum_t [Q(t-dt) - Q(t)] * DF(t) / sum_t Q(t) * dt * DF(t)
    """
    lgd   = 1 - recovery
    dt    = 1 / freq
    times = np.arange(dt, maturity + dt, dt)
    Q     = np.exp(-hazard * times)
    df    = np.exp(-r * times)

    # Protection leg: expected discounted loss
    Q_prev = np.concatenate([[1.0], Q[:-1]])
    prot   = lgd * float(np.sum((Q_prev - Q) * df))

    # Premium leg: expected discounted premium payments
    prem   = float(np.sum(Q * df)) * dt

    return (prot / prem * 10000) if prem > 1e-10 else 0.0


def cds_mtm(hazard_orig: float, hazard_new: float, recovery: float,
             r: float, remaining_T: float, spread_bps: float,
             notional: float = 1_000_000, freq: int = 4) -> float:
    """
    MTM of an existing CDS position when hazard rate changes.
    MTM = Notional * (new_par_spread - contractual_spread) * RPV01
    """
    dt    = 1 / freq
    times = np.arange(dt, remaining_T + dt, dt)
    Q_new = np.exp(-hazard_new * times)
    df    = np.exp(-r * times)
    rpv01 = float(np.sum(Q_new * df)) * dt

    new_spread = cds_par_spread(hazard_new, recovery, r, remaining_T, freq)
    mtm = notional * (new_spread - spread_bps) / 10000 * rpv01
    return round(float(mtm), 4)


def credit_curve(spreads_bps: List[float], maturities: List[float],
                  recovery: float = 0.40, r: float = 0.04) -> Dict:
    """
    Bootstrap hazard rate curve from CDS par spreads.
    Returns hazard rates, survival probabilities, discount factors.
    """
    mats    = np.array(maturities, float)
    spreads = np.array(spreads_bps, float)
    n       = len(mats)
    hazards = np.zeros(n)

    for i in range(n):
        target_spread = spreads[i]
        T_i    = mats[i]

        def residual(h_i):
            h_vec = np.concatenate([hazards[:i], [max(h_i, 1e-6)]])
            mats_i = mats[:i+1]
            dt_vec = np.diff(np.concatenate([[0], mats_i]))
            Q      = np.exp(-np.cumsum(h_vec * dt_vec))
            df     = np.exp(-r * mats_i)
            Q_prev = np.concatenate([[1.0], Q[:-1]])
            lgd    = 1 - recovery
            prot   = lgd * float(np.sum((Q_prev - Q) * df))
            freq   = 4
            dt_p   = 1 / freq
            times_p = np.arange(dt_p, T_i + dt_p, dt_p)
            # Interpolate hazard for fine grid
            if i == 0:
                h_fine = np.full(len(times_p), h_vec[-1])
            else:
                h_fine = np.interp(times_p, mats_i, h_vec)
            Q_fine = np.exp(-h_fine.cumsum() * dt_p)
            df_fine = np.exp(-r * times_p)
            prem   = float(np.sum(Q_fine * df_fine)) * dt_p
            calc_spread = (prot / prem * 10000) if prem > 1e-10 else 0
            return calc_spread - target_spread

        try:
            h_i = brentq(residual, 1e-6, 2.0, xtol=1e-8)
        except Exception:
            h_i = max(spreads[i] / ((1 - recovery) * 10000), 1e-6)
        hazards[i] = h_i

    # Compute survival probs at fine grid
    fine = np.linspace(0.25, max(mats), 100)
    h_fine = np.interp(fine, mats, hazards)
    dt_f   = np.diff(np.concatenate([[0], fine]))
    Q_fine = np.exp(-np.cumsum(h_fine * dt_f))
    df_fine = np.exp(-r * fine)

    return {
        "maturities":       mats.tolist(),
        "hazard_rates":     np.round(hazards * 100, 4).tolist(),      # % per year
        "implied_spreads":  [round(cds_par_spread(h, recovery, r, T), 2) for h, T in zip(hazards, mats)],
        "fine_mats":        np.round(fine, 4).tolist(),
        "survival_probs":   np.round(Q_fine, 6).tolist(),
        "discount_factors": np.round(df_fine, 6).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════
#  CVA — Credit Valuation Adjustment
# ══════════════════════════════════════════════════════════════════════════

def cva_calculation(exposure_profile: List[float],
                     times: List[float],
                     hazard: float,
                     recovery: float,
                     r: float) -> Dict:
    """
    CVA = LGD * sum_t [ PD(t-1→t) * DF(t) * EE(t) ]
    where EE = Expected Exposure (positive MtM of derivative).

    exposure_profile: Expected Exposure at each time step
    """
    lgd  = 1 - recovery
    T    = np.array(times, float)
    EE   = np.array(exposure_profile, float)
    DF   = np.exp(-r * T)
    Q    = np.exp(-hazard * T)
    Q_prev = np.concatenate([[1.0], Q[:-1]])
    PD_marg = np.maximum(Q_prev - Q, 0)

    cva_components = lgd * PD_marg * DF * EE
    total_cva      = float(cva_components.sum())

    # DVA (Debit Valuation Adjustment) — symmetric, assume same credit quality
    dva_components = cva_components * 0.0   # simplified: DVA=0 for basic case

    return {
        "total_cva":        round(total_cva, 4),
        "cva_components":   np.round(cva_components, 4).tolist(),
        "exposure":         np.round(EE, 4).tolist(),
        "survival_probs":   np.round(Q, 6).tolist(),
        "pd_marginal":      np.round(PD_marg, 6).tolist(),
        "discount_factors": np.round(DF, 6).tolist(),
        "times":            T.tolist(),
        "lgd":              round(lgd, 4),
        "hazard":           round(hazard, 6),
    }


# ══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO CREDIT RISK
# ══════════════════════════════════════════════════════════════════════════

def portfolio_credit_loss(notionals: List[float],
                           pds: List[float],
                           lgds: List[float],
                           correlation: float = 0.2,
                           n_sim: int = 50_000) -> Dict:
    """
    Monte Carlo simulation of portfolio credit loss.
    One-factor Gaussian copula: X_i = sqrt(rho)*Z + sqrt(1-rho)*eps_i
    Default if X_i < Phi^{-1}(PD_i).
    """
    n = len(notionals)
    N_arr = np.array(notionals, float)
    PD    = np.array(pds,       float)
    LGD   = np.array(lgds,      float)
    EL    = N_arr * PD * LGD    # expected loss per name

    rho   = correlation
    rng   = np.random.default_rng(42)
    Z     = rng.standard_normal(n_sim)       # systematic factor
    eps   = rng.standard_normal((n_sim, n))  # idiosyncratic
    X     = np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * eps

    thresholds = norm.ppf(np.clip(PD, 1e-10, 1 - 1e-10))
    defaults   = X < thresholds[None, :]    # (n_sim, n)
    losses     = (defaults * (N_arr * LGD)[None, :]).sum(axis=1)

    el_port  = float(losses.mean())
    ul_port  = float(losses.std(ddof=1))
    var_95   = float(np.percentile(losses, 95))
    var_99   = float(np.percentile(losses, 99))
    cvar_99  = float(losses[losses >= var_99].mean())

    # Loss distribution histogram
    counts, edges = np.histogram(losses, bins=80)

    return {
        "expected_loss":   round(el_port, 2),
        "unexpected_loss": round(ul_port, 2),
        "var_95":          round(var_95, 2),
        "var_99":          round(var_99, 2),
        "cvar_99":         round(cvar_99, 2),
        "el_per_name":     np.round(EL, 4).tolist(),
        "loss_hist":       {"c": counts.tolist(), "e": np.round(edges, 2).tolist()},
        "correlation":     correlation,
        "n_names":         n,
    }
