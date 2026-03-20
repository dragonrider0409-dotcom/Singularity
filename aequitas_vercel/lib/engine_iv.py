"""
vol_surface/engine.py
=====================
Implied Volatility Surface & Model Calibration Engine

Directly extends mc_engine_v2:
  - Imports SimConfig, black_scholes, implied_vol from mc_engine_v2
  - Adds SABR closed-form (Hagan 2002) for fast surface generation
  - Adds Heston characteristic-function pricing (exact, no MC noise)
  - Adds calibration: fit SABR or Heston to a set of market prices
  - Builds full K × T surface grids ready for 3D plotting

Usage (same folder as mc_engine_v2.py):
  from vol_surface.engine import build_surface, calibrate_sabr, calibrate_heston
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize, differential_evolution
from scipy.integrate import quad
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings, logging

# ── Import from the Monte Carlo engine ────────────────────────────────────
from mc_engine_v2 import SimConfig, black_scholes, implied_vol

warnings.filterwarnings("ignore")
log = logging.getLogger("vol_surface")


# ══════════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES IV SOLVER  (standalone, faster than mc_engine_v2.implied_vol)
# ══════════════════════════════════════════════════════════════════════════

def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, opt: str = "call") -> float:
    """Vectorisable B-S price. Returns 0 for bad inputs."""
    if sigma <= 0 or T <= 0:
        return max(S * np.exp(-q*T) - K * np.exp(-r*T), 0) if opt=="call" \
               else max(K * np.exp(-r*T) - S * np.exp(-q*T), 0)
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / sT
    d2 = d1 - sT
    if opt == "call":
        return float(S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))
    return float(K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1))


def bs_iv(price: float, S: float, K: float, T: float, r: float,
          q: float = 0.0, opt: str = "call") -> float:
    """Brent's method IV inversion. Returns nan if no solution."""
    if T <= 0:
        return float("nan")
    intrinsic = max((S*np.exp(-q*T) - K*np.exp(-r*T)) * (1 if opt=="call" else -1), 0)
    if price < intrinsic - 1e-5:
        return float("nan")
    try:
        return float(brentq(
            lambda sig: bs_price(S, K, T, r, q, sig, opt) - price,
            1e-5, 10.0, xtol=1e-8, maxiter=500
        ))
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════
#  SABR CLOSED-FORM  (Hagan et al. 2002)
# ══════════════════════════════════════════════════════════════════════════

def sabr_iv(F: float, K: float, T: float,
            alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    SABR implied volatility (Hagan 2002 approximation).
    F: forward price  K: strike  T: expiry
    alpha: initial vol  beta: CEV exponent  rho: F-vol corr  nu: vol-of-vol
    """
    if alpha <= 0 or nu < 0 or T <= 0:
        return float("nan")
    F  = max(F, 1e-10)
    K  = max(K, 1e-10)
    eps = 1e-7
    if abs(F - K) < eps:
        # ATM formula
        fkb   = F**(1 - beta)
        term1 = alpha / fkb
        term2 = 1 + ((1-beta)**2/24 * alpha**2 / fkb**2
                     + rho*beta*nu*alpha / (4*fkb)
                     + (2 - 3*rho**2)*nu**2/24) * T
        return max(term1 * term2, 1e-6)

    log_fk = np.log(F / K)
    fkb    = (F*K)**((1-beta)/2)
    z      = nu / alpha * fkb * log_fk
    rho_sq = max(1 - rho**2, 0)
    x_z_num = np.sqrt(rho_sq + (2*rho*z - z**2 + z**2*rho_sq)**0.0) if False else \
              np.sqrt(1 - 2*rho*z + z**2)
    # Standard formula
    x_z    = np.log((x_z_num + z - rho) / (1 - rho + 1e-12))
    if abs(x_z) < 1e-10:
        x_z = 1.0

    term1  = alpha / (fkb * (1 + (1-beta)**2/24 * log_fk**2
                              + (1-beta)**4/1920 * log_fk**4))
    term2  = z / x_z if abs(z) > 1e-10 else 1.0
    term3  = 1 + ((1-beta)**2/24 * alpha**2 / fkb**2
                  + rho*beta*nu*alpha / (4*fkb)
                  + (2 - 3*rho**2)*nu**2/24) * T
    return max(float(term1 * term2 * term3), 1e-6)


def sabr_price(S: float, K: float, T: float, r: float, q: float,
               alpha: float, beta: float, rho: float, nu: float,
               opt: str = "call") -> float:
    """SABR model price via IV -> B-S."""
    F  = S * np.exp((r - q) * T)
    iv = sabr_iv(F, K, T, alpha, beta, rho, nu)
    if np.isnan(iv):
        return float("nan")
    return bs_price(S, K, T, r, q, iv, opt)


# ══════════════════════════════════════════════════════════════════════════
#  HESTON CHARACTERISTIC-FUNCTION PRICING  (exact, no MC noise)
# ══════════════════════════════════════════════════════════════════════════

def heston_price_cf(S: float, K: float, T: float, r: float, q: float,
                    v0: float, kappa: float, theta: float,
                    xi: float, rho: float, opt: str = "call") -> float:
    """
    Heston (1993) semi-analytic price via characteristic function integration.
    No Monte Carlo — exact up to numerical integration error.
    """
    if T <= 0 or v0 <= 0:
        return float("nan")

    def integrand(phi, j):
        """Real part of the integrand for P1 and P2."""
        u  = 0.5 if j == 1 else -0.5
        b  = kappa - rho*xi if j == 1 else kappa
        d  = np.sqrt((rho*xi*1j*phi - b)**2 - xi**2*(2*u*1j*phi - phi**2))
        g  = (b - rho*xi*1j*phi + d) / (b - rho*xi*1j*phi - d)
        # Avoid numerical issues with exp of large imaginary numbers
        try:
            e_dT  = np.exp(-d*T)
            denom = 1 - g*e_dT
            if abs(denom) < 1e-12:
                return 0.0
            C = r*1j*phi*T + kappa*theta/xi**2 * (
                (b - rho*xi*1j*phi + d)*T - 2*np.log((1 - g*e_dT)/(1 - g))
            )
            D = (b - rho*xi*1j*phi + d)/xi**2 * (1 - e_dT)/denom
            cf = np.exp(C + D*v0 + 1j*phi*np.log(S*np.exp((r-q)*T)))
            return np.real(np.exp(-1j*phi*np.log(K)) * cf / (1j*phi))
        except Exception:
            return 0.0

    P = []
    for j in [1, 2]:
        try:
            val, _ = quad(lambda phi: integrand(phi, j), 0, 200,
                          limit=200, epsabs=1e-8, epsrel=1e-6)
            P.append(0.5 + val/np.pi)
        except Exception:
            P.append(0.5)

    call = S*np.exp(-q*T)*P[0] - K*np.exp(-r*T)*P[1]
    call = max(call, max(S*np.exp(-q*T) - K*np.exp(-r*T), 0))
    if opt == "call":
        return float(call)
    # Put via put-call parity
    return float(call - S*np.exp(-q*T) + K*np.exp(-r*T))


def heston_iv(S: float, K: float, T: float, r: float, q: float,
              v0: float, kappa: float, theta: float,
              xi: float, rho: float, opt: str = "call") -> float:
    """Heston model -> IV (via exact CF price -> BS IV)."""
    price = heston_price_cf(S, K, T, r, q, v0, kappa, theta, xi, rho, opt)
    if np.isnan(price):
        return float("nan")
    return bs_iv(price, S, K, T, r, q, opt)


# ══════════════════════════════════════════════════════════════════════════
#  CALIBRATION
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CalibResult:
    model:     str
    params:    Dict[str, float]
    rmse:      float          # root mean sq error in IV points
    mae:       float          # mean absolute error in IV points
    n_points:  int
    success:   bool
    model_ivs: List[float]   # model IVs at each calibration point


def calibrate_sabr(strikes: np.ndarray, market_ivs: np.ndarray,
                   S: float, T: float, r: float, q: float = 0.0,
                   beta: float = 0.5) -> CalibResult:
    """
    Calibrate SABR (alpha, rho, nu) to market IVs at fixed beta.
    Uses Nelder-Mead with multiple starts for robustness.
    """
    F    = S * np.exp((r - q) * T)
    ivs  = np.array(market_ivs, dtype=float)
    valid = ~np.isnan(ivs) & (ivs > 0)
    Kv   = strikes[valid]
    iv_v = ivs[valid]

    if len(Kv) < 3:
        return CalibResult("SABR", {}, float("nan"), float("nan"), 0, False, [])

    def loss(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu < 0 or abs(rho) >= 0.999:
            return 1e6
        errs = []
        for K, iv_mkt in zip(Kv, iv_v):
            iv_mod = sabr_iv(F, K, T, alpha, beta, rho, nu)
            if np.isnan(iv_mod):
                return 1e6
            errs.append((iv_mod - iv_mkt)**2)
        return float(np.mean(errs))

    best_res = None
    # Multiple starts
    for a0, r0, n0 in [(0.20,-0.3,0.4),(0.30,-0.5,0.6),(0.15,-0.1,0.3),(0.40,-0.7,0.8)]:
        try:
            res = minimize(loss, [a0, r0, n0], method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-8})
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        except Exception:
            pass

    if best_res is None or best_res.fun > 0.5:
        return CalibResult("SABR", {}, float("nan"), float("nan"), len(Kv), False, [])

    alpha, rho, nu = best_res.x
    model_ivs = [sabr_iv(F, K, T, alpha, beta, rho, nu) for K in strikes]
    errors_bp = [abs(m - i)*100 for m, i in zip(
        [sabr_iv(F, K, T, alpha, beta, rho, nu) for K in Kv], iv_v)]
    return CalibResult(
        model="SABR",
        params={"alpha": round(alpha,6), "beta": beta,
                "rho":   round(rho,6),   "nu":   round(nu,6)},
        rmse=round(float(np.sqrt(best_res.fun))*100, 4),
        mae=round(float(np.mean(errors_bp)), 4),
        n_points=int(len(Kv)),
        success=best_res.success or best_res.fun < 0.01,
        model_ivs=model_ivs,
    )


def calibrate_heston(strikes: np.ndarray, market_ivs: np.ndarray,
                     S: float, T: float, r: float, q: float = 0.0,
                     v0_init: float = 0.04) -> CalibResult:
    """
    Calibrate Heston (kappa, theta, xi, rho) to market IVs.
    Uses differential evolution for global optimum, then refines with L-BFGS-B.
    """
    ivs   = np.array(market_ivs, dtype=float)
    valid = ~np.isnan(ivs) & (ivs > 0)
    Kv    = strikes[valid]
    iv_v  = ivs[valid]

    if len(Kv) < 4:
        return CalibResult("Heston", {}, float("nan"), float("nan"), 0, False, [])

    v0 = v0_init

    def loss(params):
        kappa, theta, xi, rho = params
        # Feller condition preference (soft)
        if kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) >= 0.999:
            return 1e6
        errs = []
        for K, iv_mkt in zip(Kv, iv_v):
            try:
                iv_mod = heston_iv(S, K, T, r, q, v0, kappa, theta, xi, rho)
                if np.isnan(iv_mod) or iv_mod <= 0:
                    return 1e6
                errs.append((iv_mod - iv_mkt)**2)
            except Exception:
                return 1e6
        return float(np.mean(errs)) + max(0, xi**2 - 2*kappa*theta) * 0.1

    bounds = [(0.1, 20), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.99)]
    try:
        de_res = differential_evolution(loss, bounds, maxiter=300, tol=1e-6,
                                         seed=42, workers=1, popsize=12)
        res = minimize(loss, de_res.x, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-10})
        best = res if res.fun < de_res.fun else de_res
    except Exception as e:
        log.warning(f"Heston calibration failed: {e}")
        return CalibResult("Heston", {}, float("nan"), float("nan"), len(Kv), False, [])

    kappa, theta, xi, rho = best.x
    model_ivs = [heston_iv(S, K, T, r, q, v0, kappa, theta, xi, rho) for K in strikes]
    errors_bp = [abs(m - i)*100 for m, i in zip(
        [heston_iv(S, K, T, r, q, v0, kappa, theta, xi, rho) for K in Kv], iv_v)
        if not np.isnan(m)]
    return CalibResult(
        model="Heston",
        params={"v0": round(v0,6), "kappa": round(kappa,6), "theta": round(theta,6),
                "xi":  round(xi,6),  "rho":   round(rho,6)},
        rmse=round(float(np.sqrt(best.fun))*100, 4),
        mae=round(float(np.mean(errors_bp)) if errors_bp else float("nan"), 4),
        n_points=int(len(Kv)),
        success=best.fun < 0.05,
        model_ivs=model_ivs,
    )


# ══════════════════════════════════════════════════════════════════════════
#  SURFACE BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_surface(S: float, r: float, q: float,
                  expiries_yr: List[float],
                  moneyness_grid: List[float],
                  model: str = "sabr",
                  params_per_expiry: Optional[List[dict]] = None,
                  flat_sigma: float = 0.25,
                  beta: float = 0.5) -> Dict:
    """
    Build a full K×T implied vol surface.

    model: "flat" | "sabr" | "heston"
    params_per_expiry: list of param dicts (one per expiry), OR None for defaults
    moneyness_grid: list of K/S values, e.g. [0.7, 0.8, ..., 1.3]

    Returns dict with:
      T_grid, K_grid, iv_surface (2D np.array), moneyness_grid
    """
    T_grid = np.array(expiries_yr)
    K_grid = np.array(moneyness_grid) * S   # absolute strikes

    iv_surf = np.full((len(T_grid), len(K_grid)), float("nan"))

    for i, T in enumerate(T_grid):
        F = S * np.exp((r - q) * T)
        p = (params_per_expiry[i] if params_per_expiry else None) or {}

        for j, K in enumerate(K_grid):
            if model == "flat":
                iv_surf[i, j] = flat_sigma
            elif model == "sabr":
                alpha = p.get("alpha", 0.25)
                rho   = p.get("rho",   -0.30)
                nu    = p.get("nu",    0.40)
                iv_surf[i, j] = sabr_iv(F, K, T, alpha, beta, rho, nu)
            elif model == "heston":
                v0    = p.get("v0",    0.0625)
                kappa = p.get("kappa", 2.0)
                theta = p.get("theta", 0.0625)
                xi    = p.get("xi",    0.30)
                rho   = p.get("rho",   -0.70)
                iv_surf[i, j] = heston_iv(S, K, T, r, q, v0, kappa, theta, xi, rho)

    # Compute BS call prices for the surface
    price_surf = np.full_like(iv_surf, float("nan"))
    for i, T in enumerate(T_grid):
        for j, K in enumerate(K_grid):
            iv = iv_surf[i, j]
            if not np.isnan(iv):
                price_surf[i, j] = bs_price(S, K, T, r, q, iv, "call")

    return {
        "T_grid":       T_grid.tolist(),
        "K_grid":       K_grid.tolist(),
        "mono_grid":    moneyness_grid,
        "iv_surface":   np.where(np.isnan(iv_surf),  None, np.round(iv_surf  * 100, 4)).tolist(),
        "price_surface":np.where(np.isnan(price_surf),None, np.round(price_surf, 4)).tolist(),
        "S": S, "r": r, "q": q,
    }


# ══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC MARKET DATA  (for demo / when no live options chain)
# ══════════════════════════════════════════════════════════════════════════

def synthetic_market(S: float, r: float, q: float, T: float,
                     n_strikes: int = 13,
                     sigma_atm: float = 0.28,
                     skew: float = -0.15,
                     smile: float = 0.08) -> pd.DataFrame:
    """
    Generate synthetic option market prices with skew + smile.
    Useful for testing calibration when live data is unavailable.
    """
    F = S * np.exp((r - q) * T)
    moneyness = np.linspace(0.75, 1.25, n_strikes)
    strikes   = moneyness * S

    # Vol parameterisation: sigma(K) = ATM_vol + skew*(K/F-1) + smile*(K/F-1)^2
    m      = strikes / F
    iv_true = sigma_atm + skew*(m - 1) + smile*(m - 1)**2
    iv_true = np.maximum(iv_true, 0.05)

    rows = []
    rng = np.random.default_rng(42)
    for K, iv in zip(strikes, iv_true):
        mid  = bs_price(S, K, T, r, q, iv, "call")
        noise = rng.normal(0, mid * 0.005)   # 0.5% spread noise
        bid  = max(mid * 0.995 + noise - 0.05, 0.01)
        ask  = mid * 1.005 + noise + 0.05
        mid_clean = (bid + ask) / 2
        rec_iv = bs_iv(mid_clean, S, K, T, r, q, "call")
        rows.append({
            "strike":          round(K, 2),
            "moneyness":       round(K/F, 4),
            "bid":             round(bid, 3),
            "ask":             round(ask, 3),
            "mid":             round(mid_clean, 3),
            "market_iv":       round(rec_iv * 100, 3) if not np.isnan(rec_iv) else None,
            "true_iv":         round(iv * 100, 3),
            "volume":          int(rng.integers(50, 5000)),
            "open_interest":   int(rng.integers(500, 50000)),
        })
    return pd.DataFrame(rows)
