"""
fixed_income/engine.py
=======================
Fixed Income Dashboard Engine

Covers:
  1. Yield Curve Models — Nelson-Siegel, Svensson, bootstrap from par rates
  2. Bond Pricing       — Price, YTM, accrued interest, clean/dirty price
  3. Risk Metrics       — Modified duration, Macaulay duration, convexity, DV01, BPV
  4. Scenario Analysis  — Parallel shift, twist, butterfly shifts; P&L attribution
  5. Swap Pricing       — Par swap rate, DV01, fixed/floating cash flows
  6. Spread Analysis    — Z-spread, OAS, I-spread vs benchmark

Run from parent folder:
  python fixed_income/app.py  ->  http://127.0.0.1:5003
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.interpolate import CubicSpline, interp1d
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings, logging

warnings.filterwarnings("ignore")
log = logging.getLogger("fi_engine")

# ── Standard maturities used throughout ──────────────────────────────────────
STD_MATS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

# ── US Treasury benchmark yields (approximate, for demo when live data unavailable)
DEMO_YIELDS = {
    0.25: 5.27, 0.5: 5.30, 1.0: 5.10, 2.0: 4.75, 3.0: 4.60,
    5.0: 4.50, 7.0: 4.48, 10.0: 4.45, 20.0: 4.65, 30.0: 4.55,
}


# ══════════════════════════════════════════════════════════════════════════
#  YIELD CURVE MODELS
# ══════════════════════════════════════════════════════════════════════════

def nelson_siegel(t: np.ndarray, beta0: float, beta1: float,
                   beta2: float, tau: float) -> np.ndarray:
    """
    Nelson-Siegel (1987) yield curve parametrisation.
    beta0: long-run level  beta1: slope  beta2: curvature  tau: decay
    """
    t   = np.maximum(np.asarray(t, float), 1e-6)
    f1  = (1 - np.exp(-t / tau)) / (t / tau)
    f2  = f1 - np.exp(-t / tau)
    return beta0 + beta1 * f1 + beta2 * f2


def svensson(t: np.ndarray, b0: float, b1: float, b2: float, b3: float,
              tau1: float, tau2: float) -> np.ndarray:
    """
    Svensson (1994) extension of Nelson-Siegel with second hump.
    Allows more flexible mid-term curvature.
    """
    t   = np.maximum(np.asarray(t, float), 1e-6)
    f1  = (1 - np.exp(-t / tau1)) / (t / tau1)
    f2  = f1 - np.exp(-t / tau1)
    f3  = (1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2)
    return b0 + b1 * f1 + b2 * f2 + b3 * f3


def fit_nelson_siegel(maturities: np.ndarray, yields: np.ndarray) -> Dict:
    """Fit NS model to observed yields using least squares."""
    def objective(params):
        b0, b1, b2, tau = params
        if tau <= 0 or b0 <= 0:
            return 1e6
        fitted = nelson_siegel(maturities, b0, b1, b2, tau)
        return float(np.sum((fitted - yields) ** 2))

    best_res, best_val = None, 1e9
    for b0_0 in [0.04, 0.05, 0.06]:
        for tau_0 in [1.0, 2.0, 5.0]:
            try:
                from scipy.optimize import minimize as _min
                r = _min(objective, [b0_0, -0.01, 0.01, tau_0],
                         method="Nelder-Mead",
                         options={"maxiter": 10000, "xatol": 1e-9, "fatol": 1e-10})
                if r.fun < best_val:
                    best_val, best_res = r.fun, r
            except Exception:
                pass

    if best_res is None:
        return {"beta0": 0.045, "beta1": -0.01, "beta2": 0.01, "tau": 2.0,
                "rmse": 999.0, "success": False}

    b0, b1, b2, tau = best_res.x
    fitted = nelson_siegel(maturities, b0, b1, b2, tau)
    rmse   = float(np.sqrt(np.mean((fitted - yields) ** 2))) * 100  # in bps

    return {
        "beta0": round(float(b0), 6), "beta1": round(float(b1), 6),
        "beta2": round(float(b2), 6), "tau":   round(float(tau), 6),
        "rmse_bps": round(rmse, 4), "success": bool(best_val < 1e-4),
    }


def fit_svensson(maturities: np.ndarray, yields: np.ndarray) -> Dict:
    """Fit Svensson model to observed yields."""
    def objective(p):
        b0, b1, b2, b3, tau1, tau2 = p
        if tau1 <= 0 or tau2 <= 0 or b0 <= 0:
            return 1e6
        return float(np.sum((svensson(maturities, b0, b1, b2, b3, tau1, tau2) - yields)**2))

    best_res, best_val = None, 1e9
    for init in [[0.045, -0.01, 0.01, 0.005, 1.5, 5.0],
                 [0.05,  -0.02, 0.02, 0.01,  2.0, 7.0]]:
        try:
            r = minimize(objective, init, method="Nelder-Mead",
                         options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
            if r.fun < best_val:
                best_val, best_res = r.fun, r
        except Exception:
            pass

    if best_res is None:
        p = [0.045, -0.01, 0.01, 0.005, 1.5, 5.0]
    else:
        p = best_res.x.tolist()

    b0, b1, b2, b3, tau1, tau2 = p
    fitted = svensson(maturities, b0, b1, b2, b3, tau1, tau2)
    rmse   = float(np.sqrt(np.mean((fitted - yields)**2))) * 100

    return {
        "b0": round(b0, 6), "b1": round(b1, 6), "b2": round(b2, 6),
        "b3": round(b3, 6), "tau1": round(tau1, 6), "tau2": round(tau2, 6),
        "rmse_bps": round(rmse, 4),
    }


def bootstrap_zero_curve(par_maturities: np.ndarray,
                           par_rates: np.ndarray,
                           freq: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap zero (spot) rates from par yields using bootstrapping.
    Returns (maturities, zero_rates).
    """
    mats  = np.asarray(par_maturities, float)
    pars  = np.asarray(par_rates, float)
    zeros = np.zeros(len(mats))

    for i, (T, par) in enumerate(zip(mats, pars)):
        n_periods = int(round(T * freq))
        if n_periods <= 1:
            # Short end: approximate zero ≈ par
            zeros[i] = par
            continue

        coupon = par / freq
        # Cash flows at prior periods (use linear interpolation for zeros)
        pv_prior = 0.0
        for j in range(1, n_periods):
            t_j  = j / freq
            if t_j <= mats[0]:
                z_j = zeros[0]
            elif t_j >= mats[i-1] if i > 0 else 0:
                z_j = zeros[i-1] if i > 0 else pars[0]
            else:
                # Linear interpolation
                z_j = float(np.interp(t_j, mats[:i], zeros[:i]))
            pv_prior += coupon * np.exp(-z_j * t_j)

        # Solve for zero at maturity T
        try:
            z_T = -np.log((1 - pv_prior) / (1 + coupon)) / T
            zeros[i] = max(z_T, 1e-6)
        except Exception:
            zeros[i] = par

    return mats, zeros


def forward_rates(maturities: np.ndarray, zero_rates: np.ndarray,
                   forward_tenor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous and period forward rates from zero curve.
    forward_tenor: tenor of forward rate (default 0.5yr = 6M fwd)
    """
    cs    = CubicSpline(maturities, zero_rates)
    t_fwd = maturities[maturities > forward_tenor]
    fwds  = []
    for t in t_fwd:
        t0  = t - forward_tenor
        if t0 < maturities[0]:
            t0 = maturities[0]
        z0  = float(cs(t0))
        z1  = float(cs(t))
        fwd = (z1 * t - z0 * t0) / forward_tenor
        fwds.append(fwd)

    return t_fwd, np.array(fwds)


def discount_factors(maturities: np.ndarray, zero_rates: np.ndarray) -> np.ndarray:
    """Compute discount factors P(0,T) = exp(-r(T)*T)."""
    return np.exp(-np.asarray(zero_rates) * np.asarray(maturities))


# ══════════════════════════════════════════════════════════════════════════
#  BOND PRICING AND RISK
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Bond:
    face:        float = 1000.0
    coupon_rate: float = 0.05       # annual rate
    maturity:    float = 10.0       # years to maturity
    freq:        int   = 2          # coupon payments per year (2=semi-annual)
    currency:    str   = "USD"
    name:        str   = "Bond"


def bond_price(bond: Bond, ytm: float) -> float:
    """Full (dirty) price of a bond given YTM."""
    c  = bond.face * bond.coupon_rate / bond.freq
    r  = ytm / bond.freq
    n  = int(round(bond.maturity * bond.freq))
    t  = np.arange(1, n + 1)
    cf = np.full(n, c)
    cf[-1] += bond.face
    return float(np.sum(cf / (1 + r) ** t))


def bond_ytm(bond: Bond, price: float) -> float:
    """Yield to maturity given price (Brent's method)."""
    def f(ytm):
        return bond_price(bond, ytm) - price

    try:
        return float(brentq(f, 1e-6, 0.99, xtol=1e-10, maxiter=500))
    except ValueError:
        return float("nan")


def macaulay_duration(bond: Bond, ytm: float) -> float:
    """Macaulay duration in years."""
    c  = bond.face * bond.coupon_rate / bond.freq
    r  = ytm / bond.freq
    n  = int(round(bond.maturity * bond.freq))
    t  = np.arange(1, n + 1)
    cf = np.full(n, c)
    cf[-1] += bond.face
    pv   = cf / (1 + r) ** t
    p    = float(pv.sum())
    if p < 1e-10:
        return 0.0
    return float(np.sum(pv * t / bond.freq)) / p


def modified_duration(bond: Bond, ytm: float) -> float:
    """Modified duration (price sensitivity per unit yield change)."""
    mac = macaulay_duration(bond, ytm)
    return mac / (1 + ytm / bond.freq)


def convexity(bond: Bond, ytm: float) -> float:
    """Convexity (second-order price sensitivity)."""
    c  = bond.face * bond.coupon_rate / bond.freq
    r  = ytm / bond.freq
    n  = int(round(bond.maturity * bond.freq))
    t  = np.arange(1, n + 1)
    cf = np.full(n, c)
    cf[-1] += bond.face
    p    = bond_price(bond, ytm)
    if p < 1e-10:
        return 0.0
    conv = float(np.sum(cf * t * (t + 1) / (1 + r) ** (t + 2)))
    return conv / (p * bond.freq ** 2)


def dv01(bond: Bond, ytm: float) -> float:
    """DV01: dollar value of 1bp move in yield (per $1 face)."""
    p_up   = bond_price(bond, ytm + 0.0001)
    p_down = bond_price(bond, ytm - 0.0001)
    return (p_down - p_up) / 2.0


def price_change_approx(bond: Bond, ytm: float, delta_y: float) -> Dict:
    """
    Approximate price change using duration + convexity Taylor expansion.
    dP/P ≈ -D_mod * dy + 0.5 * C * dy^2
    """
    p0   = bond_price(bond, ytm)
    d    = modified_duration(bond, ytm)
    c    = convexity(bond, ytm)
    dp_pct = -d * delta_y + 0.5 * c * delta_y ** 2
    dp     = p0 * dp_pct
    p1_approx = p0 + dp
    p1_exact  = bond_price(bond, ytm + delta_y)

    return {
        "price_initial":   round(p0, 6),
        "price_approx":    round(p1_approx, 6),
        "price_exact":     round(p1_exact, 6),
        "approx_error":    round(abs(p1_approx - p1_exact), 6),
        "pct_change":      round(dp_pct * 100, 6),
        "duration_effect": round(-d * delta_y * p0, 6),
        "convexity_effect":round(0.5 * c * delta_y ** 2 * p0, 6),
    }


def full_analytics(bond: Bond, ytm: float) -> Dict:
    """Compute all bond analytics in one call."""
    p  = bond_price(bond, ytm)
    md = modified_duration(bond, ytm)
    mac= macaulay_duration(bond, ytm)
    cv = convexity(bond, ytm)
    d1 = dv01(bond, ytm)
    return {
        "price":              round(p, 6),
        "ytm":                round(ytm, 6),
        "coupon_rate":        round(bond.coupon_rate, 6),
        "modified_duration":  round(md, 6),
        "macaulay_duration":  round(mac, 6),
        "convexity":          round(cv, 6),
        "dv01":               round(d1, 6),
        "bpv":                round(d1, 6),           # same as DV01 per $1 face
        "price_pct_face":     round(p / bond.face * 100, 4),
        "premium_discount":   round(p - bond.face, 4),
    }


# ══════════════════════════════════════════════════════════════════════════
#  SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def parallel_shift(maturities: np.ndarray, yields: np.ndarray,
                    shifts_bps: List[float]) -> Dict:
    """Price change for parallel shifts of -300 to +300bps."""
    results = {}
    for shift_bps in shifts_bps:
        shifted = yields + shift_bps / 10000
        results[shift_bps] = np.round(shifted * 100, 4).tolist()
    return results


def scenario_pnl(bond: Bond, ytm: float,
                  scenarios: Dict[str, float]) -> pd.DataFrame:
    """
    P&L attribution for each yield scenario.
    scenarios: {"Parallel +100bps": 0.01, "Parallel -100bps": -0.01, ...}
    """
    p0 = bond_price(bond, ytm)
    rows = []
    for name, dy in scenarios.items():
        p1 = bond_price(bond, ytm + dy)
        dp = p1 - p0
        rows.append({
            "Scenario":      name,
            "Yield_Change":  round(dy * 10000, 1),
            "New_Price":     round(p1, 4),
            "P&L_$":         round(dp, 4),
            "P&L_%":         round(dp / p0 * 100, 4),
        })
    return pd.DataFrame(rows)


def curve_scenarios(maturities: np.ndarray, yields: np.ndarray) -> Dict:
    """Standard curve scenarios used in fixed income risk management."""
    y = np.asarray(yields)
    short_end = np.array([1 if t <= 2 else 0 for t in maturities], float)
    long_end  = np.array([1 if t >= 10 else 0 for t in maturities], float)
    mid       = np.array([1 if 2 < t < 10 else 0 for t in maturities], float)
    return {
        "base":              (maturities.tolist(), (y * 100).tolist()),
        "parallel_up_100":   (maturities.tolist(), ((y + 0.01) * 100).tolist()),
        "parallel_down_100": (maturities.tolist(), ((y - 0.01) * 100).tolist()),
        "steepener":         (maturities.tolist(), ((y + 0.01*long_end - 0.01*short_end)*100).tolist()),
        "flattener":         (maturities.tolist(), ((y - 0.01*long_end + 0.01*short_end)*100).tolist()),
        "butterfly":         (maturities.tolist(), ((y + 0.01*mid - 0.005*(short_end+long_end))*100).tolist()),
    }


# ══════════════════════════════════════════════════════════════════════════
#  INTEREST RATE SWAPS
# ══════════════════════════════════════════════════════════════════════════

def par_swap_rate(zero_maturities: np.ndarray, zero_rates: np.ndarray,
                   swap_maturity: float, freq: int = 4) -> float:
    """
    Compute par (fair) swap rate for a fixed-for-floating swap.
    freq: payment frequency (4 = quarterly, 2 = semi-annual)
    """
    n       = int(round(swap_maturity * freq))
    times   = np.array([(i + 1) / freq for i in range(n)])
    # Interpolate zero rates at payment times
    cs      = CubicSpline(zero_maturities, zero_rates, extrapolate=True)
    z_times = np.clip(cs(times), 0.001, 0.99)
    dfs     = np.exp(-z_times * times)
    # Par swap rate: numerator = 1 - last DF; denominator = sum of DFs / freq
    num     = 1.0 - float(dfs[-1])
    denom   = float(dfs.sum()) / freq
    return num / denom if denom > 1e-10 else 0.0


def swap_cashflows(notional: float, fixed_rate: float, swap_maturity: float,
                    zero_mats: np.ndarray, zero_rates: np.ndarray,
                    freq: int = 4) -> pd.DataFrame:
    """
    Compute fixed and floating leg cash flows and their present values.
    The floating leg is approximated using forward rates from the curve.
    """
    n     = int(round(swap_maturity * freq))
    times = np.array([(i + 1) / freq for i in range(n)])
    cs    = CubicSpline(zero_mats, zero_rates, extrapolate=True)
    z     = np.clip(cs(times), 0.001, 0.99)
    dfs   = np.exp(-z * times)

    # Fixed leg
    fixed_cf = np.full(n, notional * fixed_rate / freq)

    # Floating leg: use forward rates
    fwd_rates = np.zeros(n)
    for i, t in enumerate(times):
        t0 = times[i-1] if i > 0 else 0.0
        z0 = float(np.clip(cs(t0), 0.001, 0.99)) if t0 > 0 else z[0]
        z1 = z[i]
        dt = t - t0
        fwd_rates[i] = (z1 * t - z0 * t0) / dt if dt > 1e-8 else z1
    float_cf = notional * fwd_rates / freq

    rows = []
    for i, t in enumerate(times):
        rows.append({
            "time":       round(float(t), 4),
            "discount":   round(float(dfs[i]), 6),
            "fixed_cf":   round(float(fixed_cf[i]), 4),
            "float_cf":   round(float(float_cf[i]), 4),
            "fixed_pv":   round(float(fixed_cf[i] * dfs[i]), 4),
            "float_pv":   round(float(float_cf[i] * dfs[i]), 4),
            "net_cf":     round(float((fixed_cf[i] - float_cf[i])), 4),
            "net_pv":     round(float((fixed_cf[i] - float_cf[i]) * dfs[i]), 4),
        })

    df = pd.DataFrame(rows)
    df["cumulative_pv"] = df["net_pv"].cumsum().round(4)
    return df


def swap_dv01(notional: float, fixed_rate: float, swap_maturity: float,
               zero_mats: np.ndarray, zero_rates: np.ndarray,
               freq: int = 4) -> float:
    """DV01 of a swap: price sensitivity to 1bp parallel shift in zero curve."""
    sr_base = par_swap_rate(zero_mats, zero_rates, swap_maturity, freq)
    sr_up   = par_swap_rate(zero_mats, zero_rates + 0.0001, swap_maturity, freq)
    sr_dn   = par_swap_rate(zero_mats, zero_rates - 0.0001, swap_maturity, freq)
    # Approximate DV01 via finite difference on the MTM
    n     = int(round(swap_maturity * freq))
    times = np.array([(i + 1) / freq for i in range(n)])
    cs    = CubicSpline(zero_mats, zero_rates, extrapolate=True)
    dfs   = np.exp(-np.clip(cs(times), 0.001, 0.99) * times)
    mtm_base = notional * (fixed_rate - sr_base) / freq * dfs.sum()
    cs_up = CubicSpline(zero_mats, zero_rates + 0.0001, extrapolate=True)
    dfs_up = np.exp(-np.clip(cs_up(times), 0.001, 0.99) * times)
    sr_up2 = par_swap_rate(zero_mats, zero_rates + 0.0001, swap_maturity, freq)
    mtm_up = notional * (fixed_rate - sr_up2) / freq * dfs_up.sum()
    return float(mtm_up - mtm_base)


# ══════════════════════════════════════════════════════════════════════════
#  SPREAD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def z_spread(bond: Bond, price: float,
              zero_mats: np.ndarray, zero_rates: np.ndarray) -> float:
    """
    Z-spread: constant spread added to zero curve such that PV = market price.
    Standard measure for corporate bonds vs risk-free curve.
    """
    c   = bond.face * bond.coupon_rate / bond.freq
    n   = int(round(bond.maturity * bond.freq))
    ts  = np.array([(i + 1) / bond.freq for i in range(n)])
    cfs = np.full(n, c)
    cfs[-1] += bond.face
    cs  = CubicSpline(zero_mats, zero_rates, extrapolate=True)

    def pv(spread):
        zs  = np.clip(cs(ts), 0.001, 0.99) + spread
        dfs = np.exp(-zs * ts)
        return float(np.sum(cfs * dfs)) - price

    try:
        return float(brentq(pv, -0.05, 0.50, xtol=1e-10, maxiter=500))
    except ValueError:
        return float("nan")


def i_spread(ytm: float, zero_mats: np.ndarray, zero_rates: np.ndarray,
              maturity: float) -> float:
    """
    I-spread (interpolated spread): YTM minus swap/benchmark rate at same maturity.
    """
    cs         = CubicSpline(zero_mats, zero_rates, extrapolate=True)
    bench_rate = float(np.clip(cs(maturity), 0.001, 0.99))
    return ytm - bench_rate


# ══════════════════════════════════════════════════════════════════════════
#  BOND PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════

def portfolio_duration(bonds: List[Bond], ytms: List[float],
                        weights: List[float]) -> Dict:
    """
    Portfolio-level duration, convexity, and DV01 as weighted averages.
    weights: market value weights (sum to 1).
    """
    w   = np.array(weights)
    w  /= w.sum()
    mds = np.array([modified_duration(b, y) for b, y in zip(bonds, ytms)])
    cvs = np.array([convexity(b, y) for b, y in zip(bonds, ytms)])
    d1s = np.array([dv01(b, y) * b.face for b, y in zip(bonds, ytms)])
    return {
        "portfolio_duration":  round(float(w @ mds), 6),
        "portfolio_convexity": round(float(w @ cvs), 6),
        "portfolio_dv01":      round(float(w @ d1s), 4),
        "asset_durations":     [round(float(d), 4) for d in mds],
    }


# ══════════════════════════════════════════════════════════════════════════
#  UTILITY: Build full curve output dict
# ══════════════════════════════════════════════════════════════════════════

def full_curve_output(maturities: np.ndarray, par_yields: np.ndarray,
                       model: str = "nelson_siegel") -> Dict:
    """
    One-call function: from par yields, produce:
      - Fitted model curve (NS or Svensson)
      - Bootstrapped zero curve
      - Forward rates
      - Discount factors
      - Curve scenarios
    """
    mats = np.asarray(maturities, float)
    pars = np.asarray(par_yields, float) / 100.0  # convert % to decimal

    # Fit model
    if model == "svensson":
        fit  = fit_svensson(mats, pars)
        fine = np.linspace(0.25, 30, 200)
        model_yields = svensson(fine, fit["b0"], fit["b1"], fit["b2"],
                                 fit["b3"], fit["tau1"], fit["tau2"])
    else:  # nelson_siegel
        fit  = fit_nelson_siegel(mats, pars)
        fine = np.linspace(0.25, 30, 200)
        model_yields = nelson_siegel(fine, fit["beta0"], fit["beta1"],
                                      fit["beta2"], fit["tau"])

    # Bootstrap zero curve at standard maturities
    z_mats, z_rates = bootstrap_zero_curve(mats, pars)

    # Forward rates
    if len(z_mats) >= 3:
        fwd_mats, fwd_rates = forward_rates(z_mats, z_rates)
    else:
        fwd_mats, fwd_rates = z_mats, z_rates

    # Fine zero curve (interpolated)
    if len(z_mats) >= 2:
        cs_zero = CubicSpline(z_mats, z_rates, extrapolate=True)
        zero_fine = np.clip(cs_zero(fine), 0.001, 0.99)
    else:
        zero_fine = model_yields

    # Discount factors
    dfs = np.exp(-zero_fine * fine)

    # Curve scenarios
    scenarios = curve_scenarios(mats, pars)

    return {
        "maturities":    mats.tolist(),
        "par_yields":    (pars * 100).tolist(),
        "model":         model,
        "model_params":  fit,
        "fine_mats":     np.round(fine, 4).tolist(),
        "model_yields":  np.round(model_yields * 100, 4).tolist(),
        "zero_mats":     z_mats.tolist(),
        "zero_rates":    np.round(z_rates * 100, 4).tolist(),
        "zero_fine":     np.round(zero_fine * 100, 4).tolist(),
        "fwd_mats":      fwd_mats.tolist(),
        "fwd_rates":     np.round(fwd_rates * 100, 4).tolist(),
        "discount_factors": np.round(dfs, 6).tolist(),
        "scenarios":     scenarios,
    }
