"""Shared code for the notebooks (modified Chiarella agent-based model).

Simulators, summary statistics, GMM, state reconstruction, least-squares calibration,
prediction and strategy evaluation.

Usage from a notebook in `notebooks/`:

    from abm import chiarella_fast, stats_2, evaluate, ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import direct, least_squares, minimize
from scipy.stats import binomtest, kurtosis, skew

# =====================================================================================================
# Reference parameters
# =====================================================================================================

# Parameters from the group project slides, order: beta, kappa, gamma, alpha, lam, sigma
theta0 = {"beta": 0.05, "kappa": 0.05, "gamma": 10.0, "alpha": 0.1, "lam": 0.02, "sigma": 0.003}

# Regime where fundamentalists are visible (g = 3·kappa·sigma²/(2·lam) ≈ 0.07): the "true θ" of Parts 2 to 6
theta_active = {**theta0, "kappa": 50, "lam": 0.01}
THETA_TRUE = theta_active

ESTIMATED_PARAMS = ["beta", "kappa", "sigma"]   # estimated by the GMM (gamma, alpha, lam fixed)

# Reserved seeds, so that the same noise is never reused in two places
MODEL_SEEDS = range(1000, 1010)    # simulations averaged in the GMM objective (E-GMM)
V_SEEDS = range(3000, 3200)        # simulations used to estimate the moment covariance V (Hansen)
DATA_SEEDS = range(10000, 10050)   # the 50 "fake markets" of the Monte Carlo studies


# =====================================================================================================
# Simulators
# =====================================================================================================

def chiarella(beta, kappa, gamma, alpha, lam, sigma, T, rng, burn_in=500):
    """Simulate the modified Chiarella model (plain Python: slow but readable).

    Returns (p, m, v): log-price, trend, fundamental value, of length T
    (the first burn_in steps are discarded). If the price diverges, the rest is NaN.
    """
    n = T + burn_in
    p, m, v = np.zeros(n), np.zeros(n), np.zeros(n)
    eps = rng.normal(0.0, sigma, n)

    for t in range(1, n):
        # 1) update the state from p_t ...
        m[t] = (1 - alpha) * m[t - 1] + alpha * (p[t] - p[t - 1])
        v[t] = (1 - lam) * v[t - 1] + lam * p[t]
        if t == n - 1:  # last step: the state is up to date, no p_{t+1} to compute
            break
        # 2) ... then compute p_{t+1}
        fundamentalists = kappa * ((v[t] - p[t]) ** 3 - (v[t - 1] - p[t - 1]) ** 3)
        chartists = beta * (np.tanh(gamma * m[t]) - np.tanh(gamma * m[t - 1]))
        p[t + 1] = p[t] + fundamentalists + chartists + eps[t + 1]

        if not np.isfinite(p[t + 1]) or abs(p[t + 1]) > 1e3:  # explosion
            p[t + 1:] = m[t + 1:] = v[t + 1:] = np.nan
            break

    return p[burn_in:], m[burn_in:], v[burn_in:]


@njit
def _chiarella_kernel(beta, kappa, gamma, alpha, lam, eps):
    """Same loop as chiarella, compiled to machine code by numba. The noise eps is drawn outside."""
    n = len(eps)
    p, m, v = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(1, n):
        m[t] = (1 - alpha) * m[t - 1] + alpha * (p[t] - p[t - 1])
        v[t] = (1 - lam) * v[t - 1] + lam * p[t]
        if t == n - 1:
            break
        fundamentalists = kappa * ((v[t] - p[t]) ** 3 - (v[t - 1] - p[t - 1]) ** 3)
        chartists = beta * (np.tanh(gamma * m[t]) - np.tanh(gamma * m[t - 1]))
        p[t + 1] = p[t] + fundamentalists + chartists + eps[t + 1]
        if not np.isfinite(p[t + 1]) or abs(p[t + 1]) > 1e3:  # explosion
            p[t + 1:] = np.nan
            m[t + 1:] = np.nan
            v[t + 1:] = np.nan
            break
    return p, m, v


def chiarella_fast(beta, kappa, gamma, alpha, lam, sigma, T, rng, burn_in=500):
    """Same interface and same output as chiarella (bit for bit), about 90 times faster."""
    eps = rng.normal(0.0, sigma, T + burn_in)  # same draw as in chiarella
    p, m, v = _chiarella_kernel(beta, kappa, gamma, alpha, lam, eps)
    return p[burn_in:], m[burn_in:], v[burn_in:]


# =====================================================================================================
# Summary statistics (moments)
# =====================================================================================================

def acf(x, lag):
    """Sample autocorrelation at `lag` (denominator = total variance, as in statsmodels)."""
    xc = x - x.mean()
    return np.dot(xc[:-lag], xc[lag:]) / np.dot(xc, xc)


def hill(r, fraction=0.05):
    """Hill estimator of the tail exponent, on the `fraction` largest |r|."""
    x = np.sort(np.abs(r))[::-1]  # |r| sorted from largest to smallest
    k = int(fraction * len(x))
    return k / np.sum(np.log(x[:k] / x[k]))


STAT_NAMES = ["std", "tail", "kurt", "skew", "acf1", "acf2", "acf3", "acf5",
              "acfabs1", "acfabs5", "acfabs10", "acfabs20"]


def stats(r):
    """The 12 candidate statistics of Part 2.1, in the order of STAT_NAMES. NaN if unusable."""
    if np.isnan(r).any() or np.std(r) == 0:  # explosion or constant series
        return np.full(len(STAT_NAMES), np.nan)
    a = np.abs(r)
    return np.array([
        np.std(r), hill(r), kurtosis(r), skew(r),
        acf(r, 1), acf(r, 2), acf(r, 3), acf(r, 5),
        acf(a, 1), acf(a, 5), acf(a, 10), acf(a, 20),
    ])


STAT_NAMES_2 = ["std", "acf1", "acf2", "acfabs1"]


def stats_2(r):
    """The 4 statistics kept after the sensitivity analysis. NaN if the series is unusable."""
    if np.isnan(r).any() or np.std(r) == 0:  # explosion or constant series
        return np.full(len(STAT_NAMES_2), np.nan)
    return np.array([np.std(r), acf(r, 1), acf(r, 2), acf(np.abs(r), 1)])


def get_moments(theta, seed, T=5000):
    """The 12 statistics of one simulation (fixed seed: same seed, same noise)."""
    p, _, _ = chiarella_fast(**theta, T=T, rng=np.random.default_rng(seed))
    return stats(np.diff(p))


def get_moments_2(theta, seed, T=5000):
    """The 4 retained statistics of one simulation (fixed seed)."""
    p, _, _ = chiarella_fast(**theta, T=T, rng=np.random.default_rng(seed))
    return stats_2(np.diff(p))


def compute_jacobian(theta, seeds, step=0.05, moments=get_moments):
    """Normalized Jacobian: J[k, j] = effect of a ±5 % change of θ_j on statistic k,
    in noise standard deviations of that statistic (common seeds on both sides)."""
    names = list(theta.keys())
    c_ref = np.array([moments(theta, s) for s in seeds])
    noise = c_ref.std(axis=0)

    J = np.zeros((c_ref.shape[1], len(names)))
    for j, param in enumerate(names):
        delta = step * theta[param] if theta[param] != 0 else step
        c_up = np.array([moments({**theta, param: theta[param] + delta}, s) for s in seeds]).mean(axis=0)
        c_down = np.array([moments({**theta, param: theta[param] - delta}, s) for s in seeds]).mean(axis=0)
        J[:, j] = (c_up - c_down) / 2 / noise  # = derivative × delta / noise
    return J


# =====================================================================================================
# GMM (Part 2)
# =====================================================================================================

def full_theta(x, theta_true=THETA_TRUE, params=ESTIMATED_PARAMS):
    """Optimizer array (e.g. [beta, kappa, sigma]) -> full parameter dictionary for the simulator."""
    return {**theta_true, **dict(zip(params, x))}


def model_moments(x, theta_true=THETA_TRUE, seeds=MODEL_SEEDS, T=5000):
    """Model moments averaged over N fixed seeds (E-GMM)."""
    return np.mean([get_moments_2(full_theta(x, theta_true), s, T) for s in seeds], axis=0)


def gmm_objective(x, c_obs, W, theta_true=THETA_TRUE, seeds=MODEL_SEEDS):
    """D_W(x) = gᵀ W g with g = c_obs − model moments. Penalty 1e6 if the model explodes."""
    c_abm = model_moments(x, theta_true, seeds)
    if np.isnan(c_abm).any():  # explosion
        return 1e6
    g = c_obs - c_abm
    return g @ W @ g


def hansen_matrix(x, theta_true=THETA_TRUE, seeds=V_SEEDS, n_model=len(MODEL_SEEDS)):
    """W* = V⁻¹, V = covariance of the moments at x (over `seeds`), corrected by the factor (1 + 1/N)."""
    M = np.array([get_moments_2(full_theta(x, theta_true), s) for s in seeds])
    return np.linalg.pinv(np.cov(M, rowvar=False) * (1 + 1 / n_model))


def gmm_bounds(c_obs):
    """Bounds chosen without knowing the true value: beta, kappa, sigma (around the observed std)."""
    std_obs = c_obs[0]
    return [(0, 0.5), (0, 200), (0.5 * std_obs, 1.5 * std_obs)]


def estimate(c_obs, W, budget_direct=300, budget_nm=100):
    """Full GMM calibration of one dataset: DIRECT (global) then Nelder-Mead (local refinement).

    Returns (θ̂, D_W at the minimum).
    """
    bounds = gmm_bounds(c_obs)
    res = direct(gmm_objective, bounds, args=(c_obs, W), maxfun=budget_direct, locally_biased=False)
    res = minimize(gmm_objective, x0=res.x, args=(c_obs, W), method="Nelder-Mead",
                   bounds=bounds, options={"maxfev": budget_nm, "xatol": 1e-7})
    return res.x, res.fun


# =====================================================================================================
# State reconstruction and trajectory calibration (Parts 4 and 6)
# =====================================================================================================

@njit
def _reconstruct(p, alpha, lam, m0, v0):
    n = len(p)
    m, v = np.zeros(n), np.zeros(n)
    m[0], v[0] = m0, v0
    for t in range(1, n):
        m[t] = (1 - alpha) * m[t - 1] + alpha * (p[t] - p[t - 1])
        v[t] = (1 - lam) * v[t - 1] + lam * p[t]
    return m, v


def reconstruct_states(p, alpha, lam, m0=0.0, v0=None):
    """Rebuild (m_t, v_t) from observed prices with the model equations.

    Uses the past only. The initial state is unknown on real data: m0 = 0 and v0 = p[0] by default;
    the resulting error fades like (1 − λ)^t.
    """
    return _reconstruct(np.asarray(p, dtype=np.float64), alpha, lam, m0, p[0] if v0 is None else v0)


def log_returns(p):
    """r[t] = p[t] − p[t−1]; r[0] does not exist (NaN)."""
    return np.concatenate([[np.nan], np.diff(p)])


def regressors(p, alpha, lam, gamma):
    """Regressors F_t (fundamentalists) and C_t (chartists), aligned on t.

    The model reads r(t+1) = κ F[t] + β C[t] + ε. F[0] and C[0] do not exist (NaN).
    """
    m, v = reconstruct_states(p, alpha, lam)
    x = v - p
    F = np.full(len(p), np.nan)
    C = np.full(len(p), np.nan)
    F[1:] = x[1:] ** 3 - x[:-1] ** 3
    C[1:] = np.tanh(gamma * m[1:]) - np.tanh(gamma * m[:-1])
    return F, C


def calibrate(F, C, r, t_start, t_end):
    """Least squares of r(t+1) on F[t], C[t] for t_start ≤ t < t_end − 1. Returns (beta, kappa)."""
    t = np.arange(t_start, t_end - 1)
    (kappa, beta), *_ = np.linalg.lstsq(np.column_stack([F[t], C[t]]), r[t + 1], rcond=None)
    return beta, kappa


def calibrate_winsor(F, C, r, t_start, t_end, k=3.0):
    """Least squares after clipping the window's returns at ±k standard deviations (of that window)."""
    t = np.arange(t_start, t_end - 1)
    y = r[t + 1]
    y = np.clip(y, -k * y.std(), k * y.std())
    (kappa, beta), *_ = np.linalg.lstsq(np.column_stack([F[t], C[t]]), y, rcond=None)
    return beta, kappa


def calibrate_huber(F, C, r, t_start, t_end):
    """Huber loss: quadratic for small residuals, linear beyond 1.345 standard deviations."""
    t = np.arange(t_start, t_end - 1)
    X, y = np.column_stack([F[t], C[t]]), r[t + 1]
    start, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = least_squares(lambda th: X @ th - y, start, loss="huber", f_scale=1.345 * y.std(), x_scale="jac")
    kappa, beta = res.x
    return beta, kappa


def estimate_ols(p, gamma, alpha, lam, skip=500):
    """γ, α, λ fixed: least squares over the whole series (Part 4.2a).

    Returns beta, kappa, sigma, their theoretical standard errors σ²(XᵀX)⁻¹ and the correlation of F and C.
    """
    F, C = regressors(p, alpha, lam, gamma)
    r = log_returns(p)
    t = np.arange(skip + 1, len(p) - 1)   # skip the first steps (unknown initial state)
    X, y = np.column_stack([F[t], C[t]]), r[t + 1]
    (kappa, beta), *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ np.array([kappa, beta])
    sigma = e.std()
    cov = sigma ** 2 * np.linalg.inv(X.T @ X)
    return {"beta": beta, "kappa": kappa, "sigma": sigma,
            "se_beta": np.sqrt(cov[1, 1]), "se_kappa": np.sqrt(cov[0, 0]),
            "corr_F_C": np.corrcoef(F[t], C[t])[0, 1]}


def predict(F, C, beta, kappa):
    """r_hat[t+1] = κ F[t] + β C[t]: the forecast of the return of t+1, known at the end of day t."""
    r_hat = np.full(len(F), np.nan)
    r_hat[1:] = kappa * F[:-1] + beta * C[:-1]
    return r_hat


# =====================================================================================================
# Prediction and strategy evaluation (Part 6)
# =====================================================================================================

def evaluate(position, r, cost=0.0001, periods_per_year=252):
    """Performance of a strategy.

    position[t] : position held during period t (decided at the end of t−1).
    r[t]        : return realized during period t.
    cost        : proportional cost at each position change (0.0001 = 0.01 %).
    """
    position, r = np.asarray(position, float), np.asarray(r, float)
    ok = ~np.isnan(position) & ~np.isnan(r)
    position, r = position[ok], r[ok]

    gains = position * r - cost * np.abs(np.diff(position, prepend=0.0))
    active = (position != 0) & (r != 0)
    hits = np.sign(position[active]) == np.sign(r[active])
    n_active = int(active.sum())
    up, down = active & (position > 0), active & (position < 0)
    return {
        "hit ratio": hits.mean() if n_active else np.nan,
        "p-value (vs 0.5)": binomtest(int(hits.sum()), n_active, 0.5).pvalue if n_active else np.nan,
        "precision up": np.mean(r[up] > 0) if up.any() else np.nan,
        "precision down": np.mean(r[down] < 0) if down.any() else np.nan,
        "gross PnL": np.sum(position * r),
        "net PnL": gains.sum(),
        "Sharpe (annualized)": gains.mean() / gains.std() * np.sqrt(periods_per_year) if gains.std() > 0 else np.nan,
        "days": len(r),
    }


def benchmark_strategies(r):
    """Positions of the naive strategies, decided with yesterday's return only."""
    yesterday = np.concatenate([[np.nan], r[:-1]])
    return {
        "buy & hold": np.ones(len(r)),
        "naive momentum: sign(r_t)": np.sign(yesterday),
        "naive mean reversion: -sign(r_t)": -np.sign(yesterday),
    }


def sheppard(rho):
    """Theoretical hit ratio ½ + arcsin(ρ)/π of a forecast with correlation ρ to the return (normal law)."""
    return 0.5 + np.arcsin(rho) / np.pi


def backtest(p, alpha, lam, gamma, calibration=calibrate, window=500, step=20, start=1000,
             threshold_pct=80, cap=2.0):
    """Rolling window: calibrate on `window` steps, predict the next `step` steps, shift by `step`.

    The state is rebuilt once (it only uses the past). Position sizes are tuned on the calibration window
    only. Returns (positions for the three sizings, estimated parameters per window, indexed by the first
    predicted step).
    """
    r = log_returns(p)
    F, C = regressors(p, alpha, lam, gamma)
    sizes = {name: np.full(len(p), np.nan) for name in ["sign", "proportional", "threshold (80th pct)"]}
    params = []
    for t0 in range(start, len(p), step):
        beta, kappa = calibration(F, C, r, t0 - window, t0)  # data up to t0 − 1 only
        r_hat = predict(F, C, beta, kappa)
        past = r_hat[t0 - window + 1:t0]
        scale, threshold = past.std(), np.percentile(np.abs(past), threshold_pct)
        future = slice(t0, min(t0 + step, len(p)))
        sizes["sign"][future] = np.sign(r_hat[future])
        sizes["proportional"][future] = np.clip(r_hat[future] / scale, -cap, cap)
        sizes["threshold (80th pct)"][future] = np.where(np.abs(r_hat[future]) > threshold, np.sign(r_hat[future]), 0.0)
        params.append({"t0": t0, "beta": beta, "kappa": kappa})
    return sizes, pd.DataFrame(params).set_index("t0")


def max_drawdown(gains):
    """Largest fall of the cumulative PnL from a previous peak (in log)."""
    cumul = np.cumsum(np.nan_to_num(gains))
    return np.max(np.maximum.accumulate(cumul) - cumul)


def clean_fx(rates, k=5.0, vol_window=60):
    """Replace one-day errors in an FX series (pandas) by the previous rate: a jump larger than k times the
    volatility of the previous `vol_window` days, followed the next day by a move of opposite sign of at least
    half its size. Returns (cleaned series, mask of corrected days)."""
    r = np.log(rates).diff()
    vol = r.rolling(vol_window, min_periods=20).std().shift(1)
    next_day = r.shift(-1)
    error = (r.abs() > k * vol) & (r * next_day < 0) & (next_day.abs() > 0.5 * r.abs())
    return rates.mask(error).ffill(), error
