"""Exact results for the k-birth Yule process.

Let F(s,t) = E[s^{Z_t}], Z_0 = 1. By the branching property F solves the
Kolmogorov backward equation

    dF/dt = lam * (F^{k+1} - F),      F(s,0) = s.

Substituting u = F^{-k} gives the linear ODE u' = k*lam*(u - 1), hence

    F(s,t) = s * p^{1/k} / (1 - (1 - p) s^k)^{1/k},   p = exp(-k*lam*t).

Reading this off: Z_t = 1 + k*N_t with N_t ~ NegBin(r = 1/k, p),
    P(N_t = n) = Gamma(n + 1/k) / (n! Gamma(1/k)) * p^{1/k} * (1-p)^n.

k = 1 recovers the classical geometric law of the Yule process.
Consequences:
  * E[Z_t] = exp(k*lam*t); M_t = exp(-k*lam*t) Z_t is a martingale;
  * exp(-k*lam*t) N_t  -> Gamma(shape 1/k, scale 1)  a.s. and in L^1,
    so exp(-k*lam*t) Z_t -> W ~ Gamma(shape 1/k, scale k).
    For k = 1: W ~ Exponential(1). For k = 2: W ~ chi-square with 1 d.f.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mean_size(lam: float, k: int, t: float) -> float:
    return float(np.exp(k * lam * t))


def var_size(lam: float, k: int, t: float) -> float:
    p = np.exp(-k * lam * t)
    r = 1.0 / k
    var_n = r * (1 - p) / p**2  # NegBin variance
    return float(k**2 * var_n)


def pmf_size(z: np.ndarray, lam: float, k: int, t: float) -> np.ndarray:
    """P(Z_t = z). Zero unless z = 1 + k n for some n >= 0."""
    z = np.asarray(z)
    p = np.exp(-k * lam * t)
    n = (z - 1) / k
    valid = (n >= 0) & (np.abs(n - np.round(n)) < 1e-12)
    out = np.zeros_like(z, dtype=float)
    out[valid] = stats.nbinom.pmf(np.round(n[valid]).astype(int), 1.0 / k, p)
    return out


def limit_law(k: int) -> stats.rv_continuous:
    """Law of W = lim exp(-k*lam*t) Z_t: Gamma(shape=1/k, scale=k)."""
    return stats.gamma(a=1.0 / k, scale=float(k))


def limit_law_name(k: int) -> str:
    if k == 1:
        return "Exponential(1)"
    if k == 2:
        return r"$\chi^2_1$  (Gamma($\frac{1}{2}$, scale 2))"
    return f"Gamma(1/{k}, scale {k})"
