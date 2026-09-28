"""Maximum-likelihood estimation of the birth rate from one observed path.

If a path is observed on [0, T] with jumps at t_1 < ... < t_n and population
z_i on [t_i, t_{i+1}), the likelihood of the holding times is

    L(lam) = prod_i (lam z_i) exp(-lam z_i tau_i) * exp(-lam z_n (T - t_n)),

where tau_i is the holding time in state z_i. Hence, with the total
"exposure" A = sum_i z_i tau_i + z_n (T - t_n),

    lam_hat = n / A,      Fisher information n / lam^2.

The estimator does not depend on k: the number of children per birth only
changes the sequence z_i, not the form of the likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .simulate import Path


@dataclass
class MLEResult:
    lam_hat: float
    n_births: int
    exposure: float
    ci_low: float
    ci_high: float


def fit_rate(path: Path, T: float, level: float = 0.95) -> MLEResult:
    times = np.append(path.times, T)
    holding = np.diff(times)
    exposure = float(np.sum(path.sizes * holding))
    n = len(path.times) - 1
    lam_hat = n / exposure
    # Exact CI: 2*lam*A ~ chi^2 with 2n d.f. (sum of exponentials)
    alpha = 1 - level
    lo = stats.chi2.ppf(alpha / 2, 2 * n) / (2 * exposure)
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * n) / (2 * exposure)
    return MLEResult(lam_hat, n, exposure, lo, hi)
