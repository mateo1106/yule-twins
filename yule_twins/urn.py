"""Pólya-urn correspondence for the k-birth Yule process.

Start the process from two founders and let Z^(1), Z^(2) be their
descendances (independent k-birth Yule processes). At each birth the parent
is chosen with probability proportional to the current descendance, which
then grows by k. Looking only at the sequence of births, this is a Pólya urn
with reinforcement k: draw a ball, return it together with k balls of the
same colour, starting from one ball of each colour.

Classical urn result: the fraction of colour 1 converges a.s. to
Beta(1/k, 1/k). For k = 1 this is Uniform(0,1); for twins it is
Beta(1/2, 1/2), the arcsine law. This matches Section 3 of the note: the
limit fraction is W_1 / (W_1 + W_2) with W_i i.i.d. Gamma(1/k, scale k),
and a ratio of two independent Gamma(1/k) variables is Beta(1/k, 1/k).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .simulate import population_at


def urn_fraction(k: int, n_draws: int, n_urns: int, rng: np.random.Generator) -> np.ndarray:
    """Fraction of colour 1 after ``n_draws`` draws, for ``n_urns`` independent urns."""
    a = np.ones(n_urns)
    b = np.ones(n_urns)
    for _ in range(n_draws):
        pick_a = rng.random(n_urns) < a / (a + b)
        a += k * pick_a
        b += k * (~pick_a)
    return a / (a + b)


def two_founders_fraction(lam: float, k: int, t: float, n_paths: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Fraction Z^(1)_t / (Z^(1)_t + Z^(2)_t) for two independent founders."""
    z1 = population_at(lam, k, t, n_paths, rng)
    z2 = population_at(lam, k, t, n_paths, rng)
    return z1 / (z1 + z2)


def limit_fraction_law(k: int) -> stats.rv_continuous:
    return stats.beta(1.0 / k, 1.0 / k)
