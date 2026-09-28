"""Exact simulation of the Yule process and its twin-birth variant.

A single individual at time 0. Each living individual, independently and at
rate ``lam``, gives birth to ``k`` children (``k=1`` classical Yule,
``k=2`` twins). The population size jumps ``z -> z + k``; the holding time
in state ``z`` is Exponential(lam * z). This is the Gillespie / Doob
algorithm, exact for the continuous-time Markov chain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Path:
    """One realisation: jump times and population sizes right after each jump."""

    times: np.ndarray  # t_0 = 0 < t_1 < ... < t_n
    sizes: np.ndarray  # z_0 = 1, z_1, ..., z_n
    lam: float
    k: int

    def size_at(self, t: float) -> int:
        idx = np.searchsorted(self.times, t, side="right") - 1
        return int(self.sizes[idx])

    def sizes_at(self, grid: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.times, grid, side="right") - 1
        return self.sizes[idx]


def simulate_path(lam: float, k: int, t_max: float, rng: np.random.Generator,
                  max_births: int = 200_000) -> Path:
    """Simulate one path of the k-birth Yule process up to time ``t_max``."""
    times = [0.0]
    sizes = [1]
    t, z = 0.0, 1
    while True:
        tau = rng.exponential(1.0 / (lam * z))
        if t + tau > t_max or len(times) > max_births:
            break
        t += tau
        z += k
        times.append(t)
        sizes.append(z)
    return Path(np.asarray(times), np.asarray(sizes), lam, k)


def population_at(lam: float, k: int, t: float, n_paths: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Population size Z_t at a fixed time for ``n_paths`` independent runs.

    Vectorised over paths: all paths advance together, one jump at a time.
    """
    z = np.ones(n_paths, dtype=np.int64)
    clock = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)  # "alive" = still before t
    while alive.any():
        idx = np.flatnonzero(alive)
        tau = rng.exponential(1.0 / (lam * z[idx]))
        clock[idx] += tau
        done = clock[idx] > t
        z[idx[~done]] += k
        alive[idx[done]] = False
    return z


def simulate_until(lam: float, k: int, n_births: int, rng: np.random.Generator) -> Path:
    """Simulate until exactly ``n_births`` births have occurred (sequential design).

    With this stopping rule the total exposure A = sum_i z_i tau_i satisfies
    2 lam A ~ chi^2 with 2 n d.f. exactly, so the MLE confidence interval in
    :mod:`yule_twins.mle` is exact.
    """
    times = [0.0]
    sizes = [1]
    t, z = 0.0, 1
    for _ in range(n_births):
        t += rng.exponential(1.0 / (lam * z))
        z += k
        times.append(t)
        sizes.append(z)
    return Path(np.asarray(times), np.asarray(sizes), lam, k)
