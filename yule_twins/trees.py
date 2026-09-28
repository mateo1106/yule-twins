"""Genealogical tree left by the k-birth Yule process.

Conditionally on the jump chain, at each birth the parent is uniform among
the individuals alive at that moment (all have the same rate), and ``k``
children are attached to it. For k = 1 this is the random recursive tree
(RRT). For k = 2 every attachment adds a pair of sibling leaves.

Heuristic for the leaf fraction: with L leaves among m = 1 + k n
individuals, E[L_{n+1} - L_n] = k - L_n / m. If L_n ~ ell * m, then
ell * k = k - ell, so ell = k / (k + 1): 1/2 for the RRT, 2/3 for twins.
"""
from __future__ import annotations

import numpy as np


def grow_tree(n_births: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return parent array; parent[0] = -1 (root), parent[i] < i."""
    m = 1 + k * n_births
    parent = np.full(m, -1, dtype=np.int64)
    size = 1
    for _ in range(n_births):
        p = rng.integers(size)
        parent[size:size + k] = p
        size += k
    return parent


def leaf_fraction(parent: np.ndarray) -> float:
    m = len(parent)
    has_child = np.zeros(m, dtype=bool)
    has_child[parent[1:]] = True
    return float(1 - has_child.mean())


def depths(parent: np.ndarray) -> np.ndarray:
    d = np.zeros(len(parent), dtype=np.int64)
    for i in range(1, len(parent)):
        d[i] = d[parent[i]] + 1
    return d


def degree_distribution(parent: np.ndarray, max_deg: int = 12) -> np.ndarray:
    """Empirical distribution of the number of children, truncated."""
    counts = np.bincount(parent[1:], minlength=len(parent))
    hist = np.bincount(np.minimum(counts, max_deg), minlength=max_deg + 1)
    return hist / hist.sum()


def radial_layout(parent: np.ndarray, rng: np.random.Generator):
    """Simple radial layout: radius = depth, angle = share of subtree size."""
    m = len(parent)
    children = [[] for _ in range(m)]
    for i in range(1, m):
        children[parent[i]].append(i)
    sub = np.ones(m, dtype=np.int64)
    for i in range(m - 1, 0, -1):
        sub[parent[i]] += sub[i]
    d = depths(parent)
    theta = np.zeros(m)
    span = np.zeros((m, 2))
    span[0] = (0, 2 * np.pi)
    for i in range(m):
        a, b = span[i]
        theta[i] = 0.5 * (a + b)
        start = a
        for c in children[i]:
            w = (b - a) * sub[c] / max(sub[i] - 1, 1)
            span[c] = (start, start + w)
            start += w
    r = d.astype(float)
    return r * np.cos(theta), r * np.sin(theta)
