"""Checks that the closed-form results match simulation and each other."""
import numpy as np
from scipy import stats

from yule_twins import simulate, theory, mle, trees


def test_pgf_matches_negbin():
    # Numerically verify F(s,t) = s p^{1/k} (1-(1-p)s^k)^{-1/k} solves dF/dt = lam (F^{k+1} - F)
    lam, k, s = 0.7, 2, 0.6
    def F(t):
        p = np.exp(-k * lam * t)
        return s * p**(1/k) / (1 - (1 - p) * s**k)**(1/k)
    t, h = 1.3, 1e-6
    lhs = (F(t + h) - F(t - h)) / (2 * h)
    rhs = lam * (F(t)**(k + 1) - F(t))
    assert abs(lhs - rhs) < 1e-6


def test_pmf_sums_to_one_and_mean():
    lam, k, t = 1.0, 2, 1.0
    z = np.arange(1, 4000, 1)
    pmf = theory.pmf_size(z, lam, k, t)
    assert abs(pmf.sum() - 1) < 1e-9
    assert abs((z * pmf).sum() - theory.mean_size(lam, k, t)) < 1e-6
    assert np.all(pmf[1::2] == 0)  # even sizes unreachable


def test_simulation_mean_and_law():
    rng = np.random.default_rng(1)
    lam, k, t = 1.0, 2, 1.0
    z = simulate.population_at(lam, k, t, 20000, rng)
    assert abs(z.mean() - theory.mean_size(lam, k, t)) < 0.3
    support = np.arange(1, int(z.max()) + 1)
    emp = np.bincount(z, minlength=len(support) + 1)[1:] / len(z)
    tv = 0.5 * np.abs(emp - theory.pmf_size(support, lam, k, t)).sum()
    assert tv < 0.03


def test_martingale_limit_moments():
    rng = np.random.default_rng(2)
    lam, k, t = 1.0, 2, 4.0
    w = np.exp(-k * lam * t) * simulate.population_at(lam, k, t, 20000, rng)
    law = theory.limit_law(k)  # chi2_1: mean 1, var 2
    assert abs(w.mean() - law.mean()) < 0.05
    assert abs(w.var() - law.var()) < 0.15


def test_mle_exact_ci_coverage():
    rng = np.random.default_rng(3)
    cover = 0
    for _ in range(400):
        p = simulate.simulate_until(1.0, 2, 60, rng)
        r = mle.fit_rate(p, T=float(p.times[-1]))
        cover += r.ci_low <= 1.0 <= r.ci_high
    assert 0.92 <= cover / 400 <= 0.98


def test_leaf_fraction_heuristic():
    rng = np.random.default_rng(4)
    for k, target in [(1, 0.5), (2, 2 / 3)]:
        lf = np.mean([trees.leaf_fraction(trees.grow_tree(5000, k, rng)) for _ in range(10)])
        assert abs(lf - target) < 0.01


def test_urn_limit_is_arcsine_for_twins():
    from yule_twins import urn
    rng = np.random.default_rng(5)
    f = urn.urn_fraction(2, 1000, 10000, rng)
    law = urn.limit_fraction_law(2)  # Beta(1/2,1/2): mean 1/2, var 1/8
    assert abs(f.mean() - 0.5) < 0.02
    assert abs(f.var() - law.var()) < 0.01
    assert stats.kstest(f, law.cdf).statistic < 0.03
