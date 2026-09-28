"""Reproduce every figure and number in notes/note.md.

    python run_all.py            # ~1-2 min on a laptop

Figures land in figures/, a summary table in figures/summary.txt.
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from yule_twins import simulate, theory, mle, trees

SEED = 20260928
LAM = 1.0
rng = np.random.default_rng(SEED)

C_YULE, C_TWIN = "#1f5f8b", "#c0392b"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 10.5, "figure.dpi": 150, "savefig.bbox": "tight",
})
summary = {}
t0 = time.time()

# ----------------------------------------------------------------------
# Figure 1: sample paths + mean growth, classical vs twins
# ----------------------------------------------------------------------
T_by_k = {1: 5.0, 2: 2.5}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, k, col, name in [(axes[0], 1, C_YULE, "Classical Yule ($k=1$)"),
                         (axes[1], 2, C_TWIN, "Twin births ($k=2$)")]:
    T = T_by_k[k]; grid = np.linspace(0, T, 400)
    paths = [simulate.simulate_path(LAM, k, T, rng) for _ in range(25)]
    for p in paths:
        ax.step(grid, p.sizes_at(grid), where="post", color=col, alpha=0.25, lw=0.8)
    mean_emp = np.mean([p.sizes_at(grid) for p in paths], axis=0)
    ax.plot(grid, mean_emp, color=col, lw=2, label="empirical mean (25 paths)")
    ax.plot(grid, [theory.mean_size(LAM, k, t) for t in grid], "k--", lw=1.2,
            label=r"$e^{k\lambda t}$")
    ax.set_yscale("log"); ax.set_xlabel("time $t$"); ax.set_ylabel("$Z_t$")
    ax.set_title(name); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle(r"Twin births double the growth exponent: $E[Z_t]=e^{k\lambda t}$", y=1.02)
fig.savefig("figures/fig1_paths.png"); plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: exact law of Z_t (negative binomial) vs simulation
# ----------------------------------------------------------------------
t_fix_by_k, n_paths = {1: 2.0, 2: 1.0}, 40_000  # same k*lam*t
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, k, col, name in [(axes[0], 1, C_YULE, "Classical: $Z_t$ geometric"),
                         (axes[1], 2, C_TWIN, "Twins: $Z_t = 1+2N_t$, $N_t\\sim$ NegBin($\\frac{1}{2}$, $e^{-2\\lambda t}$)")]:
    t_fix = t_fix_by_k[k]
    z = simulate.population_at(LAM, k, t_fix, n_paths, rng)
    zmax = int(np.quantile(z, 0.995))
    support = np.arange(1, zmax + 1)
    emp = np.bincount(z, minlength=zmax + 1)[1:zmax + 1] / n_paths
    th = theory.pmf_size(support, LAM, k, t_fix)
    ax.bar(support, emp, width=0.9 * k, color=col, alpha=0.45, label=f"simulation ({n_paths:,} paths)")
    ax.plot(support[th > 0], th[th > 0], "k.", ms=3, label="exact pmf")
    ax.set_xlabel(f"$Z_t$ at $t={t_fix}$"); ax.set_ylabel("probability")
    ax.set_title(name, fontsize=9.5); ax.legend(frameon=False, fontsize=8)
    # goodness of fit: total variation distance
    tv = 0.5 * np.sum(np.abs(emp - th))
    summary[f"tv_distance_k{k}"] = float(tv)
    summary[f"mean_k{k}"] = {"empirical": float(z.mean()), "exact": theory.mean_size(LAM, k, t_fix)}
    summary[f"var_k{k}"] = {"empirical": float(z.var()), "exact": theory.var_size(LAM, k, t_fix)}
fig.suptitle("The twin process has an exact negative-binomial law (only odd sizes are reachable)", y=1.02)
fig.savefig("figures/fig2_exact_law.png"); plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: martingale limit W = lim e^{-k lam t} Z_t
# ----------------------------------------------------------------------
t_lim_by_k = {1: 8.0, 2: 4.0}; n_paths = 20_000
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
x = np.linspace(1e-3, 6, 600)
for ax, k, col in [(axes[0], 1, C_YULE), (axes[1], 2, C_TWIN)]:
    t_lim = t_lim_by_k[k]  # k*lam*t = 8 in both cases: E[Z_t] ~ 3000
    z = simulate.population_at(LAM, k, t_lim, n_paths, rng)
    w = np.exp(-k * LAM * t_lim) * z
    ax.hist(w, bins=80, range=(0, 6), density=True, color=col, alpha=0.45,
            label=r"$e^{-k\lambda t}Z_t$, $t=%g$" % t_lim)
    ax.plot(x, theory.limit_law(k).pdf(x), "k-", lw=1.4, label="limit: " + theory.limit_law_name(k))
    if k == 2:
        ax.plot(x, stats.expon.pdf(x), color=C_YULE, ls="--", lw=1.2, label="Exponential(1) (classical)")
    ax.set_ylim(0, 1.6 if k == 2 else 1.1)
    ax.set_xlabel("$W$"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Classical Yule" if k == 1 else "Twin births")
    ks = stats.kstest(w, theory.limit_law(k).cdf)
    summary[f"ks_limit_k{k}"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue),
                                 "mean_W_emp": float(w.mean()), "mean_W_exact": float(theory.limit_law(k).mean())}
fig.suptitle(r"The martingale limit changes law: Exponential(1) $\to$ $\chi^2_1$ (mass piles up near 0)", y=1.02)
fig.savefig("figures/fig3_martingale_limit.png"); plt.close(fig)

# ----------------------------------------------------------------------
# Figure 4: MLE of lambda from one observed path (both processes)
# ----------------------------------------------------------------------
N_OBS, n_rep = 100, 2000
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, k, col in [(axes[0], 1, C_YULE), (axes[1], 2, C_TWIN)]:
    est, cover = [], 0
    for _ in range(n_rep):
        p = simulate.simulate_until(LAM, k, N_OBS, rng)
        r = mle.fit_rate(p, T=float(p.times[-1]))
        est.append(r.lam_hat); cover += (r.ci_low <= LAM <= r.ci_high)
    est = np.asarray(est)
    ax.hist(est, bins=50, color=col, alpha=0.5, density=True)
    ax.axvline(LAM, color="k", lw=1.2, label=r"true $\lambda$")
    ax.axvline(est.mean(), color=col, lw=1.2, ls="--", label=r"mean $\hat\lambda$ = %.3f" % est.mean())
    ax.set_xlabel(r"$\hat\lambda = n / \sum_i z_i \tau_i$"); ax.set_ylabel("density")
    ax.set_title(("Classical" if k == 1 else "Twins") + f" — 95% CI coverage {cover/n_rep:.3f}")
    ax.legend(frameon=False, fontsize=8)
    summary[f"mle_k{k}"] = {"mean": float(est.mean()), "sd": float(est.std()), "coverage95": cover / n_rep}
fig.suptitle("Rate estimation from one path observed until 100 births: the likelihood has the same form for both processes", y=1.02)
fig.savefig("figures/fig4_mle.png"); plt.close(fig)

# ----------------------------------------------------------------------
# Figure 5: genealogical trees — leaf fraction and depth
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 6.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.5, wspace=0.35)
for j, (k, col, name) in enumerate([(1, C_YULE, "random recursive tree ($k=1$)"),
                                     (2, C_TWIN, "twin tree ($k=2$)")]):
    ax = fig.add_subplot(gs[0, j])
    par = trees.grow_tree(120 if k == 1 else 60, k, rng)
    xs, ys = trees.radial_layout(par, rng)
    for i in range(1, len(par)):
        ax.plot([xs[par[i]], xs[i]], [ys[par[i]], ys[i]], color="0.55", lw=0.7, zorder=1)
    d = trees.depths(par)
    is_leaf = ~np.isin(np.arange(len(par)), par[1:])
    ax.scatter(xs, ys, s=[16 if l else 10 for l in is_leaf], c=[col if l else "k" for l in is_leaf], zorder=2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{name}, {len(par)} nodes\nleaves in colour", fontsize=9.5)

ax = fig.add_subplot(gs[0, 2])
sizes = np.array([200, 500, 1000, 2000, 5000, 10000, 20000])
for k, col, lab, target in [(1, C_YULE, "$k=1$", 1 / 2), (2, C_TWIN, "$k=2$", 2 / 3)]:
    lf = []
    for m in sizes:
        n_b = int((m - 1) // k)
        lf.append(np.mean([trees.leaf_fraction(trees.grow_tree(n_b, k, rng)) for _ in range(20)]))
    ax.plot(sizes, lf, "o-", color=col, label=f"{lab} (simulation)")
    ax.axhline(target, color=col, ls="--", lw=1)
    summary[f"leaf_fraction_k{k}"] = {"largest_size": int(sizes[-1]), "empirical": float(lf[-1]), "heuristic": target}
ax.set_xscale("log"); ax.set_xlabel("number of nodes"); ax.set_ylabel("fraction of leaves")
ax.set_title(r"Leaf fraction $\to k/(k+1)$: $\frac{1}{2}$ vs $\frac{2}{3}$"); ax.legend(frameon=False, fontsize=8)

ax = fig.add_subplot(gs[1, 0:2])
m_big = 20001
for k, col, lab in [(1, C_YULE, "$k=1$"), (2, C_TWIN, "$k=2$")]:
    dd = np.concatenate([trees.depths(trees.grow_tree((m_big - 1) // k, k, rng)) for _ in range(10)])
    vals, cnt = np.unique(dd, return_counts=True)
    ax.plot(vals, cnt / cnt.sum(), "o-", ms=3, color=col, label=f"{lab}: mean depth {dd.mean():.2f}")
    summary[f"mean_depth_k{k}"] = {"nodes": m_big, "mean_depth": float(dd.mean()), "log_nodes": float(np.log(m_big))}
ax.set_xlabel("depth of a node"); ax.set_ylabel("fraction of nodes"); ax.set_title(f"Depth profile, trees with {m_big:,} nodes")
ax.legend(frameon=False, fontsize=8)

ax = fig.add_subplot(gs[1, 2])
for k, col, lab in [(1, C_YULE, "$k=1$"), (2, C_TWIN, "$k=2$")]:
    dist = np.mean([trees.degree_distribution(trees.grow_tree((m_big - 1) // k, k, rng)) for _ in range(5)], axis=0)
    ax.plot(np.arange(len(dist)), dist, "o-", ms=3, color=col, label=lab)
ax.set_yscale("log"); ax.set_xlabel("number of children"); ax.set_ylabel("fraction of nodes")
ax.set_title("Out-degree (children come in $k$-tuples)"); ax.legend(frameon=False, fontsize=8)
fig.suptitle("The genealogical tree: twins change the leaf fraction and the degree structure, not the log-depth scale", y=1.0)
fig.savefig("figures/fig5_trees.png"); plt.close(fig)

summary["runtime_seconds"] = round(time.time() - t0, 1)
with open("figures/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
