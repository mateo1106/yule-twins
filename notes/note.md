# The Yule process with twin births: exact law, martingale limit and genealogical tree

**Mateo Castañeda Cardona** · Universidad Nacional de Colombia, Manizales · September 2026

*Preliminary note prepared for the CIMPA Collaborative Workshop, Neuchâtel 2027 (group of Prof. Jean Bertoin: "Yule process, random recursive tree, and some applications"). Code, figures and tests: `github.com/mateo1106/yule-twins`.*

## 1. Setting

The classical Yule process is a pure-birth Markov chain $Z=(Z_t)_{t\ge 0}$, $Z_0=1$, in which every living individual begets one child at rate $\lambda$, independently of the others. The project abstract asks what changes when individuals beget **twins**. I take this literally: every individual, at rate $\lambda$, produces $k=2$ children at once, so $Z$ jumps from $z$ to $z+2$ at total rate $\lambda z$. I keep $k$ general because nothing below depends on $k=2$ except the numbers.

Three classical facts serve as the benchmark ($k=1$): $Z_t$ is geometric with parameter $e^{-\lambda t}$; $M_t=e^{-\lambda t}Z_t$ is a martingale converging a.s. and in $L^1$ to $W\sim\mathrm{Exp}(1)$; and the genealogy is the random recursive tree, whose leaf fraction tends to $1/2$.

## 2. Exact law of $Z_t$ (derivation)

Let $F(s,t)=\mathbb E[s^{Z_t}]$. By the branching property (each individual founds an independent copy), $F$ solves the backward equation
$$\partial_t F=\lambda\,(F^{k+1}-F),\qquad F(s,0)=s .$$
The substitution $u=F^{-k}$ linearises it: $u'=k\lambda(u-1)$, so $u-1=(s^{-k}-1)e^{k\lambda t}$ and
$$F(s,t)=\frac{s\,p^{1/k}}{\bigl(1-(1-p)s^{k}\bigr)^{1/k}},\qquad p=e^{-k\lambda t}.$$
This is the generating function of $1+kN_t$ with $N_t\sim\mathrm{NegBin}(r=1/k,\ p)$:
$$\mathbb P(Z_t=1+kn)=\frac{\Gamma(n+1/k)}{n!\,\Gamma(1/k)}\,p^{1/k}(1-p)^n .$$
For $k=1$ this is the geometric law; for twins, $Z_t$ lives on the odd integers and $N_t$ (the number of births) is negative binomial with shape $1/2$. Consequences: $\mathbb E Z_t=e^{k\lambda t}$, $\operatorname{Var}Z_t=k\,e^{k\lambda t}(e^{k\lambda t}-1)$, and $M_t=e^{-k\lambda t}Z_t$ is a martingale.

**Numerical check (Fig. 2).** With 40 000 simulated paths at $k\lambda t=2$, the total-variation distance between the empirical and exact laws is $0.008$ for both $k=1$ and $k=2$; means and variances agree to three digits (`figures/summary.json`).

## 3. The martingale limit changes law

Since $N_t\sim\mathrm{NegBin}(1/k,p)$ with $p\to 0$, the standard limit $p\,N_t\Rightarrow\mathrm{Gamma}(1/k,1)$ gives
$$e^{-k\lambda t}Z_t\ \longrightarrow\ W\sim\mathrm{Gamma}\bigl(\text{shape }1/k,\ \text{scale }k\bigr),$$
a.s. by the martingale convergence theorem and in $L^1$ since $\mathbb E W=1=\mathbb E M_0$. For twins, $W\sim\chi^2_1$: the exponential limit of the classical process becomes a chi-square with one degree of freedom. The qualitative change is visible in Fig. 3: mass piles up near $0$ (density $\sim w^{-1/2}$), i.e. with twins it is much more likely that the population lags far behind its mean, while the right tail is heavier ($\propto e^{-w/2}$ instead of $e^{-w}$).

**Numerical check (Fig. 3).** At $k\lambda t=8$, 20 000 paths: empirical mean of $W$ is $1.003$ (exact $1$); the Kolmogorov–Smirnov statistic against $\chi^2_1$ is $0.015$, and decreases with $t$ (it is $0.040$ at $k\lambda t=6$), consistent with a finite-$t$ lattice effect near $0$ where the $\chi^2_1$ density is unbounded.

## 4. Estimating $\lambda$ from one observed path

If the path is observed until $n$ births with holding times $\tau_i$ in states $z_i$, the likelihood is $\prod_i \lambda z_i e^{-\lambda z_i\tau_i}$, so $\hat\lambda=n/A$ with exposure $A=\sum_i z_i\tau_i$, and $2\lambda A\sim\chi^2_{2n}$ exactly. The estimator has the same form for $k=1$ and $k=2$; the number of children per birth only changes the sequence $z_i$. This is the same likelihood machinery I used to calibrate a rainfall-forced SDE against a landslide catalogue in the ALLO project.

**Numerical check (Fig. 4).** 2 000 replications, $n=100$: mean $\hat\lambda=1.008$ for both processes (true $1$), 95 % interval coverage $0.951$ and $0.950$.

## 5. The genealogical tree

Conditionally on the jump chain, the parent of each birth is uniform among the individuals alive, and $k$ children are attached. For $k=1$ this is the random recursive tree; for $k=2$ every attachment adds a pair of sibling leaves. A one-line heuristic gives the leaf fraction: with $L$ leaves among $m$ nodes, $\mathbb E[\Delta L]=k-L/m$, so $L/m\to k/(k+1)$: $1/2$ classically, $2/3$ for twins. Simulation with $20\,000$ nodes gives $0.500$ and $0.666$ (Fig. 5). The out-degree distribution is supported on multiples of $k$ and decays geometrically; the mean depth of a uniform node stays of order $\log m$ ($9.48$ vs $9.06$ at $m=20\,001$; $\ln m=9.90$), with a smaller constant for twins.

## 6. The embedded Pólya urn: from the uniform to the arcsine law

Start the process from two founders and let $Z^{(1)},Z^{(2)}$ be their (independent) descendances. Conditionally on the jump chain, each birth goes to founder $i$ with probability $Z^{(i)}/(Z^{(1)}+Z^{(2)})$ and adds $k$ to that descendance. Seen at birth times this is a **Pólya urn with reinforcement $k$**: draw a ball, return it with $k$ more of its colour, starting from one ball of each colour. The classical urn theorem gives that the fraction of colour 1 converges a.s. to $\mathrm{Beta}(1/k,1/k)$ (Pitman 2006, Ch. 3): the uniform law for $k=1$ and the **arcsine law** $\mathrm{Beta}(\tfrac12,\tfrac12)$ for twins.

This is consistent with Section 3 by an independent route: the limit fraction equals $W_1/(W_1+W_2)$ with $W_i$ i.i.d. $\mathrm{Gamma}(1/k,\text{scale }k)$, and the ratio of two independent $\mathrm{Gamma}(1/k)$ variables is $\mathrm{Beta}(1/k,1/k)$. Two ways of computing the same object agree, which is the kind of check I like to build in. The qualitative content: with twins, two founding lineages tend to end up **unbalanced** (mass near $0$ and $1$), whereas classically every split is equally likely.

**Numerical check (Fig. 6).** 20 000 urns with 2 000 draws and 20 000 two-founder simulations at $k\lambda t=8$: for $k=2$, Kolmogorov–Smirnov statistics against $\mathrm{Beta}(\tfrac12,\tfrac12)$ of $0.010$ and $0.005$; variance $0.1248$ (exact $1/8$).

## 7. Questions I would like to work on in Neuchâtel

1. Make the leaf-fraction heuristic a theorem (martingale + concentration, as for the RRT), and obtain the full degree distribution of the twin tree.
2. The depth profile: identify the constant in $\mathbb E[\text{depth}]\sim c_k\log m$ and the CLT for the height, comparing with the RRT results in Mahmoud (1992).
3. Beyond two founders: the $m$-founder split is Dirichlet$(1/k,\dots,1/k)$; what is the analogue of the Chinese-restaurant / random-permutation construction of the RRT, and how do Simon's (1955) skew distributions change when arrivals come in pairs?
4. Ages and sizes of subtrees "trees within trees" (Lambert 2025) when each split produces twins.

## References

Yule (1925), *Phil. Trans. R. Soc. B* 213; Simon (1955), *Biometrika* 42; Mahmoud (1992), *Evolution of Random Search Trees*; Pitman (2006), *Combinatorial Stochastic Processes*; Lambert (2025), *Phil. Trans. R. Soc. B* 380.
