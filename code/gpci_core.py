"""
Core routines for inference on the generalized process capability index C_pyk
of a lognormal process under multiple-interval progressive Type-I (MIPT-I) censoring.

Parametrisation: X ~ LN(mu, sigma^2), Y = log X ~ N(mu, sigma^2).
Arms of C_pyk (tau1 = lower, tau2 = upper desired tail probabilities):
    A = (Phi(w_U) - 1/2)/(1/2 - tau2),  w_U = (log U - mu)/sigma   (upper arm)
    B = (Phi(w_L) - 1/2)/(1/2 - tau1),  w_L = (mu - log L)/sigma   (lower arm)
    C_pyk = min(A, B).
"""
import numpy as np
from scipy.special import ndtr, log_ndtr
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize, brentq

LOG2PI = np.log(2.0 * np.pi)
TAU_3S = 1.0 - ndtr(3.0)          # 0.0013499 : 3-sigma natural tolerance tail


# ----------------------------------------------------------------------------
# numerics
# ----------------------------------------------------------------------------
def log_diff_ndtr(a, b):
    """Stable log(Phi(b) - Phi(a)) for a <= b (broadcasting). Returns -inf if a == b."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        la, lb = log_ndtr(a), log_ndtr(b)
        low = lb + np.log1p(-np.exp(np.minimum(la - lb, 0.0)))
        lua, lub = log_ndtr(-a), log_ndtr(-b)
        up = lua + np.log1p(-np.exp(np.minimum(lub - lua, 0.0)))
    return np.where(a > 0, up, low)


def inv_mills(z):
    """phi(z)/(1-Phi(z)), stable; 0 for z = -inf."""
    z = np.asarray(z, float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = np.exp(norm.logpdf(z) - log_ndtr(-z))
    return np.where(np.isneginf(z), 0.0, out)


# ----------------------------------------------------------------------------
# the index
# ----------------------------------------------------------------------------
def wdist(mu, sig, L, U):
    """Standardised log-distances to the limits: w_U=(lnU-mu)/sig, w_L=(mu-lnL)/sig."""
    return (np.log(U) - mu) / sig, (mu - np.log(L)) / sig


def arms(mu, sig, L, U, tau1=TAU_3S, tau2=TAU_3S):
    wU, wL = wdist(mu, sig, L, U)
    return (ndtr(wU) - 0.5) / (0.5 - tau2), (ndtr(wL) - 0.5) / (0.5 - tau1)


def cpyk(mu, sig, L, U, tau1=TAU_3S, tau2=TAU_3S):
    A, B = arms(mu, sig, L, U, tau1, tau2)
    return np.minimum(A, B)


def arm_grads(mu, sig, L, U):
    """Gradients of w_U and w_L with respect to (mu, sigma)."""
    wU, wL = wdist(mu, sig, L, U)
    gU = np.array([-1.0 / sig, -wU / sig])
    gL = np.array([1.0 / sig, -wL / sig])
    return wU, wL, gU, gL


# ----------------------------------------------------------------------------
# MIPT-I design and simulation
# ----------------------------------------------------------------------------
class Design:
    """K intervals (t_{j-1}, t_j], t_0 = 0.
    exact[j]  : True  -> continuously monitored (exact failure times)
                False -> intermittently monitored (failure count at t_j)
    p[j]      : proportion of survivors withdrawn at t_{j+1}, j = 0..K-2
                (all survivors at t_K = T are right-censored)
    pf        : proportion of survivors withdrawn after EACH exact failure
                (per-failure plan; 0 for the recommended inspection-time plan)
    """

    def __init__(self, t, exact, p, pf=0.0):
        self.t = np.asarray(t, float)
        self.exact = np.asarray(exact, bool)
        self.p = np.asarray(p, float)
        self.pf = float(pf)
        self.K = len(self.t) - 1
        assert len(self.exact) == self.K and len(self.p) == self.K - 1

    @staticmethod
    def from_quantiles(mu, sig, probs, exact, p, pf=0.0):
        t = np.r_[0.0, np.exp(mu + sig * norm.ppf(np.asarray(probs, float)))]
        return Design(t, exact, p, pf)


def simulate_mipt(n, mu, sig, des, rng, lifetimes=None):
    """Run one MIPT-I life test. Withdrawals: R = floor(p * survivors), chosen at random."""
    alive = np.exp(mu + sig * rng.standard_normal(n)) if lifetimes is None else np.array(lifetimes, float)
    K = des.K
    xs, Rxs = [], []
    d = np.zeros(K, int)
    R = np.zeros(K, int)
    for j in range(K):
        b = des.t[j + 1]
        if des.exact[j]:
            if des.pf > 0:
                xj, Rj = [], []
                alive = np.sort(alive)
                while alive.size and alive[0] <= b:
                    x = alive[0]
                    alive = alive[1:]
                    m = alive.size
                    r = int(np.floor(des.pf * m))
                    if r > 0:
                        keep = np.ones(m, bool)
                        keep[rng.choice(m, r, replace=False)] = False
                        alive = alive[keep]
                    xj.append(x)
                    Rj.append(r)
                xj, Rj = np.array(xj), np.array(Rj, int)
            else:
                f = alive <= b
                xj = np.sort(alive[f])
                Rj = np.zeros(xj.size, int)
                alive = alive[~f]
            xs.append(xj)
            Rxs.append(Rj)
            d[j] = xj.size
        else:
            f = alive <= b
            d[j] = int(f.sum())
            alive = alive[~f]
            xs.append(np.array([]))
            Rxs.append(np.array([], int))
        m = alive.size
        if j < K - 1:
            r = int(np.floor(des.p[j] * m))
            if r > 0:
                keep = np.ones(m, bool)
                keep[rng.choice(m, r, replace=False)] = False
                alive = alive[keep]
            R[j] = r
        else:
            R[j] = m      # R_K includes the terminal survivors C_T
    return dict(t=des.t.copy(), exact=des.exact.copy(), x=xs, Rx=Rxs, d=d, R=R, n=n)


def prep(data):
    """Sufficient statistics and arrays used by the likelihood."""
    x = np.concatenate([xx for xx in data["x"]]) if len(data["x"]) else np.array([])
    Rx = np.concatenate([rr for rr in data["Rx"]]).astype(float) if len(data["Rx"]) else np.array([])
    y = np.log(x) if x.size else np.array([])
    with np.errstate(divide="ignore"):
        lt = np.log(np.asarray(data["t"], float))
    return dict(r=y.size, Sy=y.sum(), Syy=(y ** 2).sum(), y=y, Rx=Rx, lt=lt,
                d=np.asarray(data["d"], float), R=np.asarray(data["R"], float),
                exact=np.asarray(data["exact"], bool), x=data["x"], n=data["n"])


def summary_counts(data):
    ex = int(sum(len(v) for v in data["x"]))
    grp = int(np.sum(np.asarray(data["d"])[~np.asarray(data["exact"])]))
    wd = int(np.sum(data["R"][:-1]) + sum(int(np.sum(v)) for v in data["Rx"]))
    return ex, grp, wd, int(data["R"][-1])


# ----------------------------------------------------------------------------
# log-likelihood, gradient, observed information
# ----------------------------------------------------------------------------
def loglik(mu, sig, D):
    r = D["r"]
    ll = -D["Sy"] - r * np.log(sig) - 0.5 * r * LOG2PI \
        - (D["Syy"] - 2 * mu * D["Sy"] + r * mu * mu) / (2 * sig * sig)
    if D["Rx"].size and D["Rx"].sum() > 0:
        ll += np.sum(D["Rx"] * log_ndtr(-(D["y"] - mu) / sig))
    z = (D["lt"] - mu) / sig
    G = (~D["exact"]) & (D["d"] > 0)
    if G.any():
        ll += np.sum(D["d"][G] * log_diff_ndtr(z[:-1][G], z[1:][G]))
    pos = D["R"] > 0
    ll += np.sum(D["R"][pos] * log_ndtr(-z[1:][pos]))
    return ll


def score(mu, sig, D):
    r = D["r"]
    g_mu = (D["Sy"] - r * mu) / sig ** 2
    g_s = -r / sig + (D["Syy"] - 2 * mu * D["Sy"] + r * mu * mu) / sig ** 3
    if D["Rx"].size and D["Rx"].sum() > 0:
        z = (D["y"] - mu) / sig
        lam = inv_mills(z)
        g_mu += np.sum(D["Rx"] * lam) / sig
        g_s += np.sum(D["Rx"] * lam * z) / sig
    z = (D["lt"] - mu) / sig
    G = (~D["exact"]) & (D["d"] > 0)
    if G.any():
        za, zb = z[:-1][G], z[1:][G]
        lp = log_diff_ndtr(za, zb)
        pa = np.where(np.isneginf(za), 0.0, np.exp(norm.logpdf(za) - lp))
        pb = np.exp(norm.logpdf(zb) - lp)
        za_f = np.where(np.isneginf(za), 0.0, za)
        g_mu += np.sum(D["d"][G] * (-(pb - pa))) / sig
        g_s += np.sum(D["d"][G] * (-(pb * zb - pa * za_f))) / sig
    pos = D["R"] > 0
    zc = z[1:][pos]
    lam = inv_mills(zc)
    g_mu += np.sum(D["R"][pos] * lam) / sig
    g_s += np.sum(D["R"][pos] * lam * zc) / sig
    return np.array([g_mu, g_s])


def obs_info(mu, sig, D, fun=None):
    """Observed information (2x2) in (mu, sigma) by central differences of the score."""
    sc = (lambda m, s: score(m, s, D)) if fun is None else fun
    h = np.array([1e-5 * max(1.0, abs(mu)), 1e-5 * sig])
    J = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2)
        e[k] = h[k]
        J[:, k] = (sc(mu + e[0], sig + e[1]) - sc(mu - e[0], sig - e[1])) / (2 * h[k])
    J = -0.5 * (J + J.T)
    return J


def start_values(D):
    if D["r"] >= 2:
        mu0, s0 = D["y"].mean(), max(D["y"].std(ddof=1), 1e-2)
    else:
        lt = D["lt"][np.isfinite(D["lt"])]
        mu0, s0 = np.median(lt), 0.5
    return mu0, s0


def fit_mle(D, start=None):
    mu0, s0 = start_values(D) if start is None else start
    f = lambda th: -loglik(th[0], np.exp(th[1]), D)

    def g(th):
        s = np.exp(th[1])
        sc = score(th[0], s, D)
        return -np.array([sc[0], sc[1] * s])

    res = minimize(f, [mu0, np.log(s0)], jac=g, method="BFGS", options={"gtol": 1e-8, "maxiter": 500})
    if not res.success or not np.all(np.isfinite(res.x)):
        res = minimize(f, [mu0, np.log(s0)], method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000})
    mu, sig = res.x[0], np.exp(res.x[1])
    return mu, sig, res


# ----------------------------------------------------------------------------
# Maximum product of spacings (conditional-spacings formulation)
# ----------------------------------------------------------------------------
def mps_objective(mu, sig, D):
    """H(theta) = sum over continuously monitored intervals of the log conditional spacings
    + log-likelihood of the coarsened (interval-count / withdrawal) data.
    Equivalent: for an exact interval j with r_j failures, contribution
        sum_{i=1}^{r_j+1} log[F(x_{j,i}) - F(x_{j,i-1})] - log[F(t_j) - F(t_{j-1})].
    Zero spacings from ties are replaced by the density (Cheng & Amin, 1983)."""
    z = (D["lt"] - mu) / sig
    H = 0.0
    K = len(D["exact"])
    for j in range(K):
        if D["exact"][j]:
            x = D["x"][j]
            lP = log_diff_ndtr(z[j], z[j + 1])
            if x.size == 0:
                H += 0.0                                        # empty interval contributes log 1
                continue
            zx = (np.log(x) - mu) / sig
            pts = np.r_[z[j], zx, z[j + 1]]
            sp = log_diff_ndtr(pts[:-1], pts[1:])
            tie = ~np.isfinite(sp)
            if tie.any():                                       # density substitution
                idx = np.where(tie)[0]
                idx = np.clip(idx, 0, x.size - 1)
                sp[tie] = norm.logpdf(zx[idx]) - np.log(sig) - np.log(x[idx])
            H += np.sum(sp) - lP
        else:
            if D["d"][j] > 0:
                H += D["d"][j] * log_diff_ndtr(z[j], z[j + 1])
    pos = D["R"] > 0
    H += np.sum(D["R"][pos] * log_ndtr(-z[1:][pos]))
    return H


def fit_mps(D, start):
    f = lambda th: -mps_objective(th[0], np.exp(th[1]), D)
    res = minimize(f, [start[0], np.log(start[1])], method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
    return res.x[0], np.exp(res.x[1]), res


def mps_info(mu, sig, D):
    """Negative Hessian of H at the MPS estimate (numerical, 2nd differences)."""
    h = np.array([2e-4 * max(1.0, abs(mu)), 2e-4 * sig])
    f = lambda m, s: mps_objective(m, s, D)
    f0 = f(mu, sig)
    Hm = np.zeros((2, 2))
    Hm[0, 0] = (f(mu + h[0], sig) - 2 * f0 + f(mu - h[0], sig)) / h[0] ** 2
    Hm[1, 1] = (f(mu, sig + h[1]) - 2 * f0 + f(mu, sig - h[1])) / h[1] ** 2
    Hm[0, 1] = Hm[1, 0] = (f(mu + h[0], sig + h[1]) - f(mu + h[0], sig - h[1])
                           - f(mu - h[0], sig + h[1]) + f(mu - h[0], sig - h[1])) / (4 * h[0] * h[1])
    return -Hm


# ----------------------------------------------------------------------------
# interval estimation for C_pyk: transformed Wald and adaptive kink-robust (AKR)
# ----------------------------------------------------------------------------
_RHO_GRID = np.linspace(-0.999, 0.999, 401)
_CRIT_CACHE = {}


def simultaneous_crit(rho, alpha=0.05):
    """c such that P(|Z1|<=c, |Z2|<=c) = 1-alpha for a standard bivariate normal with corr rho."""
    key = alpha
    if key not in _CRIT_CACHE:
        cs = []
        for r in _RHO_GRID:
            cov = np.array([[1.0, r], [r, 1.0]])
            mvn = multivariate_normal(mean=[0, 0], cov=cov)

            def cover(c):
                return (mvn.cdf([c, c]) - mvn.cdf([-c, c]) - mvn.cdf([c, -c]) + mvn.cdf([-c, -c])) - (1 - alpha)

            cs.append(brentq(cover, norm.ppf(1 - alpha / 2) - 1e-6, norm.ppf(1 - alpha / 4) + 1e-3))
        _CRIT_CACHE[key] = np.array(cs)
    return np.interp(rho, _RHO_GRID, _CRIT_CACHE[key])


def intervals_classical(mu, sig, V, L, U, n, tau1=TAU_3S, tau2=TAU_3S, alpha=0.05):
    """Returns dict with transformed-Wald and AKR intervals for C_pyk.
    V: estimated covariance matrix of (mu_hat, sigma_hat)."""
    wU, wL, gU, gL = arm_grads(mu, sig, L, U)
    vU, vL = gU @ V @ gU, gL @ V @ gL
    cUL = gU @ V @ gL
    sU, sL = np.sqrt(max(vU, 1e-300)), np.sqrt(max(vL, 1e-300))
    hU = lambda w: (ndtr(w) - 0.5) / (0.5 - tau2)
    hL = lambda w: (ndtr(w) - 0.5) / (0.5 - tau1)
    A, B = hU(wU), hL(wL)
    z = norm.ppf(1 - alpha / 2)
    # transformed Wald on the active arm
    if A <= B:
        wald = (hU(wU - z * sU), hU(wU + z * sU))
    else:
        wald = (hL(wL - z * sL), hL(wL + z * sL))
    # separation statistic on the w scale: delta = w_U - omega(w_L), omega = h2^{-1} o h1
    # (omega = identity when tau1 = tau2); zero set of delta is exactly the kink set K.
    om = norm.ppf(np.clip(0.5 + (0.5 - tau2) * hL(wL), 1e-300, 1 - 1e-16))
    dom = (norm.pdf(wL) / (0.5 - tau1)) / max(norm.pdf(om) / (0.5 - tau2), 1e-300)
    var_delta = vU + dom ** 2 * vL - 2 * dom * cUL
    kappa = abs(wU - om) / np.sqrt(max(var_delta, 1e-300))
    if kappa > np.sqrt(np.log(n)):
        akr = wald
        sep = True
    else:
        rho = np.clip(cUL / (sU * sL), -0.999, 0.999)
        c = simultaneous_crit(rho, alpha)
        loA, upA = hU(wU - c * sU), hU(wU + c * sU)
        loB, upB = hL(wL - c * sL), hL(wL + c * sL)
        akr = (min(loA, loB), min(upA, upB))
        sep = False
    return dict(A=A, B=B, C=min(A, B), wald=wald, akr=akr, kappa=kappa, separated=sep)


# ----------------------------------------------------------------------------
# Hierarchical Bayes: vectorised Metropolis-Hastings-within-Gibbs over many datasets
# ----------------------------------------------------------------------------
def stack_suff(Ds):
    """Stack sufficient statistics of several datasets (same K) for vectorised MCMC."""
    return dict(r=np.array([D["r"] for D in Ds], float),
                Sy=np.array([D["Sy"] for D in Ds]),
                Syy=np.array([D["Syy"] for D in Ds]),
                lt=np.array([D["lt"] for D in Ds]),
                d=np.array([D["d"] for D in Ds]),
                R=np.array([D["R"] for D in Ds]),
                exact=Ds[0]["exact"])


def loglik_vec(mu, s2, S):
    s = np.sqrt(s2)
    ll = -S["r"] * np.log(s) - (S["Syy"] - 2 * mu * S["Sy"] + S["r"] * mu ** 2) / (2 * s2)
    z = (S["lt"] - mu[:, None]) / s[:, None]
    G = (~S["exact"])[None, :] & (S["d"] > 0)
    ld = log_diff_ndtr(z[:, :-1], z[:, 1:])
    ll += np.sum(np.where(G, S["d"] * ld, 0.0), axis=1)
    lsv = log_ndtr(-z[:, 1:])
    ll += np.sum(np.where(S["R"] > 0, S["R"] * lsv, 0.0), axis=1)
    return ll


DEFAULT_PRIOR = dict(mu0=0.0, k0=0.01, a0=2.0, c0=1.0, d0=1.0)


def hb_mcmc(S, init_mu, init_s2, rng, prior=DEFAULT_PRIOR, N=6000, burn=2000,
            adapt_every=50, target=0.44, return_eta=False):
    """MH-within-Gibbs for (mu, sigma^2, eta); chains run in parallel (one per dataset).
    Proposal scales are adapted during burn-in only (then frozen)."""
    mu0, k0, a0, c0, d0 = (prior[k] for k in ("mu0", "k0", "a0", "c0", "d0"))
    B = len(init_mu)
    mu = np.array(init_mu, float).copy()
    s2 = np.array(init_s2, float).copy()
    eta = rng.gamma(a0 + c0, 1.0 / (d0 + 1.0 / s2))
    sd_mu = 1.5 * np.sqrt(s2 / np.maximum(S["r"], 5.0))
    sd_ls = np.full(B, 0.4)
    ll = loglik_vec(mu, s2, S)
    M = N - burn
    out_mu = np.empty((M, B), np.float64)
    out_s2 = np.empty((M, B), np.float64)
    out_eta = np.empty((M, B), np.float64) if return_eta else None
    acc_mu = np.zeros(B)
    acc_s = np.zeros(B)
    acc_mu_tot = np.zeros(B)
    acc_s_tot = np.zeros(B)
    for it in range(N):
        # --- mu | sigma^2, eta (random-walk MH)
        mu_p = mu + sd_mu * rng.standard_normal(B)
        ll_p = loglik_vec(mu_p, s2, S)
        lr = ll_p - ll - k0 * ((mu_p - mu0) ** 2 - (mu - mu0) ** 2) / (2 * s2)
        a = np.log(rng.random(B)) < lr
        mu = np.where(a, mu_p, mu)
        ll = np.where(a, ll_p, ll)
        acc_mu += a
        # --- sigma^2 | mu, eta (random walk on lambda = log sigma^2, Jacobian included)
        lam = np.log(s2)
        lam_p = lam + sd_ls * rng.standard_normal(B)
        s2_p = np.exp(lam_p)
        ll_p = loglik_vec(mu, s2_p, S)
        q = k0 * (mu - mu0) ** 2 / 2 + eta
        lp_new = (-a0 - 1.5) * lam_p - q / s2_p + lam_p
        lp_old = (-a0 - 1.5) * lam - q / s2 + lam
        lr = ll_p - ll + lp_new - lp_old
        a = np.log(rng.random(B)) < lr
        s2 = np.where(a, s2_p, s2)
        ll = np.where(a, ll_p, ll)
        acc_s += a
        # --- eta | sigma^2 (Gibbs)
        eta = rng.gamma(a0 + c0, 1.0 / (d0 + 1.0 / s2))
        if it < burn:
            if (it + 1) % adapt_every == 0:
                ra_mu, ra_s = acc_mu / adapt_every, acc_s / adapt_every
                sd_mu *= np.exp(ra_mu - target)
                sd_ls *= np.exp(ra_s - target)
                acc_mu[:] = 0
                acc_s[:] = 0
        else:
            k = it - burn
            out_mu[k] = mu
            out_s2[k] = s2
            if return_eta:
                out_eta[k] = eta
            acc_s_tot += a
    res = dict(mu=out_mu, s2=out_s2)
    if return_eta:
        res["eta"] = out_eta
    return res


def hb_summaries(draws, L, U, n, tau1=TAU_3S, tau2=TAU_3S, alpha=0.05):
    """Posterior summaries per chain (column)."""
    mu, s2 = draws["mu"], draws["s2"]
    sig = np.sqrt(s2)
    A, B = arms(mu, sig, L, U, tau1, tau2)
    C = np.minimum(A, B)
    out = dict(mu_mean=mu.mean(0), sig_mean=sig.mean(0), s2_mean=s2.mean(0),
               C_mean=C.mean(0), C_med=np.median(C, 0))
    q = np.quantile(C, [alpha / 2, 1 - alpha / 2], axis=0)
    out["et"] = (q[0], q[1])
    out["mu_ci"] = tuple(np.quantile(mu, [alpha / 2, 1 - alpha / 2], axis=0))
    out["sig_ci"] = tuple(np.quantile(sig, [alpha / 2, 1 - alpha / 2], axis=0))
    # adaptive kink-robust credible interval (separation measured on the w scale)
    wU, wL = wdist(mu, sig, L, U)
    om = norm.ppf(np.clip(0.5 + (0.5 - tau2) * (ndtr(wL) - 0.5) / (0.5 - tau1), 1e-300, 1 - 1e-16))
    Dlt = wU - om
    kap = np.abs(Dlt.mean(0)) / np.maximum(Dlt.std(0), 1e-300)
    sep = kap > np.sqrt(np.log(n))
    qA = np.quantile(A, [alpha / 4, 1 - alpha / 4], axis=0)
    qB = np.quantile(B, [alpha / 4, 1 - alpha / 4], axis=0)
    lo_k, up_k = np.minimum(qA[0], qB[0]), np.minimum(qA[1], qB[1])
    out["akr"] = (np.where(sep, q[0], lo_k), np.where(sep, q[1], up_k))
    out["separated"] = sep
    return out


# ----------------------------------------------------------------------------
# convergence diagnostics (split R-hat and bulk ESS; Gelman et al. 2013, Vehtari et al. 2021)
# ----------------------------------------------------------------------------
def split_rhat(chains):
    """chains: (n_iter, n_chains)."""
    n = chains.shape[0] // 2
    sp = np.concatenate([chains[:n], chains[n:2 * n]], axis=1)
    m = sp.shape[1]
    W = sp.var(0, ddof=1).mean()
    Bv = n * sp.mean(0).var(ddof=1)
    var_hat = (n - 1) / n * W + Bv / n
    return np.sqrt(var_hat / W)


def ess(chains):
    """Effective sample size (Geyer initial monotone sequence) pooled over chains."""
    n, m = chains.shape
    x = chains - chains.mean(0)
    var = x.var(0).mean()
    acf = np.zeros(n)
    for c in range(m):
        f = np.fft.rfft(x[:, c], 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        acf += ac / ac[0]
    acf /= m
    s = 0.0
    t = 1
    prev = np.inf
    while t + 1 < n:
        pair = acf[t] + acf[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        s += pair
        prev = pair
        t += 2
    tau = max(1.0 + 2.0 * s, 1e-8)
    return n * m / tau


# ----------------------------------------------------------------------------
# expected Fisher information of an MIPT-I design (inspection-time withdrawals)
# ----------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


def design_info(mu, sig, des, return_parts=False):
    """Per-unit expected Fisher information (2x2, in (mu, sigma)) of an MIPT-I design
    with withdrawals at inspection times only.  I(theta; xi) = sum_j w_j I_j."""
    with np.errstate(divide="ignore"):
        z = (np.log(des.t) - mu) / sig
    K = des.K
    I = np.zeros((2, 2))
    w = 1.0
    parts = []
    lS = log_ndtr(-z)
    sS = np.stack([inv_mills(z) / sig, inv_mills(z) * np.where(np.isfinite(z), z, 0.0) / sig])  # grad log S
    wts, fails, ttt = [], [], []
    for j in range(K):
        za, zb = z[j], z[j + 1]
        q = np.exp(lS[j + 1] - lS[j])
        dq = sS[:, j + 1] - sS[:, j]                     # grad log q
        if des.exact[j]:
            lo = max(za, -12.0)
            xs = 0.5 * (zb - lo) * _GL_X + 0.5 * (zb + lo)
            ws = 0.5 * (zb - lo) * _GL_W
            dens = np.exp(norm.logpdf(xs) - lS[j])        # conditional density of Z given > za
            s1 = xs / sig - sS[0, j]
            s2 = (xs ** 2 - 1) / sig - sS[1, j]
            Ij = np.array([[np.sum(ws * dens * s1 * s1), np.sum(ws * dens * s1 * s2)],
                           [np.sum(ws * dens * s1 * s2), np.sum(ws * dens * s2 * s2)]])
            Ij += q * np.outer(dq, dq)
        else:
            gq = q * dq
            Ij = np.outer(gq, gq) / max(q * (1 - q), 1e-300)
        I += w * Ij
        parts.append(Ij)
        wts.append(w)
        fails.append(w * (1 - q))
        w = w * q * (1 - (des.p[j] if j < K - 1 else 1.0))
    if return_parts:
        return I, dict(weights=np.array(wts), fail_frac=np.array(fails), parts=parts)
    return I


def expected_ttt(mu, sig, des, n=1):
    """Expected total time on test and expected number of withdrawals (per n units)."""
    K = des.K
    w = 1.0
    ttt = 0.0
    W = 0.0
    with np.errstate(divide="ignore"):
        z = (np.log(des.t) - mu) / sig
    lS = log_ndtr(-z)
    for j in range(K):
        a, b = des.t[j], des.t[j + 1]
        # int_a^b S(x) dx / S(a) by Gauss-Legendre on [a, b]
        xs = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        ws = 0.5 * (b - a) * _GL_W
        Sx = np.exp(log_ndtr(-(np.log(xs) - mu) / sig) - lS[j])
        ttt += w * np.sum(ws * Sx)
        q = np.exp(lS[j + 1] - lS[j])
        pj = des.p[j] if j < K - 1 else 0.0
        W += w * q * pj
        w = w * q * (1 - pj)
    return n * ttt, n * W
