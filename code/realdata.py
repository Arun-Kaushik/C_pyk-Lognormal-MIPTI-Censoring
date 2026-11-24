"""Real-data analyses: ball bearings (Lieblein & Zelen, 1956) and guinea pigs (Bjerkedal, 1960)."""
import json
import numpy as np
from scipy import stats
from scipy.stats import norm
from gpci_core import *
from design import PATTERNS, make_design, crit_values, test_cost, COST

OUT = "/home/dev/gpci_paper/results"
TAU = TAU_3S

DATA = {
    "BB": dict(
        x=np.array([17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.40, 51.84, 51.96, 54.12, 55.56, 67.80,
                    68.64, 68.64, 68.88, 84.12, 93.12, 98.64, 105.12, 105.84, 127.92, 128.04, 173.40]),
        L=12.0, U=350.0, unit="million revolutions",
        t=[0, 40, 55, 70, 85, 105, 130], R=[1, 1, 1, 1, 1], seed=101),
    "GP": dict(
        x=np.array([12, 15, 22, 24, 24, 32, 32, 33, 34, 38, 38, 43, 44, 48, 52, 53, 54, 54, 55, 56, 57, 58,
                    58, 59, 60, 60, 60, 60, 61, 62, 63, 65, 65, 67, 68, 70, 70, 72, 73, 75, 76, 76, 81, 83,
                    84, 85, 87, 91, 95, 96, 98, 99, 109, 110, 121, 127, 129, 131, 143, 146, 146, 175, 175,
                    211, 233, 258, 258, 263, 297, 341, 341, 376], float),
        L=10.0, U=1000.0, unit="days",
        t=[0, 40, 60, 80, 100, 140, 200], R=[3, 2, 1, 2, 1], seed=202),
}
EXACT = np.array([True, False, True, False, True, False])


def apply_mipt_fixed(x, t, exact, Rcounts, rng):
    """Apply an MIPT-I plan with fixed withdrawal counts to a complete sample (random withdrawals)."""
    alive = np.sort(np.asarray(x, float))
    K = len(t) - 1
    xs, Rxs, d, R = [], [], np.zeros(K, int), np.zeros(K, int)
    withdrawn = []
    for j in range(K):
        b = t[j + 1]
        f = alive <= b
        if exact[j]:
            xs.append(np.sort(alive[f]))
        else:
            xs.append(np.array([]))
        Rxs.append(np.zeros(len(xs[-1]), int))
        d[j] = int(f.sum())
        alive = alive[~f]
        if j < K - 1:
            r = min(Rcounts[j], alive.size)
            if r > 0:
                idx = rng.choice(alive.size, r, replace=False)
                withdrawn.append(alive[idx].tolist())
                keep = np.ones(alive.size, bool)
                keep[idx] = False
                alive = alive[keep]
            else:
                withdrawn.append([])
            R[j] = r
        else:
            R[j] = alive.size
    return dict(t=np.asarray(t, float), exact=np.asarray(exact, bool), x=xs, Rx=Rxs, d=d, R=R,
                n=len(x)), withdrawn


def gof_table(x):
    rows = {}
    n = len(x)
    for nm, dist in [("Lognormal", stats.lognorm), ("Weibull", stats.weibull_min),
                     ("Gamma", stats.gamma), ("Log-logistic", stats.fisk)]:
        pr = dist.fit(x, floc=0)
        ll = dist.logpdf(x, *pr).sum()
        u = np.sort(dist.cdf(x, *pr))
        i = np.arange(1, n + 1)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        A2 = -n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))
        W2 = 1 / (12 * n) + np.sum((u - (2 * i - 1) / (2 * n)) ** 2)
        ks = stats.kstest(x, dist.cdf, args=pr)
        rows[nm] = dict(ll=ll, AIC=-2 * ll + 4, BIC=-2 * ll + 2 * np.log(n), KS=ks.statistic,
                        KSp=ks.pvalue, A2=A2, W2=W2, params=[float(v) for v in pr])
    return rows


def boot_p(mu, sig, t, exact, Rcounts, n, L, U, B, rng):
    vals = []
    for _ in range(B):
        x = np.exp(mu + sig * rng.standard_normal(n))
        data, _ = apply_mipt_fixed(x, t, exact, Rcounts, rng)
        D = prep(data)
        m, s, _ = fit_mle(D)
        vals.append(cpyk(m, s, L, U, TAU, TAU))
    vals = np.array(vals)
    return np.quantile(vals, [0.025, 0.975]), vals


def analyse(key, spec):
    rng = np.random.default_rng(spec["seed"])
    x, L, U = spec["x"], spec["L"], spec["U"]
    n = len(x)
    res = dict(name=key, n=n, L=L, U=U)
    res["gof"] = gof_table(x)
    y = np.log(x)
    mu_c, sig_c = y.mean(), y.std()
    res["complete"] = dict(mu=mu_c, sig=sig_c, C=float(cpyk(mu_c, sig_c, L, U, TAU, TAU)),
                           Cstar=float(min(wdist(mu_c, sig_c, L, U)) / 3))
    # naive normal-theory Cpk on the original scale, and empirical nonconforming
    xb, s = x.mean(), x.std(ddof=1)
    res["naive"] = dict(Cpk=float(min((U - xb) / (3 * s), (xb - L) / (3 * s))),
                        mean=xb, sd=s, p_below=float(np.mean(x < L)), p_above=float(np.mean(x > U)),
                        normal_p_below=float(norm.cdf((L - xb) / s)),
                        ln_p_below=float(norm.cdf((np.log(L) - mu_c) / sig_c)),
                        ln_p_above=float(norm.sf((np.log(U) - mu_c) / sig_c)))
    # MIPT-I censored sample
    data, withdrawn = apply_mipt_fixed(x, spec["t"], EXACT, spec["R"], rng)
    D = prep(data)
    res["censored"] = dict(t=spec["t"], d=data["d"].tolist(), R=data["R"].tolist(),
                           x=[v.tolist() for v in data["x"]], withdrawn=withdrawn,
                           counts=summary_counts(data))
    # MLE
    m, sg, _ = fit_mle(D)
    V = np.linalg.inv(obs_info(m, sg, D))
    ic = intervals_classical(m, sg, V, L, U, n, TAU, TAU)
    # MPS
    mp, sp_, _ = fit_mps(D, (m, sg))
    Vp = np.linalg.inv(mps_info(mp, sp_, D))
    icp = intervals_classical(mp, sp_, Vp, L, U, n, TAU, TAU)
    z = norm.ppf(0.975)
    res["mle"] = dict(mu=m, sig=sg, se=np.sqrt(np.diag(V)).tolist(), C=ic["C"], A=ic["A"], B=ic["B"],
                      wald=ic["wald"], akr=ic["akr"], kappa=ic["kappa"], separated=ic["separated"],
                      mu_ci=(m - z * np.sqrt(V[0, 0]), m + z * np.sqrt(V[0, 0])),
                      sig_ci=(sg * np.exp(-z * np.sqrt(V[1, 1]) / sg), sg * np.exp(z * np.sqrt(V[1, 1]) / sg)),
                      Cstar=float(min(wdist(m, sg, L, U)) / 3))
    res["mps"] = dict(mu=mp, sig=sp_, se=np.sqrt(np.diag(Vp)).tolist(), C=icp["C"], akr=icp["akr"],
                      wald=icp["wald"],
                      mu_ci=(mp - z * np.sqrt(Vp[0, 0]), mp + z * np.sqrt(Vp[0, 0])),
                      sig_ci=(sp_ * np.exp(-z * np.sqrt(Vp[1, 1]) / sp_), sp_ * np.exp(z * np.sqrt(Vp[1, 1]) / sp_)))
    ci_b, bvals = boot_p(m, sg, spec["t"], EXACT, spec["R"], n, L, U, 2000, rng)
    res["boot"] = dict(ci=ci_b.tolist(), mean=float(bvals.mean()))
    # HB: 4 dispersed chains
    priors = {"P1": DEFAULT_PRIOR, "P2": dict(mu0=1.0, k0=0.1, a0=2.0, c0=1.0, d0=1.0),
              "P3": dict(mu0=0.0, k0=0.001, a0=0.5, c0=0.5, d0=1.0)}   # half-Cauchy(0,1) on sigma
    res["hb"] = {}
    for pk, pr in priors.items():
        nch = 4
        S = stack_suff([D] * nch)
        init_mu = m + np.array([-2, -1, 1, 2]) * np.sqrt(V[0, 0])
        init_s2 = (sg ** 2) * np.exp(np.array([-0.6, 0.6, -0.3, 0.3]))
        dr = hb_mcmc(S, init_mu, init_s2, rng, prior=pr, N=25000, burn=5000, return_eta=True)
        hs = hb_summaries(dict(mu=dr["mu"].reshape(-1, 1), s2=dr["s2"].reshape(-1, 1)), L, U, n, TAU, TAU)
        sig_d = np.sqrt(dr["s2"])
        Cd = cpyk(dr["mu"], sig_d, L, U, TAU, TAU)
        diag = {nm: dict(rhat=float(split_rhat(v)), ess=float(ess(v)))
                for nm, v in [("mu", dr["mu"]), ("sigma", sig_d), ("Cpyk", Cd), ("eta", dr["eta"])]}
        res["hb"][pk] = dict(prior=pr, mu=float(hs["mu_mean"][0]), sig=float(hs["sig_mean"][0]),
                             mu_ci=[float(v[0]) for v in hs["mu_ci"]], sig_ci=[float(v[0]) for v in hs["sig_ci"]],
                             C_mean=float(hs["C_mean"][0]), C_med=float(hs["C_med"][0]),
                             et=[float(v[0]) for v in hs["et"]], akr=[float(v[0]) for v in hs["akr"]],
                             P_capable=float(np.mean(Cd >= 1.0)), diag=diag)
        if pk == "P1":
            np.savez_compressed(f"{OUT}/real_{key}_draws.npz", mu=dr["mu"], s2=dr["s2"], C=Cd, boot=bvals)
    # cost-efficient future test plan at the censored-data MLE: generalised MIPT-I (all 2^6 patterns)
    from design_patterns import PATS
    wU_, wL_ = wdist(m, sg, L, U)
    act = "both" if abs(wU_ - wL_) < 0.25 else ("U" if wU_ < wL_ else "L")
    best, best_alt = None, None
    for pat in PATS:
        ex = list(pat)
        for piK in np.linspace(0.4, 0.99, 16):
            for g in np.linspace(0.5, 2.0, 7):
                for pp in np.linspace(0, 0.5, 6):
                    des = make_design(m, sg, piK, g, [pp] * 5, ex)
                    cv = crit_values(m, sg, des, act)
                    c = test_cost(m, sg, des, COST)
                    val = c * cv["C"] / COST["n"]
                    rec = (val, "".join("E" if e else "G" for e in ex), piK, g, pp, des.t.tolist(), c, cv["C"])
                    if best is None or val < best[0]:
                        best = rec
                    if rec[1] == "EGEGEG" and (best_alt is None or val < best_alt[0]):
                        best_alt = rec
    res["opt_plan"] = dict(CE=best[0], pattern=best[1], piK=best[2], gamma=best[3], p=best[4],
                           t=best[5], cost=best[6], C=best[7])
    res["opt_plan_alt"] = dict(CE=best_alt[0], pattern=best_alt[1], piK=best_alt[2], gamma=best_alt[3],
                               p=best_alt[4], t=best_alt[5], cost=best_alt[6], C=best_alt[7])
    res["active"] = act
    return res


if __name__ == "__main__":
    allres = {}
    for k, spec in DATA.items():
        r = analyse(k, spec)
        allres[k] = r
        print(k, "complete C", r["complete"]["C"], "MLE", r["mle"]["C"], r["mle"]["wald"], r["mle"]["akr"],
              "MPS", r["mps"]["C"], "HB", r["hb"]["P1"]["C_mean"], r["hb"]["P1"]["et"], r["hb"]["P1"]["akr"],
              "boot", r["boot"]["ci"], flush=True)
        print("  counts", r["censored"]["counts"], "naive", r["naive"], flush=True)
        print("  diag", r["hb"]["P1"]["diag"], flush=True)
        print("  plan", r["opt_plan"], r["opt_plan_alt"], flush=True)
    json.dump(allres, open(f"{OUT}/realdata.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
