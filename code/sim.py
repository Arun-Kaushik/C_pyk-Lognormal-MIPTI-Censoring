"""Monte Carlo study: MLE, MPS and hierarchical Bayes for C_pyk under MIPT-I censoring."""
import sys, time, json
import numpy as np
from multiprocessing import Pool
from gpci_core import *

L, U = 3.0, 18.0
TAU = TAU_3S
CONFIGS = {"C1": (1.75, 0.30), "C2": (2.00, 0.28), "C3": (2.30, 0.20)}
SCHEMES = {
    "S1": dict(probs=[0.15, 0.30, 0.45, 0.60, 0.75, 0.90], p=[0.05, 0.10, 0.05, 0.10, 0.05]),
    "S2": dict(probs=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60], p=[0.10, 0.15, 0.10, 0.15, 0.10]),
}
EXACT = [True, False, True, False, True, False]
NS = [30, 50, 100, 200]
B = 2000
OUT = "/home/dev/gpci_paper/results"


def run_cell(args):
    cfg, sch, n = args
    mu, sig = CONFIGS[cfg]
    seed = abs(hash((cfg, sch, n))) % (2 ** 31)
    seed = {"C1": 11, "C2": 22, "C3": 33}[cfg] * 1000 + {"S1": 1, "S2": 2}[sch] * 100 + n
    rng = np.random.default_rng(seed)
    des = Design.from_quantiles(mu, sig, SCHEMES[sch]["probs"], EXACT, SCHEMES[sch]["p"])
    t0 = time.time()
    rec = {k: [] for k in ["cnt", "mle", "mps", "C_mle", "C_mps", "wald", "akr", "akr_mps",
                           "kappa", "mu_ci", "sig_ci", "ok"]}
    Ds = []
    for b in range(B):
        data = simulate_mipt(n, mu, sig, des, rng)
        D = prep(data)
        rec["cnt"].append(summary_counts(data))
        m, s, res = fit_mle(D)
        ok = bool(np.isfinite(m) and np.isfinite(s) and s > 1e-4 and s < 50)
        try:
            J = obs_info(m, s, D)
            V = np.linalg.inv(J)
            if not np.all(np.isfinite(V)) or V[0, 0] <= 0 or V[1, 1] <= 0:
                raise np.linalg.LinAlgError
        except Exception:
            ok = False
            V = np.diag([s ** 2 / n, s ** 2 / (2 * n)])
        mp, sp_, _ = fit_mps(D, (m, s))
        try:
            Vp = np.linalg.inv(mps_info(mp, sp_, D))
            if not np.all(np.isfinite(Vp)) or Vp[0, 0] <= 0 or Vp[1, 1] <= 0:
                raise np.linalg.LinAlgError
        except Exception:
            Vp = V
        ic = intervals_classical(m, s, V, L, U, n, TAU, TAU)
        icp = intervals_classical(mp, sp_, Vp, L, U, n, TAU, TAU)
        rec["mle"].append((m, s))
        rec["mps"].append((mp, sp_))
        rec["C_mle"].append(ic["C"])
        rec["C_mps"].append(icp["C"])
        rec["wald"].append(ic["wald"])
        rec["akr"].append(ic["akr"])
        rec["akr_mps"].append(icp["akr"])
        rec["kappa"].append(ic["kappa"])
        z = norm.ppf(0.975)
        rec["mu_ci"].append((m - z * np.sqrt(V[0, 0]), m + z * np.sqrt(V[0, 0])))
        sl = np.sqrt(V[1, 1]) / s                          # log-transformed CI for sigma
        rec["sig_ci"].append((s * np.exp(-z * sl), s * np.exp(z * sl)))
        rec["ok"].append(ok)
        Ds.append(D)
    t1 = time.time()
    S = stack_suff(Ds)
    mle = np.array(rec["mle"])
    dr = hb_mcmc(S, mle[:, 0], mle[:, 1] ** 2, rng, prior=DEFAULT_PRIOR, N=6000, burn=2000)
    hs = hb_summaries(dr, L, U, n, TAU, TAU)
    t2 = time.time()
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(hb_mu=hs["mu_mean"], hb_sig=hs["sig_mean"], hb_C=hs["C_mean"], hb_Cmed=hs["C_med"],
               hb_et=np.array(hs["et"]).T, hb_akr=np.array(hs["akr"]).T,
               hb_mu_ci=np.array(hs["mu_ci"]).T, hb_sig_ci=np.array(hs["sig_ci"]).T,
               true=np.array([mu, sig, cpyk(mu, sig, L, U, TAU, TAU)]), n=n,
               times=np.array([t1 - t0, t2 - t1]))
    np.savez_compressed(f"{OUT}/sim_{cfg}_{sch}_n{n}.npz", **out)
    msg = f"{cfg} {sch} n={n}: classical {t1-t0:.0f}s, HB {t2-t1:.0f}s, ok={out['ok'].mean():.4f}"
    print(msg, flush=True)
    return msg


if __name__ == "__main__":
    cells = [(c, s, n) for c in CONFIGS for s in SCHEMES for n in NS]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        B = 50
        cells = cells[:2]
    simultaneous_crit(0.0)  # build cache
    with Pool(2) as pool:
        for m in pool.imap_unordered(run_cell, cells):
            pass
    print("ALL DONE", flush=True)
