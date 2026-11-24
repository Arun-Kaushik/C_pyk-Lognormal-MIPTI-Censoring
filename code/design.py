"""Optimal MIPT-I designs: information saturation, constrained D/A/C-optimality, cost-efficiency."""
import json, time
import numpy as np
from scipy.stats import norm
from gpci_core import *

L, U = 3.0, 18.0
CONFIGS = {"C1": (1.75, 0.30), "C2": (2.00, 0.28), "C3": (2.30, 0.20)}
OUT = "/home/dev/gpci_paper/results"
K = 6
PATTERNS = {
    "MIPT-I (E first)": [True, False, True, False, True, False],
    "MIPT-I (G first)": [False, True, False, True, False, True],
    "All exact (PT-I)": [True] * 6,
    "All grouped (PIT-I)": [False] * 6,
}


def make_design(mu, sig, piK, gamma, p, exact):
    probs = piK * (np.arange(1, K + 1) / K) ** gamma
    return Design.from_quantiles(mu, sig, probs, exact, p)


def crit_values(mu, sig, des, active):
    I = design_info(mu, sig, des)
    Iinv = np.linalg.inv(I)
    wU, wL, gU, gL = arm_grads(mu, sig, L, U)
    vU, vL = gU @ Iinv @ gU, gL @ Iinv @ gL
    if active == "U":
        vc = vU
    elif active == "L":
        vc = vL
    else:
        vc = max(vU, vL)
    return dict(logdet=np.log(np.linalg.det(I)), A=np.trace(Iinv), C=vc, I=I)


def active_arm(mu, sig):
    wU, wL = wdist(mu, sig, L, U)
    if abs(wU - wL) < 0.25:
        return "both"
    return "U" if wU < wL else "L"


def p_profile(mu, sig, probs, exact, rho, omega):
    """Withdrawal proportions p_j = min(0.95, kappa * exp(rho*(j-1)/(K-2))) solving E[W]/n = omega."""
    shares = np.exp(rho * np.arange(K - 1) / (K - 2))

    def EW(kap):
        p = np.minimum(0.95, kap * shares)
        des = Design.from_quantiles(mu, sig, probs, exact, p)
        return expected_ttt(mu, sig, des)[1]

    lo, hi = 0.0, 50.0
    if EW(hi) < omega:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if EW(mid) < omega:
            lo = mid
        else:
            hi = mid
    return np.minimum(0.95, hi * shares)


# ---------------------------------------------------------------------------
# 1. information saturation under per-failure proportional withdrawal
# ---------------------------------------------------------------------------
def saturation_study(reps=400):
    mu, sig = CONFIGS["C2"]
    probs = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    ex = PATTERNS["MIPT-I (E first)"]
    planA = Design.from_quantiles(mu, sig, probs, ex, [0.10] * 5)               # inspection-time
    planB = Design.from_quantiles(mu, sig, probs, ex, [0.0] * 5, pf=0.10)       # per-failure
    ns = [25, 50, 100, 200, 400, 800, 1600]
    rng = np.random.default_rng(2026)
    res = {"n": ns, "A": [], "B": []}
    for plan, key in [(planA, "A"), (planB, "B")]:
        for n in ns:
            ex_f, tot_f, trinv = [], [], []
            for _ in range(reps):
                data = simulate_mipt(n, mu, sig, plan, rng)
                D = prep(data)
                c = summary_counts(data)
                ex_f.append(c[0])
                tot_f.append(c[0] + c[1])
                J = obs_info(mu, sig, D)                 # observed information at true theta
                try:
                    trinv.append(n * np.trace(np.linalg.inv(J)))
                except Exception:
                    pass
            res[key].append(dict(exact=float(np.mean(ex_f)), total=float(np.mean(tot_f)),
                                 n_trinv=float(np.median(trinv))))
            print("sat", key, n, res[key][-1], flush=True)
    # deterministic bound for plan B: per continuously monitored interval
    p = 0.10
    res["boundB"] = [3 * (2 + 1 / p + np.log(max(p * n, 1)) / abs(np.log(1 - p))) for n in ns]
    res["IA_trinv"] = float(np.trace(np.linalg.inv(design_info(mu, sig, planA))))
    return res


# ---------------------------------------------------------------------------
# 2. constrained D-, A-, C-optimal designs (fixed T and expected withdrawal fraction)
# ---------------------------------------------------------------------------
def constrained_opt(piK=0.80, omegas=(0.10, 0.20, 0.30)):
    gammas = np.linspace(0.4, 2.5, 22)
    rhos = np.linspace(-4, 4, 17)
    ex = PATTERNS["MIPT-I (E first)"]
    out = {}
    for cfg, (mu, sig) in CONFIGS.items():
        act = active_arm(mu, sig)
        for om in omegas:
            grid = []
            for g in gammas:
                probs = piK * (np.arange(1, K + 1) / K) ** g
                for rho in rhos:
                    p = p_profile(mu, sig, probs, ex, rho, om)
                    if p is None:
                        continue
                    des = Design.from_quantiles(mu, sig, probs, ex, p)
                    cv = crit_values(mu, sig, des, act)
                    grid.append((g, rho, cv["logdet"], cv["A"], cv["C"], p.tolist()))
            arr = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in grid])
            iD, iA, iC = np.argmax(arr[:, 2]), np.argmin(arr[:, 3]), np.argmin(arr[:, 4])
            # reference: gamma = 1, rho = 0
            probs = piK * (np.arange(1, K + 1) / K)
            pref = p_profile(mu, sig, probs, ex, 0.0, om)
            ref = crit_values(mu, sig, Design.from_quantiles(mu, sig, probs, ex, pref), act)

            def effs(i):
                return dict(D=float(np.exp(0.5 * (arr[i, 2] - arr[iD, 2]))),
                            A=float(arr[iA, 3] / arr[i, 3]), C=float(arr[iC, 4] / arr[i, 4]))

            refeff = dict(D=float(np.exp(0.5 * (ref["logdet"] - arr[iD, 2]))),
                          A=float(arr[iA, 3] / ref["A"]), C=float(arr[iC, 4] / ref["C"]))
            out[f"{cfg}_w{om}"] = dict(
                active=act,
                D=dict(gamma=arr[iD, 0], rho=arr[iD, 1], p=grid[iD][5], eff=effs(iD)),
                A=dict(gamma=arr[iA, 0], rho=arr[iA, 1], p=grid[iA][5], eff=effs(iA)),
                C=dict(gamma=arr[iC, 0], rho=arr[iC, 1], p=grid[iC][5], eff=effs(iC)),
                ref=refeff,
                surface=dict(gammas=gammas.tolist(), rhos=rhos.tolist(),
                             C=[[float(r[4]) for r in grid if abs(r[0] - g) < 1e-12] for g in gammas]))
            print("opt", cfg, om, {k: (out[f'{cfg}_w{om}'][k]['gamma'], out[f'{cfg}_w{om}'][k]['rho'])
                                   for k in 'DAC'}, flush=True)
    return out


# ---------------------------------------------------------------------------
# 3. cost-efficiency
# ---------------------------------------------------------------------------
COST = dict(n=50, cu=1.0, ch=0.2, cm=20.0, cI=1.0, cT=10.0, vw=0.5)


def test_cost(mu, sig, des, cost=COST):
    """Expected cost of one MIPT-I test (time in units of the median life e^mu)."""
    m = np.exp(mu)
    n = cost["n"]
    ttt, W = expected_ttt(mu, sig, des, n)
    t = des.t / m
    mon = np.sum(np.diff(t)[des.exact])
    n_insp = int(np.sum(~des.exact))
    return (n * cost["cu"] + cost["ch"] * ttt / m + cost["cm"] * mon + cost["cI"] * n_insp
            + cost["cT"] * t[-1] - cost["vw"] * W)


def cost_study(cfg="C3", cost=COST):
    mu, sig = CONFIGS[cfg]
    act = active_arm(mu, sig)
    piKs = np.linspace(0.40, 0.99, 24)
    gammas = np.linspace(0.5, 2.0, 7)
    ps = np.linspace(0.0, 0.5, 11)
    rows = []
    for name, ex in PATTERNS.items():
        for piK in piKs:
            for g in gammas:
                for pp in ps:
                    des = make_design(mu, sig, piK, g, [pp] * (K - 1), ex)
                    cv = crit_values(mu, sig, des, act)
                    c = test_cost(mu, sig, des, cost)
                    avar = cv["C"] / cost["n"]
                    rows.append((name, piK, g, pp, c, avar, c * avar, cv["logdet"]))
    return rows


def cm_sensitivity(cfg="C3", cms=np.linspace(0, 40, 21)):
    mu, sig = CONFIGS[cfg]
    act = active_arm(mu, sig)
    piKs = np.linspace(0.40, 0.99, 16)
    gammas = np.linspace(0.5, 2.0, 7)
    ps = np.linspace(0.0, 0.5, 6)
    cache = []
    for name, ex in PATTERNS.items():
        for piK in piKs:
            for g in gammas:
                for pp in ps:
                    des = make_design(mu, sig, piK, g, [pp] * (K - 1), ex)
                    cv = crit_values(mu, sig, des, act)
                    m = np.exp(mu)
                    n = COST["n"]
                    ttt, W = expected_ttt(mu, sig, des, n)
                    t = des.t / m
                    mon = np.sum(np.diff(t)[des.exact])
                    base = (n * COST["cu"] + COST["ch"] * ttt / m + COST["cI"] * np.sum(~des.exact)
                            + COST["cT"] * t[-1] - COST["vw"] * W)
                    cache.append((name, base, mon, cv["C"] / n))
    res = {name: [] for name in PATTERNS}
    for cm in cms:
        best = {}
        for name, base, mon, av in cache:
            v = (base + cm * mon) * av
            if name not in best or v < best[name]:
                best[name] = v
        for name in PATTERNS:
            res[name].append(best[name])
    return dict(cms=list(map(float, cms)), best=res)


if __name__ == "__main__":
    t0 = time.time()
    sat = saturation_study()
    json.dump(sat, open(f"{OUT}/design_saturation.json", "w"), indent=1)
    con = constrained_opt()
    json.dump(con, open(f"{OUT}/design_constrained.json", "w"), indent=1, default=float)
    allrows = {}
    for cfg in CONFIGS:
        rows = cost_study(cfg)
        allrows[cfg] = rows
        best = {}
        for r in rows:
            if r[0] not in best or r[6] < best[r[0]][6]:
                best[r[0]] = r
        print("cost", cfg, {k: (round(v[1], 3), v[2], v[3], round(v[4], 1), v[6]) for k, v in best.items()}, flush=True)
    json.dump(allrows, open(f"{OUT}/design_cost.json", "w"), default=float)
    sens = {cfg: cm_sensitivity(cfg) for cfg in CONFIGS}
    json.dump(sens, open(f"{OUT}/design_cm_sens.json", "w"), indent=1, default=float)
    print("done", time.time() - t0)
