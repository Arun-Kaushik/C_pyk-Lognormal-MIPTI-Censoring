"""Cost-efficiency optimisation over all 2^K monitoring patterns (generalised MIPT-I)."""
import json, itertools, time
import numpy as np
from gpci_core import *
from design import CONFIGS, COST, make_design, crit_values, active_arm, K

OUT = "/home/claude/gpci_paper/results"
PATS = [tuple(bool(b) for b in bits) for bits in itertools.product([1, 0], repeat=K)]


def evaluate(cfg):
    mu, sig = CONFIGS[cfg]
    act = active_arm(mu, sig)
    m = np.exp(mu)
    n = COST["n"]
    rows = []
    for pat in PATS:
        ex = list(pat)
        for piK in np.linspace(0.40, 0.99, 16):
            for g in np.linspace(0.5, 2.0, 7):
                for pp in np.linspace(0.0, 0.5, 6):
                    des = make_design(mu, sig, piK, g, [pp] * (K - 1), ex)
                    I = design_info(mu, sig, des)
                    Iinv = np.linalg.inv(I)
                    wU, wL, gU, gL = arm_grads(mu, sig, 3.0, 18.0)
                    vU, vL = gU @ Iinv @ gU, gL @ Iinv @ gL
                    vc = vU if act == "U" else (vL if act == "L" else max(vU, vL))
                    ttt, W = expected_ttt(mu, sig, des, n)
                    t = des.t / m
                    mon = float(np.sum(np.diff(t)[des.exact]))
                    base = (n * COST["cu"] + COST["ch"] * ttt / m + COST["cI"] * np.sum(~des.exact)
                            + COST["cT"] * t[-1] - COST["vw"] * W)
                    rows.append((pat, piK, g, pp, base, mon, vc / n, np.trace(Iinv) / n,
                                 float(np.linalg.det(I)), des.t.tolist()))
    return rows


def best_by_cm(rows, cms, crit="C"):
    out = []
    for cm in cms:
        best = None
        for r in rows:
            cost = r[4] + cm * r[5]
            if crit == "C":
                v = cost * r[6]
            elif crit == "A":
                v = cost * r[7]
            else:  # D: maximise sqrt(det(n I))/cost  -> minimise cost/sqrt(det)
                v = cost / np.sqrt(r[8] * COST["n"] ** 2)
            if best is None or v < best[0]:
                best = (v, r)
        out.append(best)
    return out


if __name__ == "__main__":
    t0 = time.time()
    cms = np.linspace(0, 40, 21)
    res = {}
    for cfg in CONFIGS:
        rows = evaluate(cfg)
        res[cfg] = {}
        for crit in ["C", "A", "D"]:
            b = best_by_cm(rows, cms, crit)
            res[cfg][crit] = [dict(cm=float(cm), val=float(v), pattern="".join("E" if e else "G" for e in r[0]),
                                   piK=float(r[1]), gamma=float(r[2]), p=float(r[3]), cost=float(r[4] + cm * r[5]),
                                   avar=float(r[6]), t=r[9]) for cm, (v, r) in zip(cms, b)]
        # fixed-pattern families at each cm (C criterion) for the figure
        fam = {"EGEGEG": None, "GEGEGE": None, "EEEEEE": None, "GGGGGG": None}
        famres = {k: [] for k in fam}
        for cm in cms:
            for k in fam:
                pat = tuple(c == "E" for c in k)
                vals = [(r[4] + cm * r[5]) * r[6] for r in rows if r[0] == pat]
                famres[k].append(float(min(vals)))
        res[cfg]["families"] = famres
        res[cfg]["cms"] = cms.tolist()
        # Pareto cloud at the baseline cm=20 (cost, avar) for the C criterion
        cloud = [(float(r[4] + 20 * r[5]), float(r[6]), "".join("E" if e else "G" for e in r[0])) for r in rows]
        res[cfg]["cloud"] = cloud
        print(cfg, "time", time.time() - t0, flush=True)
        for crit in "CAD":
            print("  ", crit, [(d["cm"], d["pattern"], round(d["piK"], 3), d["gamma"], d["p"]) for d in res[cfg][crit][::4]], flush=True)
    json.dump(res, open(f"{OUT}/design_patterns.json", "w"))
    print("done", time.time() - t0)
