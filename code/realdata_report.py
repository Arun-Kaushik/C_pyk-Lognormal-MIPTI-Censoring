"""Tables and figures for the real-data analyses."""
import json, os
import numpy as np
from scipy import stats
from scipy.stats import norm, gaussian_kde
from figstyle import setup, PAL, TEXTW, panel_label, plt
from realdata import DATA, EXACT

setup()
RES = "/home/dev/gpci_paper/results"
FIG = "/home/dev/gpci_paper/figs"
TAB = "/home/dev/gpci_paper/tex/tables"
R = json.load(open(f"{RES}/realdata.json"))
NAMES = {"BB": "Ball bearings", "GP": "Guinea pigs"}


def m(x, d=4):
    return f"{x:.{d}f}".replace("-", "$-$")


def ci(a, d=4):
    return f"({m(a[0], d)}, {m(a[1], d)})"


def tab_gof():
    L = []
    for k in ["BB", "GP"]:
        g = R[k]["gof"]
        best = min(g, key=lambda q: g[q]["AIC"])
        for i, (nm, v) in enumerate(g.items()):
            lead = rf"\multirow{{4}}{{*}}{{{NAMES[k]} ($n={R[k]['n']}$)}}" if i == 0 else ""
            aic = m(v["AIC"], 2)
            if nm == best:
                aic = r"\textbf{" + aic + "}"
            L.append(f"{lead} & {nm} & {m(-v['ll'], 2)} & {aic} & {m(v['BIC'], 2)} & {m(v['KS'], 3)} & "
                     f"{m(v['KSp'], 3)} & {m(v['A2'], 3)} & {m(v['W2'], 3)} \\\\")
        L.append(r"\midrule" if k == "BB" else "")
    open(f"{TAB}/tab_gof.tex", "w").write("\n".join(L))


def tab_censored():
    L = []
    for k in ["BB", "GP"]:
        c = R[k]["censored"]
        t = c["t"]
        for j in range(6):
            lead = rf"\multirow{{6}}{{*}}{{{NAMES[k]}}}" if j == 0 else ""
            iv = f"$({t[j]:g}, {t[j+1]:g}]$"
            if EXACT[j]:
                xs = ", ".join((f"{v:.2f}" if k == "BB" else f"{v:g}") for v in c["x"][j])
                obs = f"exact: {xs}" if xs else "exact: none"
                mon = "C"
            else:
                obs = f"$d_{{{j+1}}}={c['d'][j]}$"
                mon = "I"
            Rj = c["R"][j]
            L.append(f"{lead} & {j+1} & {iv} & {mon} & \\parbox[t]{{7.2cm}}{{\\raggedright {obs}}} & {Rj} \\\\")
        L.append(r"\midrule" if k == "BB" else "")
    open(f"{TAB}/tab_censored.tex", "w").write("\n".join(L))


def tab_estimates():
    L = []
    for k in ["BB", "GP"]:
        r = R[k]
        cpl = r["complete"]
        h = r["hb"]["P1"]
        rows = [
            ("Complete-data MLE", m(cpl["mu"]), "", m(cpl["sig"]), "", m(cpl["C"]), ""),
            ("MLE", m(r["mle"]["mu"]), ci(r["mle"]["mu_ci"], 3), m(r["mle"]["sig"]), ci(r["mle"]["sig_ci"], 3),
             m(r["mle"]["C"]), r"Wald " + ci(r["mle"]["wald"]) + r"; AKR " + ci(r["mle"]["akr"])),
            ("", "", "", "", "", "", r"Boot-p " + ci(r["boot"]["ci"])),
            ("MPS", m(r["mps"]["mu"]), ci(r["mps"]["mu_ci"], 3), m(r["mps"]["sig"]), ci(r["mps"]["sig_ci"], 3),
             m(r["mps"]["C"]), r"AKR " + ci(r["mps"]["akr"])),
            ("HB (mean)", m(h["mu"]), ci(h["mu_ci"], 3), m(h["sig"]), ci(h["sig_ci"], 3), m(h["C_mean"]),
             r"ET " + ci(h["et"]) + r"; AKR " + ci(h["akr"])),
            ("HB (median)", "", "", "", "", m(h["C_med"]), rf"$P(C_{{pyk}}\ge 1\mid\mathcal{{D}})={h['P_capable']:.3f}$"),
        ]
        for i, row in enumerate(rows):
            lead = rf"\multirow{{6}}{{*}}{{\rotatebox{{90}}{{{NAMES[k]}}}}}" if i == 0 else ""
            L.append(lead + " & " + " & ".join(row) + r" \\")
        L.append(r"\midrule" if k == "BB" else "")
    open(f"{TAB}/tab_real_est.tex", "w").write("\n".join(L))


def tab_prior():
    L = []
    lab = {"P1": r"P1: $(0,0.01,2,1,1)$", "P2": r"P2: $(1,0.1,2,1,1)$", "P3": r"P3: $(0,0.001,\frac12,\frac12,1)$"}
    for k in ["BB", "GP"]:
        for i, pk in enumerate(["P1", "P2", "P3"]):
            h = R[k]["hb"][pk]
            d = h["diag"]
            lead = rf"\multirow{{3}}{{*}}{{{NAMES[k]}}}" if i == 0 else ""
            L.append(f"{lead} & {lab[pk]} & {m(h['mu'], 3)} & {m(h['sig'], 3)} & {m(h['C_mean'])} & {m(h['C_med'])} & "
                     f"{ci(h['et'])} & {h['P_capable']:.3f} & {d['Cpyk']['rhat']:.4f} & {d['Cpyk']['ess']:.0f} \\\\")
        L.append(r"\midrule" if k == "BB" else "")
    open(f"{TAB}/tab_real_prior.tex", "w").write("\n".join(L))


def tab_naive():
    L = []
    for k in ["BB", "GP"]:
        r = R[k]
        nv = r["naive"]
        L.append(f"{NAMES[k]} & {r['L']:g} & {r['U']:g} & {m(nv['Cpk'], 3)} & {m(nv['normal_p_below'], 4)} & "
                 f"{m(r['complete']['Cstar'], 3)} & {m(r['complete']['C'], 4)} & {m(nv['ln_p_below'], 4)} & "
                 f"{m(nv['ln_p_above'], 4)} & {nv['p_below']:.0f} / {nv['p_above']:.0f} \\\\")
    open(f"{TAB}/tab_real_naive.tex", "w").write("\n".join(L))


def fig_fit():
    fig, axs = plt.subplots(2, 2, figsize=(TEXTW, 4.4), gridspec_kw=dict(hspace=0.45, wspace=0.28))
    for j, k in enumerate(["BB", "GP"]):
        x = np.sort(DATA[k]["x"])
        n = len(x)
        r = R[k]
        ax = axs[0, j]
        pp = (np.arange(1, n + 1) - 0.375) / (n + 0.25)                # Blom positions
        q = norm.ppf(pp)
        ax.plot(q, np.log(x), "o", ms=3.2, mfc="white", mec=PAL["mle"], mew=0.9, label="observed")
        mu, sg = r["complete"]["mu"], r["complete"]["sig"]
        qq = np.linspace(q.min() - 0.3, q.max() + 0.3, 50)
        ax.plot(qq, mu + sg * qq, color=PAL["dark"], lw=1.1, label=rf"fit: $\hat\mu={mu:.3f},\ \hat\sigma={sg:.3f}$")
        # pointwise 95% band from order-statistic simulation
        sims = np.sort(mu + sg * np.random.default_rng(1).standard_normal((4000, n)), axis=1)
        lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
        ax.fill_between(q, lo, hi, color=PAL["sky"], alpha=0.25, lw=0, label="95% simulation envelope")
        ax.set_xlabel("standard normal quantile")
        ax.set_ylabel(r"$\ln x_{(i)}$")
        ax.set_title(f"{NAMES[k]}: lognormal probability plot", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")
        panel_label(ax, "(" + "ab"[j] + ")", x=-0.17)
        ax = axs[1, j]
        xs = np.exp(np.linspace(np.log(min(x.min(), r["L"]) * 0.8), np.log(max(x.max(), r["U"]) * 1.2), 400))
        ecdf = np.arange(1, n + 1) / n
        ax.step(x, ecdf, where="post", color=PAL["dark"], lw=1.0, label="empirical")
        cols = {"Lognormal": PAL["mle"], "Weibull": PAL["mps"], "Gamma": PAL["hb"], "Log-logistic": PAL["hbmed"]}
        dists = {"Lognormal": stats.lognorm, "Weibull": stats.weibull_min, "Gamma": stats.gamma, "Log-logistic": stats.fisk}
        for nm, dist in dists.items():
            prm = r["gof"][nm]["params"]
            ax.plot(xs, dist.cdf(xs, *prm), color=cols[nm], lw=1.1 if nm == "Lognormal" else 0.9,
                    ls="-" if nm == "Lognormal" else "--", label=nm)
        for lim, nm in [(r["L"], "L"), (r["U"], "U")]:
            ax.axvline(lim, color=PAL["orange"], lw=1.0)
            ax.text(lim * 1.08, 0.5, f"{nm}={lim:g}", color=PAL["orange"], fontsize=7, rotation=90, va="center", ha="left")
        ax.set_xscale("log")
        ax.set_xlabel(f"lifetime ({DATA[k]['unit']}, log scale)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{NAMES[k]}: fitted models and limits", fontsize=8)
        ax.legend(fontsize=6.3, loc="upper left")
        panel_label(ax, "(" + "cd"[j] + ")", x=-0.17)
    fig.savefig(f"{FIG}/fig_real_fit.pdf")
    plt.close(fig)


def fig_post():
    from gpci_core import TAU_3S
    tau = TAU_3S
    inv = lambda C: norm.ppf(np.clip(0.5 + np.asarray(C) * (0.5 - tau), 1e-15, 1 - 1e-15)) / 3.0
    fwd = lambda c: (norm.cdf(3 * np.asarray(c)) - 0.5) / (0.5 - tau)
    fig, axs = plt.subplots(2, 2, figsize=(TEXTW, 4.9), gridspec_kw=dict(hspace=0.95, wspace=0.28, height_ratios=[1.35, 1]))
    for j, k in enumerate(["BB", "GP"]):
        d = np.load(f"{RES}/real_{k}_draws.npz")
        r = R[k]
        h = r["hb"]["P1"]
        ax = axs[0, j]
        cs = inv(d["C"].ravel())
        bs = inv(d["boot"])
        lo, hi = np.percentile(np.r_[cs, bs], [0.3, 99.7])
        xs = np.linspace(lo, hi, 500)
        kd = gaussian_kde(cs[::4])
        ax.fill_between(xs, kd(xs), color=PAL["hb"], alpha=0.18, lw=0)
        ax.plot(xs, kd(xs), color=PAL["hb"], lw=1.3, label="posterior (HB, prior P1)")
        bk = gaussian_kde(bs)
        ax.plot(xs, bk(xs), color=PAL["mle"], lw=1.0, ls="--", label="bootstrap of censored MLE")
        ymax = max(kd(xs).max(), bk(xs).max())
        for val, col, ls, lab in [(r["complete"]["C"], PAL["dark"], "-", "complete-data MLE"),
                                  (r["mle"]["C"], PAL["mle"], ":", "censored MLE"),
                                  (r["mps"]["C"], PAL["mps"], ":", "MPS"),
                                  (h["C_mean"], PAL["hb"], "-.", "HB posterior mean")]:
            ax.axvline(inv(val), color=col, ls=ls, lw=1.0, label=lab)
        ax.axvline(1.0, color=PAL["orange"], lw=1.4)
        for yy, iv, col, lab in [(1.06, h["et"], PAL["hb"], "HB-ET"), (1.18, r["mle"]["akr"], PAL["mle"], "MLE-AKR")]:
            a, b = inv(iv[0]), min(inv(iv[1]), hi)
            ax.plot([a, b], [yy * ymax] * 2, color=col, lw=1.2)
            ax.plot([a, a], [(yy - 0.03) * ymax, (yy + 0.03) * ymax], color=col, lw=1.2)
            ax.text(a, (yy + 0.035) * ymax, lab, color=col, fontsize=6.3)
        ax.set_ylim(0, 1.32 * ymax)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"log-scale $C^\ast_{pk}=\min(w_U,w_L)/3$")
        ax.set_ylabel("density")
        sec = ax.secondary_xaxis("top", functions=(lambda c: c, lambda c: c))
        tick_c = np.array([0.9, 0.95, 0.99, 0.999, 1.002])
        tick_w = inv(tick_c)
        keep = (tick_w > lo) & (tick_w < hi)
        sec.set_xticks(tick_w[keep])
        sec.set_xticklabels([f"{v:g}" for v in tick_c[keep]], fontsize=6)
        sec.set_xlabel(r"$C_{pyk}$", fontsize=7, labelpad=2)
        ax.set_title(rf"{NAMES[k]}: $P(C_{{pyk}}\geq 1\mid\mathcal{{D}})={h['P_capable']:.3f}$", fontsize=8, pad=22)
        if j == 1:
            hs, ls_ = ax.get_legend_handles_labels()
            fig.legend(hs, ls_, loc="center", bbox_to_anchor=(0.5, 0.455), ncol=3, fontsize=6.6)
        panel_label(ax, "(" + "ab"[j] + ")", x=-0.17, y=1.18)
        ax = axs[1, j]
        mu = d["mu"]
        for c in range(mu.shape[1]):
            ax.plot(np.arange(3000), mu[:3000, c], lw=0.4, alpha=0.8,
                    color=[PAL["mle"], PAL["mps"], PAL["hb"], PAL["hbmed"]][c])
        ax.set_xlabel("iteration after burn-in")
        ax.set_ylabel(r"$\mu$")
        dg = h["diag"]["mu"]
        ax.set_title(rf"traces of $\mu$ (4 chains): $\widehat R={dg['rhat']:.4f}$, ESS$={dg['ess']:.0f}$", fontsize=7.5)
        panel_label(ax, "(" + "cd"[j] + ")", x=-0.17)
    fig.savefig(f"{FIG}/fig_real_post.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    tab_gof()
    tab_censored()
    tab_estimates()
    tab_prior()
    tab_naive()
    fig_fit()
    fig_post()
    for k in ["BB", "GP"]:
        print(k, "plan", R[k]["opt_plan"], "\n alt", R[k]["opt_plan_alt"])
        print(" hb", {p: (round(R[k]["hb"][p]["C_mean"], 4), round(R[k]["hb"][p]["P_capable"], 3)) for p in R[k]["hb"]})
        print(" mle kappa", R[k]["mle"]["kappa"], R[k]["mle"]["separated"], "A,B", R[k]["mle"]["A"], R[k]["mle"]["B"])
