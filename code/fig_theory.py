"""Figures: MIPT-I schematic, C_pyk landscape, information saturation, design/cost-efficiency."""
import json
import numpy as np
from scipy.stats import norm
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from figstyle import setup, PAL, TEXTW, panel_label, plt
from gpci_core import *

setup()
FIG = "/home/dev/gpci_paper/figs"
RES = "/home/dev/gpci_paper/results"
L, U = 3.0, 18.0
CONFIGS = {"C1": (1.75, 0.30), "C2": (2.00, 0.28), "C3": (2.30, 0.20)}


# ---------------------------------------------------------------------------
def fig_scheme():
    fig = plt.figure(figsize=(TEXTW, 3.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.62)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.3, 10.6)
    ax.set_ylim(-1.15, 1.9)
    ax.axis("off")
    ts = [0, 1.6, 3.0, 4.7, 6.1, 7.9, 9.6]
    labs = [r"$t_0=0$", r"$t_1$", r"$t_2$", r"$t_3$", r"$t_4$", r"$t_5$", r"$t_6=T$"]
    for j in range(6):
        a, b = ts[j], ts[j + 1]
        if j % 2 == 0:
            ax.add_patch(Rectangle((a, -0.32), b - a, 0.64, facecolor=PAL["mle"], alpha=0.18, edgecolor=PAL["mle"], lw=0.8))
            rng = np.random.default_rng(j + 3)
            for xx in np.sort(rng.uniform(a + 0.1, b - 0.1, 3 + j % 3)):
                ax.plot([xx, xx], [-0.25, 0.25], color=PAL["mle"], lw=1.3)
            ax.text((a + b) / 2, 0.47, "exact times", ha="center", fontsize=6.4, color=PAL["mle"])
        else:
            ax.add_patch(Rectangle((a, -0.32), b - a, 0.64, facecolor="white", edgecolor=PAL["mps"], hatch="////", lw=0.8, alpha=0.75))
            ax.text((a + b) / 2, 0.47, rf"count $d_{j+1}$", ha="center", fontsize=6.4, color=PAL["mps"])
    ax.annotate("", xy=(10.4, 0), xytext=(-0.1, 0), arrowprops=dict(arrowstyle="-|>", color=PAL["dark"], lw=0.8))
    for j, t in enumerate(ts):
        ax.plot([t, t], [-0.42, 0.42], color=PAL["dark"], lw=0.9)
        ax.text(t, -0.72, labs[j], ha="center", fontsize=7.5)
        if 1 <= j <= 5:
            ax.annotate("", xy=(t, 1.55), xytext=(t, 0.82),
                        arrowprops=dict(arrowstyle="-|>", color=PAL["hb"], lw=1.0))
            ax.text(t + 0.07, 1.22, rf"$R_{j}$", fontsize=7.5, color=PAL["hb"])
    ax.annotate("", xy=(9.6, 1.55), xytext=(9.6, 0.82), arrowprops=dict(arrowstyle="-|>", color=PAL["grey"], lw=1.0))
    ax.text(9.68, 1.22, r"$R_K$ (incl. $C_T$)", fontsize=7.0, color=PAL["grey"])
    ax.text(10.45, -0.08, "time", fontsize=7.5, va="top")
    handles = [Rectangle((0, 0), 1, 1, facecolor=PAL["mle"], alpha=0.25, edgecolor=PAL["mle"]),
               Rectangle((0, 0), 1, 1, facecolor="white", hatch="////", edgecolor=PAL["mps"]),
               Line2D([0], [0], color=PAL["hb"], marker=r"$\uparrow$", lw=0, markersize=8)]
    ax.legend(handles, ["continuous monitoring ($\\delta_j=1$)", "intermittent inspection ($\\delta_j=0$)",
                        "progressive withdrawal at $t_j$"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.55),
              fontsize=7, handlelength=1.6, columnspacing=1.4)
    panel_label(ax, "(a)", x=-0.02, y=0.92)

    # (b) a realisation: number at risk
    ax2 = fig.add_subplot(gs[1])
    mu, sig = CONFIGS["C2"]
    probs = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    ex = [True, False, True, False, True, False]
    des = Design.from_quantiles(mu, sig, probs, ex, [0.05, 0.10, 0.05, 0.10, 0.05])
    rng = np.random.default_rng(4)
    data = simulate_mipt(60, mu, sig, des, rng)
    t = des.t
    n_at = 60
    xs_all, ys_all = [0], [60]
    for j in range(des.K):
        a, b = t[j], t[j + 1]
        col = PAL["mle"] if ex[j] else PAL["mps"]
        ax2.axvspan(a, b, color=col, alpha=0.07 if ex[j] else 0.05, lw=0)
        if ex[j]:
            for x in data["x"][j]:
                xs_all += [x, x]
                ys_all += [n_at, n_at - 1]
                n_at -= 1
                ax2.plot([x], [n_at], "|", color=PAL["mle"], ms=4)
            xs_all += [b]
            ys_all += [n_at]
        else:
            xs_all += [b, b]
            ys_all += [n_at, n_at - data["d"][j]]
            ax2.annotate(rf"$d_{{{j+1}}}={data['d'][j]}$", xy=(b, n_at - data["d"][j] / 2), xytext=(b - 0.5, n_at - data["d"][j] / 2 + 6),
                         fontsize=6.5, color=PAL["mps"], ha="right",
                         arrowprops=dict(arrowstyle="-", color=PAL["mps"], lw=0.5))
            n_at -= data["d"][j]
        if j < des.K - 1 and data["R"][j] > 0:
            xs_all += [b]
            ys_all += [n_at - data["R"][j]]
            ax2.annotate(rf"$R_{{{j+1}}}={data['R'][j]}$", xy=(b, n_at - data["R"][j]), xytext=(b + 0.25, n_at + 4),
                         fontsize=6.5, color=PAL["hb"], arrowprops=dict(arrowstyle="-", color=PAL["hb"], lw=0.5))
            n_at -= data["R"][j]
    ax2.plot(xs_all, ys_all, color=PAL["dark"], lw=1.1, drawstyle="default")
    ax2.text(t[-1] + 0.05, n_at, rf"$R_6={data['R'][-1]}$", fontsize=6.5, color=PAL["grey"], va="center")
    for tt in t[1:]:
        ax2.axvline(tt, color="#BBBBBB", lw=0.5, ls=":")
    ax2.set_xlim(0, t[-1] * 1.08)
    ax2.set_ylim(0, 63)
    ax2.set_xlabel("time")
    ax2.set_ylabel("units at risk")
    ax2.set_title(r"One MIPT-I test ($n=60$, $\mu=2,\ \sigma=0.28$, scheme S1)", fontsize=8)
    panel_label(ax2, "(b)", x=-0.08)
    fig.savefig(f"{FIG}/fig_scheme.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_landscape():
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.8), gridspec_kw=dict(wspace=0.55))
    ax = axs[0]
    mus = np.linspace(1.35, 2.65, 300)
    sigs = np.linspace(0.12, 0.48, 300)
    M, S = np.meshgrid(mus, sigs)
    W = np.minimum(*wdist(M, S, L, U)) / 3.0          # log-scale C*_pk
    C = cpyk(M, S, L, U)
    lev = [0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
    cf = ax.contourf(M, S, np.clip(W, 0.3, 2.2), levels=np.linspace(0.3, 2.2, 20), cmap="viridis", alpha=0.9)
    cs = ax.contour(M, S, W, levels=lev, colors="white", linewidths=0.5)
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=6)
    ax.contour(M, S, W, levels=[1.0], colors=PAL["orange"], linewidths=1.6)
    mstar = 0.5 * (np.log(L) + np.log(U))
    ax.axvline(mstar, color="white", ls="--", lw=0.9)
    ax.text(mstar + 0.03, 0.47, r"$\mu^\ast$", color="white", fontsize=8, va="top")
    for k, (m, s) in CONFIGS.items():
        ax.plot(m, s, "o", ms=5, mfc="white", mec="black", mew=0.8)
        ax.text(m + 0.03, s + 0.012, k, fontsize=7.5, color="white", fontweight="bold")
    cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(r"log-scale $C^\ast_{pk}$", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.grid(False)
    ax.set_title(r"Landscape of $C^\ast_{pk}$; orange: $C_{pyk}=1$", fontsize=7.5)
    panel_label(ax, "(a)", x=-0.17)

    ax = axs[1]
    mus = np.linspace(1.2, 2.8, 600)
    cols = [PAL["mle"], PAL["hb"], PAL["mps"], PAL["hbmed"]]
    for s, c in zip([0.20, 0.25, 0.30, 0.35], cols):
        ax.plot(mus, cpyk(mus, s, L, U), color=c, lw=1.3, label=rf"$\sigma={s}$")
        A, B = arms(mus, s, L, U)
        ax.plot(mus, A, color=c, lw=0.6, ls=":")
        ax.plot(mus, B, color=c, lw=0.6, ls="--")
    ax.axvline(mstar, color=PAL["grey"], lw=0.7, ls="--")
    ax.axhline(1.0, color=PAL["orange"], lw=1.0)
    ax.axhline(1 / (1 - 2 * TAU_3S), color=PAL["grey"], lw=0.6, ls="-.")
    ax.text(1.22, 1 / (1 - 2 * TAU_3S) + 0.0004, r"upper bound $1/(1-2\tau)$", fontsize=6.5, color=PAL["grey"], va="bottom")
    ax.set_ylim(0.955, 1.0045)
    ax.set_xlim(1.2, 2.8)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$C_{pyk}$")
    ax.set_title(r"$C_{pyk}$ (solid) and its two arms", fontsize=7.5)
    ax.legend(loc="lower center", ncol=2, fontsize=6.8)
    panel_label(ax, "(b)", x=-0.3)
    fig.savefig(f"{FIG}/fig_landscape.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_saturation():
    s = json.load(open(f"{RES}/design_saturation.json"))
    ns = np.array(s["n"])
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.45), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    ax.plot(ns, [r["exact"] for r in s["A"]], "o-", color=PAL["mle"], ms=3.5, lw=1.2, label="Plan A: withdrawals at inspections")
    ax.plot(ns, [r["exact"] for r in s["B"]], "s-", color=PAL["mps"], ms=3.5, lw=1.2, label="Plan B: withdrawal after each failure")
    ax.plot(ns, s["boundB"], "--", color=PAL["mps"], lw=0.8, label="deterministic bound (Plan B)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"sample size $n$")
    ax.set_ylabel("mean number of exact failures")
    ax.legend(fontsize=6.6, loc="upper left")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    panel_label(ax, "(a)", x=-0.17)
    ax = axs[1]
    ax.plot(ns, [r["n_trinv"] for r in s["A"]], "o-", color=PAL["mle"], ms=3.5, lw=1.2, label="Plan A")
    ax.plot(ns, [r["n_trinv"] for r in s["B"]], "s-", color=PAL["mps"], ms=3.5, lw=1.2, label="Plan B")
    ax.axhline(s["IA_trinv"], color=PAL["mle"], lw=0.8, ls="--", label=r"Theorem 1: $\mathrm{tr}\,\mathcal{I}(\theta;\xi)^{-1}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_yticks([0.15, 0.3, 0.6, 1.2, 2.4])
    ax.set_yticklabels(["0.15", "0.3", "0.6", "1.2", "2.4"])
    ax.set_xlabel(r"sample size $n$")
    ax.set_ylabel(r"$n\cdot\mathrm{tr}\,\{\mathcal{I}_n(\theta)^{-1}\}$")
    ax.legend(fontsize=6.6, loc="upper left")
    panel_label(ax, "(b)", x=-0.17)
    fig.savefig(f"{FIG}/fig_saturation.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_design():
    r = json.load(open(f"{RES}/design_patterns.json"))
    famstyle = {"EGEGEG": (PAL["mle"], "-", "MIPT-I (EGEGEG)"), "GEGEGE": (PAL["sky"], "-", "MIPT-I (GEGEGE)"),
                "EEEEEE": (PAL["mps"], "--", "all exact (PT-I)"), "GGGGGG": (PAL["hbmed"], "--", "all grouped (PIT-I)")}
    fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 2.35), sharey=True, gridspec_kw=dict(wspace=0.12))
    for k, cfg in enumerate(["C1", "C2", "C3"]):
        ax = axs[k]
        cms = np.array(r[cfg]["cms"])
        best = np.array([d["val"] for d in r[cfg]["C"]])
        for fam, (c, ls, lab) in famstyle.items():
            ax.plot(cms, np.array(r[cfg]["families"][fam]) / best, color=c, ls=ls, lw=1.2, label=lab)
        ax.axhline(1.0, color=PAL["dark"], lw=1.6, label="optimal generalised MIPT-I")
        ax.set_ylim(0.97, 1.55)
        ax.set_xlim(0, 40)
        ax.set_xlabel(r"monitoring cost rate $c_m$")
        arm = {"C1": "lower arm binding", "C2": "near the kink", "C3": "upper arm binding"}[cfg]
        ax.set_title(f"{cfg}: {arm}", fontsize=8)
        panel_label(ax, "(" + "abc"[k] + ")", x=-0.1, y=1.06)
    axs[0].set_ylabel(r"$\Psi_C(\xi^{\rm fam})/\Psi_C(\xi^{\rm opt})$")
    axs[1].legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=5, columnspacing=1.0, handlelength=1.8)
    fig.savefig(f"{FIG}/fig_design_ce.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.6), gridspec_kw=dict(wspace=0.32, width_ratios=[1.0, 1.15]))
    ax = axs[0]
    for k, cfg in enumerate(["C1", "C2", "C3"]):
        for d in r[cfg]["C"]:
            for j, ch in enumerate(d["pattern"]):
                ax.add_patch(Rectangle((d["cm"] - 1, k * 7 + j), 2, 1,
                                       facecolor=PAL["mle"] if ch == "E" else "#F4D6C4", edgecolor="white", lw=0.4))
    ax.set_xlim(-1, 41)
    ax.set_ylim(-0.5, 20.5)
    ax.set_yticks([0.5, 5.5, 7.5, 12.5, 14.5, 19.5])
    ax.set_yticklabels(["C1: $j=1$", "$j=6$", "C2: $j=1$", "$j=6$", "C3: $j=1$", "$j=6$"], fontsize=6.5)
    ax.set_xlabel(r"monitoring cost rate $c_m$")
    ax.set_title(r"$\Psi_C$-optimal monitoring pattern $\delta$", fontsize=8)
    ax.grid(False)
    ax.legend([Rectangle((0, 0), 1, 1, facecolor=PAL["mle"]), Rectangle((0, 0), 1, 1, facecolor="#F4D6C4")],
              ["continuous (E)", "intermittent (G)"], fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    panel_label(ax, "(a)", x=-0.3, y=1.04)
    ax = axs[1]
    cfg = "C3"
    cl = r[cfg]["cloud"]
    cost = np.array([c[0] for c in cl])
    av = np.array([c[1] for c in cl])
    pat = np.array([c[2] for c in cl])
    ax.scatter(cost, av, s=0.8, color="#D0D0D0", rasterized=True, label="other patterns")
    for fam, (c, ls, lab) in famstyle.items():
        m = pat == fam
        ax.scatter(cost[m], av[m], s=1.2, color=c, rasterized=True, alpha=0.6, label=lab)
    o = np.argsort(cost)
    fc, fv, bv = [], [], np.inf
    for i in o:
        if av[i] < bv:
            bv = av[i]
            fc.append(cost[i])
            fv.append(av[i])
    ax.plot(fc, fv, color=PAL["dark"], lw=1.3, label="Pareto frontier")
    ax.set_yscale("log")
    ax.set_xlabel(r"expected test cost $\mathcal{C}(\xi)$ ($c_m=20$)")
    ax.set_ylabel(r"$\mathrm{AVar}(\widehat w_U)$, $n=50$")
    ax.set_title("C3: cost versus precision", fontsize=8)
    ax.set_xlim(np.percentile(cost, 0.5) * 0.98, np.percentile(cost, 99.0))
    ax.set_ylim(np.percentile(av, 0.1) * 0.9, np.percentile(av, 95))
    lg = ax.legend(fontsize=6.0, loc="upper right", markerscale=5, handletextpad=0.3)
    panel_label(ax, "(b)", x=-0.2, y=1.04)
    fig.savefig(f"{FIG}/fig_design_pattern.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_scheme()
    fig_landscape()
    fig_saturation()
    fig_design()
    print("ok")
