"""Summaries, LaTeX tables and figures for the Monte Carlo study."""
import os, json
import numpy as np
from scipy.stats import norm
from figstyle import setup, PAL, TEXTW, panel_label, plt
from gpci_core import *

setup()
RES = "/home/dev/gpci_paper/results"
FIG = "/home/dev/gpci_paper/figs"
TAB = "/home/dev/gpci_paper/tex/tables"
os.makedirs(TAB, exist_ok=True)
CFGS = ["C1", "C2", "C3"]
SCH = ["S1", "S2"]
NS = [30, 50, 100, 200]
L, U = 3.0, 18.0


def load(c, s, n):
    f = f"{RES}/sim_{c}_{s}_n{n}.npz"
    return dict(np.load(f)) if os.path.exists(f) else None


def cover(ci, true):
    ci = np.asarray(ci)
    return np.mean((ci[:, 0] <= true + 1e-12) & (true - 1e-12 <= ci[:, 1])), np.mean(ci[:, 1] - ci[:, 0])


def summarise():
    S = {}
    for c in CFGS:
        for s in SCH:
            for n in NS:
                d = load(c, s, n)
                if d is None:
                    continue
                mu, sig, C = d["true"]
                ok = d["ok"].astype(bool)
                r = dict(ok=float(ok.mean()))
                cnt = d["cnt"].mean(0)
                r["cnt"] = cnt.tolist()
                est = {"MLE": d["C_mle"], "MPS": d["C_mps"], "HB": d["hb_C"], "HBmed": d["hb_Cmed"]}
                for k, v in est.items():
                    r[f"bias_{k}"] = float(np.mean(v - C))
                    r[f"rmse_{k}"] = float(np.sqrt(np.mean((v - C) ** 2)))
                for k, ci in {"Wald": d["wald"], "AKR": d["akr"], "AKRmps": d["akr_mps"],
                              "ET": d["hb_et"], "HBAKR": d["hb_akr"]}.items():
                    cp, aw = cover(ci, C)
                    r[f"cp_{k}"] = float(cp)
                    r[f"aw_{k}"] = float(aw)
                pe = {"MLE": d["mle"], "MPS": d["mps"], "HB": np.c_[d["hb_mu"], d["hb_sig"]]}
                for k, v in pe.items():
                    r[f"mu_bias_{k}"] = float(np.mean(v[:, 0] - mu))
                    r[f"mu_mse_{k}"] = float(np.mean((v[:, 0] - mu) ** 2))
                    r[f"sig_bias_{k}"] = float(np.mean(v[:, 1] - sig))
                    r[f"sig_mse_{k}"] = float(np.mean((v[:, 1] - sig) ** 2))
                r["cp_mu_ACI"], r["aw_mu_ACI"] = cover(d["mu_ci"], mu)
                r["cp_sig_ACI"], r["aw_sig_ACI"] = cover(d["sig_ci"], sig)
                r["cp_mu_HB"], r["aw_mu_HB"] = cover(d["hb_mu_ci"], mu)
                r["cp_sig_HB"], r["aw_sig_HB"] = cover(d["hb_sig_ci"], sig)
                r["sep_rate"] = float(np.mean(d["kappa"] > np.sqrt(np.log(n))))
                r["true"] = [float(mu), float(sig), float(C)]
                S[(c, s, n)] = r
    return S


def f3(x, d=2):
    if round(x, d) == 0:
        x = 0.0
    return f"{x:.{d}f}".replace("-", "$-$")


def tables(S):
    # point estimation of C_pyk
    lines = []
    for c in CFGS:
        for s in SCH:
            for i, n in enumerate(NS):
                r = S.get((c, s, n))
                if r is None:
                    continue
                lead = ""
                if i == 0:
                    lead = rf"\multirow{{4}}{{*}}{{{c}}} & \multirow{{4}}{{*}}{{{s}}}"
                else:
                    lead = " & "
                vals = []
                for k in ["MLE", "MPS", "HB", "HBmed"]:
                    vals.append(f3(1e3 * r[f"bias_{k}"]))
                for k in ["MLE", "MPS", "HB", "HBmed"]:
                    vals.append(f3(1e3 * r[f"rmse_{k}"]))
                # bold best RMSE
                rm = [r[f"rmse_{k}"] for k in ["MLE", "MPS", "HB", "HBmed"]]
                j = int(np.argmin(rm))
                vals[4 + j] = r"\textbf{" + vals[4 + j] + "}"
                lines.append(f"{lead} & {n} & " + " & ".join(vals) + r" \\")
            lines.append(r"\midrule" if not (c == "C3" and s == "S2") else "")
    open(f"{TAB}/tab_point.tex", "w").write("\n".join(lines))
    # interval estimation of C_pyk
    lines = []
    keys = ["Wald", "AKR", "AKRmps", "ET", "HBAKR"]
    for c in CFGS:
        for s in SCH:
            for i, n in enumerate(NS):
                r = S.get((c, s, n))
                if r is None:
                    continue
                lead = rf"\multirow{{4}}{{*}}{{{c}}} & \multirow{{4}}{{*}}{{{s}}}" if i == 0 else " & "
                vals = []
                for k in keys:
                    cp = r[f"cp_{k}"]
                    txt = f"{cp:.3f}"
                    if cp < 0.9304:          # below 0.95 - 3 MC s.e. (B=2000)
                        txt = r"\underline{" + txt + "}"
                    vals.append(f3(1e3 * r[f"aw_{k}"], 2) + " & " + txt)
                lines.append(f"{lead} & {n} & " + " & ".join(vals) + r" \\")
            lines.append(r"\midrule" if not (c == "C3" and s == "S2") else "")
    open(f"{TAB}/tab_interval.tex", "w").write("\n".join(lines))
    # parameters (appendix)
    lines = []
    for c in CFGS:
        for s in SCH:
            for i, n in enumerate(NS):
                r = S.get((c, s, n))
                if r is None:
                    continue
                lead = rf"\multirow{{4}}{{*}}{{{c}}} & \multirow{{4}}{{*}}{{{s}}}" if i == 0 else " & "
                vals = []
                for p in ["mu", "sig"]:
                    for k in ["MLE", "MPS", "HB"]:
                        vals.append(f3(1e3 * r[f"{p}_bias_{k}"], 1))
                    for k in ["MLE", "MPS", "HB"]:
                        vals.append(f3(1e3 * r[f"{p}_mse_{k}"], 2))
                lines.append(f"{lead} & {n} & " + " & ".join(vals) + r" \\")
            lines.append(r"\midrule" if not (c == "C3" and s == "S2") else "")
    open(f"{TAB}/tab_param.tex", "w").write("\n".join(lines))
    # parameter interval coverage + censoring summary (appendix)
    lines = []
    for c in CFGS:
        for s in SCH:
            for i, n in enumerate(NS):
                r = S.get((c, s, n))
                if r is None:
                    continue
                lead = rf"\multirow{{4}}{{*}}{{{c}}} & \multirow{{4}}{{*}}{{{s}}}" if i == 0 else " & "
                cnt = r["cnt"]
                vals = [f"{cnt[0]:.1f}", f"{cnt[1]:.1f}", f"{cnt[2]:.1f}", f"{cnt[3]:.1f}"]
                for k in ["mu_ACI", "mu_HB", "sig_ACI", "sig_HB"]:
                    vals.append(f3(r[f"aw_{k}"], 3) + " & " + f"{r[f'cp_{k}']:.3f}")
                lines.append(f"{lead} & {n} & " + " & ".join(vals) + r" \\")
            lines.append(r"\midrule" if not (c == "C3" and s == "S2") else "")
    open(f"{TAB}/tab_param_ci.tex", "w").write("\n".join(lines))


def fig_rmse(S):
    fig, axs = plt.subplots(2, 3, figsize=(TEXTW, 3.9), sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.3))
    st = {"MLE": (PAL["mle"], "o", "MLE"), "MPS": (PAL["mps"], "s", "MPS"), "HB": (PAL["hb"], "^", "HB (posterior mean)"),
          "HBmed": (PAL["hbmed"], "v", "HB (posterior median)")}
    for i, s in enumerate(SCH):
        for j, c in enumerate(CFGS):
            ax = axs[i, j]
            for k, (col, mk, lab) in st.items():
                y = [1e3 * S[(c, s, n)][f"rmse_{k}"] for n in NS if (c, s, n) in S]
                ax.plot(NS[:len(y)], y, marker=mk, color=col, ms=3.5, lw=1.1, label=lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(NS)
            ax.set_xticklabels([str(v) for v in NS])
            ax.minorticks_off()
            if (c, s, NS[0]) in S:
                C = S[(c, s, NS[0])]["true"][2]
                ax.set_title(rf"{c} ($C_{{pyk}}={C:.4f}$), {s}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel(r"RMSE$\times 10^3$")
            if i == 1:
                ax.set_xlabel(r"$n$")
            from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
    axs[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=7)
    fig.savefig(f"{FIG}/fig_sim_rmse.pdf")
    plt.close(fig)


def fig_cp(S):
    fig, axs = plt.subplots(2, 3, figsize=(TEXTW, 3.9), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.35, wspace=0.12))
    st = {"Wald": (PAL["mle"], "o", "--", "MLE transformed Wald"), "AKR": (PAL["mle"], "o", "-", "MLE AKR"),
          "AKRmps": (PAL["mps"], "s", "-", "MPS AKR"), "ET": (PAL["hb"], "^", "--", "HB equal-tailed"),
          "HBAKR": (PAL["hb"], "^", "-", "HB AKR")}
    se = np.sqrt(0.95 * 0.05 / 2000)
    for i, s in enumerate(SCH):
        for j, c in enumerate(CFGS):
            ax = axs[i, j]
            ax.axhspan(0.95 - 1.96 * se, 0.95 + 1.96 * se, color="#EDEDED", lw=0)
            ax.axhline(0.95, color=PAL["grey"], lw=0.7)
            for k, (col, mk, ls, lab) in st.items():
                y = [S[(c, s, n)][f"cp_{k}"] for n in NS if (c, s, n) in S]
                ax.plot(NS[:len(y)], y, marker=mk, color=col, ls=ls, ms=3.2, lw=1.0, label=lab)
            ax.set_xscale("log")
            ax.set_xticks(NS)
            ax.set_xticklabels([str(v) for v in NS])
            ax.minorticks_off()
            ax.set_ylim(0.80, 1.0)
            ax.set_title(f"{c}, {s}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("coverage probability")
            if i == 1:
                ax.set_xlabel(r"$n$")
    axs[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=5, fontsize=6.6, columnspacing=0.9)
    fig.savefig(f"{FIG}/fig_sim_cp.pdf")
    plt.close(fig)


def fig_kink():
    """Finite-sample distribution of sqrt(n)(w_hat_min - w_min) vs the Theorem-3 limits."""
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, 2.4), gridspec_kw=dict(wspace=0.28))
    probs = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    ex = [True, False, True, False, True, False]
    rng = np.random.default_rng(5)
    for ax, c, lab in [(axs[0], "C2", "(a)"), (axs[1], "C3", "(b)")]:
        mu, sig = {"C2": (2.0, 0.28), "C3": (2.3, 0.2)}[c]
        n = 200
        d = load(c, "S1", n)
        if d is None:
            continue
        m = d["mle"]
        w_hat = np.minimum(*wdist(m[:, 0], m[:, 1], L, U))
        w0 = min(wdist(mu, sig, L, U))
        T = np.sqrt(n) * (w_hat - w0)
        des = Design.from_quantiles(mu, sig, probs, ex, [0.05, 0.10, 0.05, 0.10, 0.05])
        Sig = np.linalg.inv(design_info(mu, sig, des))
        wU, wL, gU, gL = arm_grads(mu, sig, L, U)
        G = np.c_[gU, gL]
        Om = G.T @ Sig @ G
        Z = rng.multivariate_normal([0, 0], Om, 400000)
        dlt = np.sqrt(n) * (wU - wL)       # local drift between the arms
        lim = np.minimum(Z[:, 0] + max(dlt, 0) * 0 + (wU - w0) * np.sqrt(n), Z[:, 1] + (wL - w0) * np.sqrt(n))
        act = 0 if wU < wL else 1
        ax.hist(T, bins=60, density=True, color=PAL["sky"], alpha=0.55, label=r"simulated MLE ($n=200$, S1)")
        xs = np.linspace(np.percentile(T, 0.2), np.percentile(T, 99.8), 300)
        from scipy.stats import gaussian_kde
        kd = gaussian_kde(lim[:100000])
        ax.plot(xs, kd(xs), color=PAL["dark"], lw=1.4, label=f"Theorem 2 limit (mean {lim.mean():.2f})".replace("-0.00", "0.00"))
        ax.plot(xs, norm.pdf(xs, 0, np.sqrt(Om[act, act])), color=PAL["mps"], lw=1.1, ls="--",
                label="naive delta-method normal")
        ax.axvline(np.mean(T), color=PAL["mle"], lw=1.0, ls=":", label=f"simulated mean {np.mean(T):.2f}")
        ax.set_xlabel(r"$\sqrt{n}\,(\widehat w_{\min}-w_{\min})$")
        ax.set_ylabel("density")
        ttl = r"C2: arms almost equal ($w_U\approx w_L$)" if c == "C2" else r"C3: arms separated ($w_U<w_L$)"
        ax.set_title(ttl, fontsize=8)
        ax.legend(fontsize=6.2, loc="upper left")
        panel_label(ax, lab, x=-0.16)
    fig.savefig(f"{FIG}/fig_kink.pdf")
    plt.close(fig)


if __name__ == "__main__":
    S = summarise()
    json.dump({"|".join(map(str, k)): v for k, v in S.items()}, open(f"{RES}/sim_summary.json", "w"), indent=1)
    tables(S)
    fig_rmse(S)
    fig_cp(S)
    fig_kink()
    for k, v in sorted(S.items()):
        print(k, {kk: round(vv, 5) for kk, vv in v.items() if kk.startswith(("bias", "rmse", "cp_"))})
