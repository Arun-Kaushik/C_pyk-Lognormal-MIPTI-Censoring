import numpy as np, time
from gpci_core import *

rng = np.random.default_rng(7)
mu, sig = 2.0, 0.28
L, U = 3.0, 18.0
probs = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
exact = [True, False, True, False, True, False]
des = Design.from_quantiles(mu, sig, probs, exact, [0.05, 0.10, 0.05, 0.10, 0.05])

# 1. score vs numerical gradient
data = simulate_mipt(80, mu, sig, des, rng)
D = prep(data)
for th in [(2.0, 0.28), (1.8, 0.4), (2.3, 0.2)]:
    g = score(*th, D)
    h = 1e-6
    gn = np.array([(loglik(th[0] + h, th[1], D) - loglik(th[0] - h, th[1], D)) / (2 * h),
                   (loglik(th[0], th[1] + h, D) - loglik(th[0], th[1] - h, D)) / (2 * h)])
    print("score check", g, gn)
# per-failure design score check
desB = Design.from_quantiles(mu, sig, probs, exact, [0.05, 0.10, 0.05, 0.10, 0.05], pf=0.1)
DB = prep(simulate_mipt(80, mu, sig, desB, rng))
th = (1.9, 0.33)
g = score(*th, DB)
h = 1e-6
gn = np.array([(loglik(th[0] + h, th[1], DB) - loglik(th[0] - h, th[1], DB)) / (2 * h),
               (loglik(th[0], th[1] + h, DB) - loglik(th[0], th[1] - h, DB)) / (2 * h)])
print("score check pf", g, gn)

# 2. complete sample info
full = Design(np.r_[0, 1e9], [True], [])
print("complete info", design_info(mu, sig, full), "expected", np.diag([1 / sig**2, 2 / sig**2]))
grp = Design(np.r_[0, np.exp(mu + sig * np.array([-1, 0, 1, 1e3]))], [False] * 4, [0, 0, 0])
print("grouped info", design_info(mu, sig, grp))

# 3. MC check of info
I = design_info(mu, sig, des)
n = 200
est = []
for _ in range(2000):
    Dk = prep(simulate_mipt(n, mu, sig, des, rng))
    m, s, _ = fit_mle(Dk)
    est.append((m, s))
est = np.array(est)
print("MC n*cov", n * np.cov(est.T), "\n inv info", np.linalg.inv(I))
print("bias", est.mean(0) - [mu, sig])

# 4. MPS
m, s, _ = fit_mle(D)
mp, sp_, _ = fit_mps(D, (m, s))
print("MLE", m, s, "MPS", mp, sp_)
print("obs info", obs_info(m, s, D), "mps info", mps_info(mp, sp_, D))
V = np.linalg.inv(obs_info(m, s, D))
print(intervals_classical(m, s, V, L, U, 80))
print("crit rho=0", simultaneous_crit(0.0), "rho=.9", simultaneous_crit(0.9))

# 5. HB sanity
Ds = [prep(simulate_mipt(80, mu, sig, des, rng)) for _ in range(200)]
S = stack_suff(Ds)
init = np.array([fit_mle(Dk)[:2] for Dk in Ds])
t0 = time.time()
dr = hb_mcmc(S, init[:, 0], init[:, 1] ** 2, rng)
print("mcmc time", time.time() - t0)
hs = hb_summaries(dr, L, U, 80)
print("HB mu mean vs mle", np.mean(hs["mu_mean"] - init[:, 0]), "sig", np.mean(hs["sig_mean"] - init[:, 1]))
print("ess mu chain0", ess(dr["mu"][:, :1]), "rhat", split_rhat(dr["mu"][:, :4]))
print("true C", cpyk(mu, sig, L, U), "HB mean", hs["C_mean"][:5])
