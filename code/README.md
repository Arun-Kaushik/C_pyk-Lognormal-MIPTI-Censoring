# Reproduction code: C_pyk for lognormal processes under generalized MIPT-I censoring

Python 3 (numpy, scipy, matplotlib). Every number, table and figure in the manuscript is produced by these scripts.

| File | Purpose |
|---|---|
| `gpci_core.py` | Index and arms; MIPT-I design and simulator; log-likelihood, score, observed information; MLE; conditional-spacings MPS; transformed-Wald and adaptive kink-robust (AKR) intervals; vectorised hierarchical-Bayes MH-within-Gibbs sampler; split R-hat and ESS; closed-form expected Fisher information (Theorem 1); expected total time on test and withdrawals |
| `test_core.py` | Sanity checks: analytic vs numerical score, complete-sample information, Monte Carlo check of Theorem 1, MPS vs MLE, MCMC |
| `sim.py` | Monte Carlo study (3 configurations x 2 schemes x 4 sample sizes x 2000 replications) |
| `sim_report.py` | Simulation tables (`tex/tables/tab_point.tex`, `tab_interval.tex`, `tab_param.tex`, `tab_param_ci.tex`) and figures (`fig_sim_rmse`, `fig_sim_cp`, `fig_kink`) |
| `design.py` | Information-saturation study (Proposition 5, Figure 3) and cost model |
| `design_patterns.py` | Cost-efficiency search over all 2^6 monitoring patterns (Section 7) |
| `realdata.py` | Ball-bearing and guinea-pig analyses (goodness of fit, MIPT-I samples, MLE/MPS/HB, bootstrap, prior sensitivity, planning) |
| `realdata_report.py` | Real-data tables and figures |
| `fig_theory.py`, `figstyle.py` | Schematic, landscape, saturation and design figures |

Run order: `python3 sim.py`, `python3 design.py`, `python3 design_patterns.py`, `python3 realdata.py`, then the report and figure scripts. Seeds are fixed, so the output is reproducible.
