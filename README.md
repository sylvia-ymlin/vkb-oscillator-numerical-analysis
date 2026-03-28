# Scientific Computing, Bridging Course, 2023 — Miniproject 2: Genetic Oscillator

This report investigates the 9-species Vilar–Kueh–Barkai (VKB) circadian oscillator [1] using stiff ODE solvers and the Gillespie SSA. Parts A–D reproduce the standard assignment tasks; the extension evaluates the validity of the paper's two-dimensional QSSA reduction near the oscillatory transition boundary.

Core outcomes of the extension:
- The QSSA shifts the Hopf bifurcation threshold upward by approximately 22%, with $s_{R,\text{Full}} \approx 0.096$ and $s_{R,\text{Reduced}} \approx 0.117$;
- At $s_R = 0.088$, the full model exhibits subcritical bistability — a stable limit cycle coexists with a stable equilibrium — while this coexistence is absent in the reduced model;
- A 50×50 basin scan over the $(R,C)$ plane (Figure 12) shows the equilibrium basin as an isolated region within a dominant limit-cycle basin;
- A normalized distance-to-bifurcation metric $d_\text{bif}$ is defined to assess the reliability of the QSSA approximation.

Figures 5, 7, and 12 involve heavy computation and require the full pipeline. Reproducibility instructions are provided in [Appendix 2](#appendix-2-quick-start-and-reproducibility).

---

## 1. Background

The VKB model [1] describes a circadian clock built from an activator (A) and a repressor (R), governed by a 16-reaction mass-action network with 9 species. Vilar et al. (2002) show that the oscillator is noise-resistant using both a deterministic ODE and a Gillespie SSA. The paper also derives a 2-variable QSSA reduction in $(R,C)$, validated mainly at $s_R = 0.2$.

Numerical investigations are conducted at two characteristic parameter settings: $s_R = 0.2$ (sustained deterministic oscillations) and $s_R = 0.03$ (stable ODE equilibrium with noise-induced stochastic oscillations). Based on these baseline results, the extension evaluates whether the low-dimensional QSSA preserves the qualitative dynamical properties of the full system near the transition between steady and oscillatory regimes.

---

## 2. Assignment tasks

### Part A: Deterministic simulation and solver comparison

The 9D ODE is integrated at $s_R = 0.2$ for 400 h using BDF with analytical Jacobian. The trajectory reproduces the sustained ~24 h oscillations reported in Vilar et al. (Figure 1).

<div align="center">
<img src="figures/assignment_a_ode.png" width="500" alt="Deterministic ODE">

**Figure 1.** ODE trajectory (400 h) for activator A and repressor R at $s_R = 0.2$.
</div>

Three numerical solvers (RK45, BDF, Radau) are benchmarked over a 48 h simulation with internal step sizes recorded (Figure 2). During steep dynamical phases, RK45 adopts step sizes around $10^{-3}$ h, resulting in approximately $10^6$ function evaluations for a long-time integration. BDF and Radau allow step sizes up to $1\,\mathrm{h}$ during slow phases, consistent with the stiffness of the system.

<div align="center">
<img src="figures/solver_step_sizes.png" width="500" alt="Solver step sizes">

**Figure 2.** Internal step sizes $\Delta t$ for RK45, BDF, and Radau over 48 h.
</div>

In the 48 h benchmark, BDF completes in ~0.12 s and Radau in ~0.21 s (hardware-dependent). Absolute errors against a tight RK45 reference ($\texttt{rtol}=10^{-12}$, $\texttt{atol}=10^{-14}$) are $10^{-4}$–$10^{-6}$ for BDF and $10^{-8}$–$10^{-14}$ for Radau (Figure 3). BDF is used for long runs; Radau is used where Jacobian eigenvalues must be computed accurately, such as near a bifurcation point.

<div align="center">
<img src="figures/solver_errors.png" width="500" alt="Solver errors">

**Figure 3.** Trajectory error for BDF and Radau vs. high-precision RK45 reference ($\texttt{rtol}=10^{-12}$, $\texttt{atol}=10^{-14}$).
</div>

### Part B: Stochastic simulation at $s_R = 0.2$

The SSA is run using GillesPy2 with the same 16 reactions. A single 400 h trajectory (Figure 4) oscillates in phase with the ODE limit cycle but shows stochastic phase drift and amplitude variation.

<div align="center">
<img src="figures/assignment_b_ssa.png" width="500" alt="Stochastic SSA">

**Figure 4.** Single SSA trajectory (400 h), A and R at $s_R = 0.2$.
</div>

To quantify the variability, 50 independent runs are collected and periods and amplitudes extracted by peak detection (Figure 5; regenerate with `python main.py stats` or `python main.py all --full`). The period distribution gives $\mu_T \approx 24.3 \pm 0.4\,\mathrm{h}$ for the run shown, consistent with the ODE period.

<div align="center">
<img src="figures/ssa_stats_sR_0.2.png" width="500" alt="SSA Statistics sR 0.2">

**Figure 5.** 50 SSA trajectories at $s_R = 0.2$; period and amplitude distributions.
</div>

### Part C: Noise-induced oscillations at $s_R = 0.03$

At $s_R = 0.03$, the deterministic ODE settles to a stable equilibrium while SSA trajectories remain oscillatory (Figure 6). All eigenvalues of the Jacobian at the equilibrium have negative real parts (Figure 8), verifying linear stability of the deterministic fixed point. The oscillatory behavior in SSA is driven purely by stochastic fluctuations.

<div align="center">
<img src="figures/assignment_c_noise.png" width="500" alt="Noise-Induced Oscillations">

**Figure 6.** ODE vs SSA at $s_R = 0.03$. The ODE decays to steady state; SSA continues to oscillate.
</div>

The 50-run statistics at $s_R = 0.03$ give $\mu_T \approx 110 \pm 16\,\mathrm{h}$ (Figure 7) — a much wider distribution than at $s_R = 0.2$, reflecting irregular, noise-driven excursions rather than a stable limit cycle.

<div align="center">
<img src="figures/ssa_stats_sR_0.03.png" width="500" alt="SSA Statistics sR 0.03">

**Figure 7.** 50 SSA trajectories at $s_R = 0.03$; period distribution is much wider than at $s_R = 0.2$.
</div>

<div align="center">
<img src="figures/stiffness_comparison_part_c.png" width="500" alt="Stiffness comparison Part C">

**Figure 8.** Jacobian eigenvalue spectra along trajectories at $s_R = 0.03$ and $s_R = 0.2$.
</div>

### Part D: When to use ODE vs SSA

At $s_R = 0.2$, high molecular concentrations ensure the ODE agrees qualitatively with SSA results. At $s_R = 0.03$, the ODE fails to capture oscillatory dynamics that only emerge in SSA. This discrepancy motivates further examination of whether the QSSA reduction remains valid near the dynamical transition boundary.

---

## 3. Extension: QSSA reduction and validity

The paper's 2D QSSA reduces the 9D system to $(R, C)$ by assuming the activator species reach quasi-steady state on a fast timescale. This extension tests the accuracy of that approximation near the oscillatory transition.

**Step 1 — Hopf threshold**

The Hopf bifurcation point is located for both models by scanning $s_R$ and finding where $\max \mathrm{Re}(\lambda) = 0$ at the equilibrium. For the full model, a 7D sub-Jacobian is used, enforcing $D_A + D_A' = 1$ and $D_R + D_R' = 1$ to remove two conservation-law zero eigenvalues. Near the crossing, direct root-finding is ill-conditioned, so continuation in $s_R$ is used.

The QSSA shifts the Hopf threshold upward by ~22%: $s_{R,\text{Full}} \approx 0.096$ vs. $s_{R,\text{Reduced}} \approx 0.117$ (Figure 9). In the grey band $[0.096, 0.117]$, the full model oscillates while the reduced model predicts a stable equilibrium.

<div align="center">
<img src="figures/bifurcation_diagram.png" width="500" alt="Bifurcation diagram">

**Figure 9.** $\max \mathrm{Re}(\lambda)$ vs $s_R$ for both models. Grey band $[0.096, 0.117]$: region where the two models disagree on stability.
</div>

**Step 2 — Subcritical bistability at $s_R = 0.088$**

At $s_R = 0.088$, the full model equilibrium has $\max \mathrm{Re}(\lambda) \approx -0.036$, indicating linear stability. To probe global dynamics, 72 initial conditions are integrated: one biological IC from the reference, 64 perturbed equilibrium states at four amplitude scales (16 samples per scale, RNG seed 42), and 7 axis-aligned perturbations. Of these, 39 trajectories converge to the stable limit cycle and 33 to the equilibrium (total: 72), demonstrating coexistence of two attractors. This result implies a subcritical Hopf bifurcation in the full model. The reduced model shows no such coexistence — all tested initial conditions converge to a single attractor.

**Step 3 — Trajectory comparison**

<div align="center">
<img src="figures/reduction_comparison.png" width="500" alt="Reduction comparison">

**Figure 10.** Full vs reduced model trajectories from the same initial condition at $s_R = 0.088$. Full model → limit cycle; reduced model → equilibrium.
</div>

**Step 4 — Basin of attraction**

Figure 11 overlays the $25\times 25$ reduced-model phase flow with the 72-IC full-model probe results on the $(R,C)$ plane.

<div align="center">
<img src="figures/basin_comparison_sR_0.088.png" width="500" alt="Basin comparison">

**Figure 11.** Phase-space comparison at $s_R = 0.088$: reduced $25\times 25$ flow (background) and 72 full-model ICs (markers), colored by attractor.
</div>

For a global view, a 50×50 grid of initial conditions is scanned on a large $(R,C)$ slice. Each point is lifted to 9D via QSSA and integrated to convergence — 2500 independent ODE solves, cached in `materials/basin_scan_sR_0.088_50x50.pkl`. The result shows the equilibrium basin as an isolated region inside a dominant limit-cycle basin (Figure 12).

<div align="center">
<img src="figures/basin_grid_sR_0.088.png" width="500" alt="Basin grid scan">

**Figure 12.** Global 50×50 basin scan of the full model at $s_R = 0.088$. Hatching distinguishes the limit-cycle basin (/////) from the equilibrium basin (\\\\). Replot from cache: `python main.py basin-grid --plot-only`.
</div>

**Step 5 — Validity heuristic**

A normalized metric $d_\text{bif} = |s_R - s_{R,\text{Hopf}}| / s_{R,\text{Hopf}}$ is defined to quantify the distance to the Hopf bifurcation (Figure 13). Smaller $d_\text{bif}$ values correspond to parameter regions where the QSSA deviates from the full system. For subcritical bifurcations, the unreliable parameter range extends below the full-model Hopf point, even when the equilibrium is linearly stable.

<div align="center">
<img src="figures/qssa_validity_criterion.png" width="500" alt="QSSA validity criterion">

**Figure 13.** $d_\text{bif}$ vs $s_R$ for the full and reduced models. The shaded region marks where the QSSA reduction is unreliable.
</div>

---

## 4. Conclusion

Solver benchmarking shows BDF is suitable for long-time integration of this stiff biochemical system, while Radau provides higher accuracy for Jacobian eigenvalue computation near bifurcation points.

Comparisons between ODE and SSA establish their applicable regimes: the mass-action ODE reliably describes dynamics at $s_R = 0.2$ with high molecule counts, whereas SSA is required to capture noise-induced oscillations at $s_R = 0.03$.

The 2D QSSA agrees with the full 9D model well away from the bifurcation, but introduces systematic errors near the oscillatory transition. It shifts the Hopf threshold upward by approximately 22% and eliminates the subcritical bistability present in the full system. The proposed $d_\text{bif}$ metric provides a practical criterion for assessing QSSA reliability; the subcritical nature of the bifurcation further narrows the valid parameter range of the reduced model.

---

## Appendix

### Appendix Table 1: 16-reaction network

The following reactions are shared by the 9-variable ODE and the Gillespie SSA.

<div id="appendix-table-1" align="center">

| Reaction                                   | Propensity / Rate  | Description               |
| :----------------------------------------- | :----------------- | :------------------------ |
| $A + R \xrightarrow{\gamma_C} C$           | $\gamma_C A R$     | Complex formation         |
| $A \xrightarrow{\delta_A} \emptyset$       | $\delta_A A$       | Activator degradation     |
| $C \xrightarrow{\delta_A} R$               | $\delta_A C$       | Complex degradation (A)   |
| $R \xrightarrow{\delta_R} \emptyset$       | $\delta_R R$       | Repressor degradation     |
| $D_A + A \xrightarrow{\gamma_A} D_A'$      | $\gamma_A D_A A$   | Promoter binding (A)      |
| $D_R + A \xrightarrow{\gamma_R} D_R'$      | $\gamma_R D_R A$   | Promoter binding (R)      |
| $D_A' \xrightarrow{\theta_A} D_A + A$      | $\theta_A D_A'$    | Promoter dissociation (A) |
| $D_R' \xrightarrow{\theta_R} D_R + A$      | $\theta_R D_R'$    | Promoter dissociation (R) |
| $D_A \xrightarrow{\alpha_A} D_A + M_A$     | $\alpha_A D_A$     | Transcription (A, basal)  |
| $D_A' \xrightarrow{\alpha_A'} D_A' + M_A$  | $\alpha_A' D_A'$   | Transcription (A, active) |
| $D_R \xrightarrow{\alpha_R} D_R + M_R$     | $\alpha_R D_R$     | Transcription (R, basal)  |
| $D_R' \xrightarrow{\alpha_R'} D_R' + M_R$  | $\alpha_R' D_R'$   | Transcription (R, active) |
| $M_A \xrightarrow{\delta_{M_A}} \emptyset$ | $\delta_{M_A} M_A$ | mRNA degradation (A)      |
| $M_R \xrightarrow{\delta_{M_R}} \emptyset$ | $\delta_{M_R} M_R$ | mRNA degradation (R)      |
| $M_A \xrightarrow{\beta_A} M_A + A$        | $\beta_A M_A$      | Translation (A)           |
| $M_R \xrightarrow{\beta_R} M_R + R$        | $\beta_R M_R$      | Translation (R)           |

</div>

### Appendix 2: Quick start and reproducibility

**Environment:** Python **3.9+** recommended.

```bash
conda create -n genetic-oscillators python=3.9
conda activate genetic-oscillators
pip install -r requirements.txt
```

**Pipelines**

| Command                                 | Purpose                                                                         | Rough time                                    |
| --------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------- |
| `python main.py all`                    | Lite: figures 1–4, 6–11, 13 (no SSA stats batch, no 50×50 grid)                | CPU-dependent (~10–40 min typical)             |
| `python main.py all --full`             | Full: adds Figures 5, 7 (50 SSA trajectories) and Figure 12 (50×50 basin scan) | CPU-dependent; hours typical (2500 ODE solves) |
| `python main.py stats`                  | Only SSA statistics panels                                                      | CPU-dependent (many SSA runs)                  |
| `python main.py basin-grid`             | Only 50×50 basin scan + Fig. 12                                                 | CPU-dependent (2500 ODE solves)                |
| `python main.py basin-grid --plot-only` | Replot Fig. 12 from `materials/basin_scan_sR_0.088_50x50.pkl`                   | Seconds                                        |

**Cached data:** The 50×50 basin scan is saved as `materials/basin_scan_sR_0.088_50x50.pkl`. Regenerate from scratch with:

```bash
python main.py basin-grid
```

## References

[1] Vilar, J. M., Kueh, H. Y., Barkai, N., & Leibler, S. (2002). Mechanisms of noise-resistance in genetic oscillators. *PNAS*, 99(9), 5988-5992.
