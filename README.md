# Scientific Computing, Bridging Course, 2023 - Miniproject 2: Genetic Oscillator

## 1. Background Introduction

The Vilar-Kueh-Barkai (VKB) model [1] describes a circadian clock driven by two proteins: an activator (A) and a repressor (R), coupled through a 16-reaction, 9-variable biochemical network. The original paper analyzes this system through two frameworks: a deterministic ODE model and a stochastic SSA simulation, demonstrating that the oscillator is noise-resistant.

This project reproduces the paper's core results (Parts A–D), then investigates one extended question:

The original paper derives a 2-variable QSSA reduction and uses it to analyze the oscillation mechanism — but validates it only at $s_R = 0.2$. The assignment results show that the system undergoes a qualitative transition between $s_R = 0.03$ (stable) and $s_R = 0.2$ (oscillatory). **Does the reduced model correctly capture this transition, or does the simplification break down near the boundary?** Section 3 addresses this question.

## 2. Assignment Tasks

### Part A: Deterministic Model and Solver Benchmarks

Solving the 9-dimensional ODE system at the baseline value $s_R = 0.2$ replicates the deterministic dynamics of the VKB oscillator. The system exhibits sustained oscillations with a period of approximately 24 hours. Both activator A and repressor R concentrations oscillate strongly over a simulated 400-hour period.

<div align="center">
<img src="figures/assignment_a_ode.png" width="600" alt="Deterministic ODE">

**Figure 1.** Deterministic ODE simulation (400 hours) for Activator A and Repressor R at $s_R = 0.2$.
</div>

<div align="center">
<img src="figures/solver_step_sizes.png" width="600" alt="Solver step sizes">

**Figure 2.** Internal step sizes ($\Delta t$) for RK45, BDF, and Radau over 48 hours.
</div>

**Solver Performance and Accuracy Tradeoff:**
Deterministic integration over 400 hours requires over 1.4 million function evaluations using the explicit RK45 method. In contrast, the implicit solvers (BDF and Radau) reduce this cost by over 25x, utilizing step sizes up to $1.0$ h (Figure 2). 

A key tradeoff exists between the two implicit methods:
*   **BDF** is the most efficient choice for general simulation, completing the 48-hour benchmark in 0.12s. While its absolute error is the highest among the tested solvers ($\sim 10^{-4}$ to $10^{-6}$, Figure 3), it remains more than sufficient for reproducing stable oscillatory patterns.
*   **Radau (Implicit Runge-Kutta)** is approximately twice as slow per step (0.21s overall) because it solves a larger nonlinear system at each stage. However, it provides 2–4 orders of magnitude higher precision ($\sim 10^{-8}$ to $10^{-14}$). This superior accuracy is critical for the **bifurcation and stability analysis** conducted in Section 3, where small numerical errors could lead to incorrect stability classifications near critical thresholds.

RK45 is excluded from further analysis due to its extreme computational overhead in this stiff regime.

<div align="center">
<img src="figures/solver_errors.png" width="600" alt="Solver errors">

**Figure 3.** Cross-solver absolute difference: BDF and Radau trajectories compared against a high-precision RK45 reference ($rtol=1e-12, atol=1e-14$).
</div>

### Part B: Stochastic Model

At the cellular level, chemical reactions are intrinsically stochastic discrete events. The system is modeled as a **discrete Markov process** with a continuous time parameter. The Gillespie Stochastic Simulation Algorithm (SSA) simulates 400 hours of dynamics (Figure 4).

<div align="center">
<img src="figures/assignment_b_ssa.png" width="600" alt="Stochastic SSA">

**Figure 4.** Stochastic SSA simulation (400 hours) for Activator A and Repressor R at $s_R = 0.2$.
</div>

**Discussion of Stochasticity:**
Running the SSA simulation multiple times produces different realizations, as expected: SSA generates sample paths, not a unique solution.

At $s_R = 0.2$, the stochastic trajectories **qualitatively agree** with the deterministic ODE solution; both demonstrate regular, sustained circadian oscillations with a comparable 24-hour period. However, they do not agree **quantitatively** due to molecular noise, which manifests in two distinct ways:
*   **Phase drift**: Noise in reaction timing causes the peak positions to shift away from the rigid ODE schedule, accumulating to a drift of roughly 4–6 hours over a 400-hour period.
*   **Amplitude noise**: Peak concentrations exhibit cycle-to-cycle modulation, with local amplitude variations typically ranging from 10% to 25% relative to the deterministic steady state.

At this parameter value, the deterministic model is a reliable approximation. However, this agreement does not hold across all parameter regimes (see Part C).

### Part C: Noise-Induced Oscillations

The authors highlight that random white noise can lead to qualitative differences from the deterministic model. Lowering the repressor decay rate to $s_R = 0.03$ reproduces this phenomenon. 

<div align="center">
<img src="figures/assignment_c_noise.png" width="600" alt="Noise-Induced Oscillations">

**Figure 5.** ODE vs. SSA comparison showing noise-induced sustained oscillations at critical threshold $s_R = 0.03$.
</div>

**Difference between Deterministic and Stochastic:**
At $s_R = 0.03$, the deterministic ODE converges to a stable steady state. As noted by the original authors, the trace of the linearized system satisfies $\tau < 0$ at this parameter value, confirming that the fixed point is stable. The stochastic SSA, however, continues to produce oscillations. At low molecule counts, intrinsic fluctuations are large enough to perturb the system away from the stable equilibrium and trigger **excitable excursions** through phase space — effectively restarting the oscillatory cycle each time.

This is consistent with the mechanism described in the original paper (Vilar et al. (2002), Fig. 6): the system is excitable near the fixed point, and stochastic perturbations substitute for the deterministic instability that drives oscillations at higher $s_R$ values.

The eigenvalue spectrum along the trajectory confirms this distinction: at $s_R = 0.2$, the dominant eigenvalue pair periodically crosses $Re(\lambda) = 0$, sustaining oscillations; at $s_R = 0.03$, both dominant modes remain strictly negative.

<div align="center">
<img src="figures/stiffness_comparison_part_c.png" width="800" alt="Stiffness comparison Part C">

**Figure 6.** Jacobian eigenvalue spectrum comparing steady ($s_R = 0.03$) and oscillatory ($s_R = 0.2$) regimes. The crossings in the right panel drive the deterministic oscillations.
</div>

### Part D: Deterministic vs. Stochastic Models

**When is the deterministic model a reliable approximation?**
At $s_R = 0.2$, the ODE and SSA produce qualitatively identical oscillations (Part B), with amplitude variations of 10–25%. In this regime, protein counts reach ~1500 molecules, making relative fluctuations ($\sim 1/\sqrt{N} \approx 2.6\%$) small enough that the deterministic approximation holds. Thus, in well-mixed macroscopic environments, ODEs allow for powerful mathematical analyses like bifurcation tracking and equilibrium eigenvalue analysis efficiently.

**When does the stochastic model become necessary?**
At $s_R = 0.03$, the deterministic model predicts a stable steady state while the SSA produces sustained oscillations (Part C). The qualitative disagreement arises because the system is excitable near the fixed point, and molecular fluctuations are large enough to trigger oscillatory excursions. In this regime, the deterministic model is not merely imprecise — it gives the wrong answer. Thus, when studying gene expression, single-cell dynamics, or systems near bifurcations where noise plays a structural role, stochastic modeling is necessary.

This raises a related question for deterministic modeling: even within the ODE framework, model *reductions* (such as QSSA) can also change qualitative behavior. Section 3 investigates this trade-off between computational cost and dynamical accuracy.

---

## 3. Extended Research: QSSA Reduction and Validity

The original paper derives a 2-variable QSSA reduction ($R, C$) and uses it to analyze the oscillation mechanism — but validates it only at $s_R = 0.2$. Parts A–D show that the system undergoes a qualitative transition between $s_R = 0.03$ (stable) and $s_R = 0.2$ (oscillatory). **Does the reduced model correctly capture this transition, or does the simplification break down near the boundary?**

**Step 1: Quantifying the Bifurcation Shift (Analytical Foundation)**

We strictly computed the Hopf bifurcation thresholds by analyzing the eigenvalues of the **7D independent Jacobian subspace**. We found that the QSSA reduction introduces a significant **22% delay** in the system's transition to oscillation ($s_{R,Full} \approx 0.096$ vs. $s_{R,Reduced} \approx 0.117$).

<div align="center">
<img src="figures/bifurcation_diagram.png" width="700" alt="Bifurcation diagram">

**Figure 7.** Hopf bifurcation diagram comparing full and reduced models. QSSA delays the onset of stability, creating a "grey zone" where the models mismatch qualitatively.
</div>

**Step 2: Probing the "Conflict Zone" ($s_R = 0.088$)**

Based on this calculation, $s_R = 0.088$ was strategically selected as a "blind spot" test case. It resides in the critical window where the full model has already developed a limit cycle (bistability), but the reduced model's topology remains fixed at a steady state. We mathematically predict that the reduced model will fail qualitatively here, not just quantitatively.

**Step 3: Multi-dimensional Verification**

Experimental results confirm the mathematical prediction.

<div align="center">
<img src="figures/reduction_comparison.png" width="900" alt="Reduction comparison">

**Figure 8.** Trajectory comparison. At $s_R = 0.088$, the reduced model (dashed) fails to capture the correct oscillatory dynamics, converging to a steady state instead.
</div>

Phase-space basin analysis reveals the underlying mechanism: the QSSA reduction collapses the complex bistable topology to a single attractor — the limit cycle is lost entirely.

<div align="center">
<img src="figures/basin_comparison_sR_0.088.png" width="900" alt="Basin comparison">

**Figure 9.** Basin comparison plot. The full model (right) possesses a limit cycle; the reduced model (left) collapses the phase space to a single point attractor.
</div>

**Step 4: When can we trust QSSA?**

A bifurcation-distance criterion quantifies this: $d_\text{bif} = \min(|s_R - s_{R,\text{Hopf}}| / s_{R,\text{Hopf}})$. The empirical bound $d_\text{bif} \lesssim 10\%$ marks the reliability limit where the reduction is likely to fail.

<div align="center">
<img src="figures/qssa_validity_criterion.png" width="800" alt="QSSA validity criterion">

**Figure 10.** QSSA validity summary. The reduction is robust only when the system is far from critical stability transitions.
</div>

## 4. Conclusion

This project reproduces the VKB oscillator analysis and extends it with a QSSA validity study. Three findings:

1. **Solver choice**: BDF provides the best efficiency-accuracy tradeoff for general simulation; Radau is preferable near bifurcation boundaries where higher precision matters.

2. **Deterministic vs. stochastic**: The ODE model is reliable at $s_R = 0.2$ (high molecule counts), but qualitatively wrong at $s_R = 0.03$ where noise-induced oscillations emerge.

3. **QSSA validity**: The 2-variable reduction agrees with the full model far from bifurcation boundaries, but shifts the Hopf threshold by 22%, eliminating bistability in the near-threshold regime. The empirical bound $d_\text{bif} \lesssim 10\%$ identifies when this breakdown occurs.

## Appendix

### Appendix Table 1: 16-Reaction Chemical Network 
The following reactions form the basis for both the deterministic 9-variable ODE model and the stochastic SSA simulation of the VKB oscillator.

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

### Appendix 2: Quick Start

Recommended Python version: **Python 3.9**

```bash
conda create -n genetic-oscillators python=3.9
conda activate genetic-oscillators
pip install -r requirements.txt

# Run complete analysis pipeline to fetch all figures
python main.py all
```

## References

[1] Vilar, J. M., Kueh, H. Y., Barkai, N., & Leibler, S. (2002). Mechanisms of noise-resistance in genetic oscillators. *PNAS*, 99(9), 5988-5992.
