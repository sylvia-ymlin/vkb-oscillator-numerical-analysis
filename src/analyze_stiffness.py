import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.ode_system import vkb_ode, vkb_jac
from src.config import DEFAULT_PARAMS, PLOT_CONFIG, RC_PARAMS, ODE_TOLERANCES


# ---------------------------------------------------------------------------
# Core computation: eigenvalues + stiffness along settled trajectory
# ---------------------------------------------------------------------------

def compute_jacobian_7d_along_trajectory(t, y, params):
    """
    Compute 7×7 Jacobian in the independent subspace along a trajectory.
    
    This is consistent with the equilibrium analysis in bifurcation_analysis.py,
    which also uses the 7D independent subspace to avoid singular eigenvalues
    from conservation laws.
    
    Parameters:
        t: time
        y: 9D state vector
        params: parameter dictionary
    
    Returns:
        J_7d: 7×7 Jacobian matrix
    """
    # Compute full 9×9 Jacobian
    J_9d = vkb_jac(t, y, params)
    
    # Extract 7×7 submatrix for independent variables
    # Independent: [D_A, D_R, MA, A, MR, R, C] = indices [0, 1, 4, 5, 6, 7, 8]
    # Dependent: [D_A', D_R'] = indices [2, 3]
    
    independent_indices = [0, 1, 4, 5, 6, 7, 8]
    
    J_7d = np.zeros((7, 7))
    
    for i, row_idx in enumerate(independent_indices):
        for j, col_idx in enumerate(independent_indices):
            J_7d[i, j] = J_9d[row_idx, col_idx]
        
        # Add contributions from dependent variables
        # Since D_A' = 1 - D_A and D_R' = 1 - D_R:
        # ∂f_i/∂D_A (effective) = ∂f_i/∂D_A - ∂f_i/∂D_A'
        # ∂f_i/∂D_R (effective) = ∂f_i/∂D_R - ∂f_i/∂D_R'
        J_7d[i, 0] += -J_9d[row_idx, 2]  # D_A' contribution to D_A column
        J_7d[i, 1] += -J_9d[row_idx, 3]  # D_R' contribution to D_R column
    
    return J_7d


def compute_stiffness_along_trajectory(s_R, t_span=(0, 500), n_sample=200):
    """
    Sample Jacobian eigenvalues along the settled trajectory.
    
    IMPORTANT: Uses 7×7 Jacobian in the independent subspace to be consistent
    with equilibrium analysis and avoid numerical artifacts from conservation laws.
    """
    params = {**DEFAULT_PARAMS, 's_R': s_R}
    y0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def ode_func(t, y):
        return vkb_ode(t, y, params)

    def jac_func(t, y):
        return vkb_jac(t, y, params)

    sol = solve_ivp(ode_func, t_span, y0, method='BDF',
                    jac=jac_func, 
                    rtol=ODE_TOLERANCES['rtol'], 
                    atol=ODE_TOLERANCES['atol'], 
                    dense_output=True)

    # Sample the last 20% of the trajectory
    t_sample = np.linspace(t_span[1] * 0.8, t_span[1], n_sample)

    eigenvalues = []
    stiffness_ratios = []
    conservation_errors = []

    for t in t_sample:
        y = sol.sol(t)

        conserv_A = y[0] + y[2]
        conserv_R = y[1] + y[3]
        conservation_errors.append([conserv_A - 1.0, conserv_R - 1.0])

        # Use 7×7 Jacobian (consistent with bifurcation analysis)
        J_7d = compute_jacobian_7d_along_trajectory(t, y, params)
        eigvals = np.linalg.eigvals(J_7d)
        eigvals = eigvals[np.argsort(np.real(eigvals))]
        eigenvalues.append(eigvals)

        real_parts = np.real(eigvals)
        significant = np.abs(real_parts) > 1e-6
        if np.sum(significant) >= 2:
            real_sig = real_parts[significant]
            stiffness = np.max(np.abs(real_sig)) / np.min(np.abs(real_sig))
        else:
            stiffness = np.nan
        stiffness_ratios.append(stiffness)

    conservation_errors = np.array(conservation_errors)
    print(f"Conservation check (s_R={s_R}): "
          f"max|D_A+D_A'-1| = {np.max(np.abs(conservation_errors[:, 0])):.2e}, "
          f"max|D_R+D_R'-1| = {np.max(np.abs(conservation_errors[:, 1])):.2e}")

    return np.array(eigenvalues), np.array(stiffness_ratios), t_sample


# ---------------------------------------------------------------------------
# Plot 1: Eigenvalue spectrum + stiffness ratio
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(s_R_values=(0.03, 0.088, 0.2)):
    """2×3 grid: eigenvalue scatter in complex plane (top) + stiffness ratio (bottom).
    
    Now uses 7×7 Jacobian (7 dynamically active eigenvalues) instead of 9×9.
    Uses symlog scale for x-axis to handle extreme eigenvalue ranges.
    """
    figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
    dpi = PLOT_CONFIG['DPI']

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(2, len(s_R_values), figsize=(15, 8))
        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        eig_colors = PLOT_CONFIG['EIGENVALUE_COLORS']
        # Eigenvalues sorted ascending by Re(λ):
        #   indices 0-4 → fast eigenvalues (background, gray)
        #   index  5    → second dominant (blue)
        #   index  6    → most dominant   (red)

        for idx, s_R in enumerate(s_R_values):
            eigvals, stiffness, t = compute_stiffness_along_trajectory(s_R)

            # Top row: eigenvalues in complex plane
            ax = axes[0, idx]

            # Fast eigenvalues (λ₁–λ₅): uniform gray cloud
            for i in range(5):
                lbl = 'λ₁–λ₅ (fast)' if (i == 0 and idx == 0) else ''
                ax.scatter(np.real(eigvals[:, i]), np.imag(eigvals[:, i]),
                           s=6, alpha=0.2, color=neutral_colors['gray'],
                           label=lbl)

            # Dominant eigenvalues (λ₆, λ₇): project colors
            dom_cfg = [(5, eig_colors['dominant_2'], 'λ₆ (dominant)'),
                       (6, eig_colors['dominant_1'], 'λ₇ (dominant)')]
            for i, color, name in dom_cfg:
                ax.scatter(np.real(eigvals[:, i]), np.imag(eigvals[:, i]),
                           s=12, alpha=0.75, color=color,
                           label=name if idx == 0 else '')

            ax.axhline(0, color=neutral_colors['black'], linewidth=0.5, linestyle='--')
            ax.axvline(0, color=neutral_colors['black'], linewidth=0.5, linestyle='--')

            # Use symlog scale for x-axis to handle extreme ranges
            ax.set_xscale('symlog', linthresh=1.0)

            ax.set_xlabel('Re(λ)')
            ax.set_ylabel('Im(λ)')
            ax.set_title(f'$s_R = {s_R}$')
            if idx == 0:
                ax.legend(fontsize=8, loc='upper left')

            # Bottom row: stiffness ratio vs time
            ax = axes[1, idx]
            valid = ~np.isnan(stiffness)
            ax.semilogy(t[valid], stiffness[valid],
                        color=neutral_colors['gray_dark'], linewidth=1.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Stiffness Ratio')
            mean_stiff = np.nanmean(stiffness)
            ax.set_title(f'Mean: {mean_stiff:.2e}')

        fig.suptitle('Jacobian Eigenvalue Spectrum (7D Independent Subspace) and Stiffness Ratio', fontsize=13)
        plt.tight_layout()
        out_path = f'{figures_dir}/stiffness_analysis.png'
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Plot 2: Eigenvalue real-part evolution — compare three regimes
# ---------------------------------------------------------------------------

def plot_eigenvalue_evolution(s_R_values=(0.03, 0.088, 0.2)):
    """2×3 grid: Re(λ) vs time — dual-panel layout per column.

    Each column = one s_R regime.
    - Top panel (small, 1/4 height): ONLY the 2 dominant eigenvalues, auto-scaled y.
      Makes the slow-mode dynamics visible without fast eigenvalue compression.
    - Bottom panel (large, 3/4 height): All 7 eigenvalues at full y-range.
      Shows the full stiffness context.

    Now uses 7×7 Jacobian (7 dynamically active eigenvalues).
    """
    figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
    dpi = PLOT_CONFIG['DPI']
    labels = {0.03: 'Steady-state', 0.088: 'Near-threshold', 0.2: 'Oscillatory'}
    neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
    eig_colors = PLOT_CONFIG['EIGENVALUE_COLORS']
    colors_dominant = [eig_colors['dominant_1'], eig_colors['dominant_2']]

    n_cols = len(s_R_values)

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            2, n_cols,
            figsize=(15, 8),
            height_ratios=[1, 3],
            sharex='col',
        )

        for idx, s_R in enumerate(s_R_values):
            eigvals, _, t = compute_stiffness_along_trajectory(s_R, n_sample=100)

            # Identify the two eigenvalues with the largest (least negative) real parts
            mean_real_parts = np.mean(np.real(eigvals), axis=0)
            dominant_indices = np.argsort(mean_real_parts)[-2:]  # top 2

            ax_top = axes[0, idx]    # zoomed: dominant eigenvalues only
            ax_bot = axes[1, idx]    # full range: all eigenvalues

            # ── TOP PANEL: dominant eigenvalues only ──────────────────────────
            for plot_idx, i in enumerate(dominant_indices):
                lbl = f'λ_{i+1} (dominant)' if idx == 0 else ''
                ax_top.plot(t, np.real(eigvals[:, i]),
                            color=colors_dominant[plot_idx], lw=2.0, alpha=0.9,
                            label=lbl)

            ax_top.axhline(0, color=neutral_colors['black'],
                           linestyle='--', linewidth=0.8)
            ax_top.set_ylabel('Re(λ)\n(dominant)', fontsize=9)
            ax_top.set_title(f'$s_R = {s_R}$  ({labels[s_R]})', fontsize=11)
            if idx == 0:
                ax_top.legend(fontsize=8, loc='lower right')

            # ── BOTTOM PANEL: all 7 eigenvalues ───────────────────────────────
            for i in range(7):
                if i in dominant_indices:
                    dom_idx = list(dominant_indices).index(i)
                    ax_bot.plot(t, np.real(eigvals[:, i]),
                                color=colors_dominant[dom_idx], lw=2.0, alpha=0.9)
                else:
                    ax_bot.plot(t, np.real(eigvals[:, i]),
                                color=neutral_colors['gray'], lw=0.8, alpha=0.3)

            ax_bot.axhline(0, color=neutral_colors['black'],
                           linestyle='--', linewidth=0.8)
            ax_bot.set_xlabel('Time')
            ax_bot.set_ylabel('Re(λ)\n(all modes)', fontsize=9)

        fig.suptitle(
            'Eigenvalue Real Parts Along Trajectory (7D Independent Subspace) — Three Regimes\n'
            'Top: dominant eigenvalues (zoomed)  |  Bottom: all eigenvalues (full scale)',
            fontsize=12,
        )
        plt.tight_layout()
        out_path = f'{figures_dir}/eigenvalue_evolution.png'
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Timescale hierarchy table (console)
# ---------------------------------------------------------------------------

def print_timescale_table(s_R=0.2):
    """Print time-averaged eigenvalue timescale hierarchy for a given s_R."""
    eigvals, _, _ = compute_stiffness_along_trajectory(s_R, n_sample=50)

    eigvals_mean = np.mean(eigvals, axis=0)
    eigvals_mean = eigvals_mean[np.argsort(np.real(eigvals_mean))]

    print(f"\n{'='*70}")
    print(f"Timescale Hierarchy for s_R = {s_R}")
    print(f"{'='*70}")
    print(f"{'Mode':<6} {'Re(λ)':<18} {'Im(λ)':<18} {'τ = 1/|Re(λ)|':<18}")
    print(f"{'-'*70}")

    for i, lam in enumerate(eigvals_mean):
        re, im = np.real(lam), np.imag(lam)
        tau = 1.0 / np.abs(re) if np.abs(re) > 1e-10 else np.inf
        print(f"{i+1:<6} {re:<18.3e} {im:<18.3e} {tau:<18.3e}")

    real_parts = np.real(eigvals_mean)
    significant = np.abs(real_parts) > 1e-6
    real_sig = real_parts[significant]

    fastest = 1.0 / np.max(np.abs(real_sig))
    slowest = 1.0 / np.min(np.abs(real_sig))

    print(f"{'-'*70}")
    print(f"Fastest timescale:  {fastest:.3e}")
    print(f"Slowest timescale:  {slowest:.3e}")
    print(f"Separation ratio:   {slowest / fastest:.3e}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stiffness_analysis():
    """Entry point for main.py."""
    plot_eigenvalue_spectrum()
    plot_eigenvalue_evolution()
    print_timescale_table(s_R=0.03)
    print_timescale_table(s_R=0.088)
    print_timescale_table(s_R=0.2)


if __name__ == '__main__':
    run_stiffness_analysis()
