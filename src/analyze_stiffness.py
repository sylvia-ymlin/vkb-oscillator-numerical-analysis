import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import solve_ivp
from src.models import vkb_ode, vkb_jac
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

def plot_stiffness_diagnostics(s_R_values=(0.03, 0.088, 0.2), 
                              output_name='stiffness_analysis',
                              show_stiffness=True):
    """2×3 grid: eigenvalue scatter in complex plane (top) + stiffness ratio (bottom).
    If show_stiffness=False, only the top row (spectrum) is plotted.
    """
    figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
    dpi = PLOT_CONFIG['DPI']

    with plt.rc_context(RC_PARAMS):
        n_cols = len(s_R_values)
        if show_stiffness:
            if n_cols == 1:
                fig, axes_flat = plt.subplots(1, 2, figsize=(12, 5))
                axes = np.array([axes_flat]).T  # shape (2, 1)
            else:
                fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)
        else:
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
        
        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        eig_colors = PLOT_CONFIG['EIGENVALUE_COLORS']

        for idx, s_R in enumerate(s_R_values):
            eigvals, stiffness, t = compute_stiffness_along_trajectory(s_R)

            # Top row (or Left for n=1): eigenvalues in complex plane
            ax = axes[0, idx]

            # Fast eigenvalues (λ₁–λ₅): uniform gray cloud
            for i in range(5):
                lbl = r'$\lambda_1-\lambda_5$ (fast)' if (i == 0 and idx == 0) else ''
                ax.scatter(np.real(eigvals[:, i]), np.imag(eigvals[:, i]),
                           s=6, alpha=0.2, color=neutral_colors['gray'],
                           label=lbl)

            # Dominant eigenvalues (λ₆, λ₇): project colors
            dom_cfg = [(5, eig_colors['dominant_2'], r'$\lambda_6$ (dominant)'),
                       (6, eig_colors['dominant_1'], r'$\lambda_7$ (dominant)')]
            for i, color, name in dom_cfg:
                ax.scatter(np.real(eigvals[:, i]), np.imag(eigvals[:, i]),
                           s=12, alpha=0.75, color=color,
                           label=name if idx == 0 else '')

            ax.axhline(0, color=neutral_colors['black'], linewidth=0.5, linestyle='--')
            ax.axvline(0, color=neutral_colors['black'], linewidth=0.5, linestyle='--')

            # Use symlog scale for x-axis to handle extreme ranges
            ax.set_xscale('symlog', linthresh=1.0)

            ax.set_xlabel(r'Re($\lambda$)')
            ax.set_ylabel(r'Im($\lambda$)')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # ax.set_title(f'$s_R = {s_R}$')
            if idx == 0:
                ax.legend(fontsize=8, loc='upper left')

            if show_stiffness:
                # Bottom row: stiffness ratio vs time
                ax = axes[1, idx]
                valid = ~np.isnan(stiffness)
                ax.semilogy(t[valid], stiffness[valid],
                            color=neutral_colors['gray_dark'], linewidth=1.5)
                ax.set_xlabel('Time')
                ax.set_ylabel('Stiffness Ratio')
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                mean_stiff = np.nanmean(stiffness)
                # ax.set_title(f'Mean: {mean_stiff:.2e}')

        # fig.suptitle('Jacobian Eigenvalue Spectrum (7D Independent Subspace) and Stiffness Ratio', fontsize=13)
        plt.tight_layout()
        out_path = f'{figures_dir}/{output_name}.png'
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_path}')


def run_stiffness_analysis():
    """Entry point for main.py. Generates Figure 6 for the README."""
    # Plot: Assignment Part C comparison (sR=0.03 vs 0.2) - SPECTRUM ONLY (Figure 6)
    plot_stiffness_diagnostics(s_R_values=(0.03, 0.2), 
                               output_name='stiffness_comparison_part_c',
                               show_stiffness=False)


if __name__ == '__main__':
    run_stiffness_analysis()
