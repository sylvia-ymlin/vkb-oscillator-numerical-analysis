import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root, root_scalar
from src.config import DEFAULT_PARAMS, PLOT_CONFIG, RC_PARAMS
from src.models import vkb_ode, vkb_jac, reduced_vkb_ode, reduced_vkb_jac




def find_full_equilibrium(s_R, initial_guess=None):
    params = {**DEFAULT_PARAMS, 's_R': s_R}

    if initial_guess is None:
        initial_guess_7d = [0.5, 0.5, 5.0, 50.0, 5.0, 50.0, 50.0]
    else:
        if len(initial_guess) == 9:
            initial_guess_7d = [
                initial_guess[0],
                initial_guess[1],
                initial_guess[4],
                initial_guess[5],
                initial_guess[6],
                initial_guess[7],
                initial_guess[8],
            ]
        else:
            initial_guess_7d = initial_guess

    def system_eqs_7d(y7):
        D_A, D_R, MA, A, MR, R, C = y7
        D_A_ = 1.0 - D_A
        D_R_ = 1.0 - D_R
        y9 = [D_A, D_R, D_A_, D_R_, MA, A, MR, R, C]
        dy9 = vkb_ode(0, y9, params)
        return [dy9[0], dy9[1], dy9[4], dy9[5], dy9[6], dy9[7], dy9[8]]

    sol = root(system_eqs_7d, initial_guess_7d, method='hybr')

    if not sol.success:
        raise ValueError(f"Failed to find equilibrium for s_R={s_R}: {sol.message}")

    D_A, D_R, MA, A, MR, R, C = sol.x
    y9_star = np.array([
        D_A,
        D_R,
        1.0 - D_A,
        1.0 - D_R,
        MA,
        A,
        MR,
        R,
        C,
    ])

    if np.any(y9_star < -1e-6):
        import warnings
        warnings.warn(
            f"Non-physical equilibrium found at s_R={s_R}: "
            f"negative concentrations detected. "
            f"Min value: {np.min(y9_star):.6f}. "
            f"This may indicate numerical continuation failure or bifurcation."
        )

    return y9_star


def find_reduced_equilibrium(s_R, initial_guess=None):
    params = {**DEFAULT_PARAMS, 's_R': s_R}

    if initial_guess is None:
        initial_guess = [100.0, 350.0]

    def system_eqs(y):
        return reduced_vkb_ode(0, y, params)

    sol = root(system_eqs, initial_guess, method='hybr')

    if not sol.success:
        raise ValueError(f"Failed to find equilibrium for s_R={s_R}: {sol.message}")

    return sol.x


def full_jacobian_7d(y_star, params):
    J_9d = vkb_jac(0, y_star, params)
    independent_indices = [0, 1, 4, 5, 6, 7, 8]

    J_7d = np.zeros((7, 7))

    for i, row_idx in enumerate(independent_indices):
        for j, col_idx in enumerate(independent_indices):
            J_7d[i, j] = J_9d[row_idx, col_idx]

        J_7d[i, 0] += -J_9d[row_idx, 2]
        J_7d[i, 1] += -J_9d[row_idx, 3]

    return J_7d


def analyze_equilibrium_stability(y_star, jacobian_func, params, model_name=""):
    if jacobian_func == full_jacobian_7d:
        J = jacobian_func(y_star, params)
    elif jacobian_func == reduced_vkb_jac:
        J = jacobian_func(0, y_star, params)
    else:
        J = jacobian_func(0, y_star, params)

    eigvals = np.linalg.eigvals(J)
    eigvals = eigvals[np.argsort(-np.real(eigvals))]

    max_real = np.max(np.real(eigvals))
    is_stable = max_real < 0

    has_complex_pair = False
    for lam in eigvals:
        if np.abs(np.imag(lam)) > 1e-6:
            conj = np.conj(lam)
            if np.any(np.abs(eigvals - conj) < 1e-6):
                has_complex_pair = True
                break

    return {
        'eigenvalues': eigvals,
        'max_real_part': max_real,
        'is_stable': is_stable,
        'has_complex_pair': has_complex_pair,
        'equilibrium': y_star,
    }


def compute_max_real_part_at_equilibrium(s_R, model='reduced'):
    params = {**DEFAULT_PARAMS, 's_R': s_R}

    try:
        if model == 'reduced':
            y_star = find_reduced_equilibrium(s_R)
            stability = analyze_equilibrium_stability(
                y_star, reduced_vkb_jac, params, model_name="Reduced"
            )
        elif model == 'full':
            y_star = find_full_equilibrium(s_R)
            stability = analyze_equilibrium_stability(
                y_star, full_jacobian_7d, params, model_name="Full"
            )
        else:
            raise ValueError(f"Unknown model: {model}")

        return stability['max_real_part']

    except Exception as e:
        print(f"Warning: Failed to compute stability at s_R={s_R} for {model} model: {e}")
        return np.nan


def find_hopf_bifurcation_threshold(model='reduced', s_R_range=(0.05, 0.15), tol=1e-6, n_scan=80):
    """
    Find the Hopf bifurcation threshold using continuation-based equilibrium tracking.

    The previous cold-start implementation called find_full_equilibrium with a fixed
    default guess at every brentq evaluation.  For s_R above the full-model Hopf
    threshold (~0.096) the solver diverges to a non-physical solution, producing
    spurious eigenvalues of ~+140.  brentq detected a sign change between the last
    valid point (maxRe ≈ -0.001) and the first garbage point (+140) and converged to
    the right value by accident — but the method was unreliable.

    This version:
      1. Scans the range with continuation (compute_bifurcation_curve) to locate the
         sign-change bracket, sidestepping the cold-start failure entirely.
      2. Bisects within that bracket, always warm-starting from the last stable
         equilibrium, so the solver stays on the physical branch above the threshold.
    """
    # ── Step 1: coarse scan with continuation to bracket the sign change ──────
    s_scan = np.linspace(s_R_range[0], s_R_range[1], n_scan)
    curve  = compute_bifurcation_curve(s_scan, model=model)
    mr_vals = curve['max_real_parts']
    eq_vals  = curve['equilibria']

    s_lo = s_hi = None
    eq_lo_guess = None
    for i in range(len(mr_vals) - 1):
        lo_v, hi_v = mr_vals[i], mr_vals[i + 1]
        if np.isnan(lo_v) or np.isnan(hi_v):
            continue
        if lo_v < 0 < hi_v:          # stable → unstable crossing (Hopf)
            s_lo, s_hi   = float(s_scan[i]), float(s_scan[i + 1])
            eq_lo_guess  = eq_vals[i]
            break

    if s_lo is None:
        print(f"Warning: No Hopf threshold found in {s_R_range} for {model} model.")
        return None

    # ── Step 2: bisect within [s_lo, s_hi] using continuation from s_lo ──────
    def _max_re(s_R, guess):
        p = {**DEFAULT_PARAMS, 's_R': s_R}
        try:
            if model == 'reduced':
                y = find_reduced_equilibrium(s_R, initial_guess=guess)
                J = reduced_vkb_jac(0, y, p)
            else:
                y = find_full_equilibrium(s_R, initial_guess=guess)
                if np.min(y) < -1e-4:       # non-physical → treat as unstable
                    return np.nan, None
                J = full_jacobian_7d(y, p)
            return float(np.max(np.real(np.linalg.eigvals(J)))), y
        except Exception:
            return np.nan, None

    eq_lo = eq_lo_guess
    for _ in range(60):                     # at most 60 bisection steps → tol ≈ 1e-18
        if s_hi - s_lo < tol:
            break
        s_mid = (s_lo + s_hi) / 2.0
        mr_mid, eq_mid = _max_re(s_mid, eq_lo)

        if np.isnan(mr_mid):
            s_hi = s_mid                    # solver failed → assume unstable side
        elif mr_mid < 0:                    # still stable → threshold lies above
            s_lo = s_mid
            if eq_mid is not None:
                eq_lo = eq_mid
        else:                               # unstable → threshold lies below
            s_hi = s_mid

    return (s_lo + s_hi) / 2.0


def compute_bifurcation_curve(s_R_values, model='reduced'):
    max_real_parts = []
    equilibria = []
    all_eigenvalues = []
    current_guess = None

    for s_R in s_R_values:
        params = {**DEFAULT_PARAMS, 's_R': s_R}

        try:
            if model == 'reduced':
                y_star = find_reduced_equilibrium(s_R, initial_guess=current_guess)
                stability = analyze_equilibrium_stability(y_star, reduced_vkb_jac, params)
            else:
                y_star = find_full_equilibrium(s_R, initial_guess=current_guess)
                stability = analyze_equilibrium_stability(y_star, full_jacobian_7d, params)

            current_guess = y_star
            max_real_parts.append(stability['max_real_part'])
            equilibria.append(y_star)
            all_eigenvalues.append(stability['eigenvalues'])

        except Exception as e:
            print(f"Warning: Failed at s_R={s_R} for {model} model: {e}")
            max_real_parts.append(np.nan)
            equilibria.append(None)
            all_eigenvalues.append(None)

    return {
        's_R': s_R_values,
        'max_real_parts': np.array(max_real_parts),
        'equilibria': equilibria,
        'all_eigenvalues': all_eigenvalues,
    }


"""
Bifurcation analysis public API and reporting utilities.

Core numerical routines are implemented in `bifurcation_core.py` to keep this
module concise while preserving backward-compatible imports used elsewhere.
"""




def compare_bifurcation_thresholds(s_R_range=(0.05, 0.15), verbose=True):
    """Compare Hopf bifurcation thresholds between full and reduced models."""
    def _log(message=""):
        if verbose:
            print(message)

    _log("=" * 80)
    _log("Hopf Bifurcation Threshold Analysis")
    _log("=" * 80)
    _log()
    _log("Computing bifurcation thresholds by analyzing equilibrium point stability...")
    _log()

    s_R_full = find_hopf_bifurcation_threshold(model='full', s_R_range=s_R_range)
    s_R_reduced = find_hopf_bifurcation_threshold(model='reduced', s_R_range=s_R_range)

    _log(f"Full model:     s_R_hopf = {s_R_full:.6f}" if s_R_full else "Full model: threshold not found")
    _log(f"Reduced model:  s_R_hopf = {s_R_reduced:.6f}" if s_R_reduced else "Reduced model: threshold not found")
    _log()

    if s_R_full and s_R_reduced:
        shift = s_R_reduced - s_R_full
        shift_percent = (shift / s_R_full) * 100

        _log(f"Threshold shift: Δs_R = {shift:+.6f} ({shift_percent:+.2f}%)")
    _log("=" * 80)

    return {
        'full_threshold': s_R_full,
        'reduced_threshold': s_R_reduced,
    }


if __name__ == '__main__':
    print("Testing equilibrium finding...")

    for s_R in [0.03, 0.088, 0.2]:
        print(f"\ns_R = {s_R}:")
        try:
            y_reduced = find_reduced_equilibrium(s_R)
            print(f"  Reduced equilibrium: R*={y_reduced[0]:.3f}, C*={y_reduced[1]:.3f}")
        except Exception as e:
            print(f"  Reduced: {e}")

        try:
            y_full = find_full_equilibrium(s_R)
            print(f"  Full equilibrium found: A={y_full[5]:.3f}, R={y_full[7]:.3f}, C={y_full[8]:.3f}")
        except Exception as e:
            print(f"  Full: {e}")

    print("\n" + "=" * 80)
    compare_bifurcation_thresholds()


"""
QSSA Validity Criterion Analysis

This module provides quantitative criteria for when QSSA approximation is reliable.
Based on bifurcation analysis and error metrics, it defines validity boundaries.
"""




def assess_qssa_validity(s_R, s_R_full_hopf, s_R_reduced_hopf, threshold_margin=0.1):
    """
    Assess QSSA validity at a given s_R value based on proximity to bifurcations.
    
    Criterion: QSSA should be used with caution when within threshold_margin
    (default 10%) of either Hopf bifurcation threshold.
    
    Parameters:
        s_R: parameter value to assess
        s_R_full_hopf: full model Hopf threshold
        s_R_reduced_hopf: reduced model Hopf threshold
        threshold_margin: relative distance threshold (0.1 = 10%)
    
    Returns:
        dict with keys:
            - valid: bool, True if QSSA is reliable
            - distance_to_full: relative distance to full model threshold
            - distance_to_reduced: relative distance to reduced model threshold
            - min_distance: minimum relative distance to either threshold
            - recommendation: string describing validity status
    """
    # Compute relative distances
    dist_full = abs(s_R - s_R_full_hopf) / s_R_full_hopf
    dist_reduced = abs(s_R - s_R_reduced_hopf) / s_R_reduced_hopf
    min_dist = min(dist_full, dist_reduced)
    
    # Assess validity
    if min_dist > threshold_margin:
        valid = True
        recommendation = "QSSA is reliable (far from bifurcations)"
    else:
        valid = False
        if dist_full < threshold_margin and dist_reduced < threshold_margin:
            recommendation = "QSSA unreliable (near both bifurcation thresholds)"
        elif dist_full < threshold_margin:
            recommendation = "QSSA unreliable (near full model bifurcation)"
        else:
            recommendation = "QSSA unreliable (near reduced model bifurcation)"
    
    return {
        'valid': valid,
        'distance_to_full': dist_full,
        'distance_to_reduced': dist_reduced,
        'min_distance': min_dist,
        'recommendation': recommendation,
    }


def plot_qssa_validity_regions():
    """
    Visualize QSSA validity regions in parameter space.
    
    Creates a diagram showing where QSSA is reliable vs unreliable based on
    proximity to bifurcation thresholds.
    """
    print("Generating QSSA validity criterion visualization...")
    
    # Find bifurcation thresholds
    s_R_full = find_hopf_bifurcation_threshold(model='full', s_R_range=(0.05, 0.15))
    s_R_reduced = find_hopf_bifurcation_threshold(model='reduced', s_R_range=(0.05, 0.15))
    
    if not s_R_full or not s_R_reduced:
        print("  Error: Could not find bifurcation thresholds")
        return
    
    # Parameter range
    s_R_values = np.linspace(0.02, 0.25, 200)
    
    # Assess validity for each point
    validity = []
    min_distances = []
    
    for s_R in s_R_values:
        result = assess_qssa_validity(s_R, s_R_full, s_R_reduced, threshold_margin=0.1)
        validity.append(result['valid'])
        min_distances.append(result['min_distance'])
    
    validity = np.array(validity)
    min_distances = np.array(min_distances)
    
    # Plot (muted palette + compact canvas — Fig. 13)
    sc = PLOT_CONFIG['SUMMARY_FIG_COLORS']
    test_points = [
        (0.03, 'Steady', sc['test_low']),
        (0.088, 'Near-threshold', sc['test_mid']),
        (0.2, 'Oscillatory', sc['test_high']),
    ]
    with plt.rc_context(RC_PARAMS):
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=PLOT_CONFIG["FIGSIZE"]["bifurcation_stack"],
            sharex=True,
            constrained_layout=True,
        )

        valid_color = sc['valid_fill']
        invalid_color = sc['invalid_fill']
        full_color = sc['hopf_full']
        reduced_color = sc['hopf_reduced']

        # Panel 1: Validity regions
        ax1.fill_between(
            s_R_values, 0, 1, where=validity,
            alpha=0.5, color=valid_color, label='QSSA reliable',
        )
        ax1.fill_between(
            s_R_values, 0, 1, where=~validity,
            alpha=0.5, color=invalid_color, label='QSSA unreliable',
        )

        ax1.axvline(s_R_full, color=full_color, linestyle='--', linewidth=1.25,
                    label=f'Full Hopf {s_R_full:.3f}')
        ax1.axvline(s_R_reduced, color=reduced_color, linestyle='--', linewidth=1.25,
                    label=f'Reduced Hopf {s_R_reduced:.3f}')

        margin_full_low = s_R_full * 0.9
        margin_full_high = s_R_full * 1.1
        margin_reduced_low = s_R_reduced * 0.9
        margin_reduced_high = s_R_reduced * 1.1

        ax1.axvline(margin_full_low, color=full_color, linestyle=':', linewidth=0.9, alpha=0.45)
        ax1.axvline(margin_full_high, color=full_color, linestyle=':', linewidth=0.9, alpha=0.45)
        ax1.axvline(margin_reduced_low, color=reduced_color, linestyle=':', linewidth=0.9, alpha=0.45)
        ax1.axvline(margin_reduced_high, color=reduced_color, linestyle=':', linewidth=0.9, alpha=0.45)

        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        edge = neutral_colors['gray_dark']
        for i, (s_R_test, label, t_color) in enumerate(test_points):
            result = assess_qssa_validity(s_R_test, s_R_full, s_R_reduced)
            marker = 'o' if result['valid'] else 'x'
            ax1.plot(s_R_test, 0.5, marker, markersize=9, color=t_color,
                     markeredgecolor=edge, markeredgewidth=0.9, zorder=5)

            y_pos = 0.78 if i == 1 else 0.66
            ax1.text(s_R_test, y_pos, label, ha='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.22', facecolor='#fafafa',
                               edgecolor=neutral_colors['gray_light'], alpha=0.95, linewidth=0.4))

        ax1.set_ylabel('QSSA validity', fontsize=10)
        ax1.set_ylim(-0.08, 1.12)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Unreliable', 'Reliable'])
        ax1.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.92)
        ax1.grid(True, alpha=0.22, axis='x')

        # Panel 2: Distance metric
        ax2.plot(s_R_values, min_distances, color=sc['distance_line'], linewidth=1.6,
                 label='Min relative distance')
        ax2.axhline(0.1, color=invalid_color, linestyle='--', linewidth=1.2, alpha=0.85,
                    label='10% threshold')
        ax2.fill_between(s_R_values, 0, 0.1, alpha=0.35, color=invalid_color)
        ax2.fill_between(s_R_values, 0.1, 1, alpha=0.28, color=valid_color)

        for s_R_test, label, t_color in test_points:
            result = assess_qssa_validity(s_R_test, s_R_full, s_R_reduced)
            ax2.plot(s_R_test, result['min_distance'], 'o', markersize=8,
                     color=t_color, markeredgecolor=edge, markeredgewidth=0.9, zorder=5)

        ax2.set_xlabel('$s_R$ (repressor degradation rate)', fontsize=10)
        ax2.set_ylabel('Min relative distance\nto bifurcation', fontsize=10)
        ax2.set_ylim(0, max(0.5, np.max(min_distances) * 1.08))
        ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.92)
        ax2.grid(True, alpha=0.22)

        import os
        figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, 'qssa_validity_criterion.png')
        fig.savefig(output_path, dpi=PLOT_CONFIG['DPI'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("QSSA Validity Assessment for Test Points")
    print("="*70)
    
    for s_R_test, label, _ in test_points:
        result = assess_qssa_validity(s_R_test, s_R_full, s_R_reduced)
        print(f"\n{label} regime (s_R = {s_R_test}):")
        print(f"  Distance to full model threshold:    {result['distance_to_full']:.1%}")
        print(f"  Distance to reduced model threshold: {result['distance_to_reduced']:.1%}")
        print(f"  Minimum distance:                    {result['min_distance']:.1%}")
        print(f"  Status: {result['recommendation']}")
    
    print("="*70)


if __name__ == '__main__':
    plot_qssa_validity_regions()


"""
Visualization for bifurcation analysis.

Generates plots comparing equilibrium stability between full and reduced models.
"""





def plot_bifurcation_diagram(s_R_range=(0.05, 0.15), n_points=30):
    """
    Plot bifurcation diagram: max(Re(λ)) vs s_R for both models.
    
    This shows how equilibrium stability changes with s_R, and clearly
    demonstrates the threshold shift between full and reduced models.
    """
    figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
    dpi = PLOT_CONFIG['DPI']
    
    s_R_values = np.linspace(s_R_range[0], s_R_range[1], n_points)
    
    print("Computing bifurcation curves...")
    print(f"  Evaluating {n_points} points in range [{s_R_range[0]}, {s_R_range[1]}]")
    
    # Compute curves
    print("  - Full model...")
    full_data = compute_bifurcation_curve(s_R_values, model='full')
    
    print("  - Reduced model...")
    reduced_data = compute_bifurcation_curve(s_R_values, model='reduced')
    
    # Find exact thresholds
    print("\nFinding exact bifurcation thresholds...")
    s_R_full = find_hopf_bifurcation_threshold(model='full', s_R_range=s_R_range)
    s_R_reduced = find_hopf_bifurcation_threshold(model='reduced', s_R_range=s_R_range)
    
    # Plot — muted palette, compact canvas (Fig. 9)
    sc = PLOT_CONFIG['SUMMARY_FIG_COLORS']
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["FIGSIZE"]["bifurcation_diagram"])

        ax.plot(
            full_data['s_R'], full_data['max_real_parts'],
            'o-', label='Full (7D)', linewidth=1.5, markersize=4.5,
            color=sc['line_full'],
        )
        ax.plot(
            reduced_data['s_R'], reduced_data['max_real_parts'],
            's-', label='Reduced (2D)', linewidth=1.5, markersize=4.5,
            color=sc['line_reduced'],
        )

        if s_R_full:
            ax.axvline(
                s_R_full, color=sc['hopf_full'], linestyle='--', alpha=0.85, linewidth=1.15,
                label=f'Full $s_R^*$ = {s_R_full:.4f}',
            )
        if s_R_reduced:
            ax.axvline(
                s_R_reduced, color=sc['hopf_reduced'], linestyle='--', alpha=0.85, linewidth=1.15,
                label=f'Reduced $s_R^*$ = {s_R_reduced:.4f}',
            )

        test_points = [
            (0.03, 'Steady', sc['test_low']),
            (0.088, 'Near-threshold', sc['test_mid']),
            (0.2, 'Oscillatory', sc['test_high']),
        ]

        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        edge = neutral_colors['gray_dark']
        for s_R_test, label, color in test_points:
            idx_full = np.argmin(np.abs(full_data['s_R'] - s_R_test))
            idx_reduced = np.argmin(np.abs(reduced_data['s_R'] - s_R_test))
            y_full = full_data['max_real_parts'][idx_full]
            y_reduced = reduced_data['max_real_parts'][idx_reduced]
            ax.plot(s_R_test, y_full, 'o', color=sc['line_full'], markersize=7,
                    markeredgecolor=edge, markeredgewidth=0.9, zorder=5)
            ax.plot(s_R_test, y_reduced, 's', color=sc['line_reduced'], markersize=7,
                    markeredgecolor=edge, markeredgewidth=0.9, zorder=5)

        ax.axhline(0, color=neutral_colors['gray_dark'], linestyle='-', linewidth=0.8, alpha=0.45)

        ax.fill_between(s_R_values, -1, 0, alpha=0.55, color=sc['region_stable'])
        ax.fill_between(s_R_values, 0, 1, alpha=0.5, color=sc['region_unstable'])

        ax.set_xlabel('$s_R$ (repressor degradation rate)', fontsize=10)
        ax.set_ylabel(r'max Re($\lambda$) at equilibrium', fontsize=10)
        ax.legend(loc='upper left', fontsize=7, framealpha=0.92, ncol=2)
        ax.grid(True, alpha=0.22)

        plt.tight_layout(pad=0.6)
        out_path = f'{figures_dir}/bifurcation_diagram.png'
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved: {out_path}")
    
    return {
        'full_data': full_data,
        'reduced_data': reduced_data,
        's_R_full': s_R_full,
        's_R_reduced': s_R_reduced,
    }


def run_bifurcation_visualization():
    """Entry point for generating the bifurcation diagram (Figure 9)."""
    import os
    figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
    os.makedirs(figures_dir, exist_ok=True)
    
    print("="*80)
    print("Bifurcation Analysis Visualization")
    print("="*80)
    print()
    
    # Main bifurcation diagram (Figure 9)
    results = plot_bifurcation_diagram(s_R_range=(0.05, 0.15), n_points=25)
    
    print()
    print("="*80)
    print("Summary:")
    print("="*80)
    
    if results['s_R_full'] and results['s_R_reduced']:
        shift = results['s_R_reduced'] - results['s_R_full']
        print(f"Full model Hopf bifurcation:    s_R = {results['s_R_full']:.6f}")
        print(f"Reduced model Hopf bifurcation: s_R = {results['s_R_reduced']:.6f}")
        print(f"Threshold shift:                Δs_R = {shift:+.6f}")
    
    print("="*80)


if __name__ == '__main__':
    run_bifurcation_visualization()


