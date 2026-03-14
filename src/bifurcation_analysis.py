"""
Bifurcation analysis for both full and reduced VKB oscillator models.

This module computes Hopf bifurcation thresholds by analyzing equilibrium point
stability, NOT by sampling along trajectories.

Mathematical approach:
1. For each s_R value, solve the algebraic system f(y*) = 0 to find equilibrium
2. Compute Jacobian J at the equilibrium point y*
3. Extract eigenvalues λ of J
4. Hopf bifurcation occurs when Re(λ) crosses zero (with Im(λ) ≠ 0)

Note on conservation laws:
The full 9-variable system has two conservation laws (D_A + D_A' = 1, D_R + D_R' = 1),
making the 9×9 Jacobian singular with two zero eigenvalues. To avoid numerical artifacts,
we solve equilibrium in the 7-dimensional independent subspace and compute eigenvalues
of the 7×7 reduced Jacobian.
"""

import numpy as np
from scipy.optimize import root, root_scalar

from .config import DEFAULT_PARAMS
from .ode_system import vkb_ode, vkb_jac
from .reduced_model import reduced_vkb_ode, reduced_vkb_jac


# ==============================================================================
# Equilibrium point computation
# ==============================================================================

def find_full_equilibrium(s_R, initial_guess=None):
    """
    Find equilibrium point of the full 9-variable system by solving in the 
    7-dimensional independent subspace (enforcing conservation laws).
    
    The system has two conservation laws:
    - D_A + D_A' = 1 (activator promoter total)
    - D_R + D_R' = 1 (repressor promoter total)
    
    This makes the 9×9 Jacobian singular. To avoid numerical artifacts, we solve
    for only the 7 independent variables and reconstruct the full state.
    
    Parameters:
        s_R: repressor degradation rate
        initial_guess: initial guess (can be 9D or 7D)
    
    Returns:
        y_star: equilibrium point (9-dimensional array)
    """
    params = {**DEFAULT_PARAMS, 's_R': s_R}
    
    if initial_guess is None:
        # Initial guess for 7 independent variables: D_A, D_R, MA, A, MR, R, C
        initial_guess_7d = [0.5, 0.5, 5.0, 50.0, 5.0, 50.0, 50.0]
    else:
        if len(initial_guess) == 9:
            # Extract 7 independent variables from 9D guess (for continuation)
            initial_guess_7d = [
                initial_guess[0],  # D_A
                initial_guess[1],  # D_R
                initial_guess[4],  # MA
                initial_guess[5],  # A
                initial_guess[6],  # MR
                initial_guess[7],  # R
                initial_guess[8],  # C
            ]
        else:
            initial_guess_7d = initial_guess
    
    def system_eqs_7d(y7):
        """
        Equations for 7 independent variables.
        Enforces conservation laws: D_A' = 1 - D_A, D_R' = 1 - D_R
        """
        D_A, D_R, MA, A, MR, R, C = y7
        
        # Apply conservation laws
        D_A_ = 1.0 - D_A
        D_R_ = 1.0 - D_R
        
        # Construct full 9D state
        y9 = [D_A, D_R, D_A_, D_R_, MA, A, MR, R, C]
        
        # Compute 9D derivatives
        dy9 = vkb_ode(0, y9, params)
        
        # Return only derivatives of independent variables
        return [dy9[0], dy9[1], dy9[4], dy9[5], dy9[6], dy9[7], dy9[8]]
    
    # Solve in 7D space (non-singular Jacobian)
    sol = root(system_eqs_7d, initial_guess_7d, method='hybr')
    
    if not sol.success:
        raise ValueError(f"Failed to find equilibrium for s_R={s_R}: {sol.message}")
    
    # Reconstruct full 9D equilibrium state
    D_A, D_R, MA, A, MR, R, C = sol.x
    y9_star = np.array([
        D_A,
        D_R,
        1.0 - D_A,  # D_A' from conservation
        1.0 - D_R,  # D_R' from conservation
        MA,
        A,
        MR,
        R,
        C
    ])
    
    # Check for physical validity (non-negative concentrations)
    if np.any(y9_star < -1e-6):  # Allow small numerical errors
        import warnings
        warnings.warn(
            f"Non-physical equilibrium found at s_R={s_R}: "
            f"negative concentrations detected. "
            f"Min value: {np.min(y9_star):.6f}. "
            f"This may indicate numerical continuation failure or bifurcation."
        )
    
    return y9_star


def find_reduced_equilibrium(s_R, initial_guess=None):
    """
    Find equilibrium point of the reduced 2-variable system.
    
    Solves: dR/dt = 0, dC/dt = 0
    
    Parameters:
        s_R: repressor degradation rate
        initial_guess: initial guess for root finding (2-dimensional)
                      If None, uses a reasonable default
    
    Returns:
        y_star: equilibrium point [R*, C*]
    """
    params = {**DEFAULT_PARAMS, 's_R': s_R}
    
    if initial_guess is None:
        # Use a better initial guess that works across parameter regimes
        # For s_R near 0.088, equilibrium is around (R~95, C~320)
        initial_guess = [100.0, 350.0]
    
    def system_eqs(y):
        return reduced_vkb_ode(0, y, params)
    
    sol = root(system_eqs, initial_guess, method='hybr')
    
    if not sol.success:
        raise ValueError(f"Failed to find equilibrium for s_R={s_R}: {sol.message}")
    
    return sol.x


# ==============================================================================
# Jacobian computation at equilibrium
# ==============================================================================

def full_jacobian_7d(y_star, params):
    """
    Compute the 7×7 Jacobian of the independent subsystem at equilibrium.
    
    The full 9-variable system has two conservation laws:
    - D_A + D_A' = 1
    - D_R + D_R' = 1
    
    This makes the 9×9 Jacobian singular with two zero eigenvalues. To properly
    analyze stability, we compute the Jacobian of the 7-dimensional independent
    subsystem: [D_A, D_R, MA, A, MR, R, C].
    
    Parameters:
        y_star: equilibrium point (9-dimensional)
        params: parameter dictionary
    
    Returns:
        J_7d: 7×7 Jacobian matrix of the independent subsystem
    """
    # Compute full 9×9 Jacobian
    J_9d = vkb_jac(0, y_star, params)
    
    # Extract 7×7 submatrix corresponding to independent variables
    # Independent variables: [D_A, D_R, MA, A, MR, R, C] = indices [0, 1, 4, 5, 6, 7, 8]
    # Dependent variables: [D_A', D_R'] = indices [2, 3]
    
    # Since D_A' = 1 - D_A and D_R' = 1 - D_R, we have:
    # ∂f_i/∂D_A' = -∂f_i/∂D_A and ∂f_i/∂D_R' = -∂f_i/∂D_R
    
    # Build 7×7 Jacobian by combining columns
    independent_indices = [0, 1, 4, 5, 6, 7, 8]
    dependent_indices = [2, 3]
    
    J_7d = np.zeros((7, 7))
    
    for i, row_idx in enumerate(independent_indices):
        for j, col_idx in enumerate(independent_indices):
            J_7d[i, j] = J_9d[row_idx, col_idx]
        
        # Add contributions from dependent variables
        # ∂f_i/∂D_A (effective) = ∂f_i/∂D_A + ∂f_i/∂D_A' * (∂D_A'/∂D_A) = ∂f_i/∂D_A - ∂f_i/∂D_A'
        J_7d[i, 0] += -J_9d[row_idx, 2]  # D_A' contribution to D_A column
        J_7d[i, 1] += -J_9d[row_idx, 3]  # D_R' contribution to D_R column
    
    return J_7d


# ==============================================================================
# Stability analysis at equilibrium
# ==============================================================================

def analyze_equilibrium_stability(y_star, jacobian_func, params, model_name=""):
    """
    Analyze stability of an equilibrium point.
    
    Parameters:
        y_star: equilibrium point
        jacobian_func: function to compute Jacobian
                      - For full model: full_jacobian_7d (returns 7×7 matrix)
                      - For reduced model: reduced_vkb_jac (returns 2×2 matrix)
        params: parameter dictionary
        model_name: string for display purposes
    
    Returns:
        dict with keys:
            - eigenvalues: complex array (7 or 2 eigenvalues)
            - max_real_part: maximum real part of eigenvalues
            - is_stable: True if all Re(λ) < 0
            - has_complex_pair: True if there's a conjugate pair with Im(λ) ≠ 0
    """
    # Compute Jacobian
    if jacobian_func == full_jacobian_7d:
        J = jacobian_func(y_star, params)
    elif jacobian_func == reduced_vkb_jac:
        J = jacobian_func(0, y_star, params)
    else:
        # Legacy support for vkb_jac (9×9, not recommended)
        J = jacobian_func(0, y_star, params)
    
    eigvals = np.linalg.eigvals(J)
    
    # Sort by real part (descending)
    eigvals = eigvals[np.argsort(-np.real(eigvals))]
    
    max_real = np.max(np.real(eigvals))
    is_stable = max_real < 0
    
    # Check for complex conjugate pairs (potential Hopf bifurcation)
    has_complex_pair = False
    for i, lam in enumerate(eigvals):
        if np.abs(np.imag(lam)) > 1e-6:
            # Check if conjugate exists
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


# ==============================================================================
# Hopf bifurcation threshold computation
# ==============================================================================

def compute_max_real_part_at_equilibrium(s_R, model='reduced'):
    """
    Compute the maximum real part of eigenvalues at equilibrium for a given s_R.
    
    This is the key function for finding Hopf bifurcation: the threshold is where
    this function crosses zero.
    
    Parameters:
        s_R: repressor degradation rate
        model: 'reduced' or 'full'
    
    Returns:
        max_real_part: float
    """
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
        # If equilibrium finding fails, return NaN
        print(f"Warning: Failed to compute stability at s_R={s_R} for {model} model: {e}")
        return np.nan


def find_hopf_bifurcation_threshold(model='reduced', s_R_range=(0.05, 0.15), tol=1e-6):
    """
    Find the exact s_R value where Hopf bifurcation occurs.
    
    Uses root finding to locate where max(Re(λ)) = 0.
    
    Parameters:
        model: 'reduced' or 'full'
        s_R_range: (min, max) bracket for search
        tol: tolerance for root finding
    
    Returns:
        s_R_hopf: bifurcation threshold, or None if not found
    """
    def objective(s_R_val):
        return compute_max_real_part_at_equilibrium(s_R_val, model=model)
    
    # Check if bracket is valid
    f_min = objective(s_R_range[0])
    f_max = objective(s_R_range[1])
    
    if np.isnan(f_min) or np.isnan(f_max):
        print(f"Warning: Cannot evaluate objective at bracket endpoints for {model} model")
        return None
    
    if f_min * f_max > 0:
        print(f"Warning: No sign change in bracket [{s_R_range[0]}, {s_R_range[1]}] for {model} model")
        print(f"  f({s_R_range[0]}) = {f_min:.6f}")
        print(f"  f({s_R_range[1]}) = {f_max:.6f}")
        return None
    
    try:
        sol = root_scalar(objective, bracket=s_R_range, method='brentq', xtol=tol)
        if sol.converged:
            return sol.root
        else:
            print(f"Warning: Root finding did not converge for {model} model")
            return None
    except Exception as e:
        print(f"Error in root finding for {model} model: {e}")
        return None


# ==============================================================================
# Parameter sweep for bifurcation diagram
# ==============================================================================

def compute_bifurcation_curve(s_R_values, model='reduced'):
    """
    Compute max(Re(λ)) at equilibrium for a range of s_R values.
    
    This generates the data for a bifurcation diagram using numerical continuation:
    the equilibrium from the previous s_R value is used as the initial guess for
    the next one, ensuring smooth tracking along the bifurcation curve.
    
    Parameters:
        s_R_values: array of s_R values to evaluate
        model: 'reduced' or 'full'
    
    Returns:
        dict with keys:
            - s_R: input array
            - max_real_parts: array of max(Re(λ)) values
            - equilibria: list of equilibrium points
            - all_eigenvalues: list of eigenvalue arrays
    """
    max_real_parts = []
    equilibria = []
    all_eigenvalues = []
    
    # Numerical continuation: use previous equilibrium as next guess
    current_guess = None
    
    for s_R in s_R_values:
        params = {**DEFAULT_PARAMS, 's_R': s_R}
        
        try:
            if model == 'reduced':
                # Pass current_guess for continuation
                y_star = find_reduced_equilibrium(s_R, initial_guess=current_guess)
                stability = analyze_equilibrium_stability(
                    y_star, reduced_vkb_jac, params
                )
            else:  # full
                # Pass current_guess for continuation
                y_star = find_full_equilibrium(s_R, initial_guess=current_guess)
                stability = analyze_equilibrium_stability(
                    y_star, full_jacobian_7d, params
                )
            
            # Update guess for next iteration (numerical continuation)
            current_guess = y_star
            
            max_real_parts.append(stability['max_real_part'])
            equilibria.append(y_star)
            all_eigenvalues.append(stability['eigenvalues'])
        
        except Exception as e:
            print(f"Warning: Failed at s_R={s_R} for {model} model: {e}")
            max_real_parts.append(np.nan)
            equilibria.append(None)
            all_eigenvalues.append(None)
            # Keep current_guess unchanged to try again with next s_R
    
    return {
        's_R': s_R_values,
        'max_real_parts': np.array(max_real_parts),
        'equilibria': equilibria,
        'all_eigenvalues': all_eigenvalues,
    }


# ==============================================================================
# Comparison and reporting
# ==============================================================================

def compare_bifurcation_thresholds(s_R_range=(0.05, 0.15)):
    """
    Compare Hopf bifurcation thresholds between full and reduced models.
    
    This analysis explains why QSSA breaks down at s_R ≈ 0.088.
    """
    print("=" * 80)
    print("Hopf Bifurcation Threshold Analysis")
    print("=" * 80)
    print()
    print("Computing bifurcation thresholds by analyzing equilibrium point stability...")
    print()
    
    # Find thresholds
    s_R_full = find_hopf_bifurcation_threshold(model='full', s_R_range=s_R_range)
    s_R_reduced = find_hopf_bifurcation_threshold(model='reduced', s_R_range=s_R_range)
    
    print(f"Full model:     s_R_hopf = {s_R_full:.6f}" if s_R_full else "Full model: threshold not found")
    print(f"Reduced model:  s_R_hopf = {s_R_reduced:.6f}" if s_R_reduced else "Reduced model: threshold not found")
    print()
    
    if s_R_full and s_R_reduced:
        shift = s_R_reduced - s_R_full
        shift_percent = (shift / s_R_full) * 100
        
        print(f"Threshold shift: Δs_R = {shift:+.6f} ({shift_percent:+.2f}%)")
        print()
        print("=" * 80)
        print("Interpretation:")
        print("=" * 80)
        print()
        print(f"At s_R = 0.088:")
        
        if s_R_full < 0.088 < s_R_reduced:
            print(f"  • Full model: ABOVE threshold ({s_R_full:.4f}) → Oscillating")
            print(f"  • Reduced model: BELOW threshold ({s_R_reduced:.4f}) → Stable steady state")
            print()
            print("This explains the QSSA breakdown at s_R = 0.088:")
            print("Dimension reduction alters the phase space topology, shifting the bifurcation.")
        else:
            print("  • Both models may be in the same regime at this parameter value.")
        
        print()
        print("The QSSA approximation shifts the bifurcation threshold because it")
        print("eliminates 7 fast variables, fundamentally altering the phase space structure.")
    
    print("=" * 80)
    
    return {
        'full_threshold': s_R_full,
        'reduced_threshold': s_R_reduced,
    }


if __name__ == '__main__':
    # Test equilibrium finding
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
    
    print("\n" + "="*80)
    compare_bifurcation_thresholds()
