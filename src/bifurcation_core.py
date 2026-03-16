import numpy as np
from scipy.optimize import root, root_scalar

from .config import DEFAULT_PARAMS
from .ode_system import vkb_ode, vkb_jac
from .reduced_model import reduced_vkb_ode, reduced_vkb_jac


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


def find_hopf_bifurcation_threshold(model='reduced', s_R_range=(0.05, 0.15), tol=1e-6):
    def objective(s_R_val):
        return compute_max_real_part_at_equilibrium(s_R_val, model=model)

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
        print(f"Warning: Root finding did not converge for {model} model")
        return None
    except Exception as e:
        print(f"Error in root finding for {model} model: {e}")
        return None


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
