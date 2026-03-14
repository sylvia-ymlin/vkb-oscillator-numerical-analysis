import numpy as np

from .config import DEFAULT_PARAMS


REDUCED_SPECIES = ["R", "C"]


def activator_quasi_steady_state(R, p):
    R = float(R)
    kd = p["theta_A"] / p["r_A"]
    gamma_r = p["b_A"] / (p["s_MA"] * (p["s_A"] + p["r_C"] * R))

    linear_term = p["a_A_"] * gamma_r - kd
    discriminant = linear_term**2 + 4.0 * p["a_A"] * gamma_r * kd
    return 0.5 * (linear_term + np.sqrt(discriminant))


def promoter_qss(A, theta, binding_rate):
    denominator = theta + binding_rate * A
    free = theta / denominator
    bound = (binding_rate * A) / denominator
    return free, bound


def mr_quasi_steady_state(A, p):
    d_r, d_r_bound = promoter_qss(A, p["theta_R"], p["r_R"])
    return (p["a_R"] * d_r + p["a_R_"] * d_r_bound) / p["s_MR"]


def reduced_vkb_ode(t, y, p):
    R, C = y
    A_tilde = activator_quasi_steady_state(R, p)
    MR_tilde = mr_quasi_steady_state(A_tilde, p)

    dR = p["b_R"] * MR_tilde - p["r_C"] * A_tilde * R + p["s_A"] * C - p["s_R"] * R
    dC = p["r_C"] * A_tilde * R - p["s_A"] * C
    return [dR, dC]


def reduced_observables(result, p=None):
    if p is None:
        p = DEFAULT_PARAMS

    R = np.asarray(result["R"])
    A = np.array([activator_quasi_steady_state(value, p) for value in R])
    C = np.asarray(result["C"])
    return {
        "time": np.asarray(result["time"]),
        "A": A,
        "R": R,
        "C": C,
    }


def reduced_vkb_jac(t, y, p):
    """
    Analytical Jacobian of the reduced 2-variable VKB model.
    
    The reduced system is:
        dR/dt = b_R * MR_tilde(A_tilde(R)) - r_C * A_tilde(R) * R + s_A * C - s_R * R
        dC/dt = r_C * A_tilde(R) * R - s_A * C
    
    where A_tilde(R) and MR_tilde(A) are quasi-steady-state expressions.
    
    Parameters:
        t: time (unused, for compatibility with scipy.integrate)
        y: state vector [R, C]
        p: parameter dictionary
    
    Returns:
        J: 2×2 Jacobian matrix [[∂R'/∂R, ∂R'/∂C], [∂C'/∂R, ∂C'/∂C]]
    """
    R, C = y
    R = float(R)
    
    # Compute A_tilde and its derivative w.r.t. R
    kd = p["theta_A"] / p["r_A"]
    gamma_r = p["b_A"] / (p["s_MA"] * (p["s_A"] + p["r_C"] * R))
    
    linear_term = p["a_A_"] * gamma_r - kd
    discriminant = linear_term**2 + 4.0 * p["a_A"] * gamma_r * kd
    sqrt_disc = np.sqrt(discriminant)
    
    A_tilde = 0.5 * (linear_term + sqrt_disc)
    
    # Derivative of gamma_r w.r.t. R
    dgamma_r_dR = -p["b_A"] * p["r_C"] / (p["s_MA"] * (p["s_A"] + p["r_C"] * R)**2)
    
    # Derivative of discriminant w.r.t. R
    dlinear_dR = p["a_A_"] * dgamma_r_dR
    ddisc_dR = 2.0 * linear_term * dlinear_dR + 4.0 * p["a_A"] * kd * dgamma_r_dR
    
    # Derivative of A_tilde w.r.t. R (chain rule)
    dA_dR = 0.5 * (dlinear_dR + 0.5 * ddisc_dR / sqrt_disc)
    
    # Compute MR_tilde and its derivative w.r.t. A
    theta_R = p["theta_R"]
    r_R = p["r_R"]
    denom = theta_R + r_R * A_tilde
    
    d_r = theta_R / denom
    d_r_bound = (r_R * A_tilde) / denom
    
    MR_tilde = (p["a_R"] * d_r + p["a_R_"] * d_r_bound) / p["s_MR"]
    
    # Derivative of d_r and d_r_bound w.r.t. A
    dd_r_dA = -theta_R * r_R / denom**2
    dd_r_bound_dA = r_R * theta_R / denom**2
    
    # Derivative of MR_tilde w.r.t. A
    dMR_dA = (p["a_R"] * dd_r_dA + p["a_R_"] * dd_r_bound_dA) / p["s_MR"]
    
    # Chain rule: dMR_tilde/dR = dMR_tilde/dA * dA/dR
    dMR_dR = dMR_dA * dA_dR
    
    # Construct Jacobian
    # dR'/dR = b_R * dMR_dR - r_C * (dA_dR * R + A_tilde) - s_R
    J_RR = p["b_R"] * dMR_dR - p["r_C"] * (dA_dR * R + A_tilde) - p["s_R"]
    
    # dR'/dC = s_A
    J_RC = p["s_A"]
    
    # dC'/dR = r_C * (dA_dR * R + A_tilde)
    J_CR = p["r_C"] * (dA_dR * R + A_tilde)
    
    # dC'/dC = -s_A
    J_CC = -p["s_A"]
    
    return np.array([[J_RR, J_RC],
                     [J_CR, J_CC]])


def project_full_to_reduced_initial_condition(y_full_9d, p):
    """
    Project a 9D full model initial condition to the 2D reduced model slow manifold.
    
    This ensures that when comparing full vs reduced models, both start from
    equivalent states on the slow manifold, avoiding artifacts from initial transients.
    
    Parameters:
        y_full_9d: 9D state vector [D_A, D_R, D_A', D_R', MA, A, MR, R, C]
        p: parameter dictionary
    
    Returns:
        y_reduced_2d: [R, C] on the slow manifold
    
    Algorithm:
        1. Extract R and C directly from full state (these are slow variables)
        2. Verify that fast variables are approximately on the slow manifold
        3. If not, issue a warning (initial transient expected)
    """
    D_A, D_R, D_A_, D_R_, MA, A, MR, R, C = y_full_9d
    
    # Slow variables are directly extracted
    R_reduced = float(R)
    C_reduced = float(C)
    
    # Verify consistency: check if fast variables match QSSA predictions
    A_qss = activator_quasi_steady_state(R_reduced, p)
    
    # Check promoter occupancy
    D_A_qss, D_A_prime_qss = promoter_qss(A_qss, p["theta_A"], p["r_A"])
    D_R_qss, D_R_prime_qss = promoter_qss(A_qss, p["theta_R"], p["r_R"])
    
    # Check mRNA
    MR_qss = mr_quasi_steady_state(A_qss, p)
    MA_qss = (p["a_A"] * D_A_qss + p["a_A_"] * D_A_prime_qss) / p["s_MA"]
    
    # Compute relative errors
    errors = {
        'A': abs(A - A_qss) / (abs(A_qss) + 1e-10),
        'D_A': abs(D_A - D_A_qss) / (abs(D_A_qss) + 1e-10),
        'D_R': abs(D_R - D_R_qss) / (abs(D_R_qss) + 1e-10),
        'MA': abs(MA - MA_qss) / (abs(MA_qss) + 1e-10),
        'MR': abs(MR - MR_qss) / (abs(MR_qss) + 1e-10),
    }
    
    max_error = max(errors.values())
    
    if max_error > 0.1:  # 10% relative error threshold
        import warnings
        warnings.warn(
            f"Initial condition projection: fast variables deviate from slow manifold.\n"
            f"  Max relative error: {max_error:.2%}\n"
            f"  Errors: {errors}\n"
            f"  This will cause initial transients in the reduced model.\n"
            f"  Consider using equilibrium or quasi-steady-state initial conditions."
        )
    
    return np.array([R_reduced, C_reduced])

