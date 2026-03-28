import numpy as np
from src.config import DEFAULT_PARAMS, SPECIES


def vkb_ode(t, y, p):
    """Deterministic ordinary differential equations (ODEs) for the VKB oscillator."""
    D_A, D_R, D_A_, D_R_, MA, A, MR, R, C = y
    
    rcl = p['r_C'] * A * R
    ra_da_a = p['r_A'] * D_A * A
    rr_dr_a = p['r_R'] * D_R * A
    th_a_da_ = p['theta_A'] * D_A_
    th_r_dr_ = p['theta_R'] * D_R_
    
    dD_A = -ra_da_a + th_a_da_
    dD_R = -rr_dr_a + th_r_dr_
    dD_A_ = ra_da_a - th_a_da_
    dD_R_ = rr_dr_a - th_r_dr_
    dMA = p['a_A'] * D_A + p['a_A_'] * D_A_ - p['s_MA'] * MA
    dA = -rcl - p['s_A'] * A - ra_da_a - rr_dr_a + th_a_da_ + th_r_dr_ + p['b_A'] * MA
    dMR = p['a_R'] * D_R + p['a_R_'] * D_R_ - p['s_MR'] * MR
    dR = -rcl + p['s_A'] * C - p['s_R'] * R + p['b_R'] * MR
    dC = rcl - p['s_A'] * C
    
    return [dD_A, dD_R, dD_A_, dD_R_, dMA, dA, dMR, dR, dC]

def vkb_jac(t, y, p):
    D_A, D_R, D_A_, D_R_, MA, A, MR, R, C = y
    J = np.zeros((9, 9))
    
    # 0:D_A, 1:D_R, 2:D_A_, 3:D_R_, 4:MA, 5:A, 6:MR, 7:R, 8:C
    
    J[0, 0] = -p['r_A'] * A
    J[0, 2] = p['theta_A']
    J[0, 5] = -p['r_A'] * D_A
    
    J[1, 1] = -p['r_R'] * A
    J[1, 3] = p['theta_R']
    J[1, 5] = -p['r_R'] * D_R
    
    J[2, 0] = p['r_A'] * A
    J[2, 2] = -p['theta_A']
    J[2, 5] = p['r_A'] * D_A
    
    J[3, 1] = p['r_R'] * A
    J[3, 3] = -p['theta_R']
    J[3, 5] = p['r_R'] * D_R
    
    J[4, 0] = p['a_A']
    J[4, 2] = p['a_A_']
    J[4, 4] = -p['s_MA']
    
    J[5, 0] = -p['r_A'] * A
    J[5, 1] = -p['r_R'] * A
    J[5, 2] = p['theta_A']
    J[5, 3] = p['theta_R']
    J[5, 4] = p['b_A']
    J[5, 5] = -p['r_C'] * R - p['s_A'] - p['r_A'] * D_A - p['r_R'] * D_R
    J[5, 7] = -p['r_C'] * A
    
    J[6, 1] = p['a_R']
    J[6, 3] = p['a_R_']
    J[6, 6] = -p['s_MR']
    
    J[7, 5] = -p['r_C'] * R
    J[7, 6] = p['b_R']
    J[7, 7] = -p['r_C'] * A - p['s_R']
    J[7, 8] = p['s_A']
    
    J[8, 5] = p['r_C'] * R
    J[8, 7] = p['r_C'] * A
    J[8, 8] = -p['s_A']
    
    return J






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


