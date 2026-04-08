import numpy as np


def vkb_ode(t, y, p):
    D_A, D_R, D_A_p, D_R_p, MA, A, MR, R, C = y

    rcA, rA, rR = p['r_C']*A, p['r_A']*A, p['r_R']*A
    thA, thR = p['theta_A']*D_A_p, p['theta_R']*D_R_p

    dD_A   = -rA*D_A + thA
    dD_R   = -rR*D_R + thR
    dD_A_p = -dD_A
    dD_R_p = -dD_R

    dMA = p['a_A']*D_A + p['a_A_']*D_A_p - p['s_MA']*MA
    dMR = p['a_R']*D_R + p['a_R_']*D_R_p - p['s_MR']*MR

    dA = -rcA*R - p['s_A']*A - rA*D_A - rR*D_R + thA + thR + p['b_A']*MA
    dR = -rcA*R + p['s_A']*C - p['s_R']*R + p['b_R']*MR
    dC =  rcA*R - p['s_A']*C

    return [dD_A, dD_R, dD_A_p, dD_R_p, dMA, dA, dMR, dR, dC]


def vkb_jac(t, y, p):
    # Analytical Jacobian for BDF/Radau — avoids finite-difference overhead on a stiff system
    D_A, D_R, D_A_p, D_R_p, MA, A, MR, R, C = y
    J = np.zeros((9, 9))

    J[0, 0], J[0, 5] = -p['r_A']*A, -p['r_A']*D_A
    J[0, 2] = p['theta_A']

    J[1, 1], J[1, 5] = -p['r_R']*A, -p['r_R']*D_R
    J[1, 3] = p['theta_R']

    J[2, 0], J[2, 5] =  p['r_A']*A,  p['r_A']*D_A
    J[2, 2] = -p['theta_A']

    J[3, 1], J[3, 5] =  p['r_R']*A,  p['r_R']*D_R
    J[3, 3] = -p['theta_R']

    J[4, 0], J[4, 2], J[4, 4] = p['a_A'], p['a_A_'], -p['s_MA']
    J[6, 1], J[6, 3], J[6, 6] = p['a_R'], p['a_R_'], -p['s_MR']

    J[5, 0], J[5, 1], J[5, 2], J[5, 3], J[5, 4] = -p['r_A']*A, -p['r_R']*A, p['theta_A'], p['theta_R'], p['b_A']
    J[5, 5] = -p['r_C']*R - p['s_A'] - p['r_A']*D_A - p['r_R']*D_R
    J[5, 7] = -p['r_C']*A

    J[7, 5], J[7, 6], J[7, 7], J[7, 8] = -p['r_C']*R, p['b_R'], -p['r_C']*A - p['s_R'], p['s_A']

    J[8, 5], J[8, 7], J[8, 8] = p['r_C']*R, p['r_C']*A, -p['s_A']

    return J
