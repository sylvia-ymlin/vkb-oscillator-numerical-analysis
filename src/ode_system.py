import numpy as np

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

