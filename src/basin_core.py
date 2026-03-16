from collections import Counter

import numpy as np
from scipy.integrate import solve_ivp

from .bifurcation_analysis import find_full_equilibrium
from .config import (
    BIOLOGICAL_INITIAL_CONDITION,
    DEFAULT_PARAMS,
    ODE_TOLERANCES,
    SPECIES,
)
from .ode_system import vkb_ode
from .reduced_model import reduced_vkb_ode


def classify_trajectory_fate(time, trajectory, equilibrium_threshold=0.01, tail_time=200):
    tail_start_time = time[-1] - tail_time
    tail_mask = time >= tail_start_time
    traj_tail = trajectory[tail_mask]

    if traj_tail.ndim > 1:
        amp_R = np.max(traj_tail[:, 0]) - np.min(traj_tail[:, 0])
        amp_C = np.max(traj_tail[:, 1]) - np.min(traj_tail[:, 1])
        amplitude = max(amp_R, amp_C)
    else:
        amplitude = np.max(traj_tail) - np.min(traj_tail)

    mean_value = np.mean(traj_tail, axis=0)
    fate = 'equilibrium' if amplitude < equilibrium_threshold else 'limit_cycle'

    return {'fate': fate, 'amplitude': amplitude, 'mean_value': mean_value}


def compute_basin_of_attraction_reduced(s_R, R_range=(0, 200), C_range=(0, 200), n_grid=20, t_max=1000, verbose=True):
    params = {**DEFAULT_PARAMS, 's_R': s_R}

    R_vals = np.linspace(R_range[0], R_range[1], n_grid)
    C_vals = np.linspace(C_range[0], C_range[1], n_grid)
    R_grid, C_grid = np.meshgrid(R_vals, C_vals)

    fates = np.empty((n_grid, n_grid), dtype=object)
    amplitudes = np.zeros((n_grid, n_grid))
    details = []

    total = n_grid * n_grid
    count = 0

    for i in range(n_grid):
        for j in range(n_grid):
            count += 1
            if verbose and count % 10 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

            y0 = [R_grid[i, j], C_grid[i, j]]

            try:
                sol = solve_ivp(
                    lambda t, y: reduced_vkb_ode(t, y, params),
                    (0, t_max),
                    y0,
                    method='BDF',
                    rtol=ODE_TOLERANCES['rtol'],
                    atol=ODE_TOLERANCES['atol'],
                    dense_output=False,
                )

                if not sol.success:
                    fates[i, j] = 'failed'
                    amplitudes[i, j] = np.nan
                    details.append({'fate': 'failed', 'y0': y0})
                    continue

                result = classify_trajectory_fate(sol.t, sol.y.T)
                fates[i, j] = result['fate']
                amplitudes[i, j] = result['amplitude']
                details.append({**result, 'y0': y0})

            except Exception as e:
                if verbose:
                    print(f"    Error at R={R_grid[i,j]:.1f}, C={C_grid[i,j]:.1f}: {e}")
                fates[i, j] = 'error'
                amplitudes[i, j] = np.nan
                details.append({'fate': 'error', 'y0': y0, 'error': str(e)})

    return {
        'R_grid': R_grid,
        'C_grid': C_grid,
        'fates': fates,
        'amplitudes': amplitudes,
        'details': details,
        's_R': s_R,
    }


def probe_full_model_attractor(s_R=0.088, n_random=16, t_max=800, verbose=True):
    INDEP_IDX = [0, 1, 4, 5, 6, 7, 8]
    R_idx = SPECIES.index('R')

    params = {**DEFAULT_PARAMS, 's_R': s_R}

    if verbose:
        print(f"Finding equilibrium point for s_R = {s_R}...")
    y_star = find_full_equilibrium(s_R)
    if y_star is None:
        print("  ERROR: Could not find equilibrium point.")
        return None

    norm_ystar = np.linalg.norm(y_star)
    if verbose:
        print(f"  Equilibrium: R* = {y_star[R_idx]:.4f}, C* = {y_star[8]:.4f},  ||y*|| = {norm_ystar:.4f}")
        print()

    def _enforce_conservation(y):
        y = np.clip(np.array(y, dtype=float), 0.0, None)
        y[0] = np.clip(y[0], 0.0, 1.0)
        y[1] = np.clip(y[1], 0.0, 1.0)
        y[2] = 1.0 - y[0]
        y[3] = 1.0 - y[1]
        return y

    initial_conditions, labels = [], []
    initial_conditions.append(_enforce_conservation(BIOLOGICAL_INITIAL_CONDITION))
    labels.append('biological_IC')

    rng = np.random.default_rng(seed=42)
    for scale in [0.01, 0.05, 0.10, 0.20]:
        for _ in range(n_random):
            delta_7d = rng.standard_normal(7)
            delta_7d /= (np.linalg.norm(delta_7d) + 1e-15)
            y_p = y_star.copy()
            for k, idx in enumerate(INDEP_IDX):
                y_p[idx] += scale * norm_ystar * delta_7d[k]
            initial_conditions.append(_enforce_conservation(y_p))
            labels.append(f'rand_{scale:.0%}')

    for idx in INDEP_IDX:
        y_p = y_star.copy()
        y_p[idx] += 0.10 * norm_ystar
        initial_conditions.append(_enforce_conservation(y_p))
        labels.append(f'axis+_var{idx}')

    total = len(initial_conditions)
    if verbose:
        print(f"Testing {total} initial conditions  (1 biological + {4 * n_random} random + {len(INDEP_IDX)} axis-aligned) ...")
        print()

    fates = []
    trajectories = []
    t_eval = np.linspace(0, t_max, 3000)

    for i, y0 in enumerate(initial_conditions):
        if verbose and (i % 15 == 0 or i == total - 1):
            print(f"  [{i+1:3d}/{total}]  label = {labels[i]}")
        try:
            sol = solve_ivp(
                lambda t, y: vkb_ode(t, y, params),
                (0, t_max),
                y0.tolist(),
                method='BDF',
                t_eval=t_eval,
                rtol=ODE_TOLERANCES['rtol'],
                atol=ODE_TOLERANCES['atol'],
                dense_output=False,
            )
            if not sol.success:
                fates.append('failed')
                trajectories.append(None)
                continue
            result = classify_trajectory_fate(sol.t, sol.y[R_idx, :])
            fates.append(result['fate'])
            trajectories.append({'t': sol.t, 'R': sol.y[R_idx, :], 'C': sol.y[8, :]})
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            fates.append('error')
            trajectories.append(None)

    return {
        's_R': s_R,
        'equilibrium': y_star,
        'initial_conditions': initial_conditions,
        'labels': labels,
        'fates': fates,
        'trajectories': trajectories,
        'summary': dict(Counter(fates)),
    }
