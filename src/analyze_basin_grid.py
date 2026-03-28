import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.integrate import solve_ivp
from src.config import DEFAULT_PARAMS, ODE_TOLERANCES, ODE_TOLERANCES_LOOSE, RC_PARAMS, SPECIES, PLOT_CONFIG
from src.models import (
    vkb_ode, 
    activator_quasi_steady_state, 
    mr_quasi_steady_state, 
    promoter_qss,
    reduced_vkb_ode
)
from src.analyze_basin import classify_trajectory_fate
from src.analyze_bifurcation import find_full_equilibrium


def initialize_9d_from_rc(R, C, params):
    """
    Initialize all 9 species of the full model from slow variables R and C 
    using the Quasi-Steady State Approximation (QSSA).
    """
    R = float(R)
    C = float(C)
    
    # 1. Activator QSS based on current R
    A = activator_quasi_steady_state(R, params)
    
    # 2. Promoter QSS based on A
    DA, DA_prime = promoter_qss(A, params['theta_A'], params['r_A'])
    DR, DR_prime = promoter_qss(A, params['theta_R'], params['r_R'])
    
    # 3. mRNA QSS based on promoter states
    MA = (params['a_A'] * DA + params['a_A_'] * DA_prime) / params['s_MA']
    MR = (params['a_R'] * DR + params['a_R_'] * DR_prime) / params['s_MR']
    
    # Order: [D_A, D_R, D_A_, D_R_, MA, A, MR, R, C]
    return [DA, DR, DA_prime, DR_prime, MA, A, MR, R, C]


def simulate_single_ic(y0, params, t_max=1000, tolerances=ODE_TOLERANCES_LOOSE):
    """Integrate the full model for a single initial condition and return fate."""
    try:
        sol = solve_ivp(
            lambda t, y: vkb_ode(t, y, params),
            (0, t_max),
            y0,
            method='BDF',
            rtol=tolerances['rtol'],
            atol=tolerances['atol'],
        )
        
        if not sol.success:
            return 'failed', 0.0
            
        # SPECIES = ['D_A', 'D_R', 'D_A_', 'D_R_', 'MA', 'A', 'MR', 'R', 'C'] -> R is index 7
        result = classify_trajectory_fate(sol.t, sol.y[7, :])
        return result['fate'], result['amplitude']
        
    except Exception:
        return 'error', 0.0


def run_basin_scan(s_R=0.088, n_grid=50, R_range=(0, 2000), C_range=(0, 2500), t_max=1000, max_workers=None, tolerances=ODE_TOLERANCES_LOOSE):
    """
    Perform a parallel grid scan on the R-C plane.
    """
    params = {**DEFAULT_PARAMS, 's_R': s_R}
    
    R_vals = np.linspace(R_range[0], R_range[1], n_grid)
    C_vals = np.linspace(C_range[0], C_range[1], n_grid)
    R_grid, C_grid = np.meshgrid(R_vals, C_vals)
    
    fates = np.empty((n_grid, n_grid), dtype=object)
    amplitudes = np.zeros((n_grid, n_grid))
    
    tasks = []
    for i in range(n_grid):
        for j in range(n_grid):
            y0 = initialize_9d_from_rc(R_grid[i, j], C_grid[i, j], params)
            tasks.append((i, j, y0))
            
    print(f"Starting basin scan for s_R={s_R} on {n_grid}x{n_grid} grid ({n_grid**2} points)...")
    print(f"Using tolerances: {tolerances}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(simulate_single_ic, y0, params, t_max, tolerances): (i, j) for i, j, y0 in tasks}
        
        completed = 0
        total = n_grid**2
        for future in as_completed(futures):
            i, j = futures[future]
            fate, amp = future.result()
            fates[i, j] = fate
            amplitudes[i, j] = amp
            
            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{total} ({100*completed/total:.1f}%)")
                
    results = {
        's_R': s_R,
        'R_grid': R_grid,
        'C_grid': C_grid,
        'fates': fates,
        'amplitudes': amplitudes,
        'R_range': R_range,
        'C_range': C_range,
        'tolerances': tolerances
    }
    
    # Save results to a file
    res_path = f"materials/basin_scan_sR_{s_R:.3f}_{n_grid}x{n_grid}.pkl"
    os.makedirs('materials', exist_ok=True)
    with open(res_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {res_path}")
    
    return results


def plot_basin_grid(scan_results, output_dir='figures'):
    """Visualize the grid scan results with fate heatmap and nullclines."""
    s_R = scan_results['s_R']
    R_grid = scan_results['R_grid']
    C_grid = scan_results['C_grid']
    fates = scan_results['fates']
    
    fate_map = {'equilibrium': 0, 'limit_cycle': 1, 'failed': -1, 'error': -1}
    fate_numeric = np.vectorize(lambda x: fate_map.get(x, -1))(fates)
    
    y_star = find_full_equilibrium(s_R)
    R_star, C_star = (y_star[7], y_star[8]) if y_star is not None else (None, None)
    
    params = {**DEFAULT_PARAMS, 's_R': s_R}
    R_arr = np.linspace(max(1.0, np.min(R_grid)), np.max(R_grid), 500)
    C_nc = []
    R_nc = []
    
    for Rv in R_arr:
        Aq = activator_quasi_steady_state(Rv, params)
        MRq = mr_quasi_steady_state(Aq, params)
        C_nc.append(params['r_C'] * Aq * Rv / params['s_A'])
        R_nc.append((params['r_C'] * Aq * Rv + params['s_R'] * Rv - params['b_R'] * MRq) / params['s_A'])
    
    sc = PLOT_CONFIG['SUMMARY_FIG_COLORS']
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["FIGSIZE"]["heatmap"])

        # Hatch filling for the basin domains
        # We draw the limit_cycle basin (1) and equilibrium basin (0) with hatches instead of solid fill.

        # Contour for the equilibrium basin boundary
        if np.any(fate_numeric == 0) and np.any(fate_numeric == 1):
            ax.contour(
                R_grid, C_grid, fate_numeric, levels=[0.5],
                colors=[PLOT_CONFIG['SUMMARY_FIG_COLORS']['line_full']],
                linewidths=1.2, linestyles='-', zorder=5,
            )

        # Limit-cycle basin (fate_numeric == 1)
        with plt.rc_context({'hatch.color': sc['basin_lc'], 'hatch.linewidth': 1.0}):
            ax.contourf(
                R_grid, C_grid, fate_numeric, levels=[0.5, 1.5],
                colors='none', hatches=['////'], zorder=1,
            )

        # Equilibrium basin (fate_numeric == 0)
        with plt.rc_context({'hatch.color': sc['basin_eq'], 'hatch.linewidth': 1.0}):
            ax.contourf(
                R_grid, C_grid, fate_numeric, levels=[-0.5, 0.5],
                colors='none', hatches=['\\\\\\\\'], zorder=1,
            )

        ax.plot(R_arr, C_nc, '--', color=sc['line_full'], lw=1.1, alpha=0.75, label=r'$\dot{C}=0$ (reduced)', zorder=2)
        ax.plot(R_arr, R_nc, ':', color=sc['hopf_reduced'], lw=1.1, alpha=0.75, label=r'$\dot{R}=0$ (reduced)', zorder=2)

        if R_star is not None:
            ax.scatter(
                R_star, C_star, color=sc['eq_cross'], s=95, marker='x', linewidths=2.0,
                label='Stable equilibrium (full)', zorder=10,
            )

        ax.set_xlabel('Repressor (R)', fontsize=10)
        ax.set_ylabel('Complex (C)', fontsize=10)
        ax.set_title(f'Basin scan (full 9D), $s_R = {s_R}$', fontsize=10.5, pad=6)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        custom_lines = [
            Rectangle((0,0), 1, 1, facecolor='none', edgecolor=sc['basin_eq'], hatch=r'\\\\', linewidth=0, label='Equilibrium basin'),
            Rectangle((0,0), 1, 1, facecolor='none', edgecolor=sc['basin_lc'], hatch=r'////', linewidth=0, label='Limit-cycle basin'),
            Line2D([0], [0], color=sc['line_full'], lw=1.15, label='Basin boundary'),
            Line2D([0], [0], color=sc['eq_cross'], marker='x', ls='', ms=8, mew=1.8, label='Stable equilibrium'),
        ]
        ax.legend(handles=custom_lines, loc='upper right', fontsize=7.5, framealpha=0.92)

        # Zoom in for clearer view of the boundary based on the user's request
        ax.set_xlim([0, 1500])
        ax.set_ylim([0, 1500])

        plt.tight_layout(pad=0.5)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'basin_grid_sR_{s_R:.3f}.png')
        fig.savefig(out_path, dpi=PLOT_CONFIG['DPI'], bbox_inches='tight')
        plt.close(fig)
        print(f"Saved basin grid plot to {out_path}")
        return out_path


if __name__ == "__main__":
    # Prefer the repo entry point: `python main.py basin-grid` or `python main.py basin-grid --plot-only`
    import sys
    s_R_crit = 0.088
    n_grid = 50

    res_path = f"materials/basin_scan_sR_{s_R_crit:.3f}_{n_grid}x{n_grid}.pkl"

    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        print(f"Loading results from {res_path}...")
        with open(res_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = run_basin_scan(s_R=s_R_crit, n_grid=n_grid, tolerances=ODE_TOLERANCES_LOOSE)
        
    plot_basin_grid(results)
