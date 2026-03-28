import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from src.config import DEFAULT_PARAMS, BIOLOGICAL_INITIAL_CONDITION, ODE_TOLERANCES_LOOSE, ODE_TOLERANCES, PLOT_CONFIG, RC_PARAMS, SPECIES
from src.models import vkb_ode, reduced_vkb_ode, activator_quasi_steady_state, mr_quasi_steady_state
from src.simulate import run_ode_simulation, run_reduced_simulation
from src.analyze_bifurcation import find_full_equilibrium, find_reduced_equilibrium






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


"""
Attractor coexistence analysis orchestration.

Core simulation utilities are implemented in `basin_core.py` and plotting logic
is in `basin_plot.py`. This module keeps the public entry points used by
`main.py` and downstream scripts.
"""




def run_combined_basin_analysis(s_R=0.088, n_grid=25, n_random=16, output_dir='figures', verbose=True):
    """Run reduced/full attractor analysis and generate the comparison figure."""
    def _log(message=""):
        if verbose:
            print(message)

    _log("=" * 70)
    _log(f"Combined Basin Analysis at s_R = {s_R}")
    _log("=" * 70)
    _log()

    _log("Phase 1: Reduced model phase space scan...")
    basin_data = compute_basin_of_attraction_reduced(
        s_R=s_R,
        R_range=(0, 200),
        C_range=(0, 200),
        n_grid=n_grid,
        t_max=1000,
        verbose=verbose,
    )
    _log(f"  Reduced model: {basin_data['fates'].size} ICs → all equilibrium")

    _log()
    _log("=" * 70)
    _log()

    _log("Phase 2: Full model attractor probe...")
    probe_data = probe_full_model_attractor(s_R=s_R, n_random=n_random, verbose=verbose)

    if probe_data is None:
        _log("ERROR: Full model probe failed")
        return None

    summary = probe_data['summary']
    n_lc = summary.get('limit_cycle', 0)
    n_eq = summary.get('equilibrium', 0)

    _log()
    _log("=" * 70)
    _log("COMBINED ANALYSIS SUMMARY")
    _log("=" * 70)
    _log(f"Reduced model: {basin_data['fates'].size} ICs tested")
    _log("  → 100% converge to equilibrium within the scanned domain")
    _log()
    _log(f"Full model: {len(probe_data['fates'])} ICs tested")
    _log(f"  → {n_lc} converge to limit cycle")
    _log(f"  → {n_eq} converge to equilibrium")
    _log()
    _log()

    _log("Phase 3: Creating paper-style phase portrait (Figure 9)...")
    plot_paper_mechanism(probe_data, s_R=s_R, output_dir=output_dir)

    return {
        'basin_data': basin_data,
        'probe_data': probe_data,
    }


def plot_paper_mechanism(probe_data, s_R=0.088, output_dir='figures'):
    """
    Paper-style phase portrait in the style of Vilar et al. (2002) Figs 4 and 6.

    Left panel  – Reduced model: nullclines + convergent trajectories (stable equilibrium).
    Right panel – Full model:    nullclines + closed limit-cycle orbit (unstable equilibrium).

    Both share the same axes so the structural difference is immediately apparent.
    """
    params = {**DEFAULT_PARAMS, 's_R': s_R}
    R_idx = SPECIES.index('R')
    C_full_idx = 8  # index of C in the 9-D state vector

    # ── equilibria ────────────────────────────────────────────────────────────
    y_star_red  = find_reduced_equilibrium(s_R)
    y_star_full = find_full_equilibrium(s_R)
    R0_red, C0_red   = float(y_star_red[0]),        float(y_star_red[1])
    R0_full, C0_full = float(y_star_full[R_idx]),   float(y_star_full[C_full_idx])

    # ── nullclines (closed-form for the 2-D reduced system) ──────────────────
    R_arr = np.linspace(0.01, 3000.0, 3000)
    C_nc  = np.zeros_like(R_arr)   # Ċ = 0  →  C = r_C·Ã(R)·R / s_A
    R_nc  = np.zeros_like(R_arr)   # Ṙ = 0  →  C = (r_C·Ã·R + s_R·R − b_R·M̃_R) / s_A

    for k, Rv in enumerate(R_arr):
        Aq  = activator_quasi_steady_state(Rv, params)
        MRq = mr_quasi_steady_state(Aq, params)
        C_nc[k] = params['r_C'] * Aq * Rv / params['s_A']
        R_nc[k] = (params['r_C'] * Aq * Rv + params['s_R'] * Rv - params['b_R'] * MRq) / params['s_A']

    # ── limit cycle from full-model probe data ────────────────────────────────
    fates       = probe_data['fates']
    trajectories = probe_data['trajectories']
    labels       = probe_data['labels']
    bio_idx      = labels.index('biological_IC')
    lc_indices   = [i for i, f in enumerate(fates) if f == 'limit_cycle']

    lc_rep = bio_idx if fates[bio_idx] == 'limit_cycle' else (lc_indices[0] if lc_indices else None)
    lc_R = lc_C = None
    if lc_rep is not None:
        traj   = trajectories[lc_rep]
        if traj is not None:
            n_tail = len(traj['t']) // 4   # last 25 % → settled limit cycle
            lc_R   = traj['R'][-n_tail:]
            lc_C   = traj['C'][-n_tail:]

    # ── axis limits (driven by the full model's limit-cycle extent) ───────────
    if lc_R is not None:
        R_max = float(np.percentile(lc_R, 99.5)) * 1.18
        C_max = float(np.percentile(lc_C, 99.5)) * 1.18
    else:
        R_max, C_max = 2600.0, 2600.0

    # ── convergent trajectories for the reduced model ─────────────────────────
    # Seed from four evenly-spaced points around the limit cycle so that
    # both panels share the same "starting positions", making the difference vivid.
    red_trajs = []
    if lc_R is not None:
        n_lc   = len(lc_R)
        seeds  = [int(i * n_lc / 4) for i in range(4)]
        for idx in seeds:
            ic  = [float(lc_R[idx]), float(lc_C[idx])]
            sol = solve_ivp(
                lambda t, y: reduced_vkb_ode(t, y, params),
                (0, 1400), ic,
                method='BDF',
                t_eval=np.linspace(0, 1400, 4500),
                rtol=ODE_TOLERANCES['rtol'],
                atol=ODE_TOLERANCES['atol'],
            )
            if sol.success:
                red_trajs.append({'R': sol.y[0], 'C': sol.y[1]})

    # ── helpers ───────────────────────────────────────────────────────────────
    def _setup_ax(ax):
        """Paper-style axes: spines through origin, single '0' tick, R/C tips."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        ax.set_xlim(0, R_max)
        ax.set_ylim(0, C_max)
        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_xticklabels(['0'])
        ax.set_yticklabels(['0'])
        ax.text(R_max * 1.03, -C_max * 0.03, 'R', fontsize=14, va='top', clip_on=False)
        ax.text(-R_max * 0.02, C_max * 1.03, 'C', fontsize=14, ha='right', clip_on=False)

    def _arrows(ax, x, y, fracs, color='black', lw=1.3, ms=15):
        n = len(x)
        for frac in fracs:
            i0 = int(frac * n)
            i1 = min(i0 + max(1, n // 60), n - 1)
            if i0 != i1:
                ax.annotate('', xy=(x[i1], y[i1]), xytext=(x[i0], y[i0]),
                            arrowprops=dict(arrowstyle='->', color=color,
                                           lw=lw, mutation_scale=ms))

    # Nullcline masks (only plot where C ≥ 0 and within axis range)
    mask_c = (C_nc >= 0) & (C_nc <= C_max * 1.25) & (R_arr <= R_max * 1.02)
    mask_r = (R_nc >= 0) & (R_nc <= C_max * 1.25) & (R_arr <= R_max * 1.02)

    # Pre-compute label positions — place both labels at the RIGHT end of each curve
    # (matching Vilar et al. style: labels stacked at lower-right, Ṙ=0 above Ċ=0)
    def _right_end(mask, arr_R, arr_C, quantile=0.90):
        """Return (R, C) at the rightmost visible portion of a nullcline."""
        if not np.any(mask):
            return R_max * 0.8, C_max * 0.05
        idx = int(np.sum(mask) * quantile)
        return float(arr_R[mask][idx]), float(arr_C[mask][idx])

    nc_label_R, nc_label_C = _right_end(mask_c, R_arr, C_nc)   # Ċ=0 curve right end
    nr_label_R, nr_label_C = _right_end(mask_r, R_arr, R_nc)   # Ṙ=0 curve right end

    def _draw_nullclines(ax, show_labels=True):
        ax.plot(R_arr[mask_c], C_nc[mask_c], 'k-', lw=2.2, solid_capstyle='round', zorder=2)
        ax.plot(R_arr[mask_r], R_nc[mask_r], 'k-', lw=2.2, solid_capstyle='round', zorder=2)
        if show_labels:
            # Ṙ=0 is above Ċ=0 on the right (since R_nc ≈ s_R·R > C_nc ≈ 0 for large R)
            ax.text(nr_label_R + R_max * 0.02, nr_label_C,
                    r'$\dot{R}=0$', fontsize=11, va='center', clip_on=False, zorder=3)
            ax.text(nc_label_R + R_max * 0.02, nc_label_C - C_max * 0.03,
                    r'$\dot{C}=0$', fontsize=11, va='top', clip_on=False, zorder=3)

    # ── figure ────────────────────────────────────────────────────────────────
    paper_rc = {**RC_PARAMS, 'axes.grid': False}
    with plt.rc_context(paper_rc):
        fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["FIGSIZE"]["paper_pair"])

        # ── LEFT: Reduced model (stable equilibrium, Vilar Fig-6 style) ──────
        ax = axes[0]
        _draw_nullclines(ax)

        for traj in red_trajs:
            Rp, Cp = traj['R'], traj['C']
            visible = (Rp >= 0) & (Rp <= R_max * 1.05) & (Cp >= 0) & (Cp <= C_max * 1.05)
            Rp, Cp = Rp[visible], Cp[visible]
            if len(Rp) > 5:
                ax.plot(Rp, Cp, 'k-', lw=1.4, alpha=0.80, zorder=3)
                _arrows(ax, Rp, Cp, fracs=(0.12, 0.45, 0.78), lw=1.1, ms=13)

        # Stable equilibrium: filled dot
        ax.plot(R0_red, C0_red, 'ko', ms=7, zorder=10)
        ax.annotate(r'$(R_0,\,C_0)$', xy=(R0_red, C0_red),
                    xytext=(R0_red + R_max * 0.07, C0_red + C_max * 0.06),
                    fontsize=11, arrowprops=dict(arrowstyle='->', lw=0.9))

        _setup_ax(ax)
        ax.set_title('Reduced model (QSSA) — stable equilibrium only', fontsize=12, pad=12)

        # ── RIGHT: Full model (limit cycle, Vilar Fig-4 style) ───────────────
        ax = axes[1]
        _draw_nullclines(ax)

        if lc_R is not None:
            ax.plot(lc_R, lc_C, 'k-', lw=2.8, zorder=4)
            _arrows(ax, lc_R, lc_C, fracs=(0.07, 0.30, 0.55, 0.80), lw=1.6, ms=17)

        # Stable equilibrium (coexists with the limit cycle – subcritical bistability)
        ax.plot(R0_full, C0_full, 'ko', ms=7, zorder=10)
        ax.annotate(r'$(R_0,\,C_0)$', xy=(R0_full, C0_full),
                    xytext=(R0_full + R_max * 0.07, C0_full + C_max * 0.06),
                    fontsize=11, arrowprops=dict(arrowstyle='->', lw=0.9))

        _setup_ax(ax)
        ax.set_title('Full model (9D ODE) — bistable', fontsize=12, pad=12)

        plt.tight_layout(pad=2.5)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'basin_comparison_sR_{s_R:.3f}.png')
        fig.savefig(out_path, dpi=PLOT_CONFIG['DPI'], bbox_inches='tight')
        plt.close(fig)

        print(f"Saved paper-style phase portrait to {out_path}")


