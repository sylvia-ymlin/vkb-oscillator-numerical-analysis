import os

import matplotlib.pyplot as plt
import numpy as np

from .bifurcation_analysis import find_full_equilibrium, find_reduced_equilibrium
from .config import DEFAULT_PARAMS, PLOT_CONFIG, RC_PARAMS, SPECIES
from .reduced_model import reduced_vkb_ode


def plot_basin_comparison(basin_data, probe_data, output_dir='figures'):
    s_R = basin_data['s_R']

    y_star_reduced = find_reduced_equilibrium(s_R)
    y_star_full = find_full_equilibrium(s_R)

    R_idx = SPECIES.index('R')
    C_idx = SPECIES.index('C')

    fate_colors = PLOT_CONFIG['FATE_COLORS']
    neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']

    ICs = np.array(probe_data['initial_conditions'])
    fates = probe_data['fates']
    labels = probe_data['labels']
    trajectories = probe_data['trajectories']

    color_map = {
        'equilibrium': fate_colors['equilibrium'],
        'limit_cycle': fate_colors['limit_cycle'],
        'failed': fate_colors['failed'],
        'error': fate_colors['failed'],
    }

    bio_idx = labels.index('biological_IC')
    traj_bio = trajectories[bio_idx]
    R_bio = traj_bio['R'] if traj_bio is not None else None
    C_bio = traj_bio['C'] if traj_bio is not None else None

    lc_indices = [i for i, f in enumerate(fates) if f == 'limit_cycle']
    eq_indices = [i for i, f in enumerate(fates) if f == 'equilibrium']

    def _sample_diverse_indices(idxs, n_show=6):
        if len(idxs) <= n_show:
            return idxs
        rc = np.array([[ICs[i, R_idx], ICs[i, C_idx]] for i in idxs], dtype=float)
        center = rc.mean(axis=0)
        first = int(np.argmax(np.sum((rc - center) ** 2, axis=1)))
        selected_local = [first]
        remaining = set(range(len(idxs))) - {first}
        while len(selected_local) < n_show and remaining:
            rem_list = sorted(remaining)
            d2_list = [min(np.sum((rc[r] - rc[s]) ** 2) for s in selected_local) for r in rem_list]
            best = rem_list[int(np.argmax(d2_list))]
            selected_local.append(best)
            remaining.remove(best)
        return [idxs[k] for k in selected_local]

    sampled_eq = _sample_diverse_indices(eq_indices, n_show=6)
    sampled_lc = _sample_diverse_indices(lc_indices, n_show=6)

    lc_tail_R = lc_tail_C = None
    if lc_indices:
        lc_rep_idx = bio_idx if fates[bio_idx] == 'limit_cycle' else lc_indices[0]
        traj_lc = trajectories[lc_rep_idx]
        n_tail = len(traj_lc['t']) // 5
        lc_tail_R = traj_lc['R'][-n_tail:]
        lc_tail_C = traj_lc['C'][-n_tail:]

    R_focus = C_focus = None
    if R_bio is not None:
        r_focus_max = np.percentile(ICs[:, R_idx], 95) + 40
        c_focus_max = np.percentile(ICs[:, C_idx], 95) + 80
        focus_mask = (R_bio <= r_focus_max) & (C_bio <= c_focus_max)
        if np.any(focus_mask):
            R_focus = R_bio[focus_mask]
            C_focus = C_bio[focus_mask]
        else:
            n_focus = max(20, len(R_bio) // 3)
            R_focus = R_bio[-n_focus:]
            C_focus = C_bio[-n_focus:]

    params = {**DEFAULT_PARAMS, 's_R': s_R}

    def _vector_field(R_lim, C_lim, n=30):
        Rv = np.linspace(*R_lim, n)
        Cv = np.linspace(*C_lim, n)
        Rg, Cg = np.meshgrid(Rv, Cv)
        dR = np.zeros_like(Rg)
        dC = np.zeros_like(Cg)
        for i in range(n):
            for j in range(n):
                dy = reduced_vkb_ode(0, [Rg[i, j], Cg[i, j]], params)
                dR[i, j] = dy[0]
                dC[i, j] = dy[1]
        return Rg, Cg, dR, dC

    Rg_wide, Cg_wide, dRw, dCw = _vector_field((0, 2500), (0, 2500))
    Rg_zoom, Cg_zoom, dRz, dCz = _vector_field((0, 200), (0, 500))

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), gridspec_kw={'hspace': 0.38, 'wspace': 0.28})

        fig.text(0.005, 0.74, 'Overview', va='center', ha='left', rotation=90, fontsize=10, color='#555555', style='italic')
        fig.text(0.005, 0.27, 'Zoomed', va='center', ha='left', rotation=90, fontsize=10, color='#555555', style='italic')

        def _panel_reduced(ax, Rg, Cg, dR, dC, xlim, ylim, label):
            ax.set_facecolor('#E8F4F8')
            ax.streamplot(Rg, Cg, dR, dC, color='#2C3E50', linewidth=1.5, density=1.5, arrowsize=1.3, arrowstyle='->')
            ax.scatter(y_star_reduced[0], y_star_reduced[1], color='red', s=300, marker='x', linewidths=3.5, zorder=10)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel('R (Repressor)')
            ax.set_ylabel('C (Complex)')
            ax.set_title(f'Reduced Model (2D QSSA) — {label}')

        def _panel_full(ax, xlim, ylim, label, show_simple_legend=False):
            for ic, fate in zip(ICs, fates):
                if fate in ['failed', 'error']:
                    continue
                col = color_map.get(fate, fate_colors['failed'])
                marker = 'o' if fate == 'limit_cycle' else 's'
                ax.scatter(ic[R_idx], ic[C_idx], color=col, s=35, marker=marker, alpha=0.70, edgecolors='black', linewidths=0.3, zorder=2)

            for idx in sampled_eq:
                traj = trajectories[idx]
                if traj is None:
                    continue
                ax.plot(traj['R'], traj['C'], color=fate_colors['equilibrium'], linewidth=1.4, alpha=0.45, zorder=3)
                ax.scatter(traj['R'][0], traj['C'][0], s=85, marker='^', facecolors='white', edgecolors=fate_colors['equilibrium'], linewidths=1.5, zorder=9)

            for idx in sampled_lc:
                traj = trajectories[idx]
                if traj is None:
                    continue
                ax.plot(traj['R'], traj['C'], color=fate_colors['limit_cycle'], linewidth=1.6, alpha=0.45, zorder=3)
                ax.scatter(traj['R'][0], traj['C'][0], s=85, marker='v', facecolors='white', edgecolors=fate_colors['limit_cycle'], linewidths=1.5, zorder=9)

            if R_bio is not None:
                ax.plot(R_bio, C_bio, color='#5A5A5A', linewidth=1.2, linestyle='--', alpha=0.6, zorder=4)
                if R_focus is not None:
                    ax.plot(R_focus, C_focus, color='#111111', linewidth=1.9, linestyle='--', alpha=0.92, zorder=6)
                    for frac in (0.35, 0.65, 0.88):
                        i0 = int(frac * (len(R_focus) - 2))
                        i1 = min(i0 + 4, len(R_focus) - 1)
                        ax.annotate('', xy=(R_focus[i1], C_focus[i1]), xytext=(R_focus[i0], C_focus[i0]), arrowprops=dict(arrowstyle='->', color='#111111', lw=1.2), zorder=7)

            if lc_tail_R is not None:
                ax.plot(lc_tail_R, lc_tail_C, color=fate_colors['limit_cycle'], linewidth=3.5, alpha=0.95, zorder=5)
                mid = len(lc_tail_R) // 2
                ax.scatter(lc_tail_R[mid], lc_tail_C[mid], color=fate_colors['limit_cycle'], s=150, marker='o', edgecolors='black', linewidths=2, zorder=6)

            ax.scatter(ICs[bio_idx, R_idx], ICs[bio_idx, C_idx], color=neutral_colors['gold'], s=250, marker='*', edgecolors='black', linewidths=1.5, zorder=16)
            ax.scatter(y_star_full[R_idx], y_star_full[C_idx], color='red', s=300, marker='x', linewidths=3.5, zorder=10)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel('R (Repressor)')
            ax.set_ylabel('C (Complex)')
            ax.set_title(f'Full Model (9D ODE, R–C projection) — {label}')

            if show_simple_legend:
                from matplotlib.lines import Line2D
                leg = [
                    Line2D([0], [0], color='#222222', lw=1.9, linestyle='--', label='Biological Trajectory'),
                    Line2D([0], [0], color=fate_colors['limit_cycle'], lw=2.0, label='→ Limit Cycle'),
                    Line2D([0], [0], color=fate_colors['equilibrium'], lw=1.8, label='→ Equilibrium'),
                ]
                ax.legend(handles=leg, loc='upper right', fontsize=9, framealpha=0.95)

        _panel_reduced(axes[0, 0], Rg_wide, Cg_wide, dRw, dCw, xlim=(0, 2500), ylim=(0, 2500), label='Overview')
        _panel_full(axes[0, 1], xlim=(-125, 2625), ylim=(-125, 2625), label='Overview', show_simple_legend=False)
        _panel_reduced(axes[1, 0], Rg_zoom, Cg_zoom, dRz, dCz, xlim=(0, 200), ylim=(0, 500), label='Zoomed')
        _panel_full(axes[1, 1], xlim=(-10, 210), ylim=(-10, 510), label='Zoomed', show_simple_legend=True)

        fig.suptitle(f'Phase Space Topology at $s_R = {s_R}$: QSSA Eliminates the Limit Cycle Attractor', fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0.015, 0, 1, 0.985])

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'basin_comparison_sR_{s_R:.3f}.png')
        fig.savefig(output_path, dpi=PLOT_CONFIG['DPI'], bbox_inches='tight')
        plt.close(fig)

        print(f"Saved basin comparison plot to {output_path}")
