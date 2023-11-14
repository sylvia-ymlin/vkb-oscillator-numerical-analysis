import os
import matplotlib.pyplot as plt
from scripts.config import PLOT_CONFIG, RC_PARAMS


def plot_ode_trajectory(ode_result, output_dir="figures"):
    fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["FIGSIZE"]["dual_row"], sharex=True)
    axes[0].plot(ode_result["time"], ode_result["A"], color=PLOT_CONFIG["COLORS"]["A"][0], lw=2)
    axes[0].set_ylabel("Activator (A)")
    axes[1].plot(ode_result["time"], ode_result["R"], color=PLOT_CONFIG["COLORS"]["R"][0], lw=2)
    axes[1].set_ylabel("Repressor (R)")
    axes[1].set_xlabel("Time (hours)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "assignment_a_ode.png"), dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
    plt.close(fig)


def plot_ssa_trajectories(ssa_results, output_dir="figures"):
    a_colors, r_colors = PLOT_CONFIG["COLORS"]["A"], PLOT_CONFIG["COLORS"]["R"]
    fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["FIGSIZE"]["dual_row"], sharex=True)
    for idx, traj in enumerate(ssa_results):
        shade = 1 + idx % (len(a_colors) - 1)
        axes[0].step(traj["time"], traj["A"], where="post", color=a_colors[shade], lw=1.2, alpha=0.7)
        axes[1].step(traj["time"], traj["R"], where="post", color=r_colors[shade], lw=1.2, alpha=0.7)
    axes[0].set_ylabel("Activator (A)")
    axes[1].set_ylabel("Repressor (R)")
    axes[1].set_xlabel("Time (hours)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "assignment_b_ssa.png"), dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
    plt.close(fig)


def plot_ode_vs_ssa(ode_result, ssa_result, output_dir="figures"):
    fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["FIGSIZE"]["dual_row"], sharex=True)
    for i, sp in enumerate(["A", "R"]):
        axes[i].plot(ode_result["time"], ode_result[sp], color="black", lw=1.5, ls="--", label="ODE", zorder=3)
        axes[i].step(ssa_result["time"], ssa_result[sp], where="post",
                     color=PLOT_CONFIG["COLORS"][sp][1], lw=1.2, alpha=0.8, label="SSA", zorder=2)
        axes[i].set_ylabel(f"{'Activator' if sp=='A' else 'Repressor'} ({sp})")
        axes[i].legend(loc="upper right")
    axes[1].set_xlabel("Time (hours)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "assignment_c_noise.png"), dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
    plt.close(fig)
