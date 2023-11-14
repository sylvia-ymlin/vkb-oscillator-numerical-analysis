import os
import matplotlib.pyplot as plt
from scripts.config import PLOT_CONFIG, RC_PARAMS


def plot_step_sizes(step_data, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    colors = [PLOT_CONFIG['NEUTRAL_COLORS']['gray'],
              PLOT_CONFIG["COLORS"]["A"][0],
              PLOT_CONFIG["COLORS"]["R"][0]]

    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["FIGSIZE"]["single"])
        for i, method in enumerate(["RK45", "BDF", "Radau"]):
            d = step_data[method]
            lbl = f"{method}: {d['n_steps']:,} steps, {d['elapsed']:.2f}s"
            ax.step(d["time"], d["dt"], where="post", color=colors[i], lw=1.2, label=lbl, alpha=0.8)

        ax.set_yscale("log")
        ax.set_ylabel(r"Step Size $\Delta t$ (h)")
        ax.set_xlabel("Time (hours)")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(output_dir, "solver_step_sizes.png")
        fig.savefig(out_path, dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_solver_errors(error_data, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    a_col, r_col = PLOT_CONFIG["COLORS"]["A"][0], PLOT_CONFIG["COLORS"]["R"][0]

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["FIGSIZE"]["bifurcation_stack"], sharex=True)
        for method, ax in zip(["BDF", "Radau"], axes):
            d = error_data[method]
            t = d["time"]
            ax.plot(t, d["err_A"], color=a_col, lw=1, alpha=0.8, label="Activator A Error")
            ax.fill_between(t, 0, d["err_A"], color=a_col, alpha=0.2)
            ax.plot(t, d["err_R"], color=r_col, lw=1, alpha=0.8, label="Repressor R Error")
            ax.fill_between(t, 0, d["err_R"], color=r_col, alpha=0.2)
            ax.set_yscale("log")
            ax.set_ylabel("Abs. Error")
            ax.set_title(f"{method} Error vs. RK45 Baseline", fontsize=11)
            ax.legend(loc="upper right", fontsize=8)

        axes[1].set_xlabel("Time (hours)")
        fig.tight_layout()
        out_path = os.path.join(output_dir, "solver_errors.png")
        fig.savefig(out_path, dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
