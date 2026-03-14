import os

import matplotlib.pyplot as plt
import numpy as np

from .config import PLOT_CONFIG, RC_PARAMS, DEFAULT_PARAMS, REDUCTION_CASES, ODE_TOLERANCES
from .reduced_model import reduced_observables
from .sim_ode import run_ode_simulation
from .sim_reduced import run_reduced_simulation


def _tail_stats(values, tail_fraction=0.2):
    start = int(len(values) * (1.0 - float(tail_fraction)))
    tail = values[start:]
    return {
        "final": float(values[-1]),
        "tail_amplitude": float(np.max(tail) - np.min(tail)),
    }


def _print_reduction_summary(label, s_R, full_obs, reduced_obs):
    full_a = _tail_stats(full_obs["A"])
    full_r = _tail_stats(full_obs["R"])
    reduced_a = _tail_stats(reduced_obs["A"])
    reduced_r = _tail_stats(reduced_obs["R"])

    print(f"[{label}] s_R={s_R}")
    print(
        "  Full: "
        f"A_final={full_a['final']:.3f}, A_tail_amp={full_a['tail_amplitude']:.3f}, "
        f"R_final={full_r['final']:.3f}, R_tail_amp={full_r['tail_amplitude']:.3f}"
    )
    print(
        "  Reduced: "
        f"A_final={reduced_a['final']:.3f}, A_tail_amp={reduced_a['tail_amplitude']:.3f}, "
        f"R_final={reduced_r['final']:.3f}, R_tail_amp={reduced_r['tail_amplitude']:.3f}"
    )


def plot_reduction_comparison(cases, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    a_colors = PLOT_CONFIG["COLORS"]["A"]
    r_colors = PLOT_CONFIG["COLORS"]["R"]
    model_colors = PLOT_CONFIG['MODEL_COLORS']
    line_styles = PLOT_CONFIG['LINE_STYLES']

    # Full model: 深灰色实线
    full_color = model_colors['full']
    # Reduced model: 使用物种颜色 + 虚线区分
    reduced_a_color = a_colors[0]
    reduced_r_color = r_colors[0]

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(2, len(cases), figsize=(15, 7), sharex="col")

        if len(cases) == 1:
            axes = np.array(axes).reshape(2, 1)

        for col, case in enumerate(cases):
            full_obs    = case["full"]
            reduced_obs = case["reduced"]

            ax_a = axes[0, col]
            ax_r = axes[1, col]

            # Full model: 深灰色实线
            ax_a.plot(full_obs["time"], full_obs["A"],
                      color=full_color, lw=2, ls=line_styles['full_solid'], 
                      label="Full", zorder=3)
            ax_a.plot(reduced_obs["time"], reduced_obs["A"],
                      color=reduced_a_color, lw=2, ls=line_styles['reduced_dashed'], 
                      label="Reduced", zorder=2)

            ax_r.plot(full_obs["time"], full_obs["R"],
                      color=full_color, lw=2, ls=line_styles['full_solid'], 
                      label="Full", zorder=3)
            ax_r.plot(reduced_obs["time"], reduced_obs["R"],
                      color=reduced_r_color, lw=2, ls=line_styles['reduced_dashed'], 
                      label="Reduced", zorder=2)

            ax_a.set_title(f"$s_R = {case['s_R']}$")
            ax_a.set_ylabel("Activator $A$")
            ax_r.set_ylabel("Repressor $R$")
            ax_r.set_xlabel("Time")
            ax_a.legend()

        fig.suptitle("Full vs Reduced Model Comparison", y=1.00)
        
        output_path = os.path.join(output_dir, "reduction_comparison.png")
        fig.savefig(output_path, dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    print(f"Saved reduction comparison plot to {output_path}")


def run_reduction_analysis(output_dir="figures", t_max=400, num_points=2000):
    cases = []

    for case in REDUCTION_CASES:
        s_R = float(case["s_R"])

        full_result = run_ode_simulation(
            s_R=s_R,
            t_max=t_max,
            num_points=num_points,
            method="BDF",
            rtol=ODE_TOLERANCES['rtol'],
            atol=ODE_TOLERANCES['atol'],
            use_jacobian=True,
        )
        reduced_result = run_reduced_simulation(
            s_R=s_R,
            t_max=t_max,
            num_points=num_points,
        )

        full_obs = {
            "time": full_result["time"],
            "A": full_result["A"],
            "R": full_result["R"],
        }
        reduced_obs = reduced_observables(reduced_result, p={**DEFAULT_PARAMS, "s_R": s_R})
        reduced_obs["time"] = reduced_result["time"]

        _print_reduction_summary(case["label"], s_R, full_obs, reduced_obs)

        cases.append(
            {
                "label": case["label"],
                "s_R": s_R,
                "full": full_obs,
                "reduced": reduced_obs,
            }
        )

    plot_reduction_comparison(cases, output_dir=output_dir)
    return cases


if __name__ == "__main__":
    run_reduction_analysis()
