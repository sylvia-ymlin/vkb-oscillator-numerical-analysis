import os

import matplotlib.pyplot as plt
import numpy as np

from .config import PLOT_CONFIG, RC_PARAMS, BASELINE_CASES
from .sim_ode import run_ode_simulation
from .sim_ssa import run_ssa as run_ssa_simulation


def _series_summary(time, values, tail_fraction=0.2):
    tail_start = int(len(values) * (1.0 - float(tail_fraction)))
    tail_values = values[tail_start:]
    return {
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "value_min": float(np.min(values)),
        "value_max": float(np.max(values)),
        "value_final": float(values[-1]),
        "amplitude": float(np.max(values) - np.min(values)),
        "tail_min": float(np.min(tail_values)),
        "tail_max": float(np.max(tail_values)),
        "tail_amplitude": float(np.max(tail_values) - np.min(tail_values)),
    }


def _print_case_summary(case_label, s_R, ode_result, ssa_results):
    ode_a = _series_summary(ode_result["time"], ode_result["A"])
    ode_r = _series_summary(ode_result["time"], ode_result["R"])

    ssa_a_tail_amplitudes = [_series_summary(traj["time"], traj["A"])["tail_amplitude"] for traj in ssa_results]
    ssa_r_tail_amplitudes = [_series_summary(traj["time"], traj["R"])["tail_amplitude"] for traj in ssa_results]

    print(f"[{case_label}] s_R={s_R}")
    print(
        "  ODE: "
        f"A_final={ode_a['value_final']:.3f}, A_tail_amp={ode_a['tail_amplitude']:.3f}, "
        f"R_final={ode_r['value_final']:.3f}, R_tail_amp={ode_r['tail_amplitude']:.3f}"
    )
    print(
        "  SSA: "
        f"A_tail_amp_mean={np.mean(ssa_a_tail_amplitudes):.3f}, "
        f"R_tail_amp_mean={np.mean(ssa_r_tail_amplitudes):.3f}, "
        f"trajectories={len(ssa_results)}"
    )


def plot_baseline_verification(cases, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    a_colors = PLOT_CONFIG["COLORS"]["A"]   # dark blue, dodger blue, teal  — for activator
    r_colors = PLOT_CONFIG["COLORS"]["R"]   # orange-red, gold, light salmon — for repressor

    with plt.rc_context(RC_PARAMS):
        # 4 rows (ODE-A, SSA-A, ODE-R, SSA-R) × n_cases columns
        fig, axes = plt.subplots(
            4, len(cases),
            figsize=(13, 10),
            sharex="col",
        )

        if len(cases) == 1:
            axes = np.array(axes).reshape(4, 1)

        for col, case in enumerate(cases):
            s_R = case["s_R"]
            ode_result = case["ode"]
            ssa_results = case["ssa"]

            ax_ode_a = axes[0, col]
            ax_ssa_a = axes[1, col]
            ax_ode_r = axes[2, col]
            ax_ssa_r = axes[3, col]

            # ODE - Activator
            ax_ode_a.plot(
                ode_result["time"], ode_result["A"],
                color=a_colors[0], lw=2.5, zorder=3,
            )
            ax_ode_a.set_ylabel("ODE: $A$")
            if col == 0:
                ax_ode_a.set_title(f"$s_R = {s_R}$\nActivator (ODE)")
            else:
                ax_ode_a.set_title(f"$s_R = {s_R}$")

            # SSA - Activator
            for idx, traj in enumerate(ssa_results):
                shade = 1 + idx % (len(a_colors) - 1)
                ax_ssa_a.step(
                    traj["time"], traj["A"],
                    where="post",
                    color=a_colors[shade], lw=1.3, alpha=0.65, zorder=2,
                )
            ax_ssa_a.set_ylabel("SSA: $A$")

            # ODE - Repressor
            ax_ode_r.plot(
                ode_result["time"], ode_result["R"],
                color=r_colors[0], lw=2.5, zorder=3,
            )
            ax_ode_r.set_ylabel("ODE: $R$")

            # SSA - Repressor
            for idx, traj in enumerate(ssa_results):
                shade = 1 + idx % (len(r_colors) - 1)
                ax_ssa_r.step(
                    traj["time"], traj["R"],
                    where="post",
                    color=r_colors[shade], lw=1.3, alpha=0.65, zorder=2,
                )
            ax_ssa_r.set_ylabel("SSA: $R$")
            ax_ssa_r.set_xlabel("Time")

        fig.suptitle("Baseline Verification: ODE vs SSA", fontsize=13)
        fig.tight_layout()

        output_path = os.path.join(output_dir, "baseline_verification.png")
        fig.savefig(output_path, dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    print(f"Saved baseline verification plot to {output_path}")


def run_baseline_verification(
    t_max=400,
    ode_points=2000,
    ssa_trajectories=3,
    seed=42,
    output_dir="figures",
):
    cases = []

    for index, case in enumerate(BASELINE_CASES):
        s_R = float(case["s_R"])
        ode_result = run_ode_simulation(
            s_R=s_R,
            t_max=t_max,
            num_points=ode_points,
        )
        ssa_results = run_ssa_simulation(
            s_R=s_R,
            trajectories=ssa_trajectories,
            t_max=t_max,
            seed=int(seed) + index * 100,
        )

        _print_case_summary(case["label"], s_R, ode_result, ssa_results)

        cases.append(
            {
                "label": case["label"],
                "s_R": s_R,
                "ode": ode_result,
                "ssa": ssa_results,
            }
        )

    plot_baseline_verification(cases, output_dir=output_dir)
    return cases


if __name__ == "__main__":
    run_baseline_verification()
