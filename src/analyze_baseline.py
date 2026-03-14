import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import PLOT_CONFIG, RC_PARAMS, BASELINE_CASES
from src.simulate import run_ode_simulation
from src.simulate import run_ssa as run_ssa_simulation


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



def plot_assignment_figures(cases, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    a_colors = PLOT_CONFIG["COLORS"]["A"]
    r_colors = PLOT_CONFIG["COLORS"]["R"]

    with plt.rc_context(RC_PARAMS):
        for case in cases:
            s_R = case["s_R"]
            ode_result = case["ode"]
            ssa_results = case["ssa"]
            
            # Assignment A: ODE 0.2
            if s_R == 0.2:
                fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
                axes[0].plot(ode_result["time"], ode_result["A"], color=a_colors[0], lw=2)
                axes[0].set_ylabel("Activator (A)")
                axes[1].plot(ode_result["time"], ode_result["R"], color=r_colors[0], lw=2)
                axes[1].set_ylabel("Repressor (R)")
                axes[1].set_xlabel("Time (hours)")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, "assignment_a_ode.png"), dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
                plt.close(fig)

                # Assignment B: SSA 0.2
                fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
                for idx, traj in enumerate(ssa_results):
                    shade = 1 + idx % (len(a_colors) - 1)
                    axes[0].step(traj["time"], traj["A"], where="post", color=a_colors[shade], lw=1.2, alpha=0.7)
                axes[0].set_ylabel("Activator (A)")
                for idx, traj in enumerate(ssa_results):
                    shade = 1 + idx % (len(r_colors) - 1)
                    axes[1].step(traj["time"], traj["R"], where="post", color=r_colors[shade], lw=1.2, alpha=0.7)
                axes[1].set_ylabel("Repressor (R)")
                axes[1].set_xlabel("Time (hours)")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, "assignment_b_ssa.png"), dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
                plt.close(fig)

            # Assignment C: ODE vs SSA 0.03
            if s_R == 0.03:
                fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
                axes[0].plot(ode_result["time"], ode_result["A"], color="black", lw=1.5, linestyle="--", label="ODE", zorder=3)
                axes[1].plot(ode_result["time"], ode_result["R"], color="black", lw=1.5, linestyle="--", label="ODE", zorder=3)
                traj = ssa_results[0]
                axes[0].step(traj["time"], traj["A"], where="post", color=a_colors[1], lw=1.2, alpha=0.8, label="SSA", zorder=2)
                axes[0].set_ylabel("Activator (A)")
                axes[0].legend(loc="upper right")
                
                axes[1].step(traj["time"], traj["R"], where="post", color=r_colors[1], lw=1.2, alpha=0.8, label="SSA", zorder=2)
                axes[1].set_ylabel("Repressor (R)")
                axes[1].set_xlabel("Time (hours)")
                axes[1].legend(loc="upper right")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, "assignment_c_noise.png"), dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
                plt.close(fig)

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
    plot_assignment_figures(cases, output_dir=output_dir)
    return cases


if __name__ == "__main__":
    run_baseline_verification()
