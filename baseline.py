import os
import matplotlib.pyplot as plt

from config import PLOT_CONFIG, RC_PARAMS, BASELINE_CASES
from simulation import run_ode_simulation, run_ssa
from plot_baseline import (
    plot_ode_trajectory, plot_ssa_trajectories, plot_ode_vs_ssa,
)


def run_baseline_analysis(t_max=400, ode_points=2000, ssa_trajectories=3, seed=42, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    with plt.rc_context(RC_PARAMS):
        for idx, c in enumerate(BASELINE_CASES):
            s_R = float(c["s_R"])
            ode = run_ode_simulation(s_R=s_R, t_max=t_max, num_points=ode_points)
            ssa = run_ssa(s_R=s_R, trajectories=ssa_trajectories, t_max=t_max, seed=int(seed) + idx*100)
            if s_R == 0.2:
                plot_ode_trajectory(ode, output_dir)
                plot_ssa_trajectories(ssa, output_dir)
            elif s_R == 0.05:
                plot_ode_vs_ssa(ode, ssa[0], output_dir)


if __name__ == "__main__":
    run_baseline_analysis()
