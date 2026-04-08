import numpy as np
from simulation import run_ode_simulation
from plot_numerics import plot_step_sizes, plot_solver_errors


def run_step_size_analysis(t_max=48):
    res = {}
    for m in ["RK45", "BDF", "Radau"]:
        sim = run_ode_simulation(s_R=0.2, t_max=t_max, num_points=0, method=m,
                                 use_jacobian=True, return_solver_stats=True)
        res[m] = {"time": sim["time"][:-1], "dt": np.diff(sim["time"]),
                  "n_steps": len(sim["time"]), "elapsed": sim["solver_stats"]["elapsed_sec"]}
    return res


def run_error_analysis(t_max=48, num_points=2000):
    # High-precision RK45 as ground truth; compare BDF and Radau at standard tolerances
    base = run_ode_simulation(s_R=0.2, t_max=t_max, num_points=num_points,
                              method="RK45", rtol=1e-12, atol=1e-14)
    res = {}
    for m in ["BDF", "Radau"]:
        sim = run_ode_simulation(s_R=0.2, t_max=t_max, num_points=num_points, method=m)
        res[m] = {"time": base["time"], "err_A": np.abs(sim["A"] - base["A"]),
                  "err_R": np.abs(sim["R"] - base["R"])}
    return res


def run_numerics_analysis(output_dir="figures"):
    plot_step_sizes(run_step_size_analysis(), output_dir=output_dir)
    plot_solver_errors(run_error_analysis(), output_dir=output_dir)


if __name__ == "__main__":
    run_numerics_analysis()
