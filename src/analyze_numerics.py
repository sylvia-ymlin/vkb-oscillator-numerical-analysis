import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import PLOT_CONFIG, RC_PARAMS
from src.simulate import run_ode_simulation

def run_step_size_analysis(t_max=48):
    """Run RK45, BDF, and Radau with internal steps to capture delta_t."""
    results = {}
    for method in ["RK45", "BDF", "Radau"]:
        sim = run_ode_simulation(
            s_R=0.2,
            t_max=t_max,
            num_points=0,
            method=method,
            use_jacobian=True,
            return_solver_stats=True
        )
        t = sim["time"]
        dt = np.diff(t)
        results[method] = {
            "time": t[:-1],
            "dt": dt,
            "n_steps": len(t),
            "stats": sim["solver_stats"]
        }
    return results

def run_error_analysis(t_max=48, num_points=2000):
    """Compare BDF and Radau against a high-precision RK45 baseline."""
    # Baseline: High-precision RK45
    print("  Generating high-precision RK45 baseline...")
    baseline = run_ode_simulation(
        s_R=0.2,
        t_max=t_max,
        num_points=num_points,
        method="RK45",
        rtol=1e-12,
        atol=1e-14
    )
    
    methods = ["BDF", "Radau"]
    errors = {}
    
    for method in methods:
        print(f"  Integrating with {method}...")
        sim = run_ode_simulation(
            s_R=0.2,
            t_max=t_max,
            num_points=num_points,
            method=method
        )
        
        # Absolute error |y_method - y_baseline|
        err_A = np.abs(sim["A"] - baseline["A"])
        err_R = np.abs(sim["R"] - baseline["R"])
        
        errors[method] = {
            "time": baseline["time"],
            "err_A": err_A,
            "err_R": err_R
        }
    
    return errors

def plot_step_sizes(step_data, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
    a_colors = PLOT_CONFIG["COLORS"]["A"]
    r_colors = PLOT_CONFIG["COLORS"]["R"]

    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(10, 5))
        methods = ["RK45", "BDF", "Radau"]
        colors = [neutral_colors['gray'], a_colors[0], r_colors[0]]
        for i, method in enumerate(methods):
            data = step_data[method]
            n_steps = data["n_steps"]
            elapsed = data["stats"]["elapsed_sec"]
            label = f"{method}: {n_steps:,} steps, {elapsed:.2f}s"
            ax.step(data["time"], data["dt"], where="post", color=colors[i], lw=1.2, label=label, alpha=0.8)

        ax.set_yscale("log")
        ax.set_ylabel(r"Step Size $\Delta t$ (h)")
        ax.set_xlabel("Time (hours)")
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
        fig.tight_layout()
        output_path = os.path.join(output_dir, "solver_step_sizes.png")
        fig.savefig(output_path, dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)
    print(f"Saved consolidated solver step sizes plot to {output_path}")

def plot_solver_errors(error_data, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    a_colors = PLOT_CONFIG["COLORS"]["A"]
    r_colors = PLOT_CONFIG["COLORS"]["R"]

    with plt.rc_context(RC_PARAMS):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        methods = ["BDF", "Radau"]
        axes = [ax1, ax2]
        
        for method, ax in zip(methods, axes):
            data = error_data[method]
            t = data["time"]
            
            # Plot error as lines and shaded area
            ax.plot(t, data["err_A"], color=a_colors[0], lw=1, alpha=0.8, label=f"Activator A Error")
            ax.fill_between(t, 0, data["err_A"], color=a_colors[0], alpha=0.2)
            
            ax.plot(t, data["err_R"], color=r_colors[0], lw=1, alpha=0.8, label=f"Repressor R Error")
            ax.fill_between(t, 0, data["err_R"], color=r_colors[0], alpha=0.2)
            
            ax.set_yscale("log")
            ax.set_ylabel("Abs. Error")
            ax.set_title(f"{method} Error vs. RK45 Baseline", fontsize=11)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, which="both", ls="--", alpha=0.3)

        ax2.set_xlabel("Time (hours)")
        fig.tight_layout()
        
        output_path = os.path.join(output_dir, "solver_errors.png")
        fig.savefig(output_path, dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)
    print(f"Saved solver error comparison plot to {output_path}")

def run_numerics_analysis(output_dir="figures"):
    print("Running Solver Step Size Analysis...")
    step_data = run_step_size_analysis(t_max=48)
    plot_step_sizes(step_data, output_dir=output_dir)
    
    print("Running Solver Error Analysis...")
    error_data = run_error_analysis(t_max=48)
    plot_solver_errors(error_data, output_dir=output_dir)
    
    return {"step_data": step_data, "error_data": error_data}

if __name__ == "__main__":
    run_numerics_analysis()
