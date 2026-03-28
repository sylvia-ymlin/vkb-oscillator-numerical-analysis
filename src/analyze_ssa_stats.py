import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from src.simulate import run_ssa
from src.config import PLOT_CONFIG, RC_PARAMS

def extract_oscillation_features(t, signal, height=500, distance=50):
    """Extract periods and amplitudes from a single trajectory."""
    peaks, _ = find_peaks(signal, height=height, distance=distance)
    if len(peaks) < 2:
        return np.array([]), np.array([])
    
    t_peaks = t[peaks]
    periods = np.diff(t_peaks)
    amplitudes = signal[peaks]
    return periods, amplitudes

def plot_ssa_statistics(trajectories, s_R, output_path):
    """Generate a 3-panel statistical summary of SSA trajectories."""
    all_periods = []
    all_amplitudes = []
    
    # Process data
    for traj in trajectories:
        p, a = extract_oscillation_features(traj["time"], traj["A"])
        all_periods.extend(p)
        all_amplitudes.extend(a)
    
    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(3, 1, figsize=PLOT_CONFIG["FIGSIZE"]["triple_row"])
        
        # Panel 1: Multi-trajectory Overlap
        for i, traj in enumerate(trajectories):
            alpha = 0.2 if len(trajectories) > 10 else 0.5
            axes[0].plot(traj["time"], traj["A"], lw=0.8, alpha=alpha, color=PLOT_CONFIG["COLORS"]["A"][0])
        axes[0].set_title(f"SSA: {len(trajectories)} Independent Trajectories (sR={s_R})")
        axes[0].set_xlabel("Time (hours)")
        axes[0].set_ylabel("Activator A (molecules)")
        axes[0].grid(alpha=0.3)
        
        # Panel 2: Period Distribution
        if all_periods:
            mu_p, std_p = np.mean(all_periods), np.std(all_periods)
            axes[1].hist(all_periods, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1].axvline(mu_p, color='red', linestyle='--', label=f'Mean: {mu_p:.2f}h')
            axes[1].set_title(f"Oscillation Period Distribution (Std: {std_p:.2f}h)")
            axes[1].set_xlabel("Period (hours)")
            axes[1].set_ylabel("Frequency")
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "No periodic peaks detected", ha='center')
            
        # Panel 3: Amplitude Distribution
        if all_amplitudes:
            mu_a, std_a = np.mean(all_amplitudes), np.std(all_amplitudes)
            axes[2].hist(all_amplitudes, bins=20, alpha=0.7, color='salmon', edgecolor='black')
            axes[2].axvline(mu_a, color='red', linestyle='--', label=f'Mean: {mu_a:.1f}')
            axes[2].set_title(f"Peak Amplitude Distribution (Std: {std_a:.2f})")
            axes[2].set_xlabel("Amplitude (molecules)")
            axes[2].set_ylabel("Frequency")
            axes[2].legend()
        else:
            axes[2].text(0.5, 0.5, "No peaks detected", ha='center')

        plt.tight_layout()
        plt.savefig(output_path, dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)
    
    # Print summary statistics to terminal
    if all_periods and all_amplitudes:
        print(f"\n[SSA Stats sR={s_R}]")
        print(f"  Period: {np.mean(all_periods):.2f} ± {np.std(all_periods):.2f} h")
        print(f"  Amplitude: {np.mean(all_amplitudes):.1f} ± {np.std(all_amplitudes):.1f} molecules")

def run_ssa_statistical_analysis(s_R_list=[0.2, 0.03], n_trajectories=50, t_max=400):
    """Main execution function for SSA statistical analysis."""
    os.makedirs("figures", exist_ok=True)
    
    for s_R in s_R_list:
        print(f"Running {n_trajectories} SSA trajectories for s_R={s_R}...")
        # Note: run_ssa in our codebase returns a list of result dicts
        trajectories = run_ssa(s_R=s_R, trajectories=n_trajectories, t_max=t_max, seed=42)
        
        output_path = f"figures/ssa_stats_sR_{s_R}.png"
        plot_ssa_statistics(trajectories, s_R, output_path)
        print(f"Generated {output_path}")

if __name__ == "__main__":
    run_ssa_statistical_analysis()
