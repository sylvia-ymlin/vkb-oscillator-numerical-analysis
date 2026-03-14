import sys
from src.analyze_baseline import run_baseline_verification
from src.analyze_numerics import run_numerics_analysis
from src.analyze_reduction import run_reduction_analysis
from src.analyze_stiffness import run_stiffness_analysis
from src.analyze_bifurcation import compare_bifurcation_thresholds, run_bifurcation_visualization, plot_qssa_validity_regions
from src.analyze_basin import run_combined_basin_analysis

def run_assignment():
    print("\n### Part A, B, C: Assignment Answers (Baseline & Benchmarks) ###\n")
    run_baseline_verification()
    run_numerics_analysis()

def run_extended():
    print("\n### Section 3.1: Stiffness Analysis ###\n")
    run_stiffness_analysis()
    
    print("\n### Section 3.2: Simplified Models (QSSA Reduction) ###\n")
    run_reduction_analysis()
    compare_bifurcation_thresholds()
    print()
    run_bifurcation_visualization()
    print()
    plot_qssa_validity_regions()
    print("\n### Section 3.2: Attractor Coexistence Test ###\n")
    run_combined_basin_analysis(s_R=0.088, n_grid=25, n_random=16)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "assignment":
        run_assignment()
        
    elif cmd == "extended":
        run_extended()

    elif cmd == "all":
        print("\n" + "="*80)
        print("Running Complete Analysis Pipeline")
        print("="*80 + "\n")
        run_assignment()
        run_extended()
        print("\n" + "="*80)
        print("Complete Analysis Finished")
        print("="*80 + "\n")

    # Granular commands
    elif cmd == "baseline":
        run_baseline_verification()
    elif cmd == "numerics":
        run_numerics_analysis()
    elif cmd == "stiffness":
        run_stiffness_analysis()
    elif cmd == "reduction":
        run_reduction_analysis()
    elif cmd == "bifurcation":
        compare_bifurcation_thresholds()
        run_bifurcation_visualization()
        plot_qssa_validity_regions()
    elif cmd == "basin":
        run_combined_basin_analysis(s_R=0.088, n_grid=25, n_random=16)

    else:
        print("Invalid command. Available commands:")
        print("  assignment    - Run Parts A, B, and C (baseline & benchmarks)")
        print("  extended      - Run Section 3.1 & 3.2 (stiffness, QSSA, bifurcation, basin)")
        print("  all           - Run complete analysis pipeline")
        print("\nGranular tasks:")
        print("  baseline, numerics, stiffness, reduction, bifurcation, basin")
