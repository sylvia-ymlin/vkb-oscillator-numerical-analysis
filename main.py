import sys
from src.analyze_baseline import run_baseline_verification
from src.analyze_numerics import run_numerics_analysis
from src.analyze_reduction import run_reduction_analysis
from src.analyze_stiffness import run_stiffness_analysis
from src.bifurcation_analysis import compare_bifurcation_thresholds
from src.plot_bifurcation import run_bifurcation_visualization
from src.basin_of_attraction import run_combined_basin_analysis
from src.qssa_validity_criterion import plot_qssa_validity_regions

cmd = sys.argv[1] if len(sys.argv) > 1 else ""

if cmd == "baseline":
    run_baseline_verification()

elif cmd == "numerics":
    run_numerics_analysis()

elif cmd == "reduction":
    run_reduction_analysis()

elif cmd == "stiffness":
    run_stiffness_analysis()

elif cmd == "bifurcation":
    compare_bifurcation_thresholds()
    print()
    run_bifurcation_visualization()
    print()
    plot_qssa_validity_regions()

elif cmd == "basin":
    # Combined analysis: creates Figure 6 (2-panel comparison)
    run_combined_basin_analysis(s_R=0.088, n_grid=25, n_random=16)

elif cmd == "all":
    # Run complete analysis pipeline
    print("\n" + "="*80)
    print("Running Complete Analysis Pipeline")
    print("="*80 + "\n")
    
    print("\n### Baseline Verification ###\n")
    run_baseline_verification()
    
    print("\n### Q1 Phenomenon: Solver Benchmark ###\n")
    run_numerics_analysis()
    
    print("\n### Q1 Mechanism: Stiffness Analysis ###\n")
    run_stiffness_analysis()
    
    print("\n### Q2 Phenomenon: Trajectory Comparison ###\n")
    run_reduction_analysis()
    
    print("\n### Q2 Mechanism: Bifurcation Analysis ###\n")
    compare_bifurcation_thresholds()
    print()
    run_bifurcation_visualization()
    print()
    plot_qssa_validity_regions()
    
    print("\n### Q2 Mechanism: Attractor Coexistence Test ###\n")
    run_combined_basin_analysis(s_R=0.088, n_grid=25, n_random=16)
    
    print("\n" + "="*80)
    print("Complete Analysis Finished")
    print("="*80 + "\n")

else:
    print("Invalid command. Available commands:")
    print("  baseline      - Baseline: Verify ODE and SSA implementations")
    print("  numerics      - Q1 Phenomenon: Benchmark solvers")
    print("  stiffness     - Q1 Mechanism: Eigenvalue analysis along trajectories")
    print("  reduction     - Q2 Phenomenon: Compare full vs reduced models")
    print("  bifurcation   - Q2 Mechanism: Bifurcation analysis + QSSA validity criterion")
    print("  basin         - Q2 Mechanism: Attractor coexistence test (phase space scan + multi-initial-condition probe)")
    print("  all           - Run complete analysis pipeline")

