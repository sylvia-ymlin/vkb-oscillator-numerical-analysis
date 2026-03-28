"""
CLI for VKB genetic oscillator analyses.

Lite default (`python main.py all`): assignment + extended figures except slow runs.
Full (`python main.py all --full`): also SSA statistics (Figs 5, 7) and 50x50 basin grid (Fig 12).
"""

import argparse
import sys

from src.analyze_baseline import run_baseline_verification
from src.analyze_numerics import run_numerics_analysis
from src.analyze_reduction import run_reduction_analysis
from src.analyze_stiffness import run_stiffness_analysis
from src.analyze_bifurcation import compare_bifurcation_thresholds, run_bifurcation_visualization, plot_qssa_validity_regions
from src.analyze_basin import run_combined_basin_analysis
from src.analyze_ssa_stats import run_ssa_statistical_analysis

# Coarse basin / phase portrait (Fig. 11): reduced 25x25 scan + full-model probe with 72 ICs
# (1 biological + 4*n_random random + 7 axis, n_random=16). Not the 50x50 R–C grid (Fig. 12).
BASIN_S_R = 0.088
BASIN_REDUCED_N_GRID = 25  # Use 50 in analyze_basin_grid only for Figure 12 high-resolution scan.
BASIN_N_RANDOM = 16


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
    print("\n### Section 3.2: Attractor Coexistence Test (coarse basin, Fig. 11) ###\n")
    run_combined_basin_analysis(
        s_R=BASIN_S_R,
        n_grid=BASIN_REDUCED_N_GRID,
        n_random=BASIN_N_RANDOM,
    )


def run_all(full: bool = False):
    print("\n" + "=" * 80)
    print("Running analysis pipeline" + (" (FULL: slow steps included)" if full else " (lite)"))
    print("=" * 80 + "\n")
    run_assignment()
    run_extended()
    if full:
        print("\n### Full-only: SSA statistics (Figs 5, 7) ###\n")
        run_ssa_statistical_analysis(n_trajectories=50)
        print("\n### Full-only: 50x50 basin grid (Fig. 12) ###\n")
        run_basin_grid_scan()
    print("\n" + "=" * 80)
    print("Pipeline finished")
    print("=" * 80 + "\n")


def run_basin_grid_scan(plot_only: bool = False):
    """50x50 scan on the R–C plane; writes materials/*.pkl and figures/basin_grid_sR_*.png."""
    import pickle

    from src.analyze_basin_grid import plot_basin_grid, run_basin_scan
    from src.config import ODE_TOLERANCES_LOOSE

    s_R = BASIN_S_R
    n_grid = 50
    res_path = f"materials/basin_scan_sR_{s_R:.3f}_{n_grid}x{n_grid}.pkl"

    if plot_only:
        print(f"Loading basin scan from {res_path}...")
        with open(res_path, "rb") as f:
            results = pickle.load(f)
    else:
        results = run_basin_scan(s_R=s_R, n_grid=n_grid, tolerances=ODE_TOLERANCES_LOOSE)

    plot_basin_grid(results)


def _print_help():
    print("VKB genetic oscillator — main commands:\n")
    print("  python main.py all              Lite pipeline (~tens of minutes on a laptop; no Figs 5,7,12)")
    print("  python main.py all --full       Also SSA stats + 50x50 basin grid (hours; reproduces Figs 5, 7, 12)")
    print("  python main.py assignment       Parts A–C baseline + numerics only")
    print("  python main.py extended         Stiffness + QSSA + bifurcation + coarse basin (Fig. 11)")
    print("  python main.py stats            SSA statistics only (Figs 5, 7)")
    print("  python main.py basin-grid       50x50 basin scan + Fig. 12 (slow)")
    print("  python main.py basin-grid --plot-only   Plot Fig. 12 from cached materials/*.pkl")
    print("\nGranular: baseline, numerics, stiffness, reduction, bifurcation, basin")
    print("See README Appendix and docs/figure_pipeline.md for figure-to-command mapping.")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="VKB genetic oscillator analysis pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        help="Subcommand (default: show help)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help='With "all": run slow steps (SSA stats, 50x50 basin grid for Figs 5, 7, 12)',
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help='With "basin-grid": only plot from cached materials/basin_scan_*.pkl',
    )
    args = parser.parse_args(argv)

    cmd = args.command

    if cmd in ("help", "-h", "--help") or cmd == "":
        _print_help()
        return

    if cmd == "assignment":
        run_assignment()
    elif cmd == "extended":
        run_extended()
    elif cmd == "all":
        run_all(full=args.full)
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
        run_combined_basin_analysis(s_R=BASIN_S_R, n_grid=BASIN_REDUCED_N_GRID, n_random=BASIN_N_RANDOM)
    elif cmd == "stats":
        run_ssa_statistical_analysis(n_trajectories=50)
    elif cmd == "basin-grid":
        run_basin_grid_scan(plot_only=args.plot_only)
    else:
        print(f"Unknown command: {cmd}\n")
        _print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
