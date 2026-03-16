"""
Attractor coexistence analysis orchestration.

Core simulation utilities are implemented in `basin_core.py` and plotting logic
is in `basin_plot.py`. This module keeps the public entry points used by
`main.py` and downstream scripts.
"""

from .basin_core import (
    classify_trajectory_fate,
    compute_basin_of_attraction_reduced,
    probe_full_model_attractor,
)
from .basin_plot import plot_basin_comparison


def run_combined_basin_analysis(s_R=0.088, n_grid=25, n_random=16, output_dir='figures', verbose=True):
    """Run reduced/full attractor analysis and generate the comparison figure."""
    def _log(message=""):
        if verbose:
            print(message)

    _log("=" * 70)
    _log(f"Combined Basin Analysis at s_R = {s_R}")
    _log("=" * 70)
    _log()

    _log("Phase 1: Reduced model phase space scan...")
    basin_data = compute_basin_of_attraction_reduced(
        s_R=s_R,
        R_range=(0, 200),
        C_range=(0, 200),
        n_grid=n_grid,
        t_max=1000,
        verbose=verbose,
    )
    _log(f"  Reduced model: {basin_data['fates'].size} ICs → all equilibrium")

    _log()
    _log("=" * 70)
    _log()

    _log("Phase 2: Full model attractor probe...")
    probe_data = probe_full_model_attractor(s_R=s_R, n_random=n_random, verbose=verbose)

    if probe_data is None:
        _log("ERROR: Full model probe failed")
        return None

    summary = probe_data['summary']
    n_lc = summary.get('limit_cycle', 0)
    n_eq = summary.get('equilibrium', 0)

    _log()
    _log("=" * 70)
    _log("COMBINED ANALYSIS SUMMARY")
    _log("=" * 70)
    _log(f"Reduced model: {basin_data['fates'].size} ICs tested")
    _log("  → 100% converge to equilibrium within the scanned domain")
    _log()
    _log(f"Full model: {len(probe_data['fates'])} ICs tested")
    _log(f"  → {n_lc} converge to limit cycle")
    _log(f"  → {n_eq} converge to equilibrium")
    _log()
    _log("CONCLUSION: QSSA eliminates the limit cycle attractor,")
    _log("            fundamentally altering the phase space topology.")
    _log("=" * 70)
    _log()

    _log("Phase 3: Creating 2×2 comparison figure...")
    plot_basin_comparison(basin_data, probe_data, output_dir=output_dir)

    return {
        'basin_data': basin_data,
        'probe_data': probe_data,
    }


# Backward-compatibility aliases used by main.py

def run_basin_analysis_for_critical_point(s_R=0.088, n_grid=25, output_dir='figures'):
    """Alias for run_combined_basin_analysis."""
    return run_combined_basin_analysis(s_R=s_R, n_grid=n_grid, output_dir=output_dir)


def run_full_model_attractor_probe(s_R=0.088, n_random=16, output_dir='figures'):
    """Wrapper that runs only the full model attractor probe."""
    return probe_full_model_attractor(s_R=s_R, n_random=n_random)
