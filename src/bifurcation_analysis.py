"""
Bifurcation analysis public API and reporting utilities.

Core numerical routines are implemented in `bifurcation_core.py` to keep this
module concise while preserving backward-compatible imports used elsewhere.
"""

from .bifurcation_core import (
    analyze_equilibrium_stability,
    compute_bifurcation_curve,
    compute_max_real_part_at_equilibrium,
    find_full_equilibrium,
    find_hopf_bifurcation_threshold,
    find_reduced_equilibrium,
    full_jacobian_7d,
)


def compare_bifurcation_thresholds(s_R_range=(0.05, 0.15), verbose=True):
    """Compare Hopf bifurcation thresholds between full and reduced models."""
    def _log(message=""):
        if verbose:
            print(message)

    _log("=" * 80)
    _log("Hopf Bifurcation Threshold Analysis")
    _log("=" * 80)
    _log()
    _log("Computing bifurcation thresholds by analyzing equilibrium point stability...")
    _log()

    s_R_full = find_hopf_bifurcation_threshold(model='full', s_R_range=s_R_range)
    s_R_reduced = find_hopf_bifurcation_threshold(model='reduced', s_R_range=s_R_range)

    _log(f"Full model:     s_R_hopf = {s_R_full:.6f}" if s_R_full else "Full model: threshold not found")
    _log(f"Reduced model:  s_R_hopf = {s_R_reduced:.6f}" if s_R_reduced else "Reduced model: threshold not found")
    _log()

    if s_R_full and s_R_reduced:
        shift = s_R_reduced - s_R_full
        shift_percent = (shift / s_R_full) * 100

        _log(f"Threshold shift: Δs_R = {shift:+.6f} ({shift_percent:+.2f}%)")
        _log()
        _log("=" * 80)
        _log("Interpretation:")
        _log("=" * 80)
        _log()
        _log("At s_R = 0.088:")

        if s_R_full < 0.088 < s_R_reduced:
            _log(f"  • Full model: ABOVE threshold ({s_R_full:.4f}) → Oscillating")
            _log(f"  • Reduced model: BELOW threshold ({s_R_reduced:.4f}) → Stable steady state")
            _log()
            _log("This explains the QSSA breakdown at s_R = 0.088:")
            _log("Dimension reduction alters the phase space topology, shifting the bifurcation.")
        else:
            _log("  • Threshold-only classification is inconclusive at this parameter value.")
            _log("  • Use basin-of-attraction results to resolve attractor coexistence behavior.")

        _log()
        _log("The QSSA approximation shifts the bifurcation threshold because it")
        _log("eliminates 7 fast variables, fundamentally altering the phase space structure.")

    _log("=" * 80)

    return {
        'full_threshold': s_R_full,
        'reduced_threshold': s_R_reduced,
    }


if __name__ == '__main__':
    print("Testing equilibrium finding...")

    for s_R in [0.03, 0.088, 0.2]:
        print(f"\ns_R = {s_R}:")
        try:
            y_reduced = find_reduced_equilibrium(s_R)
            print(f"  Reduced equilibrium: R*={y_reduced[0]:.3f}, C*={y_reduced[1]:.3f}")
        except Exception as e:
            print(f"  Reduced: {e}")

        try:
            y_full = find_full_equilibrium(s_R)
            print(f"  Full equilibrium found: A={y_full[5]:.3f}, R={y_full[7]:.3f}, C={y_full[8]:.3f}")
        except Exception as e:
            print(f"  Full: {e}")

    print("\n" + "=" * 80)
    compare_bifurcation_thresholds()
