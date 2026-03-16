"""
QSSA Validity Criterion Analysis

This module provides quantitative criteria for when QSSA approximation is reliable.
Based on bifurcation analysis and error metrics, it defines validity boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt

from .config import PLOT_CONFIG, RC_PARAMS, DEFAULT_PARAMS
from .bifurcation_analysis import find_hopf_bifurcation_threshold


def assess_qssa_validity(s_R, s_R_full_hopf, s_R_reduced_hopf, threshold_margin=0.1):
    """
    Assess QSSA validity at a given s_R value based on proximity to bifurcations.
    
    Criterion: QSSA should be used with caution when within threshold_margin
    (default 10%) of either Hopf bifurcation threshold.
    
    Parameters:
        s_R: parameter value to assess
        s_R_full_hopf: full model Hopf threshold
        s_R_reduced_hopf: reduced model Hopf threshold
        threshold_margin: relative distance threshold (0.1 = 10%)
    
    Returns:
        dict with keys:
            - valid: bool, True if QSSA is reliable
            - distance_to_full: relative distance to full model threshold
            - distance_to_reduced: relative distance to reduced model threshold
            - min_distance: minimum relative distance to either threshold
            - recommendation: string describing validity status
    """
    # Compute relative distances
    dist_full = abs(s_R - s_R_full_hopf) / s_R_full_hopf
    dist_reduced = abs(s_R - s_R_reduced_hopf) / s_R_reduced_hopf
    min_dist = min(dist_full, dist_reduced)
    
    # Assess validity
    if min_dist > threshold_margin:
        valid = True
        recommendation = "QSSA is reliable (far from bifurcations)"
    else:
        valid = False
        if dist_full < threshold_margin and dist_reduced < threshold_margin:
            recommendation = "QSSA unreliable (near both bifurcation thresholds)"
        elif dist_full < threshold_margin:
            recommendation = "QSSA unreliable (near full model bifurcation)"
        else:
            recommendation = "QSSA unreliable (near reduced model bifurcation)"
    
    return {
        'valid': valid,
        'distance_to_full': dist_full,
        'distance_to_reduced': dist_reduced,
        'min_distance': min_dist,
        'recommendation': recommendation,
    }


def plot_qssa_validity_regions():
    """
    Visualize QSSA validity regions in parameter space.
    
    Creates a diagram showing where QSSA is reliable vs unreliable based on
    proximity to bifurcation thresholds.
    """
    print("Generating QSSA validity criterion visualization...")
    
    # Find bifurcation thresholds
    s_R_full = find_hopf_bifurcation_threshold(model='full', s_R_range=(0.05, 0.15))
    s_R_reduced = find_hopf_bifurcation_threshold(model='reduced', s_R_range=(0.05, 0.15))
    
    if not s_R_full or not s_R_reduced:
        print("  Error: Could not find bifurcation thresholds")
        return
    
    # Parameter range
    s_R_values = np.linspace(0.02, 0.25, 200)
    
    # Assess validity for each point
    validity = []
    min_distances = []
    
    for s_R in s_R_values:
        result = assess_qssa_validity(s_R, s_R_full, s_R_reduced, threshold_margin=0.1)
        validity.append(result['valid'])
        min_distances.append(result['min_distance'])
    
    validity = np.array(validity)
    min_distances = np.array(min_distances)
    
    # Plot
    with plt.rc_context(RC_PARAMS):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Get unified colors
        valid_color = PLOT_CONFIG['STATUS_COLORS']['valid']
        invalid_color = PLOT_CONFIG['STATUS_COLORS']['invalid']
        full_color = PLOT_CONFIG['BIFURCATION_COLORS']['full_threshold']
        reduced_color = PLOT_CONFIG['BIFURCATION_COLORS']['reduced_threshold']
        
        # Panel 1: Validity regions
        ax1.fill_between(s_R_values, 0, 1, where=validity, 
                        alpha=0.3, color=valid_color, label='QSSA Reliable')
        ax1.fill_between(s_R_values, 0, 1, where=~validity, 
                        alpha=0.3, color=invalid_color, label='QSSA Unreliable')
        
        # Mark bifurcation thresholds
        ax1.axvline(s_R_full, color=full_color, linestyle='--', linewidth=2,
                   label=f'Full Model Hopf: {s_R_full:.3f}')
        ax1.axvline(s_R_reduced, color=reduced_color, linestyle='--', linewidth=2,
                   label=f'Reduced Model Hopf: {s_R_reduced:.3f}')
        
        # Mark 10% margins
        margin_full_low = s_R_full * 0.9
        margin_full_high = s_R_full * 1.1
        margin_reduced_low = s_R_reduced * 0.9
        margin_reduced_high = s_R_reduced * 1.1
        
        ax1.axvline(margin_full_low, color=full_color, linestyle=':', linewidth=1, alpha=0.5)
        ax1.axvline(margin_full_high, color=full_color, linestyle=':', linewidth=1, alpha=0.5)
        ax1.axvline(margin_reduced_low, color=reduced_color, linestyle=':', linewidth=1, alpha=0.5)
        ax1.axvline(margin_reduced_high, color=reduced_color, linestyle=':', linewidth=1, alpha=0.5)
        
        # Mark test points — use BIFURCATION_COLORS for consistency with Fig 9
        bifurc_colors = PLOT_CONFIG['BIFURCATION_COLORS']
        test_points = [
            (0.03,  'Steady',     bifurc_colors['test_steady']),
            (0.088, 'Near-threshold', bifurc_colors['test_bistable']),
            (0.2,   'Oscillatory', bifurc_colors['test_oscillatory']),
        ]
        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        for i, (s_R_test, label, t_color) in enumerate(test_points):
            result = assess_qssa_validity(s_R_test, s_R_full, s_R_reduced)
            marker = 'o' if result['valid'] else 'x'
            ax1.plot(s_R_test, 0.5, marker, markersize=12, color=t_color,
                    markeredgecolor=neutral_colors['black'], markeredgewidth=1.5, zorder=5)
            
            # Adjust vertical position to avoid overlap
            y_pos = 0.75 if i == 1 else 0.65  # Middle point higher to avoid overlap
            ax1.text(s_R_test, y_pos, label, ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=neutral_colors['white'], 
                             edgecolor=neutral_colors['gray'], alpha=0.9, linewidth=0.5))
        
        ax1.set_ylabel('QSSA Validity', fontsize=11)
        ax1.set_ylim(-0.1, 1.2)  # Extended upper limit to accommodate labels
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Unreliable', 'Reliable'])
        ax1.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.95)
        ax1.set_title('QSSA Validity: Observed Failure Pattern\n'
                     '(Fails at 8.3% distance, reliable beyond 69%)',
                     fontsize=11, pad=10)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Panel 2: Distance metric
        ax2.plot(s_R_values, min_distances, 'k-', linewidth=2, label='Min relative distance')
        ax2.axhline(0.1, color=invalid_color, linestyle='--', linewidth=2, alpha=0.7,
               label='10% threshold')
        ax2.fill_between(s_R_values, 0, 0.1, alpha=0.2, color=invalid_color)
        ax2.fill_between(s_R_values, 0.1, 1, alpha=0.2, color=valid_color)
        
        # Mark test points (same colors as ax1)
        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        for s_R_test, label, t_color in test_points:
            result = assess_qssa_validity(s_R_test, s_R_full, s_R_reduced)
            ax2.plot(s_R_test, result['min_distance'], 'o', markersize=10,
                    color=t_color, markeredgecolor=neutral_colors['black'], markeredgewidth=1.5, zorder=5)
        
        ax2.set_xlabel('$s_R$ (Repressor Degradation Rate)', fontsize=11)
        ax2.set_ylabel('Min Relative Distance\nto Bifurcation', fontsize=11)
        ax2.set_ylim(0, max(0.5, np.max(min_distances) * 1.1))
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        import os
        figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, 'qssa_validity_criterion.png')
        fig.savefig(output_path, dpi=PLOT_CONFIG['DPI'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("QSSA Validity Assessment for Test Points")
    print("="*70)
    
    for s_R_test, label, _ in test_points:
        result = assess_qssa_validity(s_R_test, s_R_full, s_R_reduced)
        print(f"\n{label} regime (s_R = {s_R_test}):")
        print(f"  Distance to full model threshold:    {result['distance_to_full']:.1%}")
        print(f"  Distance to reduced model threshold: {result['distance_to_reduced']:.1%}")
        print(f"  Minimum distance:                    {result['min_distance']:.1%}")
        print(f"  Status: {result['recommendation']}")
    
    print("="*70)


if __name__ == '__main__':
    plot_qssa_validity_regions()
