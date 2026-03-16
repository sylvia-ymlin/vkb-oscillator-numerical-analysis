"""
Visualization for bifurcation analysis.

Generates plots comparing equilibrium stability between full and reduced models.
"""

import numpy as np
import matplotlib.pyplot as plt

from .config import PLOT_CONFIG, RC_PARAMS
from .bifurcation_analysis import (
    compute_bifurcation_curve,
    find_hopf_bifurcation_threshold,
)


def plot_bifurcation_diagram(s_R_range=(0.05, 0.15), n_points=30):
    """
    Plot bifurcation diagram: max(Re(λ)) vs s_R for both models.
    
    This shows how equilibrium stability changes with s_R, and clearly
    demonstrates the threshold shift between full and reduced models.
    """
    figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
    dpi = PLOT_CONFIG['DPI']
    
    s_R_values = np.linspace(s_R_range[0], s_R_range[1], n_points)
    
    print("Computing bifurcation curves...")
    print(f"  Evaluating {n_points} points in range [{s_R_range[0]}, {s_R_range[1]}]")
    
    # Compute curves
    print("  - Full model...")
    full_data = compute_bifurcation_curve(s_R_values, model='full')
    
    print("  - Reduced model...")
    reduced_data = compute_bifurcation_curve(s_R_values, model='reduced')
    
    # Find exact thresholds
    print("\nFinding exact bifurcation thresholds...")
    s_R_full = find_hopf_bifurcation_threshold(model='full', s_R_range=s_R_range)
    s_R_reduced = find_hopf_bifurcation_threshold(model='reduced', s_R_range=s_R_range)
    
    # Plot
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use unified model colors
        model_colors = PLOT_CONFIG['MODEL_COLORS']
        
        # Plot curves
        ax.plot(full_data['s_R'], full_data['max_real_parts'], 
                'o-', label='Full model (7D subspace)', linewidth=2, markersize=6, 
                color=model_colors['full'])
        ax.plot(reduced_data['s_R'], reduced_data['max_real_parts'], 
                's-', label='Reduced model (2D)', linewidth=2, markersize=6, 
                color=model_colors['reduced'])
        
        # Mark bifurcation thresholds
        bifurc_colors = PLOT_CONFIG['BIFURCATION_COLORS']
        if s_R_full:
            ax.axvline(s_R_full, color=bifurc_colors['full_threshold'], 
                      linestyle='--', alpha=0.7, linewidth=1.5,
                      label=f'Full: $s_R^*$ = {s_R_full:.4f}')
        if s_R_reduced:
            ax.axvline(s_R_reduced, color=bifurc_colors['reduced_threshold'], 
                      linestyle='--', alpha=0.7, linewidth=1.5,
                      label=f'Reduced: $s_R^*$ = {s_R_reduced:.4f}')
        
        # Mark the three test points
        test_points = [
            (0.03, 'Steady State', bifurc_colors['test_steady']),
            (0.088, 'Near-threshold', bifurc_colors['test_bistable']),
            (0.2, 'Oscillatory', bifurc_colors['test_oscillatory']),
        ]
        
        for s_R_test, label, color in test_points:
            ax.axvline(s_R_test, color=color, linestyle=':', alpha=0.4, linewidth=1.5)
            
            # Find corresponding y-values
            idx_full = np.argmin(np.abs(full_data['s_R'] - s_R_test))
            idx_reduced = np.argmin(np.abs(reduced_data['s_R'] - s_R_test))
            
            y_full = full_data['max_real_parts'][idx_full]
            y_reduced = reduced_data['max_real_parts'][idx_reduced]
            
            # Mark points on curves
            neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
            ax.plot(s_R_test, y_full, 'o', color=model_colors['full'], markersize=10,
                   markeredgecolor=neutral_colors['black'], markeredgewidth=1.5, zorder=5)
            ax.plot(s_R_test, y_reduced, 's', color=model_colors['reduced'], markersize=10,
                   markeredgecolor=neutral_colors['black'], markeredgewidth=1.5, zorder=5)
        
        # Zero line
        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        ax.axhline(0, color=neutral_colors['black'], linestyle='-', linewidth=1, alpha=0.3)
        
        # Shading
        status_colors = PLOT_CONFIG['STATUS_COLORS']
        ax.fill_between(s_R_values, -1, 0, alpha=0.1, color=status_colors['valid'])
        ax.fill_between(s_R_values, 0, 1, alpha=0.1, color=status_colors['invalid'])
        
        ax.set_xlabel('$s_R$ (repressor degradation rate)', fontsize=12)
        ax.set_ylabel('max Re(λ) at equilibrium', fontsize=12)
        ax.set_title('Hopf Bifurcation: Equilibrium Stability Analysis\n(Three test regimes marked)', 
                     fontsize=13, pad=15)
        
        # Simplified legend to avoid overcrowding
        ax.legend(loc='upper left', fontsize=8, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = f'{figures_dir}/bifurcation_diagram.png'
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved: {out_path}")
    
    return {
        'full_data': full_data,
        'reduced_data': reduced_data,
        's_R_full': s_R_full,
        's_R_reduced': s_R_reduced,
    }


def plot_eigenvalue_comparison_at_equilibrium(s_R_values=[0.03, 0.088, 0.2]):
    """
    Plot eigenvalue spectra at equilibrium for selected s_R values.
    
    Shows the complex plane distribution of eigenvalues for both models.
    """
    figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
    dpi = PLOT_CONFIG['DPI']
    
    from .bifurcation_analysis import (
        find_full_equilibrium,
        find_reduced_equilibrium,
        analyze_equilibrium_stability,
        full_jacobian_7d,
    )
    from .reduced_model import reduced_vkb_jac
    from .config import DEFAULT_PARAMS
    
    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(1, len(s_R_values), figsize=(15, 5))
        
        if len(s_R_values) == 1:
            axes = [axes]
        
        # Get unified colors
        model_colors = PLOT_CONFIG['MODEL_COLORS']
        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        
        for idx, s_R in enumerate(s_R_values):
            ax = axes[idx]
            params = {**DEFAULT_PARAMS, 's_R': s_R}
            
            # Full model
            try:
                y_full = find_full_equilibrium(s_R)
                stability_full = analyze_equilibrium_stability(y_full, full_jacobian_7d, params)
                eigvals_full = stability_full['eigenvalues']
                
                ax.scatter(np.real(eigvals_full), np.imag(eigvals_full),
                          s=80, marker='o', alpha=0.7, color=model_colors['full'],
                          label='Full model', edgecolors=neutral_colors['black'], linewidths=1)
            except Exception as e:
                print(f"Warning: Could not compute full model eigenvalues at s_R={s_R}: {e}")
            
            # Reduced model
            try:
                y_reduced = find_reduced_equilibrium(s_R)
                stability_reduced = analyze_equilibrium_stability(
                    y_reduced, reduced_vkb_jac, params
                )
                eigvals_reduced = stability_reduced['eigenvalues']
                
                ax.scatter(np.real(eigvals_reduced), np.imag(eigvals_reduced),
                          s=120, marker='s', alpha=0.7, color=model_colors['reduced'],
                          label='Reduced model', edgecolors=neutral_colors['black'], linewidths=1)
            except Exception as e:
                print(f"Warning: Could not compute reduced model eigenvalues at s_R={s_R}: {e}")
            
            # Formatting
            ax.axhline(0, color=neutral_colors['black'], linewidth=0.5, linestyle='--', alpha=0.5)
            ax.axvline(0, color=neutral_colors['black'], linewidth=1.5, linestyle='-', alpha=0.5)
            
            # Use symlog scale for x-axis to handle extreme ranges
            ax.set_xscale('symlog', linthresh=1.0)
            
            ax.set_xlabel('Re(λ)')
            ax.set_ylabel('Im(λ)')
            ax.set_title(f'$s_R = {s_R}$')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add stability annotation
            if idx == 0:
                ax.text(0.02, 0.98, 'Stable →', transform=ax.transAxes,
                       verticalalignment='top', fontsize=9, alpha=0.6)
        
        fig.suptitle('Eigenvalue Spectra at Equilibrium — Three Regimes', fontsize=13)
        plt.tight_layout()
        
        out_path = f'{figures_dir}/eigenvalue_equilibrium_comparison.png'
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {out_path}")


def run_bifurcation_visualization():
    """Entry point for generating all bifurcation plots."""
    import os
    figures_dir = PLOT_CONFIG.get('FIGURES_DIR', PLOT_CONFIG['IMG_DIR'])
    os.makedirs(figures_dir, exist_ok=True)
    
    print("="*80)
    print("Bifurcation Analysis Visualization")
    print("="*80)
    print()
    
    # Main bifurcation diagram
    results = plot_bifurcation_diagram(s_R_range=(0.05, 0.15), n_points=25)
    
    print()
    
    # Eigenvalue comparison
    plot_eigenvalue_comparison_at_equilibrium(s_R_values=[0.03, 0.088, 0.2])
    
    print()
    print("="*80)
    print("Summary:")
    print("="*80)
    
    if results['s_R_full'] and results['s_R_reduced']:
        shift = results['s_R_reduced'] - results['s_R_full']
        print(f"Full model Hopf bifurcation:    s_R = {results['s_R_full']:.6f}")
        print(f"Reduced model Hopf bifurcation: s_R = {results['s_R_reduced']:.6f}")
        print(f"Threshold shift:                Δs_R = {shift:+.6f}")
        print()
        print(f"At s_R = 0.088:")
        if results['s_R_full'] < 0.088 < results['s_R_reduced']:
            print("  ✓ Full model: oscillating (above threshold)")
            print("  ✗ Reduced model: steady state (below threshold)")
            print("  → This demonstrates the QSSA breakdown mechanism!")
    
    print("="*80)


if __name__ == '__main__':
    run_bifurcation_visualization()
