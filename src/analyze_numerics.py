import os
import time

import matplotlib.pyplot as plt
import numpy as np

from .config import (
    PLOT_CONFIG,
    RC_PARAMS,
    NUMERICS_BENCHMARK_METHODS,
    NUMERICS_IMPLICIT_METHODS,
    NUMERICS_TOLERANCE_GRID,
    NUMERICS_SENSITIVITY_CASES,
)
from .sim_ode import run_ode_simulation


def _tail_amplitude(values, tail_fraction=0.2):
    start = int(len(values) * (1.0 - float(tail_fraction)))
    tail = values[start:]
    return float(np.max(tail) - np.min(tail))


def _run_with_timing(**kwargs):
    start = time.perf_counter()
    result = run_ode_simulation(return_solver_stats=True, **kwargs)
    elapsed = time.perf_counter() - start
    result["solver_stats"]["elapsed_sec"] = float(elapsed)
    return result


def benchmark_solvers(s_R=0.2, t_max=400, num_points=2000):
    rows = []
    for method in NUMERICS_BENCHMARK_METHODS:
        result = _run_with_timing(
            s_R=s_R,
            t_max=t_max,
            num_points=num_points,
            method=method,
            use_jacobian=True,
        )
        stats = result["solver_stats"]
        rows.append(
            {
                "method": method,
                "elapsed_sec": stats["elapsed_sec"],
                "nfev": stats["nfev"],
                "njev": stats["njev"],
                "nlu": stats["nlu"],
                "success": stats["success"],
            }
        )
    return rows


def benchmark_jacobian_effect(s_R=0.2, t_max=400, num_points=2000):
    rows = []
    for method in NUMERICS_IMPLICIT_METHODS:
        for use_jacobian in (False, True):
            result = _run_with_timing(
                s_R=s_R,
                t_max=t_max,
                num_points=num_points,
                method=method,
                use_jacobian=use_jacobian,
            )
            stats = result["solver_stats"]
            rows.append(
                {
                    "method": method,
                    "used_jacobian": bool(use_jacobian),
                    "elapsed_sec": stats["elapsed_sec"],
                    "nfev": stats["nfev"],
                    "njev": stats["njev"],
                    "nlu": stats["nlu"],
                    "success": stats["success"],
                }
            )
    return rows


def run_tolerance_sensitivity(cases=NUMERICS_SENSITIVITY_CASES, t_max=400, num_points=2000):
    rows = []
    for case in cases:
        reference = _run_with_timing(
            s_R=case["s_R"],
            t_max=t_max,
            num_points=num_points,
            method="BDF",
            rtol=1e-8,
            atol=1e-10,
            use_jacobian=True,
        )
        ref_a_final = float(reference["A"][-1])
        ref_r_final = float(reference["R"][-1])
        ref_a_tail = _tail_amplitude(reference["A"])
        ref_r_tail = _tail_amplitude(reference["R"])

        for rtol, atol in NUMERICS_TOLERANCE_GRID:
            result = _run_with_timing(
                s_R=case["s_R"],
                t_max=t_max,
                num_points=num_points,
                method="BDF",
                rtol=rtol,
                atol=atol,
                use_jacobian=True,
            )
            stats = result["solver_stats"]
            rows.append(
                {
                    "label": case["label"],
                    "s_R": case["s_R"],
                    "rtol": rtol,
                    "atol": atol,
                    "elapsed_sec": stats["elapsed_sec"],
                    "A_final_error": abs(float(result["A"][-1]) - ref_a_final),
                    "R_final_error": abs(float(result["R"][-1]) - ref_r_final),
                    "A_tail_error": abs(_tail_amplitude(result["A"]) - ref_a_tail),
                    "R_tail_error": abs(_tail_amplitude(result["R"]) - ref_r_tail),
                }
            )
    return rows


def plot_numerics_summary(benchmark_rows, jac_rows, sensitivity_rows, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    a_colors = PLOT_CONFIG["COLORS"]["A"]   # blues/teals
    r_colors = PLOT_CONFIG["COLORS"]["R"]   # reds/oranges

    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Get unified colors
        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        
        # Panel 1: NFEV on log scale (removes confusing dual axis)
        methods = [row["method"] for row in benchmark_rows]
        nfev    = [row["nfev"]   for row in benchmark_rows]
        bar_colors = [neutral_colors['gray'], a_colors[0], r_colors[0]]

        bars = axes[0].bar(methods, nfev, color=bar_colors, width=0.55,
                           edgecolor="white", linewidth=1.2)
        axes[0].set_yscale("log")
        
        # Calculate cost ratio for title
        explicit_cost = nfev[0]  # RK45
        implicit_cost = nfev[1]  # BDF
        cost_ratio = explicit_cost / implicit_cost
        
        axes[0].set_title(f"Explicit methods require {cost_ratio:.0f}× more evaluations", 
                         fontsize=10, pad=15)
        axes[0].set_ylabel("NFEV  (log scale)")
        for bar, val in zip(bars, nfev):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                f"{val:,}", ha="center", va="bottom", fontsize=9,
            )

        # --- Panel 2: Jacobian effect on runtime ---
        # Colour scheme: BDF family = blues, Radau family = reds
        # lighter shade = no jac, darker shade = with jac
        jac_labels = [
            f"{row['method']}\n{'+ jac' if row['used_jacobian'] else 'no jac'}"
            for row in jac_rows
        ]
        jac_elapsed   = [row["elapsed_sec"] for row in jac_rows]
        jac_bar_colors = [a_colors[1], a_colors[0], r_colors[2], r_colors[0]]

        # Calculate Jacobian speedup
        bdf_no_jac = jac_elapsed[0]
        bdf_with_jac = jac_elapsed[1]
        jac_speedup = ((bdf_no_jac - bdf_with_jac) / bdf_no_jac) * 100
        
        axes[1].bar(jac_labels, jac_elapsed, color=jac_bar_colors, width=0.55,
                    edgecolor="white", linewidth=1.2)
        axes[1].set_title(f"Analytic Jacobian reduces overhead by ~{jac_speedup:.0f}%\n"
                         f"for implicit solvers", fontsize=10)
        axes[1].set_ylabel("Seconds")

        # --- Panel 3: Tolerance sensitivity ---
        # near-threshold = prominent blue; oscillatory = subtle gray
        neutral_colors = PLOT_CONFIG['NEUTRAL_COLORS']
        label_style = {
            "near_threshold": {"color": a_colors[0], "lw": 2.2, "marker": "o"},
            "oscillatory":    {"color": neutral_colors['gray_light'], "lw": 1.8, "marker": "s",
                               "linestyle": "--"},
        }
        seen = set()
        for row_label in ["near_threshold", "oscillatory"]:
            subset = sorted(
                [r for r in sensitivity_rows if r["label"] == row_label],
                key=lambda r: r["rtol"],
                reverse=True,   # loosest first → matches x-axis label order
            )
            if not subset or row_label in seen:
                continue
            seen.add(row_label)
            x = np.arange(len(subset))
            total_error = [r["A_tail_error"] + r["R_tail_error"] for r in subset]
            style = label_style.get(row_label, {"color": neutral_colors['gray_dark'], "lw": 2, "marker": "o"})
            axes[2].plot(x, total_error, label=row_label, **style)

        axes[2].set_title("Tolerance sensitivity\n(tail amplitude error vs. BDF reference)")
        axes[2].set_xticks(range(len(NUMERICS_TOLERANCE_GRID)))
        axes[2].set_xticklabels(
            [f"rtol={rtol:.0e}\natol={atol:.0e}" for rtol, atol in NUMERICS_TOLERANCE_GRID]
        )
        axes[2].set_xlabel("← looser           tighter →")
        axes[2].set_ylabel("|$A$ tail error| + |$R$ tail error|")
        axes[2].legend()

        fig.tight_layout()
        output_path = os.path.join(output_dir, "numerics_summary.png")
        fig.savefig(output_path, dpi=PLOT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    print(f"Saved numerics summary plot to {output_path}")


def _print_table(title, rows, columns):
    def _format_value(value):
        if isinstance(value, float):
            if value == 0.0 or abs(value) < 1e-4 or abs(value) >= 1e4:
                return f"{value:.3e}"
            return f"{value:.6f}"
        return str(value)

    print(title)
    for row in rows:
        parts = [f"{column}={_format_value(row[column])}" for column in columns]
        print("  " + ", ".join(parts))


def run_numerics_analysis(output_dir="figures"):
    benchmark_rows = benchmark_solvers()
    jac_rows = benchmark_jacobian_effect()
    sensitivity_rows = run_tolerance_sensitivity()

    _print_table(
        "Solver benchmark",
        benchmark_rows,
        ("method", "elapsed_sec", "nfev", "njev", "nlu", "success"),
    )
    _print_table(
        "Jacobian benchmark",
        jac_rows,
        ("method", "used_jacobian", "elapsed_sec", "nfev", "njev", "nlu", "success"),
    )
    _print_table(
        "Tolerance sensitivity",
        sensitivity_rows,
        ("label", "s_R", "rtol", "atol", "elapsed_sec", "A_tail_error", "R_tail_error"),
    )

    plot_numerics_summary(benchmark_rows, jac_rows, sensitivity_rows, output_dir=output_dir)
    return {
        "benchmark": benchmark_rows,
        "jacobian": jac_rows,
        "sensitivity": sensitivity_rows,
    }


if __name__ == "__main__":
    run_numerics_analysis()
