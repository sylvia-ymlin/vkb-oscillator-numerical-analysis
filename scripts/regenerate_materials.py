#!/usr/bin/env python3
"""
Regenerate cached basin-scan data (50x50 grid) from scratch.

This runs the same workflow as `python main.py basin-grid` and proves the
pickle under materials/ is reproducible, not hand-edited.

Expect long runtime (thousands of ODE integrations). For plotting only from
an existing pickle:

    python scripts/regenerate_materials.py --plot-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from main import run_basin_grid_scan  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate materials/basin_scan_*.pkl and Figure 12")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Load existing pickle and only regenerate the figure",
    )
    args = p.parse_args()
    run_basin_grid_scan(plot_only=args.plot_only)


if __name__ == "__main__":
    main()
