"""CLI for VKB genetic oscillator analyses."""

import argparse
import sys

def _run_analysis(module_path, fn_name, **kwargs):
    module = __import__(module_path, fromlist=[fn_name])
    getattr(module, fn_name)(**kwargs)


def run_core_analyses():
    print("\n### Core analyses ###\n")
    _run_analysis("scripts.analyze_baseline", "run_baseline_verification")
    _run_analysis("scripts.analyze_numerics", "run_numerics_analysis")

def run_all():
    print("\n" + "=" * 80)
    print("Running analyses")
    print("=" * 80 + "\n")
    run_core_analyses()
    print("\n" + "=" * 80)
    print("Finished")
    print("=" * 80 + "\n")


def _print_help():
    print("VKB genetic oscillator — main commands:\n")
    print("  python main.py all              Run the full reported analysis pipeline")
    print("  python main.py core             Baseline and numerics")
    print("  python main.py baseline         Baseline figures only")
    print("  python main.py numerics         Solver comparison only")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="VKB genetic oscillator analysis pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        help="Subcommand (default: show help)",
    )
    args = parser.parse_args(argv)

    cmd = args.command

    if cmd in ("help", "-h", "--help") or cmd == "":
        _print_help()
        return

    command_handlers = {
        "all": run_all,
        "core": run_core_analyses,
        "baseline": lambda: _run_analysis("scripts.analyze_baseline", "run_baseline_verification"),
        "numerics": lambda: _run_analysis("scripts.analyze_numerics", "run_numerics_analysis"),
    }

    handler = command_handlers.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}\n")
        _print_help()
        sys.exit(1)
    handler()


if __name__ == "__main__":
    main()
