from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_sys_path() -> None:
    path_entries = [
        REPO_ROOT / "src",
        REPO_ROOT / "src" / "Morphology_Research",
        REPO_ROOT / "src" / "Project",
        REPO_ROOT / "src" / "Optuna_TaskIndependent_Metrics",
        REPO_ROOT / "src" / "ParetoFront_CQandMC",
        REPO_ROOT / "src" / "Plot_Functions",
        REPO_ROOT / "src" / "Test_Temporary",
    ]
    for entry in reversed(path_entries):
        entry_text = str(entry)
        if entry_text not in sys.path:
            sys.path.insert(0, entry_text)


_bootstrap_sys_path()


from tims_frontier.autoresearch import export_lineage, init_run, run_loop, run_step, show_best


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autoresearch v1 loop.")
    parser.add_argument(
        "--config",
        help="Optional path to an autoresearch JSON config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize an autoresearch run.")
    subparsers.add_parser("step", help="Run one autoresearch keep/discard step.")
    loop_parser = subparsers.add_parser("loop", help="Run multiple consecutive steps.")
    loop_parser.add_argument("--iterations", type=int, default=1, help="How many steps to attempt.")
    subparsers.add_parser("best", help="Show the current best kept proposal.")
    subparsers.add_parser("export", help="Export the lineage summary.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "init":
        payload = init_run(args.config)
    elif args.command == "step":
        payload = run_step(args.config)
    elif args.command == "loop":
        payload = run_loop(args.config, iterations=int(args.iterations))
    elif args.command == "best":
        payload = show_best(args.config)
    else:
        payload = export_lineage(args.config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
