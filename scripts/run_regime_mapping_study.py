from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE_PATH = Path("configs/tims_frontier/uniform_vs_heterogeneous_regime_mapping_v1.yaml")
DEFAULT_RUN_ID: str | None = None


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
        as_text = str(entry)
        if as_text not in sys.path:
            sys.path.insert(0, as_text)


_bootstrap_sys_path()


from tims_frontier.orchestration import run_regime_mapping_study


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _collect_provenance() -> dict[str, str | None]:
    return {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "python_version": platform.python_version(),
        "hostname": platform.node() or None,
    }


def _load_study_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Study config must deserialize to a mapping: {config_path}")
    return loaded


def main() -> int:
    config_path = (REPO_ROOT / CONFIG_RELATIVE_PATH).resolve()
    study_config = _load_study_config(config_path)
    provenance = _collect_provenance()
    summary = run_regime_mapping_study(
        study_config,
        source_config_path=config_path.as_posix(),
        run_id=DEFAULT_RUN_ID,
        git_commit=provenance["git_commit"],
        git_branch=provenance["git_branch"],
        python_version=provenance["python_version"],
        hostname=provenance["hostname"],
    )
    payload = {
        "meta_study_id": summary["meta_study_id"],
        "aggregate_run_id": summary["aggregate_run_id"],
        "config_path": config_path.as_posix(),
        "artifacts": summary["artifacts"],
        "block_order": summary["block_order"],
        "child_runs": summary["child_runs"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
