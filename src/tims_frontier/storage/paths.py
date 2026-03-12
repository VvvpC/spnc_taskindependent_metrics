from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    study_root: str
    run_root: str
    processed_study_root: str
    resolved_config_path: str
    trial_results_table_path: str
    failure_log_path: str
    run_summary_path: str
    run_manifest_path: str
    frontier_points_table_path: str
    logs_dir: str

    def as_dict(self) -> dict[str, str]:
        return {
            "study_root": self.study_root,
            "run_root": self.run_root,
            "processed_study_root": self.processed_study_root,
            "resolved_config_path": self.resolved_config_path,
            "trial_results_table_path": self.trial_results_table_path,
            "failure_log_path": self.failure_log_path,
            "run_summary_path": self.run_summary_path,
            "run_manifest_path": self.run_manifest_path,
            "frontier_points_table_path": self.frontier_points_table_path,
            "logs_dir": self.logs_dir,
        }


def build_run_paths(*, study_id: str, run_id: str, raw_root: str, processed_root: str) -> RunPaths:
    """Build canonical workflow paths for one run."""

    study_root = Path(raw_root) / study_id
    run_root = study_root / run_id
    processed_study_root = Path(processed_root) / study_id
    return RunPaths(
        study_root=study_root.as_posix(),
        run_root=run_root.as_posix(),
        processed_study_root=processed_study_root.as_posix(),
        resolved_config_path=(run_root / "resolved_config.yaml").as_posix(),
        trial_results_table_path=(run_root / "trial_results_table.csv").as_posix(),
        failure_log_path=(run_root / "failure_log.jsonl").as_posix(),
        run_summary_path=(run_root / "run_summary.json").as_posix(),
        run_manifest_path=(run_root / "run_manifest.json").as_posix(),
        frontier_points_table_path=(processed_study_root / "frontier_points_table.csv").as_posix(),
        logs_dir=(run_root / "logs").as_posix(),
    )
