from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import yaml

from tims_frontier.storage.records import (
    FAILURE_LOG_FIELDNAMES,
    FRONTIER_POINTS_FIELDNAMES,
    TRIAL_RESULTS_FIELDNAMES,
)


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_run_directories(resolved_config: Mapping[str, Any]) -> dict[str, str]:
    """Create the canonical directories required for a run."""

    paths = resolved_config["storage"]["paths"]
    created = {}
    for key in ("study_root", "run_root", "processed_study_root", "logs_dir"):
        path = Path(paths[key])
        path.mkdir(parents=True, exist_ok=True)
        created[key] = path.as_posix()
    return created


def initialize_result_files(resolved_config: Mapping[str, Any]) -> dict[str, str]:
    """Create canonical empty result artifacts so paths are always materialized."""

    paths = resolved_config["storage"]["paths"]
    trial_path = _ensure_parent(paths["trial_results_table_path"])
    _write_csv_header_if_missing(trial_path, TRIAL_RESULTS_FIELDNAMES)

    frontier_path = _ensure_parent(paths["frontier_points_table_path"])
    _write_csv_header_if_missing(frontier_path, FRONTIER_POINTS_FIELDNAMES)

    failure_path = _ensure_parent(paths["failure_log_path"])
    failure_path.touch(exist_ok=True)

    return {
        "trial_results_table": trial_path.as_posix(),
        "frontier_points_table": frontier_path.as_posix(),
        "failure_log": failure_path.as_posix(),
    }


def _write_csv_header_if_missing(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()


def _append_csv_row(path: str | Path, fieldnames: list[str], row: Mapping[str, Any]) -> str:
    output_path = _ensure_parent(path)
    _write_csv_header_if_missing(output_path, fieldnames)
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow({name: row.get(name) for name in fieldnames})
    return output_path.as_posix()


def _write_csv_rows(path: str | Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    output_path = _ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return output_path.as_posix()


def _append_jsonl_row(path: str | Path, row: Mapping[str, Any]) -> str:
    output_path = _ensure_parent(path)
    with output_path.open("a", encoding="utf-8") as handle:
        json.dump(dict(row), handle, ensure_ascii=False)
        handle.write("\n")
    return output_path.as_posix()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    output_path = _ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return output_path.as_posix()


def _write_yaml(path: str | Path, payload: Mapping[str, Any]) -> str:
    output_path = _ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, allow_unicode=True, sort_keys=False)
    return output_path.as_posix()


def write_trial_results_record(resolved_config: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    """Append one trial-results row to the canonical CSV."""

    return _append_csv_row(
        resolved_config["storage"]["paths"]["trial_results_table_path"],
        TRIAL_RESULTS_FIELDNAMES,
        record,
    )


def write_frontier_points_table(
    resolved_config: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> str:
    """Overwrite the canonical frontier-points table with the provided rows."""

    return _write_csv_rows(
        resolved_config["storage"]["paths"]["frontier_points_table_path"],
        FRONTIER_POINTS_FIELDNAMES,
        records,
    )


def write_run_summary(resolved_config: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    """Write the canonical run summary JSON."""

    return _write_json(resolved_config["storage"]["paths"]["run_summary_path"], payload)


def write_resolved_config(resolved_config: Mapping[str, Any]) -> str:
    """Write the immutable resolved-config snapshot for a run."""

    return _write_yaml(resolved_config["storage"]["paths"]["resolved_config_path"], resolved_config)


def write_run_manifest(resolved_config: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Write the mutable run-manifest snapshot for a run."""

    return _write_json(resolved_config["storage"]["paths"]["run_manifest_path"], manifest)


def write_failure_log_record(resolved_config: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    """Append one failure record to the canonical JSONL log."""

    filtered = {name: record.get(name) for name in FAILURE_LOG_FIELDNAMES}
    return _append_jsonl_row(resolved_config["storage"]["paths"]["failure_log_path"], filtered)
