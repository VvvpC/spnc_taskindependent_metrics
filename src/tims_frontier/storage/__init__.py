"""Storage helpers for canonical paths and result-record assembly."""

from .paths import RunPaths, build_run_paths
from .records import (
    FAILURE_LOG_FIELDNAMES,
    FRONTIER_POINTS_FIELDNAMES,
    TRIAL_RESULTS_FIELDNAMES,
    make_failure_record,
    make_frontier_record,
    make_trial_record,
)
from .writers import (
    ensure_run_directories,
    initialize_result_files,
    write_failure_log_record,
    write_frontier_points_table,
    write_resolved_config,
    write_run_manifest,
    write_run_summary,
    write_trial_results_record,
)

__all__ = [
    "FAILURE_LOG_FIELDNAMES",
    "FRONTIER_POINTS_FIELDNAMES",
    "RunPaths",
    "TRIAL_RESULTS_FIELDNAMES",
    "build_run_paths",
    "ensure_run_directories",
    "initialize_result_files",
    "make_failure_record",
    "make_frontier_record",
    "make_trial_record",
    "write_failure_log_record",
    "write_frontier_points_table",
    "write_resolved_config",
    "write_run_manifest",
    "write_run_summary",
    "write_trial_results_record",
]
