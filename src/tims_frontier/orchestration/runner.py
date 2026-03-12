from __future__ import annotations

from typing import Any, Iterable, Mapping

from tims_frontier.config.resolver import resolve_study_config
from tims_frontier.orchestration.run_manifest import (
    create_run_manifest,
    mark_run_finished,
    mark_run_started,
    record_analysis_completed,
    record_reporting_completed,
    record_trial_persisted,
)
from tims_frontier.reporting.summary import build_run_summary
from tims_frontier.storage.writers import (
    ensure_run_directories,
    initialize_result_files,
    write_failure_log_record,
    write_frontier_points_table,
    write_resolved_config,
    write_run_manifest,
    write_run_summary,
    write_trial_results_record,
)


def prepare_run(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create the resolved config and initial run manifest for a study run."""

    resolved = resolve_study_config(
        study_config,
        source_config_path=source_config_path,
        run_id=run_id,
        created_at_utc=created_at_utc,
    )
    manifest = create_run_manifest(resolved)
    return resolved, manifest


def initialize_run(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare a run and persist its initial resolved-config + manifest snapshots."""

    resolved, manifest = prepare_run(
        study_config,
        source_config_path=source_config_path,
        run_id=run_id,
        created_at_utc=created_at_utc,
    )
    ensure_run_directories(resolved)
    initialize_result_files(resolved)
    write_resolved_config(resolved)
    mark_run_started(manifest, started_at_utc=resolved["run"]["created_at_utc"])
    write_run_manifest(resolved, manifest)
    return resolved, manifest


def persist_trial_outputs(
    resolved_config: Mapping[str, Any],
    manifest: dict[str, Any],
    *,
    trial_record: Mapping[str, Any],
    failure_record: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Persist one trial row, optional failure detail, and the updated manifest."""

    paths: dict[str, str] = {
        "trial_results_table": write_trial_results_record(resolved_config, trial_record),
    }
    failure_logged = False
    if failure_record is not None:
        paths["failure_log"] = write_failure_log_record(resolved_config, failure_record)
        failure_logged = True

    record_trial_persisted(
        manifest,
        family=str(trial_record["family"]),
        trial_id=str(trial_record["trial_id"]),
        status=str(trial_record["status"]),
        optuna_trial_number=trial_record.get("optuna_trial_number"),
        family_trial_index=trial_record.get("family_trial_index"),
        failure_logged=failure_logged,
        failure_code=trial_record.get("failure_code"),
        failure_message=trial_record.get("failure_message"),
    )
    paths["run_manifest"] = write_run_manifest(resolved_config, manifest)
    return paths


def persist_frontier_outputs(
    resolved_config: Mapping[str, Any],
    manifest: dict[str, Any],
    frontier_records: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    """Persist processed frontier rows and the updated manifest."""

    materialized = [dict(record) for record in frontier_records]
    paths = {
        "frontier_points_table": write_frontier_points_table(resolved_config, materialized),
    }
    families = sorted({str(record["family"]) for record in materialized})
    record_analysis_completed(manifest, families=families or None)
    paths["run_manifest"] = write_run_manifest(resolved_config, manifest)
    return paths


def finalize_run(
    resolved_config: Mapping[str, Any],
    manifest: dict[str, Any],
    *,
    git_commit: str | None = None,
    git_branch: str | None = None,
    python_version: str | None = None,
    hostname: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """Build and persist the canonical run summary, then close the manifest."""

    record_reporting_completed(manifest)
    mark_run_finished(manifest, status=status)
    summary = build_run_summary(
        resolved_config,
        manifest,
        git_commit=git_commit,
        git_branch=git_branch,
        python_version=python_version,
        hostname=hostname,
    )
    write_run_summary(resolved_config, summary)
    write_run_manifest(resolved_config, manifest)
    return summary
