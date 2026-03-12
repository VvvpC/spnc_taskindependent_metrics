from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


RUN_MANIFEST_SCHEMA_VERSION = "1.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _touch_manifest(manifest: dict[str, Any], *, at_utc: str | None = None) -> str:
    timestamp = at_utc or _utc_now_iso()
    manifest["lifecycle"]["updated_at_utc"] = timestamp
    return timestamp


def create_run_manifest(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the initial mutable manifest for a resolved run."""

    now = _utc_now_iso()
    run = resolved_config["run"]
    exploration = resolved_config["exploration"]
    target_trials = int(exploration["optuna"]["n_trials_per_family"])
    baseline_family = resolved_config["comparison"]["baseline_family"]
    study_suffix = f"{exploration['method']}_study"

    family_states: dict[str, Any] = {}
    for family_name, family_cfg in resolved_config["families"].items():
        family_states[family_name] = {
            "status": "pending",
            "family_role": "baseline" if family_name == baseline_family else "candidate",
            "optuna_study_name": f"{family_name}_{study_suffix}",
            "target_trials": target_trials,
            "attempted_trials": 0,
            "completed_trials": 0,
            "failed_trials": 0,
            "pruned_trials": 0,
            "next_family_trial_index": 1,
            "last_optuna_trial_number": None,
            "frontier_extracted": False,
            "enabled": bool(family_cfg["enabled"]),
        }

    paths = resolved_config["storage"]["paths"]
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "study_id": resolved_config["study"]["study_id"],
        "run_id": run["run_id"],
        "status": "created",
        "lifecycle": {
            "created_at_utc": run["created_at_utc"],
            "updated_at_utc": now,
            "started_at_utc": None,
            "finished_at_utc": None,
        },
        "resolved_config_path": paths["resolved_config_path"],
        "source_config_path": resolved_config["source_config_path"],
        "config_schema_version": resolved_config["source_config_schema_version"],
        "resolved_config_schema_version": resolved_config["schema_version"],
        "result_schema_version": resolved_config["result_schema_ref"]["schema_version"],
        "specification_ref": resolved_config["specification_ref"]["spec_path"],
        "execution_state": {
            "current_stage": "created",
            "resume_supported": True,
            "last_persisted_trial_id": None,
            "last_persisted_at_utc": None,
        },
        "family_states": family_states,
        "artifacts": {
            "run_root": paths["run_root"],
            "trial_results_table": paths["trial_results_table_path"],
            "failure_log": paths["failure_log_path"],
            "run_summary": paths["run_summary_path"],
            "frontier_points_table": paths["frontier_points_table_path"],
            "logs_dir": paths["logs_dir"],
        },
        "checkpoints": {
            "trial_table_last_flush_utc": None,
            "failure_log_last_flush_utc": None,
            "analysis_started": False,
            "analysis_completed": False,
            "reporting_completed": False,
        },
        "latest_error": None,
        "notes": [],
    }


def mark_run_started(manifest: dict[str, Any], *, started_at_utc: str | None = None) -> dict[str, Any]:
    """Mark the run as started in the manifest."""

    timestamp = started_at_utc or _utc_now_iso()
    manifest["status"] = "running"
    manifest["lifecycle"]["started_at_utc"] = timestamp
    manifest["execution_state"]["current_stage"] = "running"
    _touch_manifest(manifest, at_utc=timestamp)
    return manifest


def record_trial_persisted(
    manifest: dict[str, Any],
    *,
    family: str,
    trial_id: str,
    status: str,
    optuna_trial_number: int | None = None,
    family_trial_index: int | None = None,
    failure_logged: bool = False,
    failure_code: str | None = None,
    failure_message: str | None = None,
    persisted_at_utc: str | None = None,
) -> dict[str, Any]:
    """Update manifest counters after one trial row has been persisted."""

    timestamp = persisted_at_utc or _utc_now_iso()
    family_state = manifest["family_states"][family]
    family_state["attempted_trials"] += 1
    if status == "completed":
        family_state["completed_trials"] += 1
    elif status == "failed":
        family_state["failed_trials"] += 1
    elif status == "pruned":
        family_state["pruned_trials"] += 1
    else:
        raise ValueError(f"Unsupported trial status: {status}")

    if family_trial_index is None:
        family_state["next_family_trial_index"] += 1
    else:
        family_state["next_family_trial_index"] = max(
            int(family_state["next_family_trial_index"]),
            int(family_trial_index) + 1,
        )

    family_state["last_optuna_trial_number"] = optuna_trial_number
    family_state["status"] = (
        "completed" if family_state["attempted_trials"] >= family_state["target_trials"] else "running"
    )

    manifest["execution_state"]["current_stage"] = "running"
    manifest["execution_state"]["last_persisted_trial_id"] = trial_id
    manifest["execution_state"]["last_persisted_at_utc"] = timestamp
    manifest["checkpoints"]["trial_table_last_flush_utc"] = timestamp
    if failure_logged:
        manifest["checkpoints"]["failure_log_last_flush_utc"] = timestamp
    if status == "failed":
        manifest["latest_error"] = {
            "family": family,
            "trial_id": trial_id,
            "failure_code": failure_code,
            "message": failure_message,
            "timestamp_utc": timestamp,
        }

    _touch_manifest(manifest, at_utc=timestamp)
    return manifest


def record_analysis_completed(
    manifest: dict[str, Any],
    *,
    families: list[str] | None = None,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Mark frontier extraction as completed for the supplied families."""

    timestamp = completed_at_utc or _utc_now_iso()
    manifest["checkpoints"]["analysis_started"] = True
    manifest["checkpoints"]["analysis_completed"] = True
    manifest["execution_state"]["current_stage"] = "analysis_completed"
    target_families = families or [
        name for name, state in manifest["family_states"].items() if bool(state.get("enabled"))
    ]
    for family in target_families:
        manifest["family_states"][family]["frontier_extracted"] = True
    _touch_manifest(manifest, at_utc=timestamp)
    return manifest


def record_reporting_completed(
    manifest: dict[str, Any],
    *,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Mark reporting outputs as persisted."""

    timestamp = completed_at_utc or _utc_now_iso()
    manifest["checkpoints"]["reporting_completed"] = True
    manifest["execution_state"]["current_stage"] = "reporting_completed"
    _touch_manifest(manifest, at_utc=timestamp)
    return manifest


def mark_run_finished(
    manifest: dict[str, Any],
    *,
    finished_at_utc: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """Mark the run as finished in the manifest."""

    timestamp = finished_at_utc or _utc_now_iso()
    manifest["status"] = status
    manifest["lifecycle"]["finished_at_utc"] = timestamp
    manifest["execution_state"]["current_stage"] = status
    _touch_manifest(manifest, at_utc=timestamp)
    return manifest
