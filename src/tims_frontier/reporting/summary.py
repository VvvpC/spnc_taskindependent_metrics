from __future__ import annotations

from typing import Any, Mapping


def build_run_summary(
    resolved_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    git_commit: str | None = None,
    git_branch: str | None = None,
    python_version: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Build the final run summary payload from resolved config + manifest."""

    family_states = manifest["family_states"]
    trial_counts = {
        family: {
            "attempted": state["attempted_trials"],
            "completed": state["completed_trials"],
            "failed": state["failed_trials"],
            "pruned": state["pruned_trials"],
        }
        for family, state in family_states.items()
    }
    primary = resolved_config["comparison"]["primary_endpoint"]
    return {
        "result_schema_version": resolved_config["result_schema_ref"]["schema_version"],
        "study_id": resolved_config["study"]["study_id"],
        "run_id": resolved_config["run"]["run_id"],
        "config_schema_version": resolved_config["source_config_schema_version"],
        "specification_ref": resolved_config["specification_ref"]["spec_path"],
        "resolved_config_path": resolved_config["storage"]["paths"]["resolved_config_path"],
        "exploration_method": resolved_config["exploration"]["method"],
        "primary_endpoint": {
            "endpoint_ref": "comparison.primary_endpoint",
            "frontier_objectives": list(primary["frontier_objectives"]),
            "directions": dict(primary["directions"]),
        },
        "families": list(resolved_config["comparison"]["compared_families"]),
        "trial_counts": trial_counts,
        "artifacts": {
            "trial_results_table": resolved_config["storage"]["paths"]["trial_results_table_path"],
            "frontier_points_table": resolved_config["storage"]["paths"]["frontier_points_table_path"],
            "failure_log": resolved_config["storage"]["paths"]["failure_log_path"],
        },
        "provenance": {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "python_version": python_version,
            "hostname": hostname,
        },
        "started_at_utc": manifest["lifecycle"]["started_at_utc"],
        "finished_at_utc": manifest["lifecycle"]["finished_at_utc"],
    }
