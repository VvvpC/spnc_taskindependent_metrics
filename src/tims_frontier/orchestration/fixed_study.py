from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

from tims_frontier.analysis.frontier import extract_frontier_points
from tims_frontier.construction.builders import build_trial_spec
from tims_frontier.evaluation.tims import evaluate_trial_tims
from tims_frontier.orchestration.runner import (
    finalize_run,
    initialize_run,
    persist_frontier_outputs,
    persist_trial_outputs,
)
from tims_frontier.orchestration.seeds import derive_seed_bundle
from tims_frontier.storage.records import make_failure_record, make_trial_record


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_trial_rows(trial_results_table_path: str) -> list[dict[str, Any]]:
    table_path = Path(trial_results_table_path)
    if not table_path.exists():
        return []
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_traceback_file(logs_dir: str, trial_id: str, exception: BaseException) -> str:
    log_path = Path(logs_dir) / f"{trial_id}_traceback.txt"
    trace_text = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(trace_text, encoding="utf-8")
    return log_path.as_posix()


def _make_trial_id(family: str, family_trial_index: int) -> str:
    return f"{family}_trial_{family_trial_index:04d}"


def resolve_fixed_search_params(resolved_config: Mapping[str, Any]) -> dict[str, float]:
    """Resolve fixed shared parameters from the configured search-space section."""

    search_space = resolved_config["exploration"]["search_space"]
    fixed_params: dict[str, float] = {}
    for name in ("beta_prime", "theta", "gamma", "m0"):
        entry = search_space[name]
        if entry["type"] == "fixed":
            fixed_params[name] = float(entry["value"])
            continue

        low = float(entry["low"])
        high = float(entry["high"])
        if low == high:
            fixed_params[name] = low
            continue

        raise ValueError(
            f"Fixed study runner requires '{name}' to be fixed, but got range [{low}, {high}]."
        )
    return fixed_params


def run_fixed_parameter_study(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    git_commit: str | None = None,
    git_branch: str | None = None,
    python_version: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Run a fixed-parameter repeated TIMs comparison for all enabled families."""

    resolved_config, manifest = initialize_run(
        study_config,
        source_config_path=source_config_path,
        run_id=run_id,
    )
    fixed_params = resolve_fixed_search_params(resolved_config)
    global_seed = int(resolved_config["execution"]["seed_policy"]["global_seed"])
    logs_dir = str(resolved_config["storage"]["paths"]["logs_dir"])

    try:
        for family in resolved_config["comparison"]["compared_families"]:
            family = str(family)
            family_state = manifest["family_states"][family]
            if not bool(family_state["enabled"]):
                continue

            study_name = str(family_state["optuna_study_name"])
            target_trials = int(family_state["target_trials"])
            for family_trial_index in range(1, target_trials + 1):
                trial_id = _make_trial_id(family, family_trial_index)
                started_at_utc = _utc_now_iso()
                started_perf = time.perf_counter()
                seed_bundle = derive_seed_bundle(
                    global_seed=global_seed,
                    family=family,
                    family_trial_index=family_trial_index,
                )
                build_spec = build_trial_spec(
                    resolved_config,
                    family=family,
                    sampled_params=fixed_params,
                    seed_bundle=seed_bundle,
                )

                try:
                    metrics = evaluate_trial_tims(build_spec, resolved_config, seed_bundle=seed_bundle)
                except Exception as exc:
                    traceback_path = _write_traceback_file(logs_dir, trial_id, exc)
                    failed_record = make_trial_record(
                        resolved_config=resolved_config,
                        build_spec=build_spec,
                        trial_id=trial_id,
                        family_trial_index=family_trial_index,
                        optuna_study_name=study_name,
                        optuna_trial_number=family_trial_index - 1,
                        status="failed",
                        seed_bundle=seed_bundle,
                        sampled_params=fixed_params,
                        failure_code=type(exc).__name__,
                        failure_message=str(exc),
                        runtime_seconds=time.perf_counter() - started_perf,
                        started_at_utc=started_at_utc,
                        finished_at_utc=_utc_now_iso(),
                    )
                    failure_record = make_failure_record(
                        resolved_config=resolved_config,
                        trial_id=trial_id,
                        family=family,
                        optuna_trial_number=family_trial_index - 1,
                        failure_code=type(exc).__name__,
                        exception_type=type(exc).__name__,
                        message=str(exc),
                        traceback_path=traceback_path,
                        trial_seed=int(seed_bundle["trial_seed"]),
                    )
                    persist_trial_outputs(
                        resolved_config,
                        manifest,
                        trial_record=failed_record,
                        failure_record=failure_record,
                    )
                    continue

                completed_record = make_trial_record(
                    resolved_config=resolved_config,
                    build_spec=build_spec,
                    trial_id=trial_id,
                    family_trial_index=family_trial_index,
                    optuna_study_name=study_name,
                    optuna_trial_number=family_trial_index - 1,
                    status="completed",
                    seed_bundle=seed_bundle,
                    sampled_params=fixed_params,
                    metrics=metrics,
                    runtime_seconds=time.perf_counter() - started_perf,
                    started_at_utc=started_at_utc,
                    finished_at_utc=_utc_now_iso(),
                )
                persist_trial_outputs(resolved_config, manifest, trial_record=completed_record)

        trial_rows = _read_trial_rows(resolved_config["storage"]["paths"]["trial_results_table_path"])
        frontier_rows = extract_frontier_points(trial_rows, resolved_config)
        persist_frontier_outputs(resolved_config, manifest, frontier_rows)
        return finalize_run(
            resolved_config,
            manifest,
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=python_version,
            hostname=hostname,
            status="completed",
        )
    except Exception:
        finalize_run(
            resolved_config,
            manifest,
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=python_version,
            hostname=hostname,
            status="failed",
        )
        raise
