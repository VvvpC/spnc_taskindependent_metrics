from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import optuna
import yaml


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
        as_text = str(entry)
        if as_text not in sys.path:
            sys.path.insert(0, as_text)


_bootstrap_sys_path()


from tims_frontier.analysis.frontier import extract_frontier_points
from tims_frontier.construction.builders import build_trial_spec
from tims_frontier.evaluation.tims import evaluate_trial_tims
from tims_frontier.exploration import create_family_study, suggest_search_params
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TIMs frontier study workflow from a YAML study config.",
    )
    parser.add_argument("config", help="Path to the study config YAML.")
    parser.add_argument("--run-id", help="Optional explicit run_id to use for this study run.")
    parser.add_argument(
        "--optuna-verbosity",
        default="warning",
        choices=("debug", "info", "warning", "error"),
        help="Optuna logging verbosity for this invocation.",
    )
    return parser.parse_args()


def _load_study_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Study config must deserialize to a mapping: {config_path}")
    return loaded


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


def _set_optuna_verbosity(level: str) -> None:
    mapping = {
        "debug": optuna.logging.DEBUG,
        "info": optuna.logging.INFO,
        "warning": optuna.logging.WARNING,
        "error": optuna.logging.ERROR,
    }
    optuna.logging.set_verbosity(mapping[level])


def _write_traceback_file(logs_dir: str, trial_id: str, exception: BaseException) -> str:
    log_path = Path(logs_dir) / f"{trial_id}_traceback.txt"
    trace_text = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(trace_text, encoding="utf-8")
    return log_path.as_posix()


def _read_trial_rows(trial_results_table_path: str) -> list[dict[str, Any]]:
    table_path = Path(trial_results_table_path)
    if not table_path.exists():
        return []
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_trial_id(family: str, family_trial_index: int) -> str:
    return f"{family}_trial_{family_trial_index:04d}"


def _create_recording_objective(
    resolved_config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    family: str,
) -> Any:
    global_seed = int(resolved_config["execution"]["seed_policy"]["global_seed"])
    optuna_study_name = str(manifest["family_states"][family]["optuna_study_name"])
    logs_dir = str(resolved_config["storage"]["paths"]["logs_dir"])

    def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        family_trial_index = int(trial.number) + 1
        trial_id = _make_trial_id(family, family_trial_index)
        started_at_utc = _utc_now_iso()
        started_perf = time.perf_counter()
        sampled_params = suggest_search_params(trial, resolved_config)
        seed_bundle = derive_seed_bundle(
            global_seed=global_seed,
            family=family,
            family_trial_index=family_trial_index,
        )

        build_spec = build_trial_spec(
            resolved_config,
            family=family,
            sampled_params=sampled_params,
            seed_bundle=seed_bundle,
        )
        trial.set_user_attr("trial_id", trial_id)
        trial.set_user_attr("family", family)
        trial.set_user_attr("family_trial_index", family_trial_index)

        try:
            metrics = evaluate_trial_tims(build_spec, resolved_config, seed_bundle=seed_bundle)
        except optuna.TrialPruned as exc:
            pruned_record = make_trial_record(
                resolved_config=resolved_config,
                build_spec=build_spec,
                trial_id=trial_id,
                family_trial_index=family_trial_index,
                optuna_study_name=optuna_study_name,
                optuna_trial_number=int(trial.number),
                status="pruned",
                seed_bundle=seed_bundle,
                sampled_params=sampled_params,
                failure_message=str(exc),
                runtime_seconds=time.perf_counter() - started_perf,
                started_at_utc=started_at_utc,
                finished_at_utc=_utc_now_iso(),
            )
            persist_trial_outputs(resolved_config, manifest, trial_record=pruned_record)
            raise
        except Exception as exc:
            traceback_path = _write_traceback_file(logs_dir, trial_id, exc)
            failed_record = make_trial_record(
                resolved_config=resolved_config,
                build_spec=build_spec,
                trial_id=trial_id,
                family_trial_index=family_trial_index,
                optuna_study_name=optuna_study_name,
                optuna_trial_number=int(trial.number),
                status="failed",
                seed_bundle=seed_bundle,
                sampled_params=sampled_params,
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
                optuna_trial_number=int(trial.number),
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
            raise

        completed_record = make_trial_record(
            resolved_config=resolved_config,
            build_spec=build_spec,
            trial_id=trial_id,
            family_trial_index=family_trial_index,
            optuna_study_name=optuna_study_name,
            optuna_trial_number=int(trial.number),
            status="completed",
            seed_bundle=seed_bundle,
            sampled_params=sampled_params,
            metrics=metrics,
            runtime_seconds=time.perf_counter() - started_perf,
            started_at_utc=started_at_utc,
            finished_at_utc=_utc_now_iso(),
        )
        persist_trial_outputs(resolved_config, manifest, trial_record=completed_record)
        return float(metrics["MC"]), float(metrics["CQ"])

    return objective


def _run_family_study(
    resolved_config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    family: str,
) -> None:
    family_state = manifest["family_states"][family]
    if not bool(family_state["enabled"]):
        return

    study = create_family_study(
        resolved_config,
        family=family,
        study_name=str(family_state["optuna_study_name"]),
    )
    objective = _create_recording_objective(resolved_config, manifest, family=family)
    budget = resolved_config["exploration"]["budget"]
    n_trials = int(resolved_config["exploration"]["optuna"]["n_trials_per_family"])
    timeout = budget["timeout_seconds"]
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None if timeout in (None, "") else int(timeout),
        catch=(Exception,),
        show_progress_bar=False,
    )


def _emit_final_message(
    resolved_config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    payload = {
        "study_id": resolved_config["study"]["study_id"],
        "run_id": resolved_config["run"]["run_id"],
        "trial_results_table": resolved_config["storage"]["paths"]["trial_results_table_path"],
        "frontier_points_table": resolved_config["storage"]["paths"]["frontier_points_table_path"],
        "run_summary": resolved_config["storage"]["paths"]["run_summary_path"],
        "failure_log": resolved_config["storage"]["paths"]["failure_log_path"],
        "trial_counts": summary["trial_counts"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    args = _parse_args()
    _set_optuna_verbosity(args.optuna_verbosity)

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    study_config = _load_study_config(config_path)

    provenance = _collect_provenance()
    resolved: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None

    try:
        resolved, manifest = initialize_run(
            study_config,
            source_config_path=config_path.as_posix(),
            run_id=args.run_id,
        )
        for family in resolved["comparison"]["compared_families"]:
            _run_family_study(resolved, manifest, family=str(family))

        trial_rows = _read_trial_rows(resolved["storage"]["paths"]["trial_results_table_path"])
        frontier_rows = extract_frontier_points(trial_rows, resolved)
        persist_frontier_outputs(resolved, manifest, frontier_rows)
        summary = finalize_run(
            resolved,
            manifest,
            git_commit=provenance["git_commit"],
            git_branch=provenance["git_branch"],
            python_version=provenance["python_version"],
            hostname=provenance["hostname"],
            status="completed",
        )
    except Exception:
        if resolved is not None and manifest is not None:
            summary = finalize_run(
                resolved,
                manifest,
                git_commit=provenance["git_commit"],
                git_branch=provenance["git_branch"],
                python_version=provenance["python_version"],
                hostname=provenance["hostname"],
                status="failed",
            )
        raise

    if resolved is None or summary is None:
        raise RuntimeError("Run did not produce a summary payload.")
    _emit_final_message(resolved, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
