from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tims_frontier.orchestration.fixed_study import run_fixed_parameter_study
from tims_frontier.reporting import send_run_completion_notification


REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_seed_sweep_definition(study_config: Mapping[str, Any]) -> tuple[int, list[int]]:
    sweep_cfg = study_config.get("seed_sweep")
    if not isinstance(sweep_cfg, Mapping):
        raise ValueError("V2 fixed-geometry study requires a top-level 'seed_sweep' mapping.")

    morphology_seeds = sweep_cfg.get("morphology_seeds")
    if not isinstance(morphology_seeds, Sequence) or isinstance(morphology_seeds, (str, bytes)):
        raise ValueError("'seed_sweep.morphology_seeds' must be a sequence of integer seeds.")
    if len(morphology_seeds) == 0:
        raise ValueError("'seed_sweep.morphology_seeds' must contain at least one seed.")

    unique_seeds = [int(seed) for seed in morphology_seeds]
    if len(set(unique_seeds)) != len(unique_seeds):
        raise ValueError("'seed_sweep.morphology_seeds' must be unique.")

    trials_per_family = int(sweep_cfg.get("n_trials_per_family", 0))
    if trials_per_family <= 0:
        raise ValueError("'seed_sweep.n_trials_per_family' must be a positive integer.")

    return trials_per_family, unique_seeds


def _derive_substudy_id(base_study_id: str, seed_index: int) -> str:
    return f"{base_study_id}_seed_{seed_index:02d}"


def _derive_subrun_id(base_run_id: str | None, seed_index: int, morphology_seed: int) -> str:
    prefix = base_run_id or "fixed_geometry_beta20_v2"
    return f"{prefix}_seed_{seed_index:02d}_{morphology_seed}"


def _prepare_child_config(
    study_config: Mapping[str, Any],
    *,
    base_study_id: str,
    seed_index: int,
    morphology_seed: int,
    trials_per_family: int,
) -> dict[str, Any]:
    child = deepcopy(dict(study_config))
    child["study"]["study_id"] = _derive_substudy_id(base_study_id, seed_index)
    child["study"]["title"] = (
        f"{study_config['study']['title']} | morphology seed {seed_index:02d}"
    )
    child["study"]["description"] = (
        f"{study_config['study']['description']} "
        f"(seed_index={seed_index:02d}, morphology_seed={morphology_seed})"
    )
    child["study"]["tags"] = list(child["study"].get("tags", [])) + ["seed-sweep-child"]
    child["families"]["heterogeneous"]["construction"]["morphology"]["morphology_seed"]["mode"] = "fixed"
    child["families"]["heterogeneous"]["construction"]["morphology"]["morphology_seed"]["value"] = int(
        morphology_seed
    )
    child["exploration"]["optuna"]["n_trials_per_family"] = int(trials_per_family)
    return child


def _results_root(base_study_id: str, aggregate_run_id: str) -> tuple[Path, Path]:
    raw_root = REPO_ROOT / "results" / "raw" / base_study_id / aggregate_run_id
    processed_root = REPO_ROOT / "results" / "processed" / base_study_id
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    return raw_root, processed_root


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.as_posix()


def run_fixed_geometry_seed_sweep(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    git_commit: str | None = None,
    git_branch: str | None = None,
    python_version: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Run a fixed-geometry study across multiple heterogeneous morphology seeds."""

    trials_per_family, morphology_seeds = _load_seed_sweep_definition(study_config)
    base_study_id = str(study_config["study"]["study_id"])
    aggregate_run_id = run_id or f"{base_study_id}_seed_sweep"
    started_at_utc = _utc_now_iso()

    child_runs: list[dict[str, Any]] = []
    for seed_index, morphology_seed in enumerate(morphology_seeds, start=1):
        child_config = _prepare_child_config(
            study_config,
            base_study_id=base_study_id,
            seed_index=seed_index,
            morphology_seed=morphology_seed,
            trials_per_family=trials_per_family,
        )
        child_summary = run_fixed_parameter_study(
            child_config,
            source_config_path=source_config_path,
            run_id=_derive_subrun_id(aggregate_run_id, seed_index, morphology_seed),
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=python_version,
            hostname=hostname,
            notify_on_completion=False,
        )
        child_runs.append(
            {
                "seed_index": seed_index,
                "morphology_seed": morphology_seed,
                "study_id": child_config["study"]["study_id"],
                "run_id": child_summary["run_id"],
                "trial_counts": child_summary["trial_counts"],
                "artifacts": child_summary["artifacts"],
            }
        )

    finished_at_utc = _utc_now_iso()
    raw_root, processed_root = _results_root(base_study_id, aggregate_run_id)

    aggregate_summary = {
        "meta_study_id": base_study_id,
        "aggregate_run_id": aggregate_run_id,
        "source_config_path": source_config_path,
        "morphology_seed_count": len(morphology_seeds),
        "morphology_seeds": morphology_seeds,
        "n_trials_per_family": trials_per_family,
        "child_runs": child_runs,
        "provenance": {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "python_version": python_version,
            "hostname": hostname,
        },
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
    }
    manifest = {
        "meta_study_id": base_study_id,
        "aggregate_run_id": aggregate_run_id,
        "status": "completed",
        "child_run_count": len(child_runs),
        "child_run_ids": [entry["run_id"] for entry in child_runs],
        "morphology_seeds": morphology_seeds,
        "created_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
    }

    aggregate_summary_path = _write_json(raw_root / "seed_sweep_summary.json", aggregate_summary)
    aggregate_manifest_path = _write_json(raw_root / "seed_sweep_manifest.json", manifest)
    processed_summary_path = _write_json(processed_root / "seed_sweep_summary.json", aggregate_summary)

    send_run_completion_notification(
        study_id=base_study_id,
        run_id=aggregate_run_id,
        status="completed",
        extra_message=(
            f"morphology_seed_count={len(morphology_seeds)}, "
            f"n_trials_per_family={trials_per_family}"
        ),
    )

    return {
        "meta_study_id": base_study_id,
        "aggregate_run_id": aggregate_run_id,
        "source_config_path": source_config_path,
        "n_trials_per_family": trials_per_family,
        "morphology_seeds": morphology_seeds,
        "artifacts": {
            "seed_sweep_summary": aggregate_summary_path,
            "seed_sweep_manifest": aggregate_manifest_path,
            "processed_seed_sweep_summary": processed_summary_path,
        },
        "child_runs": child_runs,
    }
