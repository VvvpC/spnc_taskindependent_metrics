from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping

from tims_frontier.storage.paths import build_run_paths


RESULT_SCHEMA_PATH = "reports/schemas/tims_frontier_result_schema.md"
RESULT_SCHEMA_VERSION = "1.0"
RESOLVED_CONFIG_SCHEMA_VERSION = "1.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_run_id(created_at_utc: str) -> str:
    stamp = created_at_utc.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    return f"run_{stamp}"


def _config_digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def resolve_study_config(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Resolve a study config into an immutable run-level config snapshot."""

    raw = deepcopy(dict(study_config))
    created = created_at_utc or _utc_now_iso()
    run_name = run_id or _timestamp_run_id(created)

    study = deepcopy(raw["study"])
    comparison = deepcopy(raw["comparison"])
    reservoir = deepcopy(raw["reservoir"])
    families = deepcopy(raw["families"])
    evaluation = deepcopy(raw["evaluation"])
    exploration = deepcopy(raw["exploration"])
    execution = deepcopy(raw["execution"])
    storage = deepcopy(raw["storage"])
    analysis = deepcopy(raw["analysis"])
    reporting = deepcopy(raw["reporting"])

    primary_endpoint = deepcopy(comparison["primary_endpoint"])
    search_space = deepcopy(exploration["shared_non_geometric_search"])
    nvirt = int(search_space["Nvirt"]["value"])

    run_paths = build_run_paths(
        study_id=study["study_id"],
        run_id=run_name,
        raw_root=storage["raw_dir"],
        processed_root=storage["processed_dir"],
    )

    reservoir.setdefault("bindings", {})
    reservoir["bindings"]["reference_beta_param"] = reservoir["parameter_binding"]["reference_beta_param"]
    reservoir["shared_defaults"]["network"]["Nvirt"] = nvirt
    reservoir.pop("parameter_binding", None)

    hetero = families["heterogeneous"]["construction"]["morphology"]
    morphology_seed_cfg = hetero["morphology_seed"]
    morphology_seed_value = morphology_seed_cfg.get("value")
    families["heterogeneous"]["resolved_construction"] = {
        "geometry_mode": families["heterogeneous"]["construction"]["geometry_mode"],
        "morphology": {
            "scheme": hetero["scheme"]["value"],
            "reference_beta_param": "beta_prime",
            "n_instances": int(hetero["n_instances"]["value"]),
            "beta_spread": float(hetero["beta_spread"]["value"]),
            "beta_sampling_rule": hetero["beta_sampling_rule"],
            "clip_beta_to_positive": bool(hetero["clip_beta_to_positive"]),
            "weights_mode": hetero["weights_mode"]["value"],
            "morphology_seed_mode": morphology_seed_cfg["mode"],
            "morphology_seed_value": int(morphology_seed_value) if morphology_seed_value is not None else None,
        },
    }
    families["heterogeneous"].pop("construction", None)
    families["uniform"]["resolved_construction"] = {
        "geometry_mode": families["uniform"]["construction"]["geometry_mode"],
        "parameters": families["uniform"]["construction"]["parameters"],
    }
    families["uniform"].pop("construction", None)

    exploration["search_space"] = search_space
    exploration["optuna"]["objective_metrics"] = list(primary_endpoint["frontier_objectives"])
    exploration["optuna"]["objective_directions"] = deepcopy(primary_endpoint["directions"])
    exploration["optuna"].pop("objective_ref", None)
    exploration.pop("shared_non_geometric_search", None)

    analysis["frontier"]["metrics"] = list(primary_endpoint["frontier_objectives"])
    analysis["frontier"]["directions"] = deepcopy(primary_endpoint["directions"])
    analysis["frontier"].pop("endpoint_ref", None)

    kr_gr = evaluation["tims"]["kr_gr"]
    kr_gr["n_readouts_from"] = "Nvirt"
    kr_gr["n_readouts_value"] = nvirt
    kr_gr.pop("n_readouts", None)

    storage["paths"] = run_paths.as_dict()

    return {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "source_config_schema_version": raw["schema_version"],
        "source_config_path": source_config_path,
        "specification_ref": {
            "spec_id": raw["specification_ref"]["spec_id"],
            "spec_path": raw["specification_ref"]["spec_path"],
        },
        "result_schema_ref": {
            "schema_id": "tims_frontier_result_schema",
            "schema_version": RESULT_SCHEMA_VERSION,
            "schema_path": RESULT_SCHEMA_PATH,
        },
        "run": {
            "run_id": run_name,
            "created_at_utc": created,
            "config_digest": _config_digest(raw),
            "mode": "study_run",
            "status_at_creation": "created",
        },
        "study": study,
        "comparison": comparison,
        "reservoir": reservoir,
        "families": families,
        "evaluation": evaluation,
        "exploration": exploration,
        "execution": execution,
        "storage": storage,
        "analysis": analysis,
        "reporting": reporting,
        "source_trace": {
            "exploration.optuna.objectives_resolved_from": "comparison.primary_endpoint",
            "analysis.frontier_resolved_from": "comparison.primary_endpoint",
            "reservoir.shared_defaults.network.Nvirt_resolved_from": "exploration.search_space.Nvirt",
            "families.heterogeneous.resolved_construction.morphology.reference_beta_param_resolved_from": (
                "exploration.search_space.beta_prime"
            ),
        },
    }
