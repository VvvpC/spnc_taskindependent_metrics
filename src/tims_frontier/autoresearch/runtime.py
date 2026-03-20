from __future__ import annotations

import os
import traceback
from typing import Any, Mapping

from .adapter import evaluate_sampled_instances
from .archive import AutoResearchArchive
from .common import REPO_ROOT, utc_now_iso
from .config import load_autoresearch_config
from .models import Proposal
from .sampler import sample_family
from .scoring import score_family
from .summary import build_next_context_summary
from .validator import validate_proposal


SUMMARY_PREFIX = "AUTORESEARCH_SUMMARY "


def _branch_name_fallback() -> str:
    try:
        from .git_ops import current_branch

        return current_branch(REPO_ROOT)
    except Exception:
        return "manual"


def _proposal_record_from_validated(compiled, score_bundle, round_index: int) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "proposal_id": compiled.proposal_id,
        "parent_proposal_id": compiled.parent_proposal_id,
        "edit_type": compiled.edit_type,
        "primary_edit": dict(compiled.primary_edit),
        "family_type": compiled.family_definition.family_type,
        "score_s_abs": score_bundle.score_s_abs,
        "status": "completed",
        "branch_decision": "pending",
        "audit_metrics": dict(score_bundle.audit_metrics),
    }


def run_train_file(current_proposal: Mapping[str, Any], *, config_path: str | None = None) -> dict[str, Any]:
    """Run the fixed autoresearch runtime against the editable CURRENT_PROPOSAL."""

    runtime_config = load_autoresearch_config(config_path)
    active_run_dir = os.environ.get("AUTORESEARCH_ACTIVE_RUN_DIR")
    if active_run_dir:
        archive, state = AutoResearchArchive.from_run_dir(runtime_config, active_run_dir)
    else:
        archive, state = AutoResearchArchive.from_or_create_manual(
            runtime_config,
            branch_name=_branch_name_fallback(),
        )
    round_index = int(state["round_index"]) + 1
    raw_proposal = Proposal.from_mapping(dict(current_proposal))
    archive.write_round_json(round_index, "proposal_raw.json", raw_proposal.to_dict())

    parent_proposal = None
    if int(state["round_index"]) > 0:
        previous_round_record = state["history"][-1] if state.get("history") else None
        if previous_round_record is not None:
            last_payload_path = previous_round_record.get("proposal_raw_path")
            if last_payload_path:
                from .common import read_json_if_exists

                parent_payload = read_json_if_exists(last_payload_path)
                if parent_payload is not None:
                    parent_proposal = Proposal.from_mapping(parent_payload)

    state["last_attempted_proposal_id"] = raw_proposal.proposal_id
    state["last_attempted_proposal_path"] = archive.round_dir(round_index).joinpath("proposal_raw.json").as_posix()
    archive.save_state(state)

    try:
        compiled = validate_proposal(raw_proposal, runtime_config, parent_proposal=parent_proposal)
        archive.write_round_json(round_index, "proposal_validated.json", compiled.to_dict())
        archive.write_round_json(round_index, "compiled_family_spec.json", compiled.to_dict())
        sampled_instances = sample_family(compiled)
        archive.write_round_json(
            round_index,
            "sampled_instances.json",
            [instance.to_dict() for instance in sampled_instances],
        )
        evaluation_results = evaluate_sampled_instances(
            sampled_instances,
            runtime_config,
            family_seed=int(compiled.sampling_plan.seed),
        )
        archive.write_round_json(round_index, "raw_evaluator_outputs.json", evaluation_results)
        score_bundle = score_family(
            evaluation_results,
            runtime_config,
            baseline_points=list(state.get("baseline_points", [])) or None,
            archive_points=list(state.get("archive_points", [])) or None,
        )
        archive.write_round_json(round_index, "family_frontier.json", score_bundle.family_frontier)
        archive.write_round_json(round_index, "scores.json", score_bundle.to_dict())

        if state.get("baseline_hv") is None:
            state["baseline_round_id"] = f"round_{round_index:04d}"
            state["baseline_hv"] = score_bundle.baseline_hv
            state["baseline_points"] = score_bundle.baseline_points
            archive.save_baseline(score_bundle.to_dict())

        state["archive_points"] = score_bundle.combined_frontier
        state["round_index"] = round_index
        round_record = _proposal_record_from_validated(compiled, score_bundle, round_index)
        round_record["proposal_raw_path"] = archive.round_dir(round_index).joinpath("proposal_raw.json").as_posix()
        round_record["round_dir"] = archive.round_dir(round_index).as_posix()
        round_record["completed_at_utc"] = utc_now_iso()
        state.setdefault("history", []).append(round_record)
        next_context = build_next_context_summary(state, round_record, runtime_config)
        archive.write_round_json(round_index, "next_context_summary.json", next_context)
        archive.save_state(state)
        archive.export_lineage_summary(state)
        summary = {
            "proposal_id": compiled.proposal_id,
            "status": "completed",
            "baseline_hv": score_bundle.baseline_hv,
            "family_hv": score_bundle.family_hv,
            "score_s_abs": score_bundle.score_s_abs,
            "frontier_size": len(score_bundle.family_frontier),
            "round_dir": archive.round_dir(round_index).as_posix(),
            "round_index": round_index,
        }
        print(SUMMARY_PREFIX + __import__("json").dumps(summary, ensure_ascii=False, sort_keys=True))
        return summary
    except Exception as exc:
        failure_payload = {
            "proposal_id": raw_proposal.proposal_id,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        archive.write_round_json(round_index, "failure.json", failure_payload)
        state["round_index"] = round_index
        state.setdefault("recent_failures", []).append(
            {
                "round_index": round_index,
                "proposal_id": raw_proposal.proposal_id,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "round_dir": archive.round_dir(round_index).as_posix(),
            }
        )
        state["recent_failures"] = state["recent_failures"][-5:]
        state.setdefault("history", []).append(
            {
                "round_index": round_index,
                "proposal_id": raw_proposal.proposal_id,
                "parent_proposal_id": raw_proposal.parent_proposal_id,
                "edit_type": raw_proposal.edit_type,
                "primary_edit": dict(raw_proposal.primary_edit),
                "status": "failed",
                "branch_decision": "pending",
                "proposal_raw_path": archive.round_dir(round_index).joinpath("proposal_raw.json").as_posix(),
                "round_dir": archive.round_dir(round_index).as_posix(),
                "failed_at_utc": utc_now_iso(),
            }
        )
        archive.save_state(state)
        archive.export_lineage_summary(state)
        raise
