from __future__ import annotations

from typing import Any, Mapping


def build_next_context_summary(
    state: Mapping[str, Any],
    round_record: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact next-round context for the AI that edits train.py."""

    recent_rounds = list(state.get("history", []))[-int(runtime_config["summary"]["recent_rounds"]) :]
    return {
        "current_proposal": {
            "proposal_id": round_record.get("proposal_id"),
            "parent_proposal_id": round_record.get("parent_proposal_id"),
            "edit_type": round_record.get("edit_type"),
            "primary_edit": round_record.get("primary_edit"),
            "score_s_abs": round_record.get("score_s_abs"),
            "branch_decision": round_record.get("branch_decision"),
        },
        "best_proposal": {
            "proposal_id": state.get("best_proposal_id"),
            "best_score": state.get("best_score"),
            "best_commit": state.get("best_commit"),
        },
        "baseline": {
            "baseline_round_id": state.get("baseline_round_id"),
            "baseline_hv": state.get("baseline_hv"),
            "baseline_point_count": len(state.get("baseline_points", [])),
        },
        "recent_score_trend": [
            {
                "round_index": item.get("round_index"),
                "proposal_id": item.get("proposal_id"),
                "score_s_abs": item.get("score_s_abs"),
                "status": item.get("status"),
                "branch_decision": item.get("branch_decision"),
            }
            for item in recent_rounds
        ],
        "latest_frontier_expansion": round_record.get("audit_metrics", {}).get("frontier_external_span"),
        "recent_failures": list(state.get("recent_failures", []))[-3:],
        "reminder": "Next edit must be a minimal semantic change relative to the last attempted proposal. Only edit train.py.",
    }
