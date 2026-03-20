from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from .archive import AutoResearchArchive
from .common import REPO_ROOT, tail_lines, utc_now_iso, write_json
from .config import load_autoresearch_config
from .git_ops import add_and_commit, current_branch, current_commit, hard_reset_to, status_entries, validate_train_only_changes
from .models import Proposal
from .runtime import SUMMARY_PREFIX


RESULTS_TSV_FIELDNAMES = [
    "timestamp_utc",
    "round_index",
    "commit",
    "proposal_id",
    "score_s_abs",
    "baseline_hv",
    "family_hv",
    "status",
    "description",
]


def _load_train_module(train_path: Path):
    spec = importlib.util.spec_from_file_location("autoresearch_train_preview", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load train module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_current_proposal(train_path: str | Path) -> Proposal:
    module = _load_train_module(Path(train_path))
    payload = getattr(module, "CURRENT_PROPOSAL", None)
    if not isinstance(payload, Mapping):
        raise ValueError("train.py must define CURRENT_PROPOSAL as a mapping")
    return Proposal.from_mapping(payload)


def init_run(config_path: str | None = None) -> dict[str, Any]:
    runtime_config = load_autoresearch_config(config_path)
    archive, state = AutoResearchArchive.create(runtime_config, branch_name=current_branch(REPO_ROOT))
    _ensure_results_tsv(REPO_ROOT / "results.tsv")
    return {"run_id": state["run_id"], "run_dir": archive.run_dir.as_posix(), "branch_name": state["branch_name"]}


def _ensure_results_tsv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_TSV_FIELDNAMES, delimiter="\t")
        writer.writeheader()


def _append_results_row(row: Mapping[str, Any]) -> None:
    output_path = REPO_ROOT / "results.tsv"
    _ensure_results_tsv(output_path)
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_TSV_FIELDNAMES, delimiter="\t")
        writer.writerow({name: row.get(name) for name in RESULTS_TSV_FIELDNAMES})


def _run_train_process(runtime_config: Mapping[str, Any], run_dir: str) -> tuple[int, str]:
    log_path = REPO_ROOT / "run.log"
    env = os.environ.copy()
    env["AUTORESEARCH_CONFIG_PATH"] = str(runtime_config["storage"]["config_path"])
    env["AUTORESEARCH_ACTIVE_RUN_DIR"] = run_dir
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            [sys.executable, "train.py"],
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    return process.returncode, log_path.as_posix()


def _extract_summary_from_log(log_path: str | Path) -> dict[str, Any] | None:
    for line in reversed(Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.startswith(SUMMARY_PREFIX):
            continue
        return json.loads(line[len(SUMMARY_PREFIX) :])
    return None


def _capture_crash_artifacts(
    runtime_config: Mapping[str, Any],
    *,
    round_dir: Path,
    log_path: str,
    proposal: Proposal | None,
    failure_attempts: int,
) -> dict[str, Any]:
    tail_count = int(runtime_config["runtime"]["crash_tail_lines"])
    tail = tail_lines(log_path, n_lines=tail_count)
    crash_context = {
        "proposal_id": proposal.proposal_id if proposal is not None else None,
        "edit_type": proposal.edit_type if proposal is not None else None,
        "primary_edit": dict(proposal.primary_edit) if proposal is not None else None,
        "failure_attempts": failure_attempts,
        "traceback_tail": tail,
    }
    (round_dir / "crash_tail.txt").write_text("\n".join(tail) + ("\n" if tail else ""), encoding="utf-8")
    write_json(round_dir / "crash_context.json", crash_context)
    return crash_context


def _update_branch_decision(state: dict[str, Any], *, branch_decision: str, commit: str, score: float | None) -> None:
    state["current_commit"] = commit
    if not state.get("history"):
        return
    state["history"][-1]["branch_decision"] = branch_decision
    if branch_decision == "keep":
        state["best_commit"] = commit
        if score is not None:
            state["best_score"] = score
        state["best_proposal_id"] = state["history"][-1].get("proposal_id")


def run_step(config_path: str | None = None) -> dict[str, Any]:
    runtime_config = load_autoresearch_config(config_path)
    archive, state = AutoResearchArchive.from_active(runtime_config)
    editable_paths = list(runtime_config["runtime"]["editable_paths"])
    round_index = int(state["round_index"]) + 1
    commit_before = current_commit(REPO_ROOT)
    proposal_preview: Proposal | None = None
    commit_for_step = commit_before
    worktree_entries = status_entries(REPO_ROOT)

    if round_index == 1 and not worktree_entries:
        proposal_preview = load_current_proposal(REPO_ROOT / "train.py")
    else:
        disallowed = validate_train_only_changes(REPO_ROOT, editable_paths=editable_paths)
        if disallowed:
            raise ValueError(f"Only train.py may be modified before a step. Disallowed paths: {disallowed}")
        if not worktree_entries:
            raise ValueError("No train.py changes detected for this step")
        try:
            proposal_preview = load_current_proposal(REPO_ROOT / "train.py")
            commit_message = f"autoresearch step {round_index}: {proposal_preview.proposal_id}"
        except Exception:
            commit_message = f"autoresearch step {round_index}"
        commit_for_step = add_and_commit(REPO_ROOT, pathspecs=["train.py"], message=commit_message)

    return_code, log_path = _run_train_process(runtime_config, archive.run_dir.as_posix())
    summary = _extract_summary_from_log(log_path)
    round_dir = archive.round_dir(round_index)
    state = archive.load_state()

    if summary is None or return_code != 0:
        crash_context = _capture_crash_artifacts(
            runtime_config,
            round_dir=round_dir,
            log_path=log_path,
            proposal=proposal_preview,
            failure_attempts=len(state.get("recent_failures", [])),
        )
        _append_results_row(
            {
                "timestamp_utc": utc_now_iso(),
                "round_index": round_index,
                "commit": commit_for_step[:7],
                "proposal_id": crash_context.get("proposal_id"),
                "score_s_abs": 0.0,
                "baseline_hv": state.get("baseline_hv") or 0.0,
                "family_hv": 0.0,
                "status": "crash",
                "description": json.dumps(crash_context.get("primary_edit"), ensure_ascii=False),
            }
        )
        if state.get("best_commit") and commit_for_step != state.get("best_commit"):
            hard_reset_to(REPO_ROOT, str(state["best_commit"]))
        _update_branch_decision(state, branch_decision="crash", commit=current_commit(REPO_ROOT), score=None)
        archive.save_state(state)
        archive.export_lineage_summary(state)
        return {"status": "crash", "round_dir": round_dir.as_posix(), "log_path": log_path}

    score = float(summary["score_s_abs"])
    is_baseline_round = state.get("baseline_round_id") == f"round_{round_index:04d}"
    best_score = state.get("best_score")
    if best_score is None:
        best_score = None if not is_baseline_round else score
    keep = is_baseline_round or best_score is None or score > float(best_score)
    decision = "keep" if keep else "discard"
    if not keep and state.get("best_commit"):
        hard_reset_to(REPO_ROOT, str(state["best_commit"]))
    final_commit = current_commit(REPO_ROOT)
    _append_results_row(
        {
            "timestamp_utc": utc_now_iso(),
            "round_index": round_index,
            "commit": commit_for_step[:7],
            "proposal_id": summary["proposal_id"],
            "score_s_abs": summary["score_s_abs"],
            "baseline_hv": summary["baseline_hv"],
            "family_hv": summary["family_hv"],
            "status": decision,
            "description": summary["proposal_id"],
        }
    )
    _update_branch_decision(state, branch_decision=decision, commit=final_commit if keep else str(state["best_commit"] or final_commit), score=score if keep else state.get("best_score"))
    archive.save_state(state)
    archive.export_lineage_summary(state)
    return {**summary, "branch_decision": decision, "log_path": log_path}


def run_loop(config_path: str | None = None, *, iterations: int) -> list[dict[str, Any]]:
    results = []
    for _ in range(iterations):
        results.append(run_step(config_path))
    return results


def show_best(config_path: str | None = None) -> dict[str, Any]:
    runtime_config = load_autoresearch_config(config_path)
    archive, state = AutoResearchArchive.from_active(runtime_config)
    return {
        "run_id": state["run_id"],
        "best_commit": state.get("best_commit"),
        "best_score": state.get("best_score"),
        "best_proposal_id": state.get("best_proposal_id"),
        "baseline_round_id": state.get("baseline_round_id"),
        "lineage_summary": archive.lineage_path.as_posix(),
    }


def export_lineage(config_path: str | None = None) -> dict[str, Any]:
    runtime_config = load_autoresearch_config(config_path)
    archive, state = AutoResearchArchive.from_active(runtime_config)
    archive.export_lineage_summary(state)
    return {"lineage_summary": archive.lineage_path.as_posix(), "run_id": state["run_id"]}
