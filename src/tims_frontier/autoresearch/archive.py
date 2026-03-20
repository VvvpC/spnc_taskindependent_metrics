from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import REPO_ROOT, read_json_if_exists, utc_now_iso, write_json
from .config import build_run_dir, timestamp_run_id


def _round_id(index: int) -> str:
    return f"round_{index:04d}"


class AutoResearchArchive:
    """Persistent archive for run-level state, round artifacts, and summaries."""

    def __init__(self, runtime_config: Mapping[str, Any], run_dir: str) -> None:
        self.runtime_config = dict(runtime_config)
        self.run_dir = Path(run_dir)
        self.rounds_dir = self.run_dir / "rounds"
        self.state_path = self.run_dir / "run_state.json"
        self.config_snapshot_path = self.run_dir / "run_config.json"
        self.baseline_path = self.run_dir / "baseline.json"
        self.lineage_path = self.run_dir / "lineage_summary.json"

    @classmethod
    def create(
        cls,
        runtime_config: Mapping[str, Any],
        *,
        branch_name: str,
        run_id: str | None = None,
    ) -> tuple["AutoResearchArchive", dict[str, Any]]:
        resolved_run_id = run_id or f"run_{timestamp_run_id()}"
        run_dir = build_run_dir(runtime_config, run_id=resolved_run_id)
        archive = cls(runtime_config, run_dir)
        archive.rounds_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "run_id": resolved_run_id,
            "repo_root": REPO_ROOT.as_posix(),
            "branch_name": branch_name,
            "created_at_utc": utc_now_iso(),
            "updated_at_utc": utc_now_iso(),
            "round_index": 0,
            "current_commit": None,
            "best_commit": None,
            "best_score": None,
            "best_proposal_id": None,
            "last_attempted_proposal_id": None,
            "last_attempted_proposal_path": None,
            "baseline_round_id": None,
            "baseline_hv": None,
            "baseline_points": [],
            "archive_points": [],
            "recent_failures": [],
            "history": [],
            "active": True,
        }
        archive.save_config_snapshot(runtime_config)
        archive.save_state(state)
        archive.export_lineage_summary(state)
        archive.set_active_pointer()
        return archive, state

    @classmethod
    def from_active(cls, runtime_config: Mapping[str, Any]) -> tuple["AutoResearchArchive", dict[str, Any]]:
        active_pointer = Path(str(runtime_config["storage"]["active_run_pointer"]))
        pointer_payload = read_json_if_exists(active_pointer)
        if pointer_payload is None:
            raise FileNotFoundError(f"No active autoresearch run pointer found at {active_pointer}")
        run_dir = str(pointer_payload["run_dir"])
        archive = cls(runtime_config, run_dir)
        state = archive.load_state()
        return archive, state

    @classmethod
    def from_or_create_manual(cls, runtime_config: Mapping[str, Any], *, branch_name: str) -> tuple["AutoResearchArchive", dict[str, Any]]:
        try:
            return cls.from_active(runtime_config)
        except FileNotFoundError:
            return cls.create(runtime_config, branch_name=branch_name, run_id=f"manual_{timestamp_run_id()}")

    @classmethod
    def from_run_dir(cls, runtime_config: Mapping[str, Any], run_dir: str) -> tuple["AutoResearchArchive", dict[str, Any]]:
        archive = cls(runtime_config, run_dir)
        state = archive.load_state()
        return archive, state

    def set_active_pointer(self) -> str:
        payload = {"run_dir": self.run_dir.as_posix(), "updated_at_utc": utc_now_iso()}
        return write_json(self.runtime_config["storage"]["active_run_pointer"], payload)

    def save_config_snapshot(self, runtime_config: Mapping[str, Any]) -> str:
        return write_json(self.config_snapshot_path, dict(runtime_config))

    def load_state(self) -> dict[str, Any]:
        payload = read_json_if_exists(self.state_path)
        if payload is None:
            raise FileNotFoundError(f"Run state not found: {self.state_path}")
        return payload

    def save_state(self, state: Mapping[str, Any]) -> str:
        mutable = dict(state)
        mutable["updated_at_utc"] = utc_now_iso()
        return write_json(self.state_path, mutable)

    def round_dir(self, round_index: int) -> Path:
        directory = self.rounds_dir / _round_id(round_index)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_round_json(self, round_index: int, filename: str, payload: Any) -> str:
        return write_json(self.round_dir(round_index) / filename, payload)

    def write_round_text(self, round_index: int, filename: str, text: str) -> str:
        output_path = self.round_dir(round_index) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path.as_posix()

    def save_baseline(self, score_payload: Mapping[str, Any]) -> str:
        return write_json(self.baseline_path, dict(score_payload))

    def export_lineage_summary(self, state: Mapping[str, Any]) -> str:
        summary = {
            "run_id": state["run_id"],
            "branch_name": state["branch_name"],
            "round_index": state["round_index"],
            "best_commit": state["best_commit"],
            "best_score": state["best_score"],
            "best_proposal_id": state["best_proposal_id"],
            "last_attempted_proposal_id": state["last_attempted_proposal_id"],
            "baseline_round_id": state["baseline_round_id"],
            "baseline_hv": state["baseline_hv"],
            "recent_failures": list(state["recent_failures"]),
            "history": list(state["history"]),
        }
        return write_json(self.lineage_path, summary)
