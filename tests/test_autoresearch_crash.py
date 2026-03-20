from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in [REPO_ROOT / "src"]:
    entry_text = str(entry)
    if entry_text not in sys.path:
        sys.path.insert(0, entry_text)

from tims_frontier.autoresearch.loop import _capture_crash_artifacts
from tims_frontier.autoresearch.models import Proposal


class CrashInjectionTests(unittest.TestCase):
    def test_traceback_tail_is_captured_and_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            log_path = root / "run.log"
            log_path.write_text("\n".join(f"line {index}" for index in range(150)) + "\n", encoding="utf-8")
            proposal = Proposal.from_mapping(
                {
                    "proposal_id": "proposal_crash",
                    "parent_proposal_id": None,
                    "edit_type": "initial_seed",
                    "primary_edit": {"target": "initialization", "before": None, "after": "seed"},
                    "rationale": "seed",
                    "expected_effect": "baseline",
                    "family_definition": {
                        "family_type": "single_distribution",
                        "topology_name": "single_group",
                        "subgroups": [],
                        "distribution_rule": {"form": "random"},
                        "coupling_rule": {"rule": "independent"},
                        "continuous_parameters": {},
                    },
                    "sampling_plan": {"sampler_name": "random", "n_samples": 1, "seed": 1},
                }
            )
            context = _capture_crash_artifacts(
                {"runtime": {"crash_tail_lines": 100}},
                round_dir=root,
                log_path=log_path.as_posix(),
                proposal=proposal,
                failure_attempts=2,
            )
            self.assertEqual(len(context["traceback_tail"]), 100)
            self.assertEqual(context["traceback_tail"][0], "line 50")
            self.assertTrue((root / "crash_tail.txt").exists())
            self.assertTrue((root / "crash_context.json").exists())


if __name__ == "__main__":
    unittest.main()
