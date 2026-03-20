from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in [REPO_ROOT / "src"]:
    entry_text = str(entry)
    if entry_text not in sys.path:
        sys.path.insert(0, entry_text)

from tims_frontier.autoresearch.archive import AutoResearchArchive
from tims_frontier.autoresearch.config import load_autoresearch_config
from tims_frontier.autoresearch.runtime import run_train_file


def _write_mock_config(tmp_path: Path) -> Path:
    config = {
        "schema_version": "1.0",
        "study": {"study_id": "test"},
        "storage": {
            "run_root": str(tmp_path / "raw"),
            "processed_root": str(tmp_path / "processed"),
            "active_run_pointer": str(tmp_path / "raw" / "_active_run.json"),
        },
        "runtime": {"editable_paths": ["train.py"], "crash_tail_lines": 100},
        "summary": {"recent_rounds": 5},
        "constraints": {
            "beta_prime": {"min": 5.0, "max": 50.0},
            "theta": {"min": 0.01, "max": 0.8},
            "gamma": {"min": 0.001, "max": 0.3},
            "m0": {"min": 0.001, "max": 0.05},
        },
        "evaluator": {
            "mode": "mock",
            "nvirt": 10,
            "shared_defaults": {
                "physical": {"h": 0.4, "theta_H": 90, "k_s_0": 0, "phi": 45},
                "network": {"bias": True, "Nwarmup": 0},
                "simulation": {
                    "delay_feedback": 0,
                    "voltage_noise": False,
                    "johnson_noise": False,
                    "thermal_noise": False,
                },
            },
            "tims": {
                "mc": {"signal_len": 40, "delays": 10, "splits": [0.2, 0.6]},
                "kr_gr": {"n_wash": 5, "threshold": 0.001},
            },
        },
        "scoring": {
            "reference_point": {"MC": 0.0, "CQ": 0.0},
            "normalization": {
                "mode": "fixed_bounds",
                "bounds": {
                    "MC": {"min": 0.0, "max": 10.0},
                    "CQ": {"min": 0.0, "max": 100.0},
                },
            },
        },
    }
    path = tmp_path / "config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return path


def _base_proposal() -> dict:
    return {
        "proposal_id": "proposal_0001",
        "parent_proposal_id": None,
        "edit_type": "initial_seed",
        "primary_edit": {"target": "initialization", "before": None, "after": "single_distribution"},
        "rationale": "seed",
        "expected_effect": "baseline",
        "family_definition": {
            "family_type": "single_distribution",
            "topology_name": "single_group",
            "subgroups": [
                {
                    "name": "core",
                    "role": "core",
                    "count": {"kind": "fixed", "value": 3},
                    "offset_center": {"kind": "fixed", "value": 0.0},
                    "spread": {"kind": "fixed", "value": 2.5},
                }
            ],
            "distribution_rule": {"form": "random", "clip_beta_to_positive": True},
            "coupling_rule": {"rule": "independent"},
            "continuous_parameters": {
                "beta_prime": {"kind": "fixed", "value": 30.0},
                "theta": {"kind": "fixed", "value": 0.2},
                "gamma": {"kind": "fixed", "value": 0.05},
                "m0": {"kind": "fixed", "value": 0.01},
            },
        },
        "sampling_plan": {"sampler_name": "random", "n_samples": 2, "seed": 1234},
    }


class ArchiveRuntimeTests(unittest.TestCase):
    def test_archive_create_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _write_mock_config(Path(tmp_dir))
            runtime_config = load_autoresearch_config(config_path.as_posix())
            archive, state = AutoResearchArchive.create(runtime_config, branch_name="test", run_id="run_test")
            self.assertEqual(state["run_id"], "run_test")
            archive_2, state_2 = AutoResearchArchive.from_active(runtime_config)
            self.assertEqual(archive_2.run_dir, archive.run_dir)
            self.assertEqual(state_2["run_id"], "run_test")

    def test_baseline_freezes_after_first_successful_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _write_mock_config(Path(tmp_dir))
            runtime_config = load_autoresearch_config(config_path.as_posix())
            archive, _ = AutoResearchArchive.create(runtime_config, branch_name="test", run_id="run_test")
            previous_run_dir = os.environ.get("AUTORESEARCH_ACTIVE_RUN_DIR")
            try:
                os.environ["AUTORESEARCH_ACTIVE_RUN_DIR"] = archive.run_dir.as_posix()
                run_train_file(_base_proposal(), config_path=config_path.as_posix())
                state_after_first = archive.load_state()
                baseline_hv = state_after_first["baseline_hv"]

                second = _base_proposal()
                second["proposal_id"] = "proposal_0002"
                second["parent_proposal_id"] = "proposal_0001"
                second["edit_type"] = "scalar_tune"
                second["primary_edit"] = {
                    "target": "continuous_parameters.gamma.value",
                    "before": 0.05,
                    "after": 0.06,
                }
                second["family_definition"]["continuous_parameters"]["gamma"]["value"] = 0.06
                run_train_file(second, config_path=config_path.as_posix())

                state_after_second = archive.load_state()
                self.assertEqual(state_after_second["baseline_round_id"], "round_0001")
                self.assertAlmostEqual(float(state_after_second["baseline_hv"]), float(baseline_hv), places=10)
                self.assertEqual(len(state_after_second["history"]), 2)
            finally:
                if previous_run_dir is None:
                    os.environ.pop("AUTORESEARCH_ACTIVE_RUN_DIR", None)
                else:
                    os.environ["AUTORESEARCH_ACTIVE_RUN_DIR"] = previous_run_dir


if __name__ == "__main__":
    unittest.main()
