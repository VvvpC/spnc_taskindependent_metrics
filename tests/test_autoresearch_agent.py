from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in [REPO_ROOT / "src"]:
    entry_text = str(entry)
    if entry_text not in sys.path:
        sys.path.insert(0, entry_text)

from tims_frontier.autoresearch.agent import (
    _decode_json_like,
    _normalize_proposal_payload,
    _prevalidate_agent_proposal,
    replace_current_proposal_in_train,
    resolve_llm_settings,
)


class AgentHelperTests(unittest.TestCase):
    def _runtime_config(self) -> dict[str, object]:
        return {
            "constraints": {
                "beta_prime": {"min": 5.0, "max": 50.0},
                "theta": {"min": 0.01, "max": 0.8},
                "gamma": {"min": 0.001, "max": 0.3},
                "m0": {"min": 0.001, "max": 0.05},
            }
        }

    def test_decode_json_like_handles_fenced_wrapper(self) -> None:
        payload = _decode_json_like(
            """```json
            {
              "proposal": {
                "proposal_id": "proposal_0002",
                "parent_proposal_id": "proposal_0001",
                "edit_type": "scalar_tune",
                "primary_edit": {"target": "gamma", "before": 0.08, "after": 0.09},
                "rationale": "narrow change",
                "expected_effect": "small frontier shift",
                "family_definition": {
                  "family_type": "single_distribution",
                  "topology_name": "single_group",
                  "subgroups": [],
                  "distribution_rule": {"form": "random"},
                  "coupling_rule": {"rule": "independent"},
                  "continuous_parameters": {}
                },
                "sampling_plan": {"sampler_name": "random", "n_samples": 2, "seed": 1}
              }
            }
            ```"""
        )
        self.assertEqual(payload["proposal_id"], "proposal_0002")
        self.assertEqual(payload["parent_proposal_id"], "proposal_0001")

    def test_replace_current_proposal_rewrites_only_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.py"
            train_path.write_text(
                "from tims_frontier.autoresearch import run_train_file\n\n"
                "CURRENT_PROPOSAL = {'proposal_id': 'proposal_0001', 'edit_type': 'initial_seed'}\n\n"
                "def main():\n"
                "    run_train_file(CURRENT_PROPOSAL)\n",
                encoding="utf-8",
            )
            changed = replace_current_proposal_in_train(
                train_path,
                {
                    "proposal_id": "proposal_0002",
                    "parent_proposal_id": "proposal_0001",
                    "edit_type": "scalar_tune",
                    "primary_edit": {"target": "gamma", "before": 0.08, "after": 0.09},
                    "rationale": "narrow change",
                    "expected_effect": "small frontier shift",
                    "family_definition": {
                        "family_type": "single_distribution",
                        "topology_name": "single_group",
                        "subgroups": [],
                        "distribution_rule": {"form": "random"},
                        "coupling_rule": {"rule": "independent"},
                        "continuous_parameters": {},
                    },
                    "sampling_plan": {"sampler_name": "random", "n_samples": 2, "seed": 1},
                },
            )
            updated = train_path.read_text(encoding="utf-8")
            self.assertTrue(changed)
            self.assertIn("'proposal_id': 'proposal_0002'", updated)
            self.assertIn("def main()", updated)
            self.assertIn("run_train_file(CURRENT_PROPOSAL)", updated)

    def test_resolve_llm_settings_uses_moonshot_fallbacks(self) -> None:
        settings = resolve_llm_settings(
            {
                "llm": {
                    "default_base_url": "https://api.moonshot.ai/v1",
                    "default_model": "kimi-k2.5",
                }
            },
            env={"MOONSHOT_API_KEY": "secret-key"},
        )
        self.assertEqual(settings.base_url, "https://api.moonshot.ai/v1")
        self.assertEqual(settings.model, "kimi-k2.5")
        self.assertEqual(settings.api_key, "secret-key")

    def test_resolve_llm_settings_reads_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / "kimi.env"
            env_path.write_text(
                "MOONSHOT_API_KEY=file-key\n"
                "AUTORESEARCH_LLM_BASE_URL=https://api.moonshot.ai/v1\n"
                "AUTORESEARCH_LLM_MODEL=kimi-k2.5\n",
                encoding="utf-8",
            )
            settings = resolve_llm_settings(
                {
                    "llm": {
                        "env_file": env_path.as_posix(),
                        "default_base_url": "https://fallback.invalid/v1",
                        "default_model": "fallback-model",
                    }
                },
                env={},
            )
            self.assertEqual(settings.api_key, "file-key")
            self.assertEqual(settings.base_url, "https://api.moonshot.ai/v1")
            self.assertEqual(settings.model, "kimi-k2.5")

    def test_normalize_proposal_payload_maps_edit_type_alias(self) -> None:
        payload = _normalize_proposal_payload(
            {
                "proposal_id": "proposal_0002",
                "parent_proposal_id": "proposal_0001",
                "edit_type": "parameter_range_narrowing",
                "primary_edit": {"target": "continuous_parameters.gamma.high", "before": 0.08, "after": 0.07},
                "rationale": "narrow the gamma range",
                "expected_effect": "reduce dispersion",
                "family_definition": {
                    "family_type": "single_distribution",
                    "topology_name": "single_group",
                    "subgroups": [
                        {
                            "name": "core",
                            "role": "core",
                            "count": {"kind": "uniform_int", "low": 3, "high": 5},
                            "offset_center": {"kind": "fixed", "value": 0.0},
                            "spread": {"kind": "uniform_float", "low": 2.0, "high": 3.5},
                        }
                    ],
                    "distribution_rule": {"form": "random"},
                    "coupling_rule": {"rule": "independent"},
                    "continuous_parameters": {
                        "beta_prime": {"kind": "uniform_float", "low": 28.0, "high": 32.0},
                        "theta": {"kind": "uniform_float", "low": 0.16, "high": 0.24},
                        "gamma": {"kind": "uniform_float", "low": 0.04, "high": 0.07},
                        "m0": {"kind": "uniform_float", "low": 0.008, "high": 0.015},
                    },
                },
                "sampling_plan": {"sampler_name": "latin_hypercube", "n_samples": 3, "seed": 1234},
            }
        )
        self.assertEqual(payload["edit_type"], "scalar_tune")

    def test_prevalidate_agent_proposal_accepts_normalized_scalar_tune(self) -> None:
        state = {
            "last_attempted_proposal_path": None,
        }
        initial = {
            "proposal_id": "proposal_0001",
            "parent_proposal_id": None,
            "edit_type": "initial_seed",
            "primary_edit": {"target": "initialization", "before": None, "after": "single_distribution_random_independent"},
            "rationale": "baseline",
            "expected_effect": "baseline",
            "family_definition": {
                "family_type": "single_distribution",
                "topology_name": "single_group",
                "subgroups": [
                    {
                        "name": "core",
                        "role": "core",
                        "count": {"kind": "uniform_int", "low": 3, "high": 5},
                        "offset_center": {"kind": "fixed", "value": 0.0},
                        "spread": {"kind": "uniform_float", "low": 2.0, "high": 3.5},
                    }
                ],
                "distribution_rule": {"form": "random"},
                "coupling_rule": {"rule": "independent"},
                "continuous_parameters": {
                    "beta_prime": {"kind": "uniform_float", "low": 28.0, "high": 32.0},
                    "theta": {"kind": "uniform_float", "low": 0.16, "high": 0.24},
                    "gamma": {"kind": "uniform_float", "low": 0.04, "high": 0.08},
                    "m0": {"kind": "uniform_float", "low": 0.008, "high": 0.015},
                },
            },
            "sampling_plan": {"sampler_name": "latin_hypercube", "n_samples": 3, "seed": 1234},
        }
        proposal = _prevalidate_agent_proposal(initial, self._runtime_config(), state)
        self.assertEqual(proposal.edit_type, "initial_seed")


if __name__ == "__main__":
    unittest.main()
