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

from tims_frontier.autoresearch.agent import _decode_json_like, replace_current_proposal_in_train, resolve_llm_settings


class AgentHelperTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
