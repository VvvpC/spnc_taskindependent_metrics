from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in [REPO_ROOT / "src"]:
    entry_text = str(entry)
    if entry_text not in sys.path:
        sys.path.insert(0, entry_text)

from tims_frontier.autoresearch.scoring import compute_frontier, compute_hypervolume, score_family


RUNTIME_CONFIG = {
    "scoring": {
        "reference_point": {"MC": 0.0, "CQ": 0.0},
        "normalization": {
            "mode": "fixed_bounds",
            "bounds": {
                "MC": {"min": 0.0, "max": 10.0},
                "CQ": {"min": 0.0, "max": 100.0},
            },
        },
    }
}


class ScoringTests(unittest.TestCase):
    def test_hypervolume_matches_hand_calculation(self) -> None:
        frontier = compute_frontier(
            [
                {"MC": 0.2, "CQ": 0.8},
                {"MC": 0.5, "CQ": 0.6},
                {"MC": 0.7, "CQ": 0.4},
            ]
        )
        hv = compute_hypervolume(frontier, {"MC": 0.0, "CQ": 0.0})
        self.assertAlmostEqual(hv, 0.42, places=6)

    def test_baseline_relative_score_uses_fixed_baseline(self) -> None:
        baseline = [{"MC": 0.2, "CQ": 0.4}, {"MC": 0.3, "CQ": 0.3}]
        family_results = [
            {"MC": 4.0, "CQ": 50.0},
            {"MC": 2.0, "CQ": 40.0},
        ]
        bundle = score_family(family_results, RUNTIME_CONFIG, baseline_points=baseline)
        self.assertAlmostEqual(bundle.baseline_hv, 0.11, places=6)
        self.assertAlmostEqual(bundle.combined_hv, 0.20, places=6)
        self.assertAlmostEqual(bundle.score_s_abs, 0.09, places=6)


if __name__ == "__main__":
    unittest.main()
