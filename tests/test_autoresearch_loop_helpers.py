from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in [REPO_ROOT / "src"]:
    entry_text = str(entry)
    if entry_text not in sys.path:
        sys.path.insert(0, entry_text)

from tims_frontier.autoresearch.git_ops import current_commit, hard_reset_to, validate_train_only_changes


def _run(repo_root: Path, *args: str) -> None:
    subprocess.run(args, cwd=repo_root, check=True, capture_output=True, text=True)


class LoopHelperTests(unittest.TestCase):
    def test_allowlist_rejects_non_train_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir)
            _run(repo, "git", "init")
            _run(repo, "git", "config", "user.email", "test@example.com")
            _run(repo, "git", "config", "user.name", "Test User")
            (repo / "train.py").write_text("CURRENT_PROPOSAL = {}\n", encoding="utf-8")
            (repo / "notes.txt").write_text("baseline\n", encoding="utf-8")
            _run(repo, "git", "add", ".")
            _run(repo, "git", "commit", "-m", "baseline")

            (repo / "notes.txt").write_text("edited\n", encoding="utf-8")
            disallowed = validate_train_only_changes(repo, editable_paths=["train.py"])
            self.assertEqual(disallowed, ["notes.txt"])

    def test_hard_reset_restores_previous_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir)
            _run(repo, "git", "init")
            _run(repo, "git", "config", "user.email", "test@example.com")
            _run(repo, "git", "config", "user.name", "Test User")
            train_path = repo / "train.py"
            train_path.write_text("CURRENT_PROPOSAL = {'proposal_id': 'p1'}\n", encoding="utf-8")
            _run(repo, "git", "add", "train.py")
            _run(repo, "git", "commit", "-m", "first")
            first_commit = current_commit(repo)

            train_path.write_text("CURRENT_PROPOSAL = {'proposal_id': 'p2'}\n", encoding="utf-8")
            _run(repo, "git", "add", "train.py")
            _run(repo, "git", "commit", "-m", "second")
            self.assertNotEqual(current_commit(repo), first_commit)

            hard_reset_to(repo, first_commit)
            self.assertEqual(current_commit(repo), first_commit)


if __name__ == "__main__":
    unittest.main()
