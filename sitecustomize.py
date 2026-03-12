from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "src" / "Project",
    REPO_ROOT / "src" / "Morphology_Research",
    REPO_ROOT / "src" / "Optuna_TaskIndependent_Metrics",
    REPO_ROOT / "src" / "ParetoFront_CQandMC",
    REPO_ROOT / "src" / "Plot_Functions",
    REPO_ROOT / "src" / "Test_Temporary",
]

for source_dir in SOURCE_DIRS:
    source_dir_str = str(source_dir)
    if source_dir.exists() and source_dir_str not in sys.path:
        sys.path.insert(0, source_dir_str)
