"""Workflow package for the TIMs frontier study.

This package maps the accepted specification/config/result schemas into a
concrete source layout. Legacy code remains in place and is consumed through
adapters at the workflow boundaries.
"""

from .config.resolver import resolve_study_config
from .orchestration.fixed_study import resolve_fixed_search_params, run_fixed_parameter_study
from .orchestration.runner import finalize_run, initialize_run, persist_frontier_outputs, persist_trial_outputs, prepare_run
from .orchestration.run_manifest import create_run_manifest
from .construction.builders import TrialBuildSpec, build_trial_spec
from .evaluation.tims import evaluate_trial_tims
from .analysis.frontier import extract_frontier_points

__all__ = [
    "TrialBuildSpec",
    "build_trial_spec",
    "create_run_manifest",
    "evaluate_trial_tims",
    "extract_frontier_points",
    "finalize_run",
    "initialize_run",
    "persist_frontier_outputs",
    "persist_trial_outputs",
    "prepare_run",
    "resolve_fixed_search_params",
    "resolve_study_config",
    "run_fixed_parameter_study",
]
