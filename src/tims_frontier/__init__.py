"""Workflow package for the TIMs frontier study.

This package now uses lazy imports so lightweight subpackages such as
`tims_frontier.autoresearch` can be imported without pulling in every optional
runtime dependency from the legacy workflow stack.
"""

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


def __getattr__(name: str):
    if name == "resolve_study_config":
        from .config.resolver import resolve_study_config

        return resolve_study_config
    if name in {"resolve_fixed_search_params", "run_fixed_parameter_study"}:
        from .orchestration.fixed_study import resolve_fixed_search_params, run_fixed_parameter_study

        return {"resolve_fixed_search_params": resolve_fixed_search_params, "run_fixed_parameter_study": run_fixed_parameter_study}[name]
    if name in {"finalize_run", "initialize_run", "persist_frontier_outputs", "persist_trial_outputs", "prepare_run"}:
        from .orchestration.runner import (
            finalize_run,
            initialize_run,
            persist_frontier_outputs,
            persist_trial_outputs,
            prepare_run,
        )

        return {
            "finalize_run": finalize_run,
            "initialize_run": initialize_run,
            "persist_frontier_outputs": persist_frontier_outputs,
            "persist_trial_outputs": persist_trial_outputs,
            "prepare_run": prepare_run,
        }[name]
    if name == "create_run_manifest":
        from .orchestration.run_manifest import create_run_manifest

        return create_run_manifest
    if name in {"TrialBuildSpec", "build_trial_spec"}:
        from .construction.builders import TrialBuildSpec, build_trial_spec

        return {"TrialBuildSpec": TrialBuildSpec, "build_trial_spec": build_trial_spec}[name]
    if name == "evaluate_trial_tims":
        from .evaluation.tims import evaluate_trial_tims

        return evaluate_trial_tims
    if name == "extract_frontier_points":
        from .analysis.frontier import extract_frontier_points

        return extract_frontier_points
    raise AttributeError(name)
