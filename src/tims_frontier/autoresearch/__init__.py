"""Autoresearch-style fixed runtime for single-file heterogeneous family search."""

__all__ = [
    "AutoResearchArchive",
    "DEFAULT_CONFIG_PATH",
    "LLMBackendError",
    "ProposalValidationError",
    "SUMMARY_PREFIX",
    "ai_loop",
    "ai_step",
    "compile_proposal",
    "compute_frontier",
    "compute_hypervolume",
    "export_lineage",
    "init_run",
    "load_autoresearch_config",
    "run_loop",
    "run_step",
    "run_train_file",
    "sample_family",
    "score_family",
    "show_best",
    "validate_proposal",
]


def __getattr__(name: str):
    if name in {"AutoResearchArchive"}:
        from .archive import AutoResearchArchive

        return AutoResearchArchive
    if name in {"LLMBackendError", "ai_loop", "ai_step"}:
        from .agent import LLMBackendError, ai_loop, ai_step

        return {
            "LLMBackendError": LLMBackendError,
            "ai_loop": ai_loop,
            "ai_step": ai_step,
        }[name]
    if name in {"DEFAULT_CONFIG_PATH", "load_autoresearch_config"}:
        from .config import DEFAULT_CONFIG_PATH, load_autoresearch_config

        return {"DEFAULT_CONFIG_PATH": DEFAULT_CONFIG_PATH, "load_autoresearch_config": load_autoresearch_config}[name]
    if name in {"compile_proposal"}:
        from .compiler import compile_proposal

        return compile_proposal
    if name in {"export_lineage", "init_run", "run_loop", "run_step", "show_best"}:
        from .loop import export_lineage, init_run, run_loop, run_step, show_best

        return {
            "export_lineage": export_lineage,
            "init_run": init_run,
            "run_loop": run_loop,
            "run_step": run_step,
            "show_best": show_best,
        }[name]
    if name in {"SUMMARY_PREFIX", "run_train_file"}:
        from .runtime import SUMMARY_PREFIX, run_train_file

        return {"SUMMARY_PREFIX": SUMMARY_PREFIX, "run_train_file": run_train_file}[name]
    if name in {"sample_family"}:
        from .sampler import sample_family

        return sample_family
    if name in {"compute_frontier", "compute_hypervolume", "score_family"}:
        from .scoring import compute_frontier, compute_hypervolume, score_family

        return {
            "compute_frontier": compute_frontier,
            "compute_hypervolume": compute_hypervolume,
            "score_family": score_family,
        }[name]
    if name in {"ProposalValidationError", "validate_proposal"}:
        from .validator import ProposalValidationError, validate_proposal

        return {"ProposalValidationError": ProposalValidationError, "validate_proposal": validate_proposal}[name]
    raise AttributeError(name)
