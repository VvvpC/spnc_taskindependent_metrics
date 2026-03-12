# TIMs Frontier Run Manifest Schema

- Document type: Run Manifest Schema
- Schema version: `1.0`
- Status: Accepted baseline
- Scope: Phase 1 TIMs frontier workflow
- Aligned specification: `reports/specifications/tims_frontier_specification.md`
- Aligned resolved config schema: `reports/schemas/tims_frontier_resolved_config_schema.md`
- Aligned result schema: `reports/schemas/tims_frontier_result_schema.md`

This document defines the official mutable orchestration ledger for a run. The
run manifest exists to support execution tracking, resume logic, and artifact
bookkeeping while a run is in progress.

## 1. Role

The run manifest exists to answer these questions:

- what is the current execution status of the run?
- which families have progressed how far?
- which artifacts already exist?
- where should resume continue from?

It is not:

- the immutable resolved config snapshot
- the final run summary
- the raw trial results table

Recommended path:

```text
results/raw/{study_id}/{run_id}/run_manifest.json
```

## 2. Relationship to Other Files

- `resolved_config.yaml`
  - immutable run definition
- `run_manifest.json`
  - mutable execution state and bookkeeping
- `run_summary.json`
  - final stable summary after the run finishes

## 3. Core Rules

- The manifest is mutable during execution.
- It should be updated atomically after meaningful state changes.
- It must be resume-safe.
- It should summarize progress, not duplicate full trial tables.
- It should reference artifact paths rather than embed large payloads.
- Trial-level detailed outcomes belong in `trial_results_table.csv`.

## 4. Recommended Status Values

Top-level `status`:

- `created`
- `initializing`
- `running`
- `analyzing`
- `reporting`
- `completed`
- `failed`
- `aborted`
- `partial`

Family-level `status`:

- `pending`
- `running`
- `completed`
- `failed`
- `partial`

## 5. Top-Level Structure

```text
schema_version
study_id
run_id
status
lifecycle
resolved_config_path
source_config_path
config_schema_version
resolved_config_schema_version
result_schema_version
specification_ref
execution_state
family_states
artifacts
checkpoints
latest_error
notes
```

## 6. Canonical JSON

```json
{
  "schema_version": "1.0",
  "study_id": "uniform_vs_heterogeneous_tims_frontier_v1",
  "run_id": "run_20260311_120000",
  "status": "running",
  "lifecycle": {
    "created_at_utc": "2026-03-11T12:00:00Z",
    "updated_at_utc": "2026-03-11T12:35:00Z",
    "started_at_utc": "2026-03-11T12:00:05Z",
    "finished_at_utc": null
  },
  "resolved_config_path": "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/resolved_config.yaml",
  "source_config_path": "configs/studies/uniform_vs_heterogeneous_tims_frontier_v1.yaml",
  "config_schema_version": "0.3.1",
  "resolved_config_schema_version": "1.0",
  "result_schema_version": "1.0",
  "specification_ref": "reports/specifications/tims_frontier_specification.md",
  "execution_state": {
    "current_stage": "exploration",
    "resume_supported": true,
    "last_persisted_trial_id": "heterogeneous_trial_0021",
    "last_persisted_at_utc": "2026-03-11T12:34:58Z"
  },
  "family_states": {
    "uniform": {
      "status": "running",
      "family_role": "baseline",
      "optuna_study_name": "uniform_mc_cq_study",
      "target_trials": 500,
      "attempted_trials": 120,
      "completed_trials": 110,
      "failed_trials": 10,
      "pruned_trials": 0,
      "next_family_trial_index": 121,
      "last_optuna_trial_number": 119,
      "frontier_extracted": false
    },
    "heterogeneous": {
      "status": "running",
      "family_role": "candidate",
      "optuna_study_name": "heterogeneous_mc_cq_study",
      "target_trials": 500,
      "attempted_trials": 118,
      "completed_trials": 104,
      "failed_trials": 14,
      "pruned_trials": 0,
      "next_family_trial_index": 119,
      "last_optuna_trial_number": 117,
      "frontier_extracted": false
    }
  },
  "artifacts": {
    "run_root": "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000",
    "trial_results_table": "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/trial_results_table.csv",
    "failure_log": "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/failure_log.jsonl",
    "run_summary": "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/run_summary.json",
    "frontier_points_table": "results/processed/uniform_vs_heterogeneous_tims_frontier_v1/frontier_points_table.csv",
    "logs_dir": "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/logs"
  },
  "checkpoints": {
    "trial_table_last_flush_utc": "2026-03-11T12:34:58Z",
    "failure_log_last_flush_utc": "2026-03-11T12:34:59Z",
    "analysis_started": false,
    "analysis_completed": false,
    "reporting_completed": false
  },
  "latest_error": null,
  "notes": []
}
```

## 7. Field Semantics

- `status`
  - overall run state
- `lifecycle`
  - creation and progress timestamps
- `execution_state.current_stage`
  - coarse workflow stage such as `exploration`, `analysis`, or `reporting`
- `family_states`
  - per-family progress counters and resume anchors
- `artifacts`
  - registered artifact paths, whether present yet or expected later
- `checkpoints`
  - last persisted state markers for resume logic
- `latest_error`
  - optional last run-level blocking error
- `notes`
  - optional operator notes or non-fatal warnings

## 8. Ownership Notes

- The run manifest owns mutable execution bookkeeping.
- It should not duplicate full raw trial records.
- It may reference the current last trial, but not embed all trials.
- Final aggregate counts should match `run_summary.json` after completion.

## 9. Non-Goals

This schema does not include:

- the full resolved configuration payload
- full exception tracebacks
- full trial metric tables
- frontier point rows

Those belong in their dedicated files.
