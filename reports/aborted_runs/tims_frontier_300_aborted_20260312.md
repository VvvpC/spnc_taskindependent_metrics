# TIMs Frontier 300-Trial Run Aborted Record

## Status

This run was cancelled intentionally by the user before completion.

- `study_id`: `uniform_vs_heterogeneous_tims_frontier_300_v1`
- `run_id`: `study300_run_20260312`
- Final archived status: `cancelled`

## Why This Record Exists

This document exists to prevent the partial artifacts from being mistaken for a completed study run.
The run produced valid partial raw data, but it did not reach:

- the full `uniform` budget
- any `heterogeneous` trial execution
- frontier extraction
- final run summary generation

As a result, these artifacts should not be used as a completed study result.

## Config And Entry Point

- Config: [study_300.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/configs/tims_frontier/study_300.yaml)
- Script: [run_tims_frontier_study.py](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/scripts/run_tims_frontier_study.py)

## Lifecycle

- `created_at_utc`: `2026-03-12T16:24:10Z`
- `cancelled_at_utc`: `2026-03-12T19:39:48Z`

## Partial Progress At Cancellation

- `uniform`:
  - attempted: `136`
  - completed: `136`
  - failed: `0`
  - pruned: `0`
- `heterogeneous`:
  - attempted: `0`
  - completed: `0`
  - failed: `0`
  - pruned: `0`

## Artifact Status

- Valid partial raw table:
  - [trial_results_table.csv](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_300_v1/study300_run_20260312/trial_results_table.csv)
- Valid run manifest:
  - [run_manifest.json](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_300_v1/study300_run_20260312/run_manifest.json)
- Valid resolved config:
  - [resolved_config.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_300_v1/study300_run_20260312/resolved_config.yaml)
- Empty failure log:
  - [failure_log.jsonl](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_300_v1/study300_run_20260312/failure_log.jsonl)
- Canonical but not meaningful processed frontier table:
  - [frontier_points_table.csv](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/processed/uniform_vs_heterogeneous_tims_frontier_300_v1/frontier_points_table.csv)
  - This file contains only the schema header and should be treated as unusable for analysis.

## Interpretation Rule

Do not use this run for:

- uniform vs heterogeneous comparison
- family-level frontier interpretation
- report generation
- scientific conclusion making

At most, it may be reused later as:

- a partial uniform-only debug artifact
- an execution/performance reference
- a resume design reference if resume support is extended

## Archival Note

The run process was terminated, the manifest was updated to `cancelled`, and the partial artifacts were intentionally retained.
