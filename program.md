# Autoresearch V1 Program

This repository now follows a narrowed autoresearch-style loop for heterogeneous nanodot reservoir design.

## Core rules

1. Only edit `train.py`.
2. Do not edit evaluator/runtime/framework files.
3. Each round proposes exactly one heterogeneous design family, not a single point.
4. The new proposal must be a minimal semantic change relative to the last attempted proposal.
5. The first successful round becomes the fixed baseline.
6. From the second successful round onward, the only primary objective is:

   `S_abs = HV(frontier(baseline ∪ P(G))) - HV(frontier(baseline))`

7. Keep a commit only if:
   - it is the first successful baseline round, or
   - its `S_abs` strictly improves over the current best kept round.

If not, discard it and reset back to the current best commit.

## Editable surface

`train.py` must always export:

- `CURRENT_PROPOSAL`
- `main()`

`CURRENT_PROPOSAL` must contain:

- `proposal_id`
- `parent_proposal_id`
- `edit_type`
- `primary_edit`
- `rationale`
- `expected_effect`
- `family_definition`
- `sampling_plan`

Do not move the runtime call out of `main()`. The framework depends on `python train.py`.

## Run loop

Use the fixed runner:

```bash
python scripts/run_autoresearch_v1.py init
python scripts/run_autoresearch_v1.py step
```

Or a bounded loop:

```bash
python scripts/run_autoresearch_v1.py loop --iterations 3
```

The step order is fixed:

1. Validate that only `train.py` changed.
2. Commit the `train.py` experiment.
3. Run `python train.py > run.log 2>&1`.
4. Parse the machine-readable summary from the log.
5. If the summary is missing, treat the run as a crash.
6. Capture the last 50-100 lines from `run.log` and feed that traceback context into the next proposal attempt.
7. Append one row to `results.tsv`.
8. Keep or discard the commit based on the rule above.

## Crash handling

If a run crashes:

- read `run.log`
- inspect the tail
- look at the saved `crash_tail.txt` and `crash_context.json`
- make a minimal fix in `train.py`

Do not respond to a crash by editing framework files. The fixed runtime is intentionally sealed.

## Output contract

Successful `train.py` runs print one structured line:

`AUTORESEARCH_SUMMARY {...}`

The loop runner extracts:

- `proposal_id`
- `status`
- `baseline_hv`
- `family_hv`
- `score_s_abs`
- `frontier_size`
- `round_dir`

If that line is absent, the run is treated as a crash.

## Design intent

The goal is not generic agent behavior. The goal is an interpretable scientific search loop:

- one editable file
- one proposal per round
- one minimal semantic edit per round
- fixed evaluator
- fixed baseline after the first successful round
- fixed scoring target
