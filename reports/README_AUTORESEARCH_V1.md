# Autoresearch V1

## Goal

This v1 framework adapts the `karpathy/autoresearch` interaction style to heterogeneous nanodot reservoir design search.

The loop is intentionally narrow:

- one editable file: `train.py`
- one human-maintained instruction file: `program.md`
- one proposal family per round
- one minimal semantic edit per round
- one fixed evaluator pipeline
- one fixed baseline after the first successful round
- one primary score: baseline-relative hypervolume gain in MC-CQ space

## Why proposal = family, not point

The scientific object is not a single parameter point. A proposal describes a heterogeneous design family:

- family type
- subgroup topology
- distribution rule
- coupling rule
- continuous parameter space
- sampling plan

The runtime samples that family into concrete reservoir instances and evaluates the resulting family frontier.

## Why minimal semantic edits matter

Every new proposal must stay close to the last attempted proposal.

The framework tracks semantic edit types such as:

- `scalar_tune`
- `categorical_swap`
- `structural_expand`
- `structural_reduce`
- `coupling_change`
- `distribution_change`

This keeps the search interpretable and makes crash recovery and pattern learning easier for the AI.

## Baseline

In this repository's current autoresearch mode, the fixed baseline is not loaded from a uniform frontier file.

Instead:

1. the first successful heterogeneous proposal is evaluated,
2. its family frontier is frozen as the run baseline,
3. later rounds are scored relative to that fixed baseline.

The runtime stores:

- `baseline_points`
- `baseline_frontier`
- `baseline_hv`

## Primary score

For a proposal family `G`, let `P(G)` be the evaluated sample set from that family.

The only primary optimization target is:

`S_abs(G) = HV(frontier(baseline ∪ P(G))) - HV(frontier(baseline))`

Where:

- objectives are `MC` and `CQ`
- both are maximized
- the reference point is fixed in config
- normalization is fixed in config

Audit metrics such as marginal gain, span, repeat consistency, and beyond-baseline count are recorded but do not replace `S_abs`.

## Runtime layout

The fixed runtime lives under `src/tims_frontier/autoresearch/`.

It handles:

- proposal schema and validation
- auto-completion
- family sampling
- evaluator adaptation
- Pareto and hypervolume scoring
- run archive
- traceback injection
- keep/discard loop helpers

The AI must not edit these files during the experiment loop.

## How to run

Initialize:

```bash
python scripts/run_autoresearch_v1.py init
```

Run one step:

```bash
python scripts/run_autoresearch_v1.py step
```

Run several steps:

```bash
python scripts/run_autoresearch_v1.py loop --iterations 3
```

Inspect the current best:

```bash
python scripts/run_autoresearch_v1.py best
```

Export lineage:

```bash
python scripts/run_autoresearch_v1.py export
```

Minimal portable demo:

```bash
python scripts/run_autoresearch_v1.py --config configs/autoresearch_v1/demo.json init
python -c "import train; from tims_frontier.autoresearch import run_train_file; run_train_file(train.CURRENT_PROPOSAL, config_path='configs/autoresearch_v1/demo.json')"
```

The default config keeps the real evaluator wiring. The `demo.json` config uses the same fixed control loop with a mock evaluator so the framework can be exercised on lightweight environments.

## Proposal examples

Scalar tune example:

```python
{
  "proposal_id": "proposal_0002",
  "parent_proposal_id": "proposal_0001",
  "edit_type": "scalar_tune",
  "primary_edit": {
    "target": "continuous_parameters.gamma.high",
    "before": 0.08,
    "after": 0.09
  }
}
```

Structural family change example:

```python
{
  "proposal_id": "proposal_0003",
  "parent_proposal_id": "proposal_0002",
  "edit_type": "structural_expand",
  "primary_edit": {
    "target": "family_type",
    "before": "single_distribution",
    "after": "bimodal_core_tail"
  }
}
```

## Traceback injection

The loop runner always executes:

```bash
python train.py > run.log 2>&1
```

If the structured summary line is missing, the run is treated as a crash.
The runner then captures the last 50-100 lines of `run.log` and stores:

- `crash_tail.txt`
- `crash_context.json`

This makes the next round act like a closed-loop debugger.

## Replacing the AI backend

The current v1 expects the agent to edit `train.py` directly.

That is the main backend.

Later extensions can still build other frontends on top of the fixed runtime, but they should preserve the same constraints:

- only `train.py` is editable during the autonomous loop
- the runtime remains fixed
- the score remains `S_abs`

## Current limitations

- The loop currently uses direct `train.py` editing instead of a richer external proposal API.
- The supported family compiler is intentionally small: `single_distribution` and `bimodal_core_tail`.
- The real evaluator is slow, so tests use lightweight logic and should not depend on long physics runs.
- Keep/discard uses git state, so autonomous operation assumes a clean dedicated branch.
