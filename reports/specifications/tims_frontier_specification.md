# TIMs Frontier Workflow Specification

- Document type: Specification Document
- Status: Accepted baseline
- Scope: Current branch workflow design
- Authority: User-approved project brief

This document is the authoritative workflow specification for the current
research direction on this branch. Future architecture, schema, orchestration,
analysis, and reporting decisions should follow this document unless a newer
explicit specification supersedes it.

---

# Project Brief: TIMs Frontier Comparison Between Uniform and Heterogeneous Nanodot Reservoirs

## 1. Project objective

The goal of this project is to build a research workflow for comparing
**uniform nanodot reservoirs** and **heterogeneous nanodot reservoirs** in terms
of their **task-independent metrics (TIMs) frontier**.

The core research question is:

> Under a shared non-geometric hyperparameter search regime, do heterogeneous
> nanodot reservoirs exhibit an expanded or reconfigured Pareto frontier in the
> core TIMs space defined by MC, KR, GR, and CQ, compared with uniform
> nanodot reservoirs?

This project is **not** primarily about downstream task performance at this
stage.
It is also **not** primarily about comparing single best instances.
The main goal is to compare **family-level capability frontiers**.

---

## 2. Core scientific scope

### 2.1 Comparison target

We compare two reservoir families:

- **Uniform reservoir family**
  Reservoirs without heterogeneous geometric morphology distributions.

- **Heterogeneous reservoir family**
  Reservoirs whose geometric morphology includes distribution-based
  heterogeneity as part of the design rule.

The main comparison is:

- **uniform vs heterogeneous**

We are not yet focusing on comparing multiple heterogeneous schemes against
each other, although the code should ideally remain extensible.

---

## 3. Core capability space

The main TIMs space is defined by the following four metrics:

- **MC** = Memory Capacity
- **KR** = Kernel Rank
- **GR** = Generalisation Rank
- **CQ** = Computational Quality

These four metrics form the **core capability space** of this study.

Any 2D or 3D plots are only projections of this 4D core space.
The code and data structures should treat this as a **multi-objective metric
space**, not as isolated scalar scores.

---

## 4. Primary endpoint

The **primary endpoint** of this study is:

> The difference between the Pareto frontiers of the uniform and heterogeneous
> reservoir families in the MC-KR-GR-CQ space.

The primary operationalization of "expansion" is:

- whether the heterogeneous family reaches Pareto-optimal regions not covered
  or dominated by the uniform family;
- whether the frontier shifts outward along some multi-metric trade-off
  directions;
- whether the frontier geometry is reconfigured, even if not uniformly
  expanded in every direction.

This means:

- **Pareto frontier comparison is primary**
- point cloud coverage or single-metric distributions are secondary
- downstream tasks are out of scope for the current stage

---

## 5. Fair comparison protocol

The core fairness principle is **NOT** strict equality of total volume,
average volume, or average energy scale.

Instead, the main fairness constraint is:

> Uniform and heterogeneous families must be explored under the same
> **non-geometric hyperparameter search regime**.

### 5.1 Shared constraints

The following should be shared across compared families:

- the same non-geometric hyperparameter search ranges;
- the same exploration budget (e.g. number of trials);
- the same exploration strategy where applicable;
- the same evaluator definitions for MC, KR, GR, CQ;
- the same runtime and logging conventions.

### 5.2 Not part of the main fairness constraint

The geometric morphology parameter space specific to the heterogeneous
reservoir design is **not** required to match the uniform one.

Therefore:

- heterogeneous reservoirs may naturally differ in total volume or related
  geometric statistics;
- this is not automatically considered unfair;
- such differences are part of the heterogeneous design family itself.

This project compares **design families under a shared search protocol**, not
strictly resource-matched isolated perturbations.

---

## 6. Current study boundaries

At the current stage, the project should explicitly focus on:

- TIMs only
- uniform vs heterogeneous only
- MC, KR, GR, CQ as the core metric set
- Pareto frontier as the primary comparison object

The following are **not** the main focus right now:

- downstream task benchmarks such as NARMA10 or TI-46
- static-system filtering as a major research concern
- strict physical resource matching such as fixed total volume
- broad qualitative summaries without structured outputs

Basic numerical sanity checks are still welcome, but static-system concerns
should not dominate the current implementation.

---

## 7. Required workflow structure

Please organize the project around the following layers:

### 7.1 Research Specification

A structured representation of the study definition, including:

- research question
- compared families
- fairness protocol
- core metrics
- primary endpoint

### 7.2 Configuration

A configuration system defining:

- reservoir construction parameters
- non-geometric hyperparameter search space
- evaluation settings
- exploration settings
- storage/output settings

### 7.3 Reservoir Construction

A module that can construct:

- uniform reservoir instances
- heterogeneous reservoir instances
- optionally reservoir families or family generators

### 7.4 Evaluation

A module for computing TIMs for a single reservoir instance:

- MC
- KR
- GR
- CQ

This layer should only compute metrics and return structured results.
It should not perform family-level comparison logic.

### 7.5 Exploration

A module for exploring the parameter space under the shared search protocol,
for example via:

- Optuna
- random sampling
- grid search (optional)

Optuna should be treated as part of **exploration**, not as the metric
definition itself.

### 7.6 Execution / Orchestration

A layer for running studies robustly, including:

- trial execution
- run IDs
- logging
- retry / resume logic where possible
- experiment bookkeeping

### 7.7 Storage / Provenance

A layer for storing:

- study_id
- run_id
- trial_id
- resolved config
- seed
- code version or git commit if possible
- metric outputs
- runtime metadata
- failure logs where applicable

### 7.8 Analysis

A layer for:

- extracting Pareto frontiers
- comparing frontiers between families
- optionally summarizing point clouds as secondary analysis

### 7.9 Reporting

A layer for generating structured outputs that can be consumed by humans or
other AI models.

---

## 8. What Codex should help with

The immediate task is **not** to rewrite the scientific question.
The immediate task is to help translate this study definition into a
maintainable code/repository structure.

Specifically, Codex should help with the following:

1. Review the existing repository structure and identify reusable components.
2. Map existing code into the workflow layers above.
3. Propose a refactored module structure if needed.
4. Identify what is missing for:
   - TIM evaluation
   - exploration with Optuna
   - frontier extraction
   - structured result storage
5. Help define standard data schemas for:
   - instance-level metric outputs
   - trial-level outputs
   - family-level frontier comparison inputs
6. Help design a reproducible run pipeline.

---

## 9. Expected implementation style

Please prefer the following engineering principles:

- modular design;
- clear separation between construction / evaluation / exploration / analysis;
- configuration-driven execution;
- reproducible output structure;
- minimal hidden coupling;
- explicit naming and typed data structures where useful;
- avoid embedding scientific comparison logic directly inside low-level metric
  evaluators.

---

## 10. Expected outputs

The project should eventually support outputs such as:

### 10.1 Trial-level results

Structured records containing:

- family type
- parameter values
- MC / KR / GR / CQ
- trial metadata

### 10.2 Frontier-ready tables

Tables suitable for Pareto extraction and comparison.

### 10.3 Study summaries

Compact machine-readable summaries for later AI-assisted analysis.

### 10.4 Human-readable reports

Markdown summaries describing:

- what was run
- what was compared
- main frontier outcomes
- relevant caveats

---

## 11. Important interpretation rules

Please keep the following interpretation rules in mind when designing the
workflow:

- The main conclusion should not be based on one best reservoir instance.
- The main comparison object is the **family-level Pareto frontier**.
- Frontier reconfiguration matters even when there is no uniform outward
  expansion in every direction.
- The workflow should make it easy to compare:
  - frontier position
  - frontier shape
  - non-dominated capability combinations

---

## 12. Immediate deliverable requested from Codex

Please use this document to do the following:

1. infer the intended workflow architecture for this project;
2. inspect the existing repository and identify which parts already implement:
   - config handling
   - reservoir construction
   - TIM evaluation
   - Optuna exploration
   - result saving
3. propose a concrete repository/module structure aligned with this study
   definition;
4. point out ambiguities, missing abstractions, or technical debt that would
   block this workflow;
5. if possible, draft the first version of the code architecture needed to
   support this study.

Do not change the scientific objective unless absolutely necessary.
If there are design ambiguities, preserve the core logic of this specification:

- TIMs frontier problem
- uniform vs heterogeneous
- MC/KR/GR/CQ core space
- shared non-geometric search protocol
- Pareto frontier as primary endpoint
