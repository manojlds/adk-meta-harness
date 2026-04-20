# Reference Meta-Harness Adoption Plan

This plan captures what we should adopt from the Stanford reference repo while
staying aligned with this project's ADK-first architecture.

## Goals

1. Use a real search/holdout split during optimization and keep a hidden final
   test split for end-of-run reporting.
2. Add run-scoped artifacts (`pending_eval`, frontier snapshots, evolution
   summary) to make runs reproducible and easy to debug.
3. Add stronger proposer observability (request/response/session logs) and a
   structured per-iteration proposal artifact.

## Non-Goals

- Porting Harbor-specific runners or Terminal-Bench-specific agent code.
- Changing the one-candidate-per-iteration core loop in this first pass.
- Replacing `results.tsv` or candidate `meta.json` (we keep backward
  compatibility).

## Guiding Constraints

- Preserve current local/temporal runner interfaces.
- Keep changes incremental and testable in small PRs.
- Avoid breaking existing `amh eval` and `amh optimize` commands.

## Phase 1 - Real Task Splits + Final Test (High Priority)

### Scope

- Make `holdout_ratio` actually drive per-run task splitting.
- Add a hidden test split that is never used for gating.
- Evaluate best candidate on test tasks only after optimization completes.

### Design

- Add deterministic split helper (new module):
  - `src/adk_meta_harness/splits.py`
  - Input: all discovered task names, `holdout_ratio`, `test_ratio`, `seed`
  - Output: `search_task_names`, `holdout_task_names`, `test_task_names`
  - Guarantee no overlap; enforce minimums where possible.
- Persist split manifest per run:
  - `<candidates_dir>/runs/<run_id>/task_splits.json`
  - Include ratios, seed, task counts, and concrete task lists.
- Wire split usage in `optimize()`:
  - Baseline and iteration evals: search + holdout only
  - Final pass: evaluate current best on test only
- Extend `OptimizeResult`:
  - `best_test: float | None`
  - `run_id: str`

### CLI Additions

- `amh optimize` new flags:
  - `--test-ratio` (default `0.2`)
  - `--split-seed` (default `42`)
  - `--run-id` (optional; auto-generated when omitted)

### Tests

- Add/extend tests:
  - `tests/test_outer_loop.py`
  - `tests/test_cli.py`
  - new `tests/test_splits.py`
- Assertions:
  - deterministic splits for same seed
  - no overlap between search/holdout/test
  - `holdout_ratio` affects runner call task lists
  - final test evaluation runs exactly once at end

### Acceptance Criteria

- `holdout_ratio` is no longer a no-op.
- Final CLI output includes holdout, search, and final test score.

## Phase 2 - Run-Scoped Artifacts + Frontier Tracking (High Priority)

### Scope

- Add a run artifact directory independent of candidate version dirs.
- Track frontier and summary files in a stable schema.

### Run Directory Layout

`<candidates_dir>/runs/<run_id>/`

- `task_splits.json`
- `pending_eval.json` (latest proposal payload)
- `frontier_val.json`
- `evolution_summary.jsonl`
- `reports/` (reserved for iteration-level summaries)

### Design

- Add helper module:
  - `src/adk_meta_harness/run_artifacts.py`
  - helpers to init run dirs, append summary rows, update frontier, read resume
    metadata.
- Keep existing `results.tsv` updates for compatibility.
- `evolution_summary.jsonl` row schema (one row per evaluated candidate):
  - `iteration`, `version`, `parent_version`, `description`, `change_type`
  - `combined_score`, `search_score`, `holdout_score`
  - `gate_kept`, `gate_reason`, `search_delta`, `holdout_delta`
  - `validation_status`, `timing_s` (propose/eval/total)

### Resume Behavior

- If `--run-id` is provided and exists, continue writing to same run artifacts.
- If omitted, create a new run id by default.
- Do not change candidate resume logic; only add run-level bookkeeping.

### Tests

- new `tests/test_run_artifacts.py`
- extend `tests/test_outer_loop.py` resume cases to verify run artifact
  continuity.

### Acceptance Criteria

- Every optimize run creates a complete run artifact folder.
- Summary/frontier files are readable and sufficient to debug iteration history
  without opening individual candidate dirs.

## Phase 3 - Proposer Contract + Session Logging (High Priority)

### Scope

- Capture proposer IO and edit outcomes in candidate-local proposal artifacts.
- Produce a structured proposal record each iteration.

### Candidate Proposal Artifacts

`<candidate>/proposal/`

- `request.md` (instruction sent to proposer)
- `stdout.log`
- `stderr.log`
- `session.json` (command, return code, duration, change_type, diff summary)
- `pending_eval.json` (single-candidate schema for this project)

### Schema (single candidate)

```json
{
  "iteration": 3,
  "candidate": {
    "version": 7,
    "name": "v0007",
    "hypothesis": "<short, falsifiable claim>",
    "change_type": "tool|skill|system_prompt|config|routing|callback|harness|multiple",
    "diff_summary": "<compact file-level summary>"
  }
}
```

### Design

- Extend `CodingAgentCLIProposer.propose_edit()` to return structured metadata
  and write logs under `candidate/proposal/`.
- Keep retry-on-no-edit behavior.
- Keep current proposer CLI support (`opencode`, `pi`) and avoid introducing
  provider-specific protocol assumptions in first pass.

### Tests

- extend `tests/test_proposer.py` and `tests/test_proposer_extended.py`:
  - proposal artifacts are written
  - `pending_eval.json` is emitted with expected fields
  - retry behavior remains unchanged

### Acceptance Criteria

- Every iteration has a machine-readable proposal artifact and proposer session
  logs.
- Failed proposal runs are diagnosable without rerunning.

## Phase 4 - Onboarding Doc for New Domains (Medium Priority)

### Scope

- Add first-party onboarding prompt modeled after the reference.

### Deliverable

- `ONBOARDING.md` in repo root with:
  - required fields for domain spec
  - leakage/split checklist
  - budget and benchmark readiness questions
  - `domain_spec.md` template

### Acceptance Criteria

- A new user can create a complete domain spec in one guided conversation.

## PR Slicing (Recommended)

1. PR-1: task split plumbing + final test evaluation + CLI flags + tests.
2. PR-2: run artifact module + frontier/summary writing + resume integration.
3. PR-3: proposer proposal/session artifacts + tests.
4. PR-4: onboarding docs.

## Rollout and Validation Commands

- `uv run pytest tests/test_splits.py tests/test_outer_loop.py tests/test_cli.py`
- `uv run pytest tests/test_run_artifacts.py tests/test_outer_loop.py`
- `uv run pytest tests/test_proposer.py tests/test_proposer_extended.py`
- `uv run pytest`

## Recommended Defaults

- Keep one-candidate-per-iteration behavior for now.
- `test_ratio=0.2`, `split_seed=42`.
- New runs create a new `run_id` unless explicitly resumed with `--run-id`.

## Risks and Mitigations

- Very small task sets can make 3-way splits unstable:
  - Mitigation: minimum-count guardrails + documented fallback behavior.
- Resume semantics can become confusing with run ids:
  - Mitigation: explicit CLI output showing `run_id`, split counts, and artifact
    directory.
- Additional files per run may increase clutter:
  - Mitigation: keep all run files under `candidates/runs/` and add cleanup docs.
