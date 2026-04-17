# DRS Project Context

## What adk-meta-harness Is
Python framework that optimizes Google ADK harnesses through an iterative
propose-evaluate-gate loop. It automates candidate generation, evaluation on
task datasets, gating decisions, and accumulation of learning signals.

## Core Flow
- Start from an initial harness (`agent.py`, prompt, skills, tools, routing).
- Generate one candidate change per iteration using a proposer CLI.
- Evaluate on search and holdout tasks using local or Harbor runners.
- Gate candidates based on score outcomes and keep only improved versions.
- Persist traces, reward artifacts, and `learnings.md` for the next iteration.

## Architecture
- `src/adk_meta_harness/outer_loop.py`: main optimization loop orchestration.
- `src/adk_meta_harness/eval_one.py`: single-candidate evaluation entrypoint.
- `src/adk_meta_harness/proposer/`: proposer adapters (OpenCode, Pi, base CLI).
- `src/adk_meta_harness/runner/`: execution backends (local and Harbor).
- `src/adk_meta_harness/trace/`: trace export and ATIF/Harbor reward handling.
- `src/adk_meta_harness/gate.py`, `candidate.py`, `learnings.py`: selection logic,
  candidate metadata, and accumulated insights.
- `tests/`: unit and integration-style coverage for loop, runners, judges, traces,
  and proposer plumbing.

## Technology Stack
- Language: Python 3.12+
- Agent framework: Google ADK
- Evaluation runtime: Harbor (optional Docker isolation)
- LLM abstraction: `litellm`
- Packaging/build: Hatchling, `uv`
- Testing: `pytest`, `pytest-asyncio`
- Lint/format: `ruff`

## Security and Reliability Focus
- Correct isolation boundaries between task runs and candidate directories.
- Safe subprocess/CLI invocation for proposer and verifier commands.
- Deterministic gating behavior from evaluation and holdout signals.
- Robust handling of missing traces, partial failures, and adapter edge cases.

## Review Focus
- Correctness of optimization-loop state transitions and candidate lifecycle.
- Runner behavior parity between local and Harbor modes.
- Trace/reward data integrity from execution to scoring.
- CLI ergonomics, failure messages, and docs alignment with behavior.
