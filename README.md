# adk-meta-harness

Meta-harness optimization for Google ADK agents.

`adk-meta-harness` iteratively improves an agent harness (prompt, tools,
skills, callbacks, routing, config) with a propose -> evaluate -> gate loop,
following the ideas in the Meta-Harness paper.

## Install

```bash
git clone https://github.com/manojlds/adk-meta-harness.git
cd adk-meta-harness

# Core (local runner)
uv sync

# Optional extras
uv sync --extra skills    # adk-skills-agent + adk-tool-search
uv sync --extra temporal  # Temporal worker/workflow support
```

## Task Format

The project now uses a first-party task format (no Harbor dependency).

```text
tasks/
  read-file/
    instruction.md
    task.toml
    fixtures/              # copied into WORK_DIR before execution
    scripts/
      setup.sh             # optional
      teardown.sh          # optional
    tests/
      test.sh              # optional verifier, writes reward files
```

`task.toml` (minimal):

```toml
[metadata]
description = "Read a file and report its contents"

[agent]
timeout_sec = 120

[verifier]
timeout_sec = 120

[scripts]
setup_timeout_sec = 60
teardown_timeout_sec = 60

[env]
MY_FLAG = "value"
```

Verifier scripts can write either `reward.txt` or `reward.json`.

## Quick Start (Local)

```bash
amh optimize \
  --dataset examples/vanilla/tasks \
  --initial-harness examples/vanilla/initial_harness \
  --proposer opencode \
  --proposer-model opencode-go/glm-5.1 \
  --model openai/glm-5.1 \
  --iterations 10 \
  --candidates-dir ./candidates/vanilla
```

Evaluate a single candidate:

```bash
amh eval \
  --candidate ./candidates/vanilla/v0003 \
  --dataset examples/vanilla/tasks
```

## Temporal Mode

Phase 2 introduces a Temporal workflow mode for optimization orchestration.

1) Start a worker (long-lived):

```bash
amh worker --server localhost:7233 --task-queue amh-tasks
```

2) Start optimize as a workflow (fire-and-forget):

```bash
amh optimize \
  --runner temporal \
  --server localhost:7233 \
  --task-queue amh-tasks \
  --dataset examples/vanilla/tasks \
  --initial-harness examples/vanilla/initial_harness \
  --iterations 10
```

The CLI prints Workflow ID (and Run ID when available) and exits.

Notes:
- Temporal mode requires shared filesystem visibility for worker processes.
- `amh eval --runner temporal` currently uses the same local single-candidate
  evaluator to preserve the `TaskRunner` interface.

## Runners

| Runner | Flag | Use case |
|---|---|---|
| Local | `--runner local` | In-process development and debugging |
| Temporal | `--runner temporal` | Distributed optimization orchestration |

## How Scoring Works

For each task:
1. Run optional `scripts/setup.sh`
2. Run ADK agent and capture OTel spans -> ATIF trajectory
3. Run optional verifier `tests/test.sh`
4. Parse reward files when present (`reward.txt` / `reward.json`)
5. Optional judge fallback if no reward file exists
6. Run optional `scripts/teardown.sh`

Candidate-level scores:
- `search_score`: pass rate on search tasks
- `holdout_score`: pass rate on holdout tasks (or `search_score` if no holdout)
- `combined_score`: pass rate across all evaluated tasks

## Model Precedence

Model resolution order:
1. `--model`
2. `config.yaml` in candidate harness
3. agent default in `agent.py`

## CLI

```text
amh optimize --dataset PATH --initial-harness PATH [options]
amh eval --candidate PATH --dataset PATH [options]
amh worker [--server HOST:PORT] [--task-queue NAME]
```

Common optimize options:
- `--proposer`, `--proposer-model`
- `--judge`, `--judge-model`
- `--iterations`, `--holdout-ratio`, `--timeout`
- `--runner [local|temporal]`
- `--server`, `--task-queue`, `--workflow-id` (temporal mode)

## Project Layout

```text
src/adk_meta_harness/
  cli.py
  outer_loop.py
  task.py
  task_executor.py
  runner/
    __init__.py
    local.py
    temporal_runner.py
  trace/
    atif.py
    otel_to_atif.py
    reward.py
```

## Related

- Meta-Harness paper: https://arxiv.org/abs/2603.28052
- canvas-org/meta-agent: https://github.com/canvas-org/meta-agent
- agentskills.io: https://agentskills.io

## License

Apache-2.0
