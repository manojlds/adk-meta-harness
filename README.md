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

## Environment

Copy and configure environment values:

```bash
cp .env.example .env
```

Common variables:

| Variable | Purpose |
|---|---|
| `AMH_MODEL` | Default model for the agent under optimization |
| `OPENAI_API_KEY` | API key for model provider used by the harness |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` | Base URL for OpenAI-compatible endpoints |

## Task Format

The project uses a first-party task format inspired by Harbor's task
conventions (without a Harbor runtime dependency).

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

## Usage

### Local runner

Evaluate a single candidate:

```bash
amh eval \
  --candidate examples/vanilla/initial_harness \
  --dataset examples/vanilla/tasks \
  --runner local
```

Run optimization locally:

```bash
amh optimize \
  --dataset examples/vanilla/tasks \
  --initial-harness examples/vanilla/initial_harness \
  --proposer opencode \
  --proposer-model opencode-go/glm-5.1 \
  --model openai/glm-5.1 \
  --iterations 10 \
  --candidates-dir ./candidates/vanilla \
  --runner local
```

### Temporal runner

Evaluate using the Temporal runner interface:

```bash
amh eval \
  --candidate ./candidates/vanilla/v0003 \
  --dataset examples/vanilla/tasks \
  --runner temporal
```

Start a worker (long-lived):

```bash
amh worker --server localhost:7233 --task-queue amh-tasks
```

Run optimization as a Temporal workflow execution (returns immediately):

```bash
amh optimize \
  --runner temporal \
  --server localhost:7233 \
  --task-queue amh-tasks \
  --dataset examples/vanilla/tasks \
  --initial-harness examples/vanilla/initial_harness \
  --iterations 10
```

Notes:
- Temporal optimize requires shared filesystem visibility for worker processes.
- `amh eval --runner temporal` preserves the `TaskRunner` interface by using
  the same local single-candidate evaluator.

## Proposers

Supported proposer values:

| Proposer | Flag |
|---|---|
| OpenCode | `--proposer opencode` |
| Pi | `--proposer pi` |
| Custom CLI command | `--proposer <command>` |

You can set a proposer model independently with `--proposer-model`.

## Judges

Supported judge values:

| Judge | Flag |
|---|---|
| LiteLLM | `--judge litellm` |
| ADK | `--judge adk` |
| CLI (OpenCode/Pi/custom) | `--judge opencode`, `--judge pi`, or custom command |

Scoring order:
1. Use reward files when present (`reward.txt`/`reward.json`)
2. Otherwise use judge scoring when configured
3. Otherwise mark task failed (score `0.0`)

## Harness Surfaces

The proposer can modify these harness surfaces:

| Surface | Path |
|---|---|
| System prompt | `system_prompt.md` |
| Skills | `skills/*/SKILL.md` |
| Tools | `tools/*.py` |
| Callbacks | `callbacks/*.py` |
| Routing | `routing/*.yaml` |
| Agent config | `config.yaml` |
| Full harness | `agent.py` |

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

Local commands:

```text
amh eval --candidate PATH --dataset PATH --runner local [options]
amh optimize --dataset PATH --initial-harness PATH --runner local [options]
```

Temporal commands:

```text
amh eval --candidate PATH --dataset PATH --runner temporal [options]
amh optimize --dataset PATH --initial-harness PATH --runner temporal [options]
amh worker [--server HOST:PORT] [--task-queue NAME]
```

Common optimize options (local and temporal):
- `--proposer`, `--proposer-model`
- `--judge`, `--judge-model`
- `--iterations`, `--holdout-ratio`, `--timeout`
- `--runner [local|temporal]`

Temporal optimize options:
- `--server`, `--task-queue`, `--workflow-id`

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
