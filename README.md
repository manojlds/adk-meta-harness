# adk-meta-harness

Meta-harness optimization for Google ADK agents. Automatically improves your
ADK agent's harness — system prompts, skills, tools, callbacks, and routing —
through an iterative propose-evaluate-gate loop.

Follows the [Meta-Harness](https://arxiv.org/abs/2603.28052) paper.

## How it works

```
┌─────────────────────────────────────────────────────────┐
│                     Optimization Loop                     │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────┐   ┌────────┐ │
│  │ Proposer │───►│ Evaluate │───►│ Gate │──►│ Learn  │ │
│  │ (CLI)   │    │ (Runner) │    │      │   │ (.md)  │ │
│  └──────────┘    └──────────┘    └──────┘   └────────┘ │
│       ▲                              │                   │
│       │         kept ────────────────┘                   │
│       │         discarded ──► delete candidate            │
│       │                                                   │
│       └─── next iteration reads all prior candidates ◄──┘
│                                                          │
└─────────────────────────────────────────────────────────┘
```

1. **Propose** — A coding agent CLI (OpenCode, Pi) reads the filesystem of all
   prior candidates, traces, and scores, then proposes one targeted harness change.
2. **Evaluate** — The proposed harness runs on tasks (search + holdout sets).
3. **Gate** — Keep only if holdout improves. Discard otherwise.
4. **Learn** — Accumulated insights are written to `learnings.md` for the next iteration.
5. **Repeat** — Continue the loop.

The key insight: the proposer has **full filesystem access** to all prior candidates,
execution traces, and scores — up to 10M tokens of diagnostic context per step.

## Install

```bash
# From source (recommended for development)
git clone https://github.com/manojlds/adk-meta-harness.git
cd adk-meta-harness
uv sync

# Or via pip once published
pip install adk-meta-harness
```

## Setup

### Environment variables

Copy the example env file and configure:

```bash
cp .env.example .env
```

`.env` contains two sets of configuration:

| Variable | Purpose | Example |
|---|---|---|
| `AMH_MODEL` | Default model for the ADK agent being optimized | `openai/glm-5` |
| `OPENAI_API_KEY` | API key for the ADK agent's model provider | `sk-...` |
| `OPENAI_BASE_URL` | Base URL for the ADK agent's model API | `https://opencode.ai/zen/go/v1` |
| `OPENAI_API_BASE` | Alias — some libraries read this instead of `OPENAI_BASE_URL` | same as above |

`AMH_MODEL` sets the default model (overridden by `--model` or `config.yaml`).
`OPENAI_API_KEY` / `OPENAI_BASE_URL` are used by the ADK agent under optimization.
The proposer (OpenCode) has its own provider config — it uses whatever model
and API endpoint is configured in your OpenCode setup (the `go` provider for
local LLM access).

### Harbor tasks (dataset)

The optimizer runs your agent on Harbor tasks. Each task is a directory containing:

```
my-tasks/
├── task-001/
│   ├── instruction.md    # Task instruction
│   ├── environment/
│   │   └── Dockerfile     # Task environment (Docker)
│   ├── fixtures/           # Files copied into the work directory
│   └── tests/
│       └── test.sh         # Verification script (Harbor)
├── task-002/
│   ├── instruction.md
│   └── ...
└── ...
```

You bring your own tasks. See [Harbor](https://github.com/harbor-framework/harbor)
for the task format specification.

## Quick start

### 1. Choose an initial harness

Pick one of the bundled examples or bring your own:

```bash
# Vanilla — minimal agent, no skills or tools
examples/vanilla/initial_harness/

# Skills-enabled — agent with adk-skills discovery
examples/skills-enabled/initial_harness/

# Tool-search — agent with adk-tool-search dynamic discovery
examples/tool-search/initial_harness/

# Deep-research — agent with skills + tool search + deep-research skill
examples/deep-research/initial_harness/
```

### 2. Run the optimization loop

```bash
# Local runner (default) — agent runs in-process
amh optimize \
  --dataset examples/vanilla/tasks \
  --initial-harness examples/vanilla/initial_harness \
  --proposer opencode \
  --proposer-model opencode-go/glm-5.1 \
  --model openai/glm-5.1 \
  --iterations 10 \
  --candidates-dir ./candidates/vanilla

# Harbor runner — agent runs inside Docker containers
amh optimize \
  --dataset examples/vanilla/tasks \
  --initial-harness examples/vanilla/initial_harness \
  --proposer opencode \
  --runner harbor \
  --iterations 10 \
  --candidates-dir ./candidates/vanilla
```

### 3. What happens during optimization

The optimizer creates a `candidates/` directory structure:

```
candidates/
├── v0000/                    # Baseline (copy of initial_harness)
│   ├── agent.py
│   ├── system_prompt.md
│   ├── config.yaml
│   ├── evaluation/            # ATIF traces from eval runs
│   │   ├── task-001/
│   │   │   ├── trajectory.json
│   │   │   └── reward.json
│   │   └── task-002/
│   │       └── ...
│   ├── traces/                # Mirrored from evaluation/ for proposer
│   │   ├── task-001.json
│   │   └── task-002.json
│   └── .candidate_meta.json   # Score, diff, kept/discarded
├── v0001/                    # Iteration 1 — proposer edits this
│   ├── agent.py              # Proposer may have modified this
│   ├── system_prompt.md       # Or this
│   ├── skills/               # Or added skills
│   ├── evaluation/
│   ├── traces/
│   └── .candidate_meta.json
├── results.tsv               # Running score history
└── learnings.md              # Accumulated proposer insights
```

Each iteration:

1. **Copy** — The best candidate so far is copied to a new `vNNNN/` directory.
2. **Propose** — OpenCode (or Pi) reads all prior candidates, traces, and
   `learnings.md`, then makes one targeted edit.
3. **Evaluate** — The edited harness runs on search + holdout tasks. ATIF traces
   and Harbor reward files are collected.
4. **Gate** — If holdout score improves (or stays same with less complexity),
   the candidate is **kept**. Otherwise it is **discarded** and its directory
   is removed.
5. **Learn** — Failure patterns and insights are appended to `learnings.md`.

### 4. Evaluate a single candidate manually

```bash
# Local runner
amh eval \
  --candidate candidates/v0002 \
  --dataset examples/vanilla/tasks \
  --model openai/glm-5.1

# Harbor runner
amh eval \
  --candidate candidates/v0002 \
  --dataset examples/vanilla/tasks \
  --runner harbor
```

## Runners

Task evaluation can run in two modes:

| Runner | Flag | Description |
|---|---|---|
| Local | `--runner local` (default) | Agent runs in-process. No containers. Uses `os.chdir` for isolation. Verifier scripts run as local subprocesses. |
| Harbor | `--runner harbor` | Each task runs in its own Docker container. Agent executes inside the container. Full filesystem and network isolation. Verifier runs inside the container. |

### Local runner

The default. No Docker required. The ADK agent runs in the current Python process
with `os.chdir(work_dir)` as the only isolation. Verifier scripts (`tests/test.sh`)
run as local subprocesses with `LOGS_DIR` and `AGENT_RESPONSE_FILE` env vars.

Best for: development, fast iteration, trusted tasks.

### Harbor runner

Each task gets its own Docker container built from the task's `environment/Dockerfile`.
The base image `adk-meta-harness:latest` has Python, google-adk, and adk-meta-harness
pre-installed. Task Dockerfiles should use `FROM adk-meta-harness:latest` and add
only task-specific files:

```dockerfile
# examples/vanilla/tasks/read-file/read-file/environment/Dockerfile
FROM adk-meta-harness:latest
COPY fixtures/ /app/
```

The harness is uploaded into `/app/harness/` inside the container, and the agent
runs via the `AdkHarborAgent` (Harbor `BaseAgent` subclass).

Best for: final evaluation, untrusted tasks, CI/CD, reproducibility.

### Building the base image

```bash
docker build -t adk-meta-harness:latest -f docker/adk-meta-harness.Dockerfile .
```

## Model precedence

Models are resolved in this order:

| Priority | Source | Example |
|---|---|---|
| 1 (highest) | `--model` CLI flag | `amh optimize --model openai/glm-5.1` |
| 2 | `config.yaml` in the harness | `model: gemini-2.5-flash` |
| 3 (lowest) | Agent default | `Agent(model="gemini-2.5-flash")` |

The proposer **cannot** change the model unless `config.yaml` has
`allow_model_changes: true`. This prevents the proposer from swapping to a
more capable model instead of genuinely improving the harness.

## Harness surfaces

The proposer can modify any of these harness surfaces:

| Surface | File | ADK mapping |
|---|---|---|
| System prompt | `system_prompt.md` | `Agent(instruction=...)` |
| Skills | `skills/*/SKILL.md` | Agent Skills (agentskills.io spec) |
| Tools | `tools/*.py` | `Agent(tools=[...])` |
| Callbacks | `callbacks/*.py` | `before_model_callback`, `after_tool_callback` |
| Routing | `routing/*.yaml` | Multi-agent transfer rules |
| Agent config | `config.yaml` | Model, max_turns, stop_conditions |
| Full harness | `agent.py` | Top-level Agent construction |

## Proposers

The proposer is pluggable. Currently supported:

| Proposer | Flag | How it works |
|---|---|---|
| OpenCode | `--proposer opencode` | Invokes `opencode run --dir <candidate> --format json` |
| Pi | `--proposer pi` | Invokes `pi --print --mode auto` |
| Any CLI | `--proposer custom-cli-command` | Invokes the command in the candidate directory |

You can set a proposer model separately from the agent model:

```bash
amh optimize --proposer opencode --proposer-model opencode-go/glm-5.1 --model openai/glm-5.1
```

## Judges

When Harbor reward files (`reward.txt`/`reward.json`) are present, they provide
deterministic pass/fail scoring. When reward files are absent, the judge scores
the ATIF trajectory instead. Three judge backends:

| Judge | Flag | How it works |
|---|---|---|
| LiteLLM | `--judge litellm` (default) | Any model via litellm |
| ADK agent | `--judge adk` | An ADK LlmAgent as judge |
| CLI | `--judge opencode` or `--judge pi` | Coding agent CLI as judge |

Scoring flow:
1. **Harbor reward exists** → use it (deterministic)
2. **No reward, judge provided** → judge scores the trajectory
3. **No reward, no judge** → task marked as failed (score 0.0)

```bash
# Default: litellm judge with gemini-2.5-flash
amh optimize --judge litellm --judge-model openai/glm-5 ...

# ADK judge
amh optimize --judge adk --judge-model gemini-2.5-flash ...

# CLI judge (OpenCode)
amh optimize --judge opencode ...
```

## Trace pipeline

All trace collection uses a single path — a per-task `FileSpanExporter` that
captures OTel spans emitted by the ADK agent and writes them to a JSON file:

```
ADK Agent (emits OTel spans via Python SDK)
       │
       ▼
FileSpanExporter (SimpleSpanProcessor, per-task)
       │
       ▼
evaluation/<task>/agent/otel_spans.json
       │
       ▼
OtelToAtifConverter.convert_file()
       │
       ▼
AtifTrajectory
       │
       ├── evaluation/<task>/trajectory.json
       └── traces/<task>.json  (mirrored for proposer access)
```

### How it works

1. **Before agent run** — `setup_file_exporter()` creates a `FileSpanExporter`
   pointing to `evaluation/<task>/agent/otel_spans.json` and registers a
   `SimpleSpanProcessor` with the global OTel `TracerProvider`.
2. **During agent run** — ADK emits spans through the standard OTel Python SDK.
   The `SimpleSpanProcessor` synchronously exports each span to the
   `FileSpanExporter`, which accumulates them in memory.
3. **After agent run** — `teardown_file_exporter()` flushes all accumulated
   spans to the JSON file and removes the processor from the provider.
4. **On read** — `_load_collector_span_file()` locates the file, then
   `OtelToAtifConverter.convert_file()` parses the OTel spans into an ATIF
   trajectory.

If no OTel spans are captured (e.g., the agent doesn't emit them), the trajectory
is created with only agent metadata (name, model). The `_ensure_user_instruction_step()`
function prepends the task instruction as a user step so judges and proposers
always see the full conversation.

### In Harbor containers

The same `FileSpanExporter` approach is used inside containers by `eval_one.py`.
The `AdkHarborAgent` shells out to:

```bash
python -m adk_meta_harness.eval_one \
    --harness /app/harness \
    --instruction "..." \
    --output /logs/agent \
    --model openai/glm-5.1
```

`eval_one` loads the ADK agent, sets up the `FileSpanExporter`, runs the agent,
flushes spans to `/logs/agent/otel_spans.json`, converts to ATIF, and writes
`trajectory.json` and `response.txt`.

No separate OTel Collector process is needed — the `FileSpanExporter` writes
directly to disk within the same Python process.

## CLI reference

```
amh optimize \
  --dataset PATH           # Harbor task directory (required)
  --initial-harness PATH   # Initial harness directory (required)
  --proposer [opencode|pi] # Proposer CLI (default: opencode)
  --proposer-model MODEL   # Model override for proposer
  --runner [local|harbor]  # Task runner (default: local)
  --judge [litellm|adk|opencode|pi|custom] # Judge backend (default: litellm)
  --judge-model MODEL      # Model for the judge (e.g. openai/glm-5)
  --model MODEL            # Model override for ADK agent (highest priority)
  --iterations N           # Number of iterations (default: 10)
  --holdout-ratio RATIO    # Fraction of tasks held out for gating (default: 0.3)
  --candidates-dir PATH    # Where to store candidates/traces/learnings
  --timeout SECS           # Per-task timeout in seconds (default: 300)

amh eval \
  --candidate PATH         # Harness candidate directory (required)
  --dataset PATH            # Harbor task directory (required)
  --runner [local|harbor]   # Task runner (default: local)
  --judge [litellm|adk|opencode|pi|custom] # Judge backend (default: litellm)
  --judge-model MODEL       # Model for the judge
  --model MODEL             # Model override for ADK agent
  --timeout SECS            # Per-task timeout in seconds (default: 300)
```

## Project structure

```
adk-meta-harness/
├── src/adk_meta_harness/
│   ├── cli.py                   # CLI: optimize, eval (also amh alias)
│   ├── outer_loop.py            # Propose → Evaluate → Gate → Repeat
│   ├── candidate.py             # Candidate harness representation
│   ├── gate.py                  # Holdout evaluation + keep/discard
│   ├── learnings.py             # learnings.md accumulator
│   ├── eval_one.py              # Single-task evaluator for Harbor containers
│   ├── harbor_adapter.py        # ADK agent runner, model precedence, task discovery
│   ├── runner/
│   │   ├── __init__.py           # get_runner() factory
│   │   ├── base.py              # TaskRunner protocol
│   │   ├── local.py             # LocalTaskRunner (in-process)
│   │   ├── harbor_runner.py     # HarborTaskRunner (Docker containers)
│   │   └── harbor_agent.py      # AdkHarborAgent (BaseAgent for Harbor)
│   ├── proposer/
│   │   ├── base.py              # ProposerProtocol
│   │   ├── coding_agent_cli.py  # Generic CLI adapter + PROPOSER template
│   │   ├── opencode.py          # OpenCode proposer
│   │   └── pi.py                # Pi proposer
│   ├── judge/
│   │   ├── base.py              # JudgeProtocol, JudgeResult
│   │   ├── litellm_judge.py     # Any model via litellm
│   │   ├── adk_judge.py         # ADK LlmAgent as judge
│   │   └── cli_judge.py         # CLI-based judge
│   └── trace/
│       ├── atif.py              # ATIF v1.4 data models
│       ├── otel_to_atif.py      # OTel spans → ATIF conversion
│       ├── file_exporter.py     # FileSpanExporter for per-task OTel span capture
│       └── harbor_reward.py     # reward.txt / reward.json parsing
├── docker/
│   └── adk-meta-harness.Dockerfile  # Base Docker image for Harbor runner
├── examples/
│   ├── vanilla/                 # Minimal baseline agent
│   ├── skills-enabled/          # Agent with adk-skills
│   ├── tool-search/             # Agent with adk-tool-search
│   └── deep-research/           # Agent with skills + tool search
├── tests/                        # 142 tests
├── PROPOSER.md                   # Template proposer directive
├── .env.example                  # Environment variable template
└── pyproject.toml               # Package config, amh entry point
```

## Related work

- [Meta-Harness](https://arxiv.org/abs/2603.28052) — The paper this is based on
- [canvas-org/meta-agent](https://github.com/canvas-org/meta-agent) — Open-source Meta-Harness for Claude SDK
- [GEPA](https://github.com/gepa-ai/gepa) — Reflective prompt evolution (complementary inner loop)
- [Harbor](https://github.com/harbor-framework/harbor) — Agent evaluation framework
- [agentskills.io](https://agentskills.io) — Open standard for agent skills

## License

Apache 2.0