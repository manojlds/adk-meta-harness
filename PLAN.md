# Plan: Strip Harbor, Own Task Format, Add Temporal Runner

## Motivation

Harbor provides container-based task execution (Docker, E2B, Modal, etc.) but
we don't need it:

- **Local runner** already runs the ADK agent in-process without containers.
- **Temporal runner** (planned) distributes evaluation as activities across
  workers — also no containers.
- Harbor's task format (`task.toml`, `instruction.md`, `fixtures/`, `tests/`)
  is good but we only use a subset and can own it.
- Harbor adds a heavyweight dependency for value we don't use.

## Goals

1. Remove the `harbor` dependency entirely.
2. Define our own task format (Harbor-inspired, convention over configuration).
3. Add `scripts/setup.sh` and `scripts/teardown.sh` lifecycle hooks.
4. Build a Temporal runner where the **entire optimization loop** runs as a
   durable workflow with per-task evaluation as concurrent activities.
5. Keep the local runner for development/testing (sequential, no infra needed).

## Non-goals

- Container-based isolation (use Temporal worker machines for isolation instead).
- Harbor compatibility — tasks will follow our own format. Users who need
  Harbor can convert.

---

## 1. Own Task Format

### Convention-based directory layout

```
my-task/
├── instruction.md            # Required: agent prompt
├── task.toml                 # Optional: metadata + overrides
├── fixtures/                 # Optional: files copied to work dir
├── scripts/
│   ├── setup.sh              # Optional: runs before agent (discovered by convention)
│   └── teardown.sh           # Optional: runs after verifier (discovered by convention)
└── tests/
    └── test.sh               # Optional: verifier (writes reward.txt)
```

Everything is discovered at well-known paths — no config needed.
`task.toml` is only for overrides and metadata.

### `task.toml` schema (simplified from Harbor)

```toml
[metadata]
description = "Read a file and report its contents"

[agent]
timeout_sec = 120              # per-task agent timeout override

[verifier]
timeout_sec = 120              # verifier timeout override

[scripts]
setup_timeout_sec = 60         # setup.sh timeout (default: 60)
teardown_timeout_sec = 60      # teardown.sh timeout (default: 60)

[env]                          # extra env vars for setup, agent, verifier
MY_VAR = "value"
```

Removed from Harbor's schema (not applicable without containers):
- `environment.build_timeout_sec`, `cpus`, `memory_mb`, `storage_mb`, `gpus`
- `environment.docker_image`, `allow_internet`, `mcp_servers`
- `solution.env`

### New module: `task.py`

```python
@dataclass
class TaskConfig:
    name: str
    path: Path                      # resolved task directory
    instruction: str
    agent_timeout: int = 300
    verifier_timeout: int = 300
    setup_timeout: int = 60
    teardown_timeout: int = 60
    env: dict[str, str]             # extra env vars from task.toml

    # Convention-discovered paths (None if not present)
    setup_script: Path | None
    teardown_script: Path | None
    verifier_script: Path | None
    fixtures_dir: Path | None

    @classmethod
    def from_path(cls, task_path: Path, name: str) -> TaskConfig: ...

def discover_tasks(tasks_dir: Path) -> list[TaskConfig]: ...
```

---

## 2. Refactor `harbor_adapter.py` → `task_executor.py`

The core agent-running logic is ours — rename and restructure:

### What stays (renamed)

| Old | New |
|-----|-----|
| `harbor_adapter.py` | `task_executor.py` |
| `_run_agent_on_task()` | `run_agent_on_task()` (public, used by Temporal activities) |
| `_prepare_task_workspace()` | `prepare_workspace()` |
| `_run_task_verifier()` | `run_verifier()` |
| `evaluate_candidate()` | `evaluate_candidate()` (uses TaskConfig) |
| `_load_adk_agent()` | `load_adk_agent()` (public, reused) |
| `_ensure_importable()` | `ensure_importable()` |
| `EvalResult`, `EvalOutput` | same names |

### What's added

| Function | Purpose |
|----------|---------|
| `run_setup()` | Execute `scripts/setup.sh` before agent |
| `run_teardown()` | Execute `scripts/teardown.sh` after verifier |
| `run_single_task()` | Full lifecycle: setup → agent → verifier → teardown → score |

`run_single_task()` is the atomic unit that both the local runner and
Temporal activity call:

```python
async def run_single_task(
    task: TaskConfig,
    candidate_dir: Path,
    model: str,
    output_dir: Path,
    judge: JudgeProtocol | None = None,
) -> EvalResult:
    work_dir = prepare_workspace(task, output_dir)
    run_setup(task, work_dir)                         # NEW
    result = await run_agent_on_task(...)
    write_trajectory(result, output_dir)
    run_verifier(task, output_dir, work_dir)
    score_result(result, output_dir, judge)
    run_teardown(task, work_dir)                      # NEW
    return result
```

### What's removed

| Removed | Reason |
|---------|--------|
| `HarborReward` class name | Rename to `Reward` |
| `harbor_reward.py` | Rename to `reward.py` |
| All Harbor references in docstrings | No longer relevant |

### Rename `trace/harbor_reward.py` → `trace/reward.py`

- `HarborReward` → `Reward`
- `parse_reward_dir` → `parse_reward_dir` (same name, updated docstrings)
- Update all imports

---

## 3. Files to Delete

| File | Reason |
|------|--------|
| `runner/harbor_runner.py` | Harbor Job API, requires `harbor` |
| `runner/harbor_agent.py` | Harbor BaseAgent subclass |
| `eval_one.py` | Harbor container entrypoint |
| `docker/adk-meta-harness.Dockerfile` | Harbor base image |
| `examples/*/tasks/*/environment/` | Dockerfiles, no longer needed |

---

## 4. Local Runner

Stays simple — calls `evaluate_candidate()` which now uses `TaskConfig`
and `run_single_task()`:

```python
class LocalTaskRunner:
    """In-process sequential runner. No containers, no infra.

    Tasks run sequentially because os.chdir is process-global.
    For parallel execution, use the Temporal runner.
    """

    async def evaluate(self, candidate_dir, tasks_dir, ...) -> EvalOutput:
        tasks = discover_tasks(tasks_dir)
        agent, app = load_adk_agent(candidate_dir, model)
        for task in tasks:
            result = await run_single_task(task, candidate_dir, model, ...)
            # partition into search/holdout
        return output
```

---

## 5. Temporal Runner

### Architecture

The **entire optimization loop** runs as a Temporal workflow. The CLI
starts the workflow and exits (fire-and-forget). Workers pick up
activities from a task queue.

```
amh optimize --runner temporal    →  starts OptimizeWorkflow (fire & forget)
amh worker                        →  runs a Temporal worker (long-lived)
amh status --workflow-id <id>     →  check workflow status (optional)
```

### Workflow: `OptimizeWorkflow`

```python
@workflow.defn
class OptimizeWorkflow:
    @workflow.run
    async def run(self, config: OptimizeInput) -> OptimizeOutput:
        # 1. Initialize baseline
        baseline = await workflow.execute_activity(
            init_baseline, config, ...)

        # 2. Evaluate baseline (fan-out)
        baseline_results = await self._eval_candidate(
            baseline, config.tasks, config.model)

        best = baseline
        for iteration in range(1, config.iterations + 1):
            # 3. Propose
            candidate = await workflow.execute_activity(
                propose_activity, best, iteration, ...)

            # 4. Validate
            valid = await workflow.execute_activity(
                validate_activity, candidate, ...)
            if not valid:
                continue

            # 5. Evaluate (fan-out)
            results = await self._eval_candidate(
                candidate, config.tasks, config.model)

            # 6. Gate
            kept = await workflow.execute_activity(
                gate_activity, results, best, ...)

            # 7. Learn
            await workflow.execute_activity(
                learn_activity, iteration, results, kept, ...)

            if kept:
                best = candidate

        return OptimizeOutput(best=best, ...)

    async def _eval_candidate(self, candidate, tasks, model):
        """Fan out one activity per task, run concurrently."""
        return await asyncio.gather(*[
            workflow.execute_activity(
                eval_task_activity,
                EvalTaskInput(candidate=candidate, task=task, model=model),
                start_to_close_timeout=timedelta(seconds=task.timeout),
            )
            for task in tasks
        ])
```

### Activities

| Activity | What it does | Timeout |
|----------|-------------|---------|
| `init_baseline` | Copy initial harness, create candidate v0000 | 60s |
| `propose_activity` | Run proposer CLI on candidate dir | 30min |
| `validate_activity` | `validate_candidate()` | 30s |
| `eval_task_activity` | `run_single_task()` — the atomic unit | per-task |
| `gate_activity` | `gate_decision()` | 5s |
| `learn_activity` | Update `learnings.md` | 5s |

`eval_task_activity` is the only compute-heavy activity. All others are
lightweight and can run on the orchestrator worker.

### Worker

```python
# runner/temporal_worker.py
async def run_worker(
    server_url: str = "localhost:7233",
    task_queue: str = "amh-tasks",
):
    client = await Client.connect(server_url)
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[OptimizeWorkflow],
        activities=[
            init_baseline,
            propose_activity,
            validate_activity,
            eval_task_activity,
            gate_activity,
            learn_activity,
        ],
    )
    await worker.run()
```

### CLI: `amh worker`

```bash
# Start a worker (blocks, runs until killed)
amh worker --server localhost:7233 --task-queue amh-tasks

# Start optimization (fire and forget)
amh optimize \
    --runner temporal \
    --server localhost:7233 \
    --dataset ./tasks \
    --initial-harness ./harness \
    --iterations 10
# Prints workflow ID and exits immediately
```

### Shared filesystem requirement

Temporal activities need access to the candidate directories, task
fixtures, and learnings. Options:

- **Shared filesystem** (NFS, EFS) — simplest, workers mount the same path.
- **S3/GCS sync** — activities download task/harness at start, upload
  results at end. More complex but works across regions.

The plan starts with shared filesystem (same as local runner).

### Serialization

Temporal requires serializable inputs/outputs. Use `dataclasses-json` or
manual `to_dict`/`from_dict` (which we already have on ATIF types):

- `EvalTaskInput` — candidate path (str), task config (dict), model (str)
- `EvalTaskOutput` — EvalResult serialized as dict
- `OptimizeInput` — paths, model, iterations, etc.
- `OptimizeOutput` — best candidate path, scores, history

---

## 6. Dependency Changes

### `pyproject.toml`

```diff
 dependencies = [
     "google-adk>=1.0.0",
-    "adk-skills-agent>=0.1.0",
-    "adk-tool-search>=0.1.0",
-    "harbor>=0.3.0",
     "litellm>=1.0",
     "python-dotenv>=1.0",
     "pyyaml>=6.0",
     "rich>=13.0",
+    "temporalio>=1.9",
 ]
+
+[project.optional-dependencies]
+skills = ["adk-skills-agent>=0.1.0", "adk-tool-search>=0.1.0"]
```

`adk-skills-agent` and `adk-tool-search` move to optional — they're only
needed if the harness uses skills/tool-search (the examples do, but the
core framework doesn't require them).

---

## 7. Updated Project Structure

```
adk-meta-harness/
├── src/adk_meta_harness/
│   ├── cli.py                   # CLI: optimize, eval, worker
│   ├── outer_loop.py            # Local-mode outer loop (unchanged)
│   ├── candidate.py             # Candidate management
│   ├── gate.py                  # Gate logic
│   ├── learnings.py             # Learnings accumulator
│   ├── validate.py              # Candidate validation
│   ├── task.py                  # NEW: TaskConfig, discover_tasks
│   ├── task_executor.py         # RENAMED from harbor_adapter.py
│   ├── runner/
│   │   ├── __init__.py          # get_runner: local, temporal
│   │   ├── base.py              # TaskRunner protocol
│   │   ├── local.py             # LocalTaskRunner
│   │   └── temporal_runner.py   # NEW: OptimizeWorkflow + activities
│   ├── proposer/                # unchanged
│   ├── judge/                   # unchanged
│   └── trace/
│       ├── atif.py              # unchanged
│       ├── otel_to_atif.py      # unchanged
│       ├── file_exporter.py     # unchanged
│       └── reward.py            # RENAMED from harbor_reward.py
├── examples/                    # Updated: no environment/ dirs
├── tests/                       # Updated
├── PROPOSER.md
├── pyproject.toml               # harbor removed, temporalio added
└── README.md                    # Updated
```

### Deleted files

- `runner/harbor_runner.py`
- `runner/harbor_agent.py`
- `eval_one.py`
- `docker/adk-meta-harness.Dockerfile`
- `examples/*/tasks/*/environment/` directories

---

## 8. Example Task After Migration

### Before (Harbor format)

```
read-file/read-file/
├── instruction.md
├── task.toml              # Has [environment] section
├── fixtures/hello.txt
├── tests/test.sh
├── environment/
│   └── Dockerfile         # echo "hello world" > /app/hello.txt
└── README.md
```

### After (our format)

```
read-file/
├── instruction.md
├── task.toml              # Simplified, no [environment]
├── fixtures/hello.txt     # Same
├── scripts/
│   └── setup.sh           # Optional: any pre-agent work
├── tests/
│   └── test.sh            # Same verifier
└── README.md
```

The nested `read-file/read-file/` structure flattens to `read-file/`
(we keep backward compat for nested during migration but prefer flat).

---

## 9. Execution Order

### Phase 1 — Foundation (sequential, interconnected)

1. Create `task.py` with `TaskConfig` and `discover_tasks`
2. Rename `harbor_adapter.py` → `task_executor.py`, add setup/teardown lifecycle
3. Rename `trace/harbor_reward.py` → `trace/reward.py`, `HarborReward` → `Reward`
4. Update local runner to use new modules
5. Update `runner/__init__.py` — remove harbor
6. Delete Harbor-specific files
7. Update `pyproject.toml` — remove `harbor`, add `temporalio`
8. Update imports across all files
9. Fix tests

### Phase 2 — Temporal Runner (can start once Phase 1 compiles)

1. Create `runner/temporal_runner.py` with workflow + activities
2. Add `amh worker` subcommand to CLI
3. Add `--runner temporal` to CLI
4. Add Temporal-specific tests

### Phase 3 — Cleanup

1. Flatten example task directories (remove nesting)
2. Remove `environment/` from examples
3. Add `scripts/setup.sh` example to at least one task
4. Update README.md
5. Update PROPOSER.md
