"""Harbor ADK adapter — runs an ADK agent on Harbor tasks.

Uses ADK's AgentLoader for agent discovery and ATIF for trace collection.
For each task, this adapter runs task verification scripts (tests/test.sh)
that write Harbor reward files, then parses those rewards. If no reward files
are produced, it can fall back to a judge over ATIF trajectories.

Model precedence:
    --model CLI flag (runtime override) > config.yaml (harness) > agent default

Trace pipeline:
    ADK Agent (OTel spans) → OtelToAtifConverter → AtifTrajectory → trajectory.json
    Harbor verifier → reward.txt/reward.json → HarborReward
    (or) Judge → JudgeResult → score
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from adk_meta_harness.trace.atif import AtifStep, AtifTrajectory
from adk_meta_harness.trace.harbor_reward import HarborReward, parse_reward_dir
from adk_meta_harness.trace.otel_to_atif import OtelToAtifConverter

if TYPE_CHECKING:
    from adk_meta_harness.judge.base import JudgeProtocol


def load_model_from_config(candidate_dir: Path) -> str | None:
    """Read the model from the candidate's config.yaml.

    Returns None if config.yaml doesn't exist or has no model key.
    """
    config_path = candidate_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        data = yaml.safe_load(config_path.read_text())
        if isinstance(data, dict) and "model" in data:
            return str(data["model"])
    except Exception:
        pass
    return None


def set_model_in_config(candidate_dir: Path, model: str) -> None:
    """Write the model into the candidate's config.yaml.

    Creates config.yaml if it doesn't exist.
    """
    config_path = candidate_dir / "config.yaml"
    data: dict = {}
    if config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text()) or {}
        except Exception:
            data = {}
    data["model"] = model
    config_path.write_text(yaml.dump(data, default_flow_style=False))


@dataclass
class EvalResult:
    """Result from evaluating an agent on a single task."""

    task_name: str
    passed: bool
    score: float
    trajectory: AtifTrajectory | None = None
    reward: HarborReward | None = None
    error: str | None = None

    @property
    def trace_summary(self) -> str:
        """Human-readable trace summary for quick diagnosis."""
        if not self.trajectory or not self.trajectory.steps:
            return self.error or "No trace available"
        steps = self.trajectory.steps
        tool_calls = sum(len(s.tool_calls) for s in steps)
        lines = [f"Task: {self.task_name}"]
        lines.append(f"Steps: {len(steps)}, Tool calls: {tool_calls}")
        if self.trajectory.final_metrics:
            fm = self.trajectory.final_metrics
            lines.append(
                f"Tokens: {fm.total_prompt_tokens}+{fm.total_completion_tokens}, "
                f"Cost: ${fm.total_cost_usd:.4f}"
            )
        for step in steps:
            if step.message:
                preview = step.message[:100]
                lines.append(f"  [{step.source}] {preview}")
            for tc in step.tool_calls:
                lines.append(f"  [tool_call] {tc.function_name}")
        return "\n".join(lines)


@dataclass
class EvalOutput:
    """Complete output from evaluating a candidate on all tasks."""

    search_results: list[EvalResult] = field(default_factory=list)
    holdout_results: list[EvalResult] = field(default_factory=list)

    @property
    def search_score(self) -> float:
        if not self.search_results:
            return 0.0
        return sum(1 for r in self.search_results if r.passed) / len(self.search_results)

    @property
    def holdout_score(self) -> float:
        if not self.holdout_results:
            return self.search_score
        return sum(1 for r in self.holdout_results if r.passed) / len(self.holdout_results)

    @property
    def combined_score(self) -> float:
        all_results = self.search_results + self.holdout_results
        if not all_results:
            return 0.0
        return sum(1 for r in all_results if r.passed) / len(all_results)


async def evaluate_candidate(
    candidate_dir: Path,
    tasks_dir: Path,
    model: str | None = None,
    timeout: int = 300,
    search_task_names: list[str] | None = None,
    holdout_task_names: list[str] | None = None,
    output_dir: Path | None = None,
    judge: JudgeProtocol | None = None,
) -> EvalOutput:
    """Evaluate a candidate harness on search and holdout tasks.

    Scoring logic:
    1. Run task verifier script (tests/test.sh) when present
    2. If Harbor reward files (reward.txt/reward.json) exist, use them
    3. Else if a judge is provided, score the trajectory with the judge
    4. Else mark as failed (score 0.0)

    Args:
        candidate_dir: Path to the candidate harness directory.
        tasks_dir: Path to Harbor task definitions.
        model: Runtime override for the model. If provided, writes to
            config.yaml before loading. If None, reads from config.yaml
            or uses the agent's default.
        timeout: Timeout per task in seconds.
        search_task_names: Task names for the search set.
        holdout_task_names: Task names for the holdout set.
        output_dir: Directory to write trajectory.json and reward files.
            Defaults to candidate_dir / "evaluation".
        judge: Optional judge for scoring traces when Harbor rewards absent.

    Returns:
        EvalOutput with search and holdout results.
    """
    if model is not None:
        set_model_in_config(candidate_dir, model)

    resolved_model = load_model_from_config(candidate_dir) or model or "gemini-2.5-flash"

    output_dir = output_dir or candidate_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    agent, app = _load_adk_agent(candidate_dir, resolved_model)
    converter = OtelToAtifConverter()

    all_tasks = _discover_tasks(tasks_dir)
    search_set = search_task_names or [t for t in all_tasks]
    holdout_set = holdout_task_names or []

    output = EvalOutput()

    for task_name in all_tasks:
        task_path = _resolve_task_path(tasks_dir, task_name)
        if not task_path.exists():
            continue
        instruction = _read_instruction(task_path)
        task_logs_dir = output_dir / task_name
        task_logs_dir.mkdir(parents=True, exist_ok=True)
        work_dir = _prepare_task_workspace(task_path, task_logs_dir)

        result = await _run_agent_on_task(
            agent=agent,
            app=app,
            converter=converter,
            instruction=instruction,
            task_name=task_name,
            timeout=timeout,
            work_dir=work_dir,
            task_logs_dir=task_logs_dir,
        )

        # Persist trajectory and last agent response for test scripts.
        if result.trajectory:
            traj_path = task_logs_dir / "trajectory.json"
            result.trajectory.to_json_file(traj_path)
        _write_agent_response(task_logs_dir, _extract_last_agent_message(result.trajectory))

        # 1) Execute task verifier script if present.
        verifier_error = _run_task_verifier(
            task_path=task_path,
            task_logs_dir=task_logs_dir,
            work_dir=work_dir,
            timeout=timeout,
        )
        if verifier_error:
            if result.error:
                result.error = f"{result.error}; verifier: {verifier_error}"
            else:
                result.error = f"verifier: {verifier_error}"

        scored = False

        # 2) Harbor reward files from verifier scripts (authoritative)
        if _has_reward_file(task_logs_dir):
            reward = parse_reward_dir(task_logs_dir)
            result.reward = reward
            result.score = reward.score
            result.passed = reward.passed
            scored = True

        # 3) Judge fallback only if still unscored
        if not scored and judge is not None and result.trajectory is not None:
            trace_text = result.trace_summary
            judge_result = await judge.judge_trace(
                task_instruction=instruction,
                trace=trace_text,
                task_name=task_name,
            )
            result.score = judge_result.score
            result.passed = judge_result.score >= 0.5

        if task_name in holdout_set:
            output.holdout_results.append(result)
        else:
            output.search_results.append(result)

    return output


def _has_reward_file(logs_dir: Path) -> bool:
    """Return True if a Harbor reward file exists under logs_dir."""
    return any(
        p.exists()
        for p in (
            logs_dir / "verifier" / "reward.json",
            logs_dir / "verifier" / "reward.txt",
            logs_dir / "reward.json",
            logs_dir / "reward.txt",
        )
    )


def _prepare_task_workspace(task_path: Path, task_logs_dir: Path) -> Path:
    """Prepare a per-task local workspace.

    Copies optional fixtures from task_path/fixtures into task_logs_dir/work.
    """
    work_dir = (task_logs_dir / "work").resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    fixtures_dir = task_path / "fixtures"
    if fixtures_dir.exists() and fixtures_dir.is_dir():
        for item in fixtures_dir.iterdir():
            dest = work_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    return work_dir


def _extract_last_agent_message(trajectory: AtifTrajectory | None) -> str:
    """Extract the last non-empty agent message from a trajectory."""
    if not trajectory or not trajectory.steps:
        return ""
    for step in reversed(trajectory.steps):
        if step.source == "agent" and step.message:
            msg = step.message.strip()
            if msg:
                return msg
    return ""


def _write_agent_response(task_logs_dir: Path, response_text: str) -> None:
    """Write agent response in Harbor-like logs location."""
    agent_dir = task_logs_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "response.txt").write_text(response_text or "")


def _run_task_verifier(
    task_path: Path,
    task_logs_dir: Path,
    work_dir: Path,
    timeout: int,
) -> str | None:
    """Run task verification script (tests/test.sh) if present.

    Returns an error string if verifier execution failed, else None.
    """
    test_script = (task_path / "tests" / "test.sh").resolve()
    if not test_script.exists():
        return None

    env = os.environ.copy()
    logs_root = task_logs_dir.resolve()
    logs_dir = str(logs_root)
    work_dir = work_dir.resolve()
    env.update(
        {
            "LOGS_DIR": logs_dir,
            "REWARD_DIR": str(logs_root / "verifier"),
            "AGENT_DIR": str(logs_root / "agent"),
            "AGENT_RESPONSE_FILE": str(logs_root / "agent" / "response.txt"),
            "WORK_DIR": str(work_dir),
        }
    )

    proc = subprocess.run(
        ["bash", str(test_script)],
        cwd=str(work_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode == 0:
        return None
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    msg = stderr or stdout or f"Verifier exited with code {proc.returncode}"
    return msg[:1000]


def _ensure_importable(candidate_dir: Path) -> None:
    """Ensure the candidate directory is importable by ADK's AgentLoader.

    AgentLoader expects:
    - The directory to be a Python package (has __init__.py)
    - agent.py to expose a ``root_agent`` attribute (not ``agent``)

    This function creates __init__.py if missing and patches agent.py
    to add a ``root_agent`` alias if the module uses ``agent`` instead.

    It also adds the candidate directory to sys.path so that
    sub-packages (e.g. tools/, skills/) can be imported with
    absolute imports like ``from tools import ...``.
    """
    import sys

    candidate_dir = candidate_dir.resolve()
    candidate_str = str(candidate_dir)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

    init_path = candidate_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")

    agent_path = candidate_dir / "agent.py"
    if agent_path.exists():
        content = agent_path.read_text()
        if "root_agent" not in content and "agent =" in content:
            agent_path.write_text(content + "\nroot_agent = agent\n")


def _load_adk_agent(
    candidate_dir: Path,
    model: str = "gemini-2.5-flash",
) -> tuple:
    """Load the ADK agent using the official AgentLoader.

    Handles all ADK patterns:
    - agent.py with root_agent attribute
    - agent.py with app attribute (App instance)
    - __init__.py with root_agent or app
    - root_agent.yaml config

    Before loading, ensures the directory is importable (__init__.py)
    and agent.py exposes root_agent.

    The model is only set if the loaded agent doesn't already have one.
    This allows config.yaml and agent.py to control the model, with
    the CLI --model flag as a runtime override.

    Returns:
        Tuple of (root_agent, app).
    """
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.apps.app import App
    from google.adk.cli.utils.agent_loader import AgentLoader

    _ensure_importable(candidate_dir)

    parent_dir = str(candidate_dir.parent)
    agent_name = candidate_dir.name

    loader = AgentLoader(agents_dir=parent_dir)
    agent_or_app = loader.load_agent(agent_name)

    if isinstance(agent_or_app, App):
        app = agent_or_app
        root_agent = app.root_agent
    elif isinstance(agent_or_app, BaseAgent):
        root_agent = agent_or_app
        app = App(name=agent_name, root_agent=root_agent)
    else:
        msg = f"Expected BaseAgent or App from {candidate_dir}, got {type(agent_or_app)}"
        raise TypeError(msg)

    if not getattr(root_agent, "model", None):
        root_agent.model = model

    return root_agent, app


def _discover_tasks(tasks_dir: Path) -> list[str]:
    """Discover Harbor task directories.

    Handles both flat and nested directory structures:
    - Flat: tasks_dir/read-file/instruction.md
    - Nested (Harbor default): tasks_dir/read-file/read-file/instruction.md

    Returns task names that have either an instruction.md or task.toml.
    For nested structures, returns the outer directory name (e.g. "read-file").
    """
    if not tasks_dir.exists():
        return []
    tasks = []
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        # Check flat structure: tasks_dir/task-name/instruction.md
        if (d / "instruction.md").exists() or (d / "task.toml").exists():
            tasks.append(d.name)
        # Check nested structure: tasks_dir/task-name/task-name/instruction.md
        elif (d / d.name / "instruction.md").exists() or (d / d.name / "task.toml").exists():
            tasks.append(d.name)
    return tasks


def _resolve_task_path(tasks_dir: Path, task_name: str) -> Path:
    """Resolve the actual task directory.

    Harbor tasks can be nested (task-name/task-name/) or flat (task-name/).
    Returns the path containing instruction.md, task.toml, etc.
    """
    flat = tasks_dir / task_name
    nested = tasks_dir / task_name / task_name
    if (flat / "instruction.md").exists() or (flat / "task.toml").exists():
        return flat
    if (nested / "instruction.md").exists() or (nested / "task.toml").exists():
        return nested
    return flat


def _read_instruction(task_path: Path) -> str:
    """Read the task instruction."""
    instruction_file = task_path / "instruction.md"
    if instruction_file.exists():
        return instruction_file.read_text()
    return ""


_OTEL_CAPTURE_EXPORTER = None
_OTEL_CAPTURE_PROVIDER = None


def _ensure_otel_capture() -> tuple[Any | None, Any | None]:
    """Ensure an in-memory OTel span capture pipeline is available.

    Returns (exporter, provider). If OpenTelemetry SDK is unavailable,
    returns (None, None).
    """
    global _OTEL_CAPTURE_EXPORTER, _OTEL_CAPTURE_PROVIDER

    if _OTEL_CAPTURE_EXPORTER is not None and _OTEL_CAPTURE_PROVIDER is not None:
        return _OTEL_CAPTURE_EXPORTER, _OTEL_CAPTURE_PROVIDER

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except Exception:
        return None, None

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()

    if not isinstance(provider, SDKTracerProvider):
        try:
            provider = SDKTracerProvider()
            trace.set_tracer_provider(provider)
        except Exception:
            return None, None

    if not hasattr(provider, "add_span_processor"):
        return None, None

    try:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    except Exception:
        return None, None
    _OTEL_CAPTURE_EXPORTER = exporter
    _OTEL_CAPTURE_PROVIDER = provider
    return exporter, provider


def _readable_span_to_dict(span: Any) -> dict[str, Any]:
    """Convert an OpenTelemetry ReadableSpan into a dict for converter."""
    attrs = {}
    for k, v in dict(getattr(span, "attributes", {}) or {}).items():
        attrs[str(k)] = _normalize_otel_value(v)

    span_id = ""
    parent_span_id = ""
    context = getattr(span, "context", None)
    if context is not None and getattr(context, "span_id", None) is not None:
        span_id = format(context.span_id, "x")

    parent = getattr(span, "parent", None)
    if parent is not None and getattr(parent, "span_id", None) is not None:
        parent_span_id = format(parent.span_id, "x")

    return {
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "start_time": int(getattr(span, "start_time", 0) or 0),
        "name": str(getattr(span, "name", "")),
        "attributes": attrs,
    }


def _normalize_otel_value(value: Any) -> Any:
    """Normalize OTel attribute values into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, dict):
        return {str(k): _normalize_otel_value(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_normalize_otel_value(v) for v in value]
    return str(value)


def _load_collector_span_file(task_logs_dir: Path, task_name: str) -> Path | None:
    """Locate collector-exported OTel span file for a task, if present."""
    env_dir = os.getenv("AMH_OTEL_SPANS_DIR", "").strip()
    env_file = os.getenv("AMH_OTEL_SPANS_FILE", "").strip()

    candidates = [
        task_logs_dir / "agent" / "otel_spans.json",
        task_logs_dir / "otel_spans.json",
    ]
    if env_dir:
        candidates.append(Path(env_dir) / f"{task_name}.json")
    if env_file:
        candidates.append(Path(env_file))

    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _ensure_user_instruction_step(
    trajectory: AtifTrajectory,
    instruction: str,
) -> AtifTrajectory:
    """Ensure the trajectory includes the user task input as a user step."""
    prompt = instruction.strip()
    if not prompt:
        return trajectory

    for step in trajectory.steps:
        if step.source == "user" and step.message.strip() == prompt:
            return trajectory

    user_step = AtifStep(
        step_id="step-user-input",
        timestamp="",
        source="user",
        message=prompt,
    )
    trajectory.steps = [user_step, *trajectory.steps]
    trajectory.compute_final_metrics()
    return trajectory


async def _run_agent_on_task(
    agent,
    app,
    converter: OtelToAtifConverter,
    instruction: str,
    task_name: str,
    timeout: int,
    work_dir: Path | None = None,
    task_logs_dir: Path | None = None,
) -> EvalResult:
    """Run an ADK agent on a single task and collect ATIF trajectory.

    Preferred trace path: OTel spans → ATIF.
    Fallback trace path: ADK event stream → ATIF.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    run_error: Exception | None = None
    events = []

    exporter, provider = _ensure_otel_capture()
    if exporter is not None:
        exporter.clear()

    session_service = InMemorySessionService()
    runner = Runner(
        app=app,
        session_service=session_service,
    )

    # Create session before running
    await session_service.create_session(
        app_name=getattr(app, "name", "adk-meta-harness"),
        user_id="meta_harness",
        session_id=task_name,
    )

    content = types.Content(parts=[types.Part(text=instruction)], role="user")

    previous_cwd = Path.cwd()
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(work_dir)
    try:
        async for event in runner.run_async(
            user_id="meta_harness",
            session_id=task_name,
            new_message=content,
        ):
            events.append(event)
    except Exception as e:
        run_error = e
    finally:
        if work_dir is not None:
            os.chdir(previous_cwd)

    # Prefer collector-exported span file when available.
    trajectory: AtifTrajectory | None = None
    span_file = None
    if task_logs_dir is not None:
        span_file = _load_collector_span_file(task_logs_dir, task_name)
    if span_file is not None:
        try:
            trajectory = converter.convert_file(span_file)
        except Exception:
            trajectory = None

    # Otherwise, use in-memory OTel capture from this run.
    if trajectory is None and exporter is not None:
        try:
            if provider is not None and hasattr(provider, "force_flush"):
                provider.force_flush()
            spans = [_readable_span_to_dict(s) for s in exporter.get_finished_spans()]
            if spans:
                trajectory = converter.convert_spans(spans)
        except Exception:
            trajectory = None

    # Final fallback: convert ADK events directly.
    if trajectory is None:
        trajectory = converter.adk_events_to_atif(
            events,
            agent_name=getattr(agent, "name", "adk-agent"),
            model_name=getattr(agent, "model", ""),
        )

    # Ensure user instruction is present for judges/proposers.
    trajectory = _ensure_user_instruction_step(trajectory, instruction)

    if run_error and not trajectory.steps:
        trajectory.steps.append(
            AtifStep(
                step_id="step-error",
                timestamp="",
                source="system",
                message=f"Run error: {run_error}",
            )
        )
        trajectory.compute_final_metrics()

    return EvalResult(
        task_name=task_name,
        passed=False,
        score=0.0,
        trajectory=trajectory,
        error=str(run_error) if run_error else None,
    )
