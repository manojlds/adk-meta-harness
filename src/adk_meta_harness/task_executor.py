"""Task executor — runs an ADK agent on local tasks.

Uses ADK's AgentLoader for agent discovery and ATIF for trace collection.
For each task, this executor runs lifecycle hooks and verification scripts
that write reward files. If no reward files are produced, it can fall back to
a judge over ATIF trajectories.

Model precedence:
    --model CLI flag (runtime override) > config.yaml (harness) > agent default

Trace pipeline:
    ADK Agent → OTel SDK → FileSpanExporter → otel_spans.json → ATIF
    verifier → reward.txt/reward.json → Reward
    (or) Judge → JudgeResult → score
"""

from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from adk_meta_harness.task import (
    DEFAULT_AGENT_TIMEOUT_SEC,
    TaskConfig,
    discover_tasks,
    read_instruction,
    resolve_task_path,
)
from adk_meta_harness.trace.atif import AtifAgent, AtifStep, AtifTrajectory
from adk_meta_harness.trace.otel_to_atif import OtelToAtifConverter
from adk_meta_harness.trace.reward import Reward, parse_reward_dir

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
    reward: Reward | None = None
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
    timeout: int = DEFAULT_AGENT_TIMEOUT_SEC,
    search_task_names: list[str] | None = None,
    holdout_task_names: list[str] | None = None,
    output_dir: Path | None = None,
    judge: JudgeProtocol | None = None,
) -> EvalOutput:
    """Evaluate a candidate harness on search and holdout tasks.

    Scoring logic:
    1. Run task verifier script (tests/test.sh) when present
    2. If reward files (reward.txt/reward.json) exist, use them
    3. Else if a judge is provided, score the trajectory with the judge
    4. Else mark as failed (score 0.0)

    Args:
        candidate_dir: Path to the candidate harness directory.
        tasks_dir: Path to task definitions.
        model: Runtime override for the model. If provided, writes to
            config.yaml before loading. If None, reads from config.yaml
            or uses the agent's default.
        timeout: Timeout per task in seconds.
        search_task_names: Task names for the search set.
        holdout_task_names: Task names for the holdout set.
        output_dir: Directory to write trajectory.json and reward files.
            Defaults to candidate_dir / "evaluation".
        judge: Optional judge for scoring traces when reward files are absent.

    Returns:
        EvalOutput with search and holdout results.
    """
    if model is not None:
        set_model_in_config(candidate_dir, model)

    resolved_model = load_model_from_config(candidate_dir) or model or "gemini-2.5-flash"

    output_dir = output_dir or candidate_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    agent, app = load_adk_agent(candidate_dir, resolved_model)
    converter = OtelToAtifConverter()

    all_tasks = discover_tasks(tasks_dir)
    holdout_set = set(holdout_task_names or [])
    search_set = set(search_task_names or [])

    output = EvalOutput()

    for task in all_tasks:
        if search_set and task.name not in search_set and task.name not in holdout_set:
            continue

        task_to_run = task
        if timeout != DEFAULT_AGENT_TIMEOUT_SEC:
            task_to_run = replace(task, agent_timeout=timeout, verifier_timeout=timeout)

        result = await run_single_task(
            task=task_to_run,
            candidate_dir=candidate_dir,
            model=resolved_model,
            output_dir=output_dir,
            judge=judge,
            agent=agent,
            app=app,
            converter=converter,
        )

        if task.name in holdout_set:
            output.holdout_results.append(result)
        else:
            output.search_results.append(result)

    return output


async def run_single_task(
    task: TaskConfig,
    candidate_dir: Path,
    model: str,
    output_dir: Path,
    judge: JudgeProtocol | None = None,
    *,
    agent=None,
    app=None,
    converter: OtelToAtifConverter | None = None,
) -> EvalResult:
    """Run one task through setup → agent → verifier → teardown."""
    task_logs_dir = output_dir / task.name
    task_logs_dir.mkdir(parents=True, exist_ok=True)

    run_converter = converter or OtelToAtifConverter()
    run_agent = agent
    run_app = app
    if run_agent is None or run_app is None:
        run_agent, run_app = load_adk_agent(candidate_dir, model)

    result = EvalResult(task_name=task.name, passed=False, score=0.0)
    work_dir = prepare_workspace(task, task_logs_dir)

    setup_error = await run_setup(task, task_logs_dir, work_dir)
    if setup_error:
        result.error = f"setup: {setup_error}"
    else:
        result = await run_agent_on_task(
            agent=run_agent,
            app=run_app,
            converter=run_converter,
            instruction=task.instruction,
            task_name=task.name,
            timeout=task.agent_timeout,
            work_dir=work_dir,
            task_logs_dir=task_logs_dir,
            extra_env=task.env,
        )

        if result.trajectory:
            result.trajectory.to_json_file(task_logs_dir / "trajectory.json")
        _write_agent_response(task_logs_dir, _extract_last_agent_message(result.trajectory))

        verifier_error = await run_verifier(task, task_logs_dir, work_dir)
        if verifier_error:
            result.error = _merge_errors(result.error, f"verifier: {verifier_error}")

        scored = False
        if _has_reward_file(task_logs_dir):
            reward = parse_reward_dir(task_logs_dir)
            result.reward = reward
            result.score = reward.score
            result.passed = reward.passed
            scored = True

        if not scored and judge is not None and result.trajectory is not None:
            trace_text = result.trace_summary
            judge_result = await judge.judge_trace(
                task_instruction=task.instruction,
                trace=trace_text,
                task_name=task.name,
            )
            result.score = judge_result.score
            result.passed = judge_result.score >= 0.5

    teardown_error = await run_teardown(task, work_dir, task_logs_dir)
    if teardown_error:
        result.error = _merge_errors(result.error, f"teardown: {teardown_error}")

    return result


def _has_reward_file(logs_dir: Path) -> bool:
    """Return True if a reward file exists under logs_dir."""
    return any(
        p.exists()
        for p in (
            logs_dir / "verifier" / "reward.json",
            logs_dir / "verifier" / "reward.txt",
            logs_dir / "reward.json",
            logs_dir / "reward.txt",
        )
    )


def _merge_errors(current: str | None, new_error: str | None) -> str | None:
    if not new_error:
        return current
    if not current:
        return new_error
    return f"{current}; {new_error}"


def prepare_workspace(task: TaskConfig, task_logs_dir: Path) -> Path:
    """Prepare a per-task local workspace.

    Copies optional fixtures into task_logs_dir/work.
    """
    work_dir = (task_logs_dir / "work").resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if task.fixtures_dir and task.fixtures_dir.exists() and task.fixtures_dir.is_dir():
        fixtures_dir = task.fixtures_dir
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
    """Write the final agent response in the logs location."""
    agent_dir = task_logs_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "response.txt").write_text(response_text or "")


async def run_setup(
    task: TaskConfig,
    task_logs_dir: Path,
    work_dir: Path,
) -> str | None:
    """Run scripts/setup.sh if present.

    Returns an error string if setup execution failed, else None.
    """
    if task.setup_script is None or not task.setup_script.exists():
        return None
    return await _run_script(
        script_path=task.setup_script,
        task=task,
        task_logs_dir=task_logs_dir,
        work_dir=work_dir,
        timeout=task.setup_timeout,
        step_name="setup",
    )


async def run_verifier(
    task: TaskConfig,
    task_logs_dir: Path,
    work_dir: Path,
) -> str | None:
    """Run tests/test.sh if present.

    Returns an error string if verifier execution failed, else None.
    """
    if task.verifier_script is None or not task.verifier_script.exists():
        return None
    return await _run_script(
        script_path=task.verifier_script,
        task=task,
        task_logs_dir=task_logs_dir,
        work_dir=work_dir,
        timeout=task.verifier_timeout,
        step_name="verifier",
    )


async def run_teardown(
    task: TaskConfig,
    work_dir: Path,
    task_logs_dir: Path,
) -> str | None:
    """Run scripts/teardown.sh if present."""
    if task.teardown_script is None or not task.teardown_script.exists():
        return None
    return await _run_script(
        script_path=task.teardown_script,
        task=task,
        task_logs_dir=task_logs_dir,
        work_dir=work_dir,
        timeout=task.teardown_timeout,
        step_name="teardown",
    )


async def _run_script(
    script_path: Path,
    task: TaskConfig,
    task_logs_dir: Path,
    work_dir: Path,
    timeout: int,
    step_name: str,
) -> str | None:
    """Run one task lifecycle script asynchronously and return an error on failure."""
    env = _build_script_env(task, task_logs_dir, work_dir)

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            str(script_path),
            cwd=str(work_dir.resolve()),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            with suppress(Exception):
                await proc.communicate()
            return f"{step_name} timed out after {timeout}s"
    except Exception as exc:
        return f"{step_name} failed to start: {exc}"

    if proc.returncode == 0:
        return None

    stderr = (stderr_bytes or b"").decode(errors="replace").strip()
    stdout = (stdout_bytes or b"").decode(errors="replace").strip()
    msg = stderr or stdout or f"{step_name} exited with code {proc.returncode}"
    return msg[:1000]


def _build_script_env(task: TaskConfig, task_logs_dir: Path, work_dir: Path) -> dict[str, str]:
    logs_root = task_logs_dir.resolve()
    work_root = work_dir.resolve()
    env = os.environ.copy()
    env.update(
        {
            "LOGS_DIR": str(logs_root),
            "REWARD_DIR": str(logs_root / "verifier"),
            "AGENT_DIR": str(logs_root / "agent"),
            "AGENT_RESPONSE_FILE": str(logs_root / "agent" / "response.txt"),
            "WORK_DIR": str(work_root),
        }
    )
    env.update(task.env)
    return env


def ensure_importable(candidate_dir: Path) -> None:
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


def load_adk_agent(
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

    ensure_importable(candidate_dir)

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


# Backward-compatible aliases during module rename.
_ensure_importable = ensure_importable
_load_adk_agent = load_adk_agent


def _discover_tasks(tasks_dir: Path) -> list[str]:
    """Compatibility shim returning discovered task names."""
    return [task.name for task in discover_tasks(tasks_dir)]


def _resolve_task_path(tasks_dir: Path, task_name: str) -> Path:
    """Compatibility shim for resolving task paths."""
    return resolve_task_path(tasks_dir, task_name)


def _read_instruction(task_path: Path) -> str:
    """Compatibility shim for reading task instructions."""
    return read_instruction(task_path)


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


@contextmanager
def _temporary_env(extra_env: dict[str, str] | None):
    old_env: dict[str, str] = {}
    created_env_keys: set[str] = set()

    try:
        for key, value in (extra_env or {}).items():
            if key in os.environ:
                old_env[key] = os.environ[key]
            else:
                created_env_keys.add(key)
            os.environ[key] = value
        yield
    finally:
        for key in created_env_keys:
            os.environ.pop(key, None)
        for key, value in old_env.items():
            os.environ[key] = value


@contextmanager
def _temporary_cwd(work_dir: Path | None):
    if work_dir is None:
        yield
        return

    previous_cwd = Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


async def run_agent_on_task(
    agent,
    app,
    converter: OtelToAtifConverter,
    instruction: str,
    task_name: str,
    timeout: int,
    work_dir: Path | None = None,
    task_logs_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> EvalResult:
    """Run an ADK agent on a single task and collect ATIF trajectory.

    Trace collection uses a FileSpanExporter that writes OTel spans to a
    per-task JSON file. The agent emits spans via the OTel SDK; the
    exporter captures them and writes to disk on flush.

    Pipeline:
        ADK Agent → OTel SDK → FileSpanExporter → otel_spans.json → ATIF
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from adk_meta_harness.trace.file_exporter import (
        FileSpanExporter,
        setup_file_exporter,
        teardown_file_exporter,
    )

    # Determine where to write OTel spans for this task.
    span_output_path: Path | None = None
    if task_logs_dir is not None:
        span_output_path = task_logs_dir / "agent" / "otel_spans.json"

    # Set up the per-task file exporter.
    exporter: FileSpanExporter | None = None
    if span_output_path is not None:
        try:
            exporter = setup_file_exporter(span_output_path)
        except Exception:
            exporter = None

    run_error: Exception | None = None

    session_service = InMemorySessionService()
    runner = Runner(
        app=app,
        session_service=session_service,
    )

    await session_service.create_session(
        app_name=getattr(app, "name", "adk-meta-harness"),
        user_id="meta_harness",
        session_id=task_name,
    )

    content = types.Content(parts=[types.Part(text=instruction)], role="user")

    try:
        with _temporary_env(extra_env), _temporary_cwd(work_dir):

            async def _run() -> None:
                async for _ in runner.run_async(
                    user_id="meta_harness",
                    session_id=task_name,
                    new_message=content,
                ):
                    pass

            await asyncio.wait_for(_run(), timeout=timeout)
    except Exception as e:
        run_error = e

    # Flush OTel spans to file and tear down the processor.
    if exporter is not None:
        with suppress(Exception):
            teardown_file_exporter(exporter)

    # Read the span file back.
    trajectory: AtifTrajectory | None = None
    span_file = None
    if task_logs_dir is not None:
        span_file = _load_collector_span_file(task_logs_dir, task_name)
    if span_file is not None:
        try:
            trajectory = converter.convert_file(span_file)
        except Exception:
            trajectory = None

    if trajectory is None:
        trajectory = AtifTrajectory(
            agent=AtifAgent(
                name=getattr(agent, "name", "adk-agent"),
                model_name=getattr(agent, "model", ""),
            ),
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


_prepare_task_workspace = prepare_workspace
_run_task_verifier = run_verifier
_run_agent_on_task = run_agent_on_task
