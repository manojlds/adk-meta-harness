"""Harbor ADK adapter — runs an ADK agent on Harbor tasks.

Uses ADK's AgentLoader for agent discovery and ATIF for trace collection.
Harbor reward files provide pass/fail scores. When reward files are absent,
the judge module scores the ATIF trajectory instead.

Model precedence:
    --model CLI flag (runtime override) > config.yaml (harness) > agent default

Trace pipeline:
    ADK Agent (OTel spans) → OtelToAtifConverter → AtifTrajectory → trajectory.json
    Harbor verifier → reward.txt/reward.json → HarborReward
    (or) Judge → JudgeResult → score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from adk_meta_harness.trace.atif import AtifTrajectory
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
        return sum(1 for r in self.search_results if r.passed) / len(
            self.search_results
        )

    @property
    def holdout_score(self) -> float:
        if not self.holdout_results:
            return self.search_score
        return sum(1 for r in self.holdout_results if r.passed) / len(
            self.holdout_results
        )

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
    1. If Harbor reward files (reward.txt/reward.json) exist for a task,
       use them for deterministic pass/fail scoring.
    2. If reward files are absent and a judge is provided, the judge scores
       the ATIF trajectory for that task.
    3. If neither reward files nor a judge exist, the task is marked as failed
       with score 0.0.

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
        task_path = tasks_dir / task_name
        if not task_path.exists():
            continue
        instruction = _read_instruction(task_path)

        result = await _run_agent_on_task(
            agent=agent,
            app=app,
            converter=converter,
            instruction=instruction,
            task_name=task_name,
            timeout=timeout,
        )

        # Check Harbor reward files — prefer deterministic scoring
        task_logs_dir = output_dir / task_name
        reward = parse_reward_dir(task_logs_dir)
        if reward.score > 0 or reward.passed:
            result.reward = reward
            result.score = reward.score
            result.passed = reward.passed
        elif judge is not None and result.trajectory is not None:
            # No Harbor reward — use judge to score the trajectory
            trace_text = result.trace_summary
            judge_result = await judge.judge_trace(
                task_instruction=instruction,
                trace=trace_text,
                task_name=task_name,
            )
            result.score = judge_result.score
            result.passed = judge_result.score >= 0.5

        # Write trajectory to disk
        if result.trajectory:
            traj_path = task_logs_dir / "trajectory.json"
            result.trajectory.to_json_file(traj_path)

        if task_name in holdout_set:
            output.holdout_results.append(result)
        else:
            output.search_results.append(result)

    return output


def _ensure_importable(candidate_dir: Path) -> None:
    """Ensure the candidate directory is importable by ADK's AgentLoader.

    AgentLoader expects:
    - The directory to be a Python package (has __init__.py)
    - agent.py to expose a ``root_agent`` attribute (not ``agent``)

    This function creates __init__.py if missing and patches agent.py
    to add a ``root_agent`` alias if the module uses ``agent`` instead.
    """
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
        msg = (
            f"Expected BaseAgent or App from {candidate_dir}, "
            f"got {type(agent_or_app)}"
        )
        raise TypeError(msg)

    if not getattr(root_agent, "model", None):
        root_agent.model = model

    return root_agent, app


def _discover_tasks(tasks_dir: Path) -> list[str]:
    """Discover Harbor task directories."""
    if not tasks_dir.exists():
        return []
    return [
        d.name
        for d in sorted(tasks_dir.iterdir())
        if d.is_dir()
        and (
            (d / "instruction.md").exists() or (d / "task.toml").exists()
        )
    ]


def _read_instruction(task_path: Path) -> str:
    """Read the task instruction."""
    instruction_file = task_path / "instruction.md"
    if instruction_file.exists():
        return instruction_file.read_text()
    return ""


async def _run_agent_on_task(
    agent,
    app,
    converter: OtelToAtifConverter,
    instruction: str,
    task_name: str,
    timeout: int,
) -> EvalResult:
    """Run an ADK agent on a single task and collect ATIF trajectory.

    Captures the full event stream and converts to ATIF format.
    Falls back to Harbor reward files for scoring.
    """
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

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
        events = []

        async for event in runner.run_async(
            user_id="meta_harness",
            session_id=task_name,
            new_message=content,
        ):
            events.append(event)

        # Convert ADK events to ATIF trajectory
        trajectory = converter.adk_events_to_atif(
            events,
            agent_name=getattr(agent, "name", "adk-agent"),
            model_name=getattr(agent, "model", ""),
        )

        return EvalResult(
            task_name=task_name,
            passed=False,
            score=0.0,
            trajectory=trajectory,
        )

    except Exception as e:
        return EvalResult(
            task_name=task_name,
            passed=False,
            score=0.0,
            trajectory=None,
            error=str(e),
        )