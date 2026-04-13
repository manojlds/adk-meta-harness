"""Harbor ADK adapter — runs an ADK agent on Harbor tasks.

Uses ADK's AgentLoader for agent discovery and ATIF for trace collection.
Harbor reward files provide pass/fail scores.

Trace pipeline:
    ADK Agent (OTel spans) → OtelToAtifConverter → AtifTrajectory → trajectory.json
    Harbor verifier → reward.txt/reward.json → HarborReward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from adk_meta_harness.trace.atif import AtifTrajectory
from adk_meta_harness.trace.harbor_reward import HarborReward, parse_reward_dir
from adk_meta_harness.trace.otel_to_atif import OtelToAtifConverter


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
    model: str = "gemini-2.5-flash",
    timeout: int = 300,
    search_task_names: list[str] | None = None,
    holdout_task_names: list[str] | None = None,
    output_dir: Path | None = None,
) -> EvalOutput:
    """Evaluate a candidate harness on search and holdout tasks.

    Args:
        candidate_dir: Path to the candidate harness directory.
        tasks_dir: Path to Harbor task definitions.
        model: Model to use for the ADK agent.
        timeout: Timeout per task in seconds.
        search_task_names: Task names for the search set.
        holdout_task_names: Task names for the holdout set.
        output_dir: Directory to write trajectory.json and reward files.
            Defaults to candidate_dir / "evaluation".

    Returns:
        EvalOutput with search and holdout results.
    """
    output_dir = output_dir or candidate_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    agent, app = _load_adk_agent(candidate_dir, model)
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
            model=model,
        )

        # Check Harbor reward files
        task_logs_dir = output_dir / task_name
        reward = parse_reward_dir(task_logs_dir)
        if reward.score > 0 or reward.passed:
            result.reward = reward
            result.score = reward.score
            result.passed = reward.passed

        # Write trajectory to disk
        if result.trajectory:
            traj_path = task_logs_dir / "trajectory.json"
            result.trajectory.to_json_file(traj_path)

        if task_name in search_set:
            output.search_results.append(result)
        if task_name in holdout_set:
            output.holdout_results.append(result)
        elif not holdout_set:
            output.search_results.append(result)

    return output


def _load_adk_agent(
    candidate_dir: Path,
    model: str = "gemini-2.5-flash",
) -> tuple:
    """Load the ADK agent using the official AgentLoader."""
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.apps.app import App
    from google.adk.cli.utils.agent_loader import AgentLoader

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
    model: str = "gemini-2.5-flash",
) -> EvalResult:
    """Run an ADK agent on a single task and collect ATIF trajectory.

    Captures the full event stream and converts to ATIF format.
    Falls back to Harbor reward files for scoring.
    """
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        session_service = InMemorySessionService()
        runner = Runner(
            app=app,
            session_service=session_service,
        )

        content = {"parts": [{"text": instruction}]}
        events = []

        async for event in runner.run_async(
            user_id="meta-harness",
            session_id=task_name,
            new_message=content,
        ):
            events.append(event)

        # Convert ADK events to ATIF trajectory
        trajectory = converter.adk_events_to_atif(
            events,
            agent_name=getattr(agent, "name", "adk-agent"),
            model_name=model,
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