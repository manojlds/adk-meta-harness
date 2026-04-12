"""Harbor ADK adapter — runs an ADK agent on Harbor tasks.

Uses ADK's AgentLoader to properly discover and load agent apps,
supporting all ADK patterns: agent.py, __init__.py, App instances,
and YAML configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalResult:
    """Result from evaluating an agent on a single task."""

    task_name: str
    passed: bool
    score: float
    trace: str
    error: str | None = None


async def evaluate_candidate(
    candidate_dir: Path,
    tasks_dir: Path,
    model: str = "gemini-2.5-flash",
    timeout: int = 300,
    search_task_names: list[str] | None = None,
    holdout_task_names: list[str] | None = None,
) -> tuple[list[EvalResult], list[EvalResult]]:
    """Evaluate a candidate harness on search and holdout tasks.

    Args:
        candidate_dir: Path to the candidate harness directory.
            This directory is treated as an ADK agents_dir parent.
            The candidate itself is a subdirectory containing agent.py
            or other ADK entry points.
        tasks_dir: Path to Harbor task definitions.
        model: Model to use for the ADK agent.
        timeout: Timeout per task in seconds.
        search_task_names: Task names for the search set.
        holdout_task_names: Task names for the holdout set.

    Returns:
        Tuple of (search_results, holdout_results).
    """
    search_results = []
    holdout_results = []

    agent, app = _load_adk_agent(candidate_dir, model)

    all_tasks = _discover_tasks(tasks_dir)
    search_set = search_task_names or [t for t in all_tasks]
    holdout_set = holdout_task_names or []

    for task_name in all_tasks:
        task_path = tasks_dir / task_name
        if not task_path.exists():
            continue
        instruction = _read_instruction(task_path)
        result = await _run_agent_on_task(
            agent=agent,
            app=app,
            instruction=instruction,
            task_name=task_name,
            timeout=timeout,
        )

        if task_name in search_set:
            search_results.append(result)
        if task_name in holdout_set:
            holdout_results.append(result)
        elif not holdout_set:
            search_results.append(result)

    return search_results, holdout_results


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

    If the loaded agent doesn't specify a model, the provided
    model parameter is used as a default.

    Returns:
        Tuple of (root_agent, app) where app may be None if
        only a bare BaseAgent was found.
    """
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
    instruction: str,
    task_name: str,
    timeout: int,
) -> EvalResult:
    """Run an ADK agent on a single task and collect the result.

    Uses the App-wrapped Runner for full ADK lifecycle support
    including plugins, context caching, and resumability.
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
        trace_parts = []

        async for event in runner.run_async(
            user_id="meta-harness",
            session_id=task_name,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        trace_parts.append(part.text)
                    elif part.function_call:
                        trace_parts.append(
                            f"[tool_call: {part.function_call.name}]"
                        )
                    elif part.function_response:
                        trace_parts.append(
                            f"[tool_response: {part.function_response.name}]"
                        )

        trace = "\n".join(trace_parts)
        return EvalResult(
            task_name=task_name,
            passed=False,
            score=0.0,
            trace=trace,
        )

    except Exception as e:
        return EvalResult(
            task_name=task_name,
            passed=False,
            score=0.0,
            trace="",
            error=str(e),
        )