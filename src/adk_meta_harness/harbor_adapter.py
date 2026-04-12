"""Harbor ADK adapter — runs an ADK agent on Harbor tasks."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalResult:
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
        tasks_dir: Path to Harbor task definitions.
        model: Model to use for the ADK agent.
        timeout: Timeout per task in seconds.
        search_task_names: Task names for the search set (proposer sees traces).
        holdout_task_names: Task names for the holdout set (gate uses scores only).

    Returns:
        Tuple of (search_results, holdout_results).
    """
    search_results = []
    holdout_results = []

    agent = await _load_adk_agent(candidate_dir, model)

    all_tasks = _discover_tasks(tasks_dir)
    search_set = search_task_names or [t for t in all_tasks]
    holdout_set = holdout_task_names or []

    for task_name in all_tasks:
        task_path = tasks_dir / task_name
        if not task_path.exists():
            continue
        instruction = _read_instruction(task_path)
        result = await _run_agent_on_task(agent, instruction, task_name, timeout)

        if task_name in search_set:
            search_results.append(result)
        if task_name in holdout_set:
            holdout_results.append(result)
        elif not holdout_set:
            search_results.append(result)

    return search_results, holdout_results


async def _load_adk_agent(candidate_dir: Path, model: str):
    """Load the ADK agent from a candidate directory.

    The candidate's agent.py must export an `agent` or `create_agent` function.
    """
    agent_file = candidate_dir / "agent.py"
    if not agent_file.exists():
        raise FileNotFoundError(f"No agent.py found in {candidate_dir}")

    spec = importlib.util.spec_from_file_location("harness_agent", str(agent_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent from {agent_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["harness_agent"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "create_agent"):
        agent = module.create_agent(model=model)
    elif hasattr(module, "agent"):
        agent = module.agent
    else:
        raise AttributeError(f"{agent_file} must export 'agent' or 'create_agent'")

    return agent


def _discover_tasks(tasks_dir: Path) -> list[str]:
    """Discover Harbor task directories."""
    if not tasks_dir.exists():
        return []
    return [
        d.name
        for d in sorted(tasks_dir.iterdir())
        if (d.is_dir() and (d / "instruction.md").exists()) or (d / "task.toml").exists()
    ]


def _read_instruction(task_path: Path) -> str:
    """Read the task instruction."""
    instruction_file = task_path / "instruction.md"
    if instruction_file.exists():
        return instruction_file.read_text()
    return ""


async def _run_agent_on_task(agent, instruction: str, task_name: str, timeout: int) -> EvalResult:
    """Run an ADK agent on a single task and collect the result.

    This is a simplified version. In production, this would use Harbor's
    container-based evaluation for full isolation.
    """
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        session_service = InMemorySessionService()
        runner = Runner(agent=agent, session_service=session_service, app_name="adk-meta-harness")

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
                        trace_parts.append(f"[tool_call: {part.function_call.name}]")
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