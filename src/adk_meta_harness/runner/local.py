from __future__ import annotations

from pathlib import Path

from adk_meta_harness.judge.base import JudgeProtocol
from adk_meta_harness.task import DEFAULT_AGENT_TIMEOUT_SEC
from adk_meta_harness.task_executor import EvalOutput, evaluate_candidate


class LocalTaskRunner:
    """Runs tasks locally using the in-process ADK Runner.

    No containers, no sandboxing. The agent runs in the current process
    with ``os.chdir`` as the only isolation. Verifier scripts run as
    local subprocesses.

    Tasks run sequentially because ``os.chdir`` and temporary
    ``os.environ`` updates are process-global — concurrent tasks would
    clobber each other's working directory and environment.
    For parallel execution, use the Temporal runner once enabled.
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "local"

    async def evaluate(
        self,
        candidate_dir: Path,
        tasks_dir: Path,
        *,
        model: str | None = None,
        timeout: int = DEFAULT_AGENT_TIMEOUT_SEC,
        search_task_names: list[str] | None = None,
        holdout_task_names: list[str] | None = None,
        judge: JudgeProtocol | None = None,
    ) -> EvalOutput:

        return await evaluate_candidate(
            candidate_dir=candidate_dir,
            tasks_dir=tasks_dir,
            model=model,
            timeout=timeout,
            search_task_names=search_task_names,
            holdout_task_names=holdout_task_names,
            judge=judge,
        )
