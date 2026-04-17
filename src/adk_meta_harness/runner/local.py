from __future__ import annotations

from pathlib import Path

from adk_meta_harness.harbor_adapter import EvalOutput, evaluate_candidate
from adk_meta_harness.judge.base import JudgeProtocol


class LocalTaskRunner:
    """Runs tasks locally using the in-process ADK Runner.

    No containers, no sandboxing. The agent runs in the current process
    with ``os.chdir`` as the only isolation. Verifier scripts run as
    local subprocesses.

    This is the default runner and preserves the original behavior of
    ``evaluate_candidate()``.
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
        timeout: int = 300,
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
