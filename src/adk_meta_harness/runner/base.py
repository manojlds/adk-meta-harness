from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from adk_meta_harness.task import DEFAULT_AGENT_TIMEOUT_SEC
from adk_meta_harness.task_executor import EvalOutput


@runtime_checkable
class TaskRunner(Protocol):
    @property
    def name(self) -> str: ...

    async def evaluate(
        self,
        candidate_dir: Path,
        tasks_dir: Path,
        *,
        model: str | None = None,
        timeout: int = DEFAULT_AGENT_TIMEOUT_SEC,
        search_task_names: list[str] | None = None,
        holdout_task_names: list[str] | None = None,
        judge: object | None = None,
    ) -> EvalOutput: ...
