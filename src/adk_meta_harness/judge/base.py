"""Judge protocol and result type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class JudgeResult:
    """Result from judging an agent trace."""

    score: float
    reasoning: str
    model: str
    task_name: str
    raw_output: str = ""


class JudgeProtocol(Protocol):
    """Protocol that all judges must implement.

    A judge reads a task instruction and agent execution trace,
    then produces a score (0.0-1.0) and reasoning.
    """

    async def judge_trace(
        self,
        task_instruction: str,
        trace: str,
        task_name: str = "",
        expected_outcome: str | None = None,
    ) -> JudgeResult:
        """Judge a single trace.

        Args:
            task_instruction: The instruction given to the agent.
            trace: The agent's execution trace.
            task_name: Name/ID of the task.
            expected_outcome: Optional description of expected outcome.

        Returns:
            JudgeResult with score and reasoning.
        """
        ...

    @property
    def name(self) -> str:
        """Name of this judge (e.g. 'litellm', 'opencode')."""
        ...
