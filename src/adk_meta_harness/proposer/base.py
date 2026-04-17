"""Proposer protocol and base class."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ProposerProtocol(Protocol):
    """Protocol that all proposers must implement.

    A proposer reads the filesystem of prior candidates, traces, and scores,
    then proposes a targeted edit to the current harness.
    """

    async def propose_edit(
        self,
        candidate_dir: Path,
        filesystem_dir: Path,
        learnings: str,
        instruction: str,
    ) -> dict[str, str]:
        """Propose an edit to the harness.

        Args:
            candidate_dir: Working copy of the current best harness.
            filesystem_dir: Root directory containing all prior candidates,
                traces, scores, and learnings.
            learnings: Contents of learnings.md.
            instruction: Natural language instruction for the proposer.

        Returns:
            Dict with keys:
                - "description": Short description of the change.
                - "change_type": One of "system_prompt", "skill", "tool",
                    "callback", "routing", "config", "harness".
                - "diff_summary": Human-readable summary of what changed.
        """
        ...

    @property
    def name(self) -> str:
        """Name of this proposer (e.g. 'opencode', 'pi')."""
        ...
