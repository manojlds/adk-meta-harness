"""OpenCode CLI proposer."""

from __future__ import annotations

from pathlib import Path

from adk_meta_harness.proposer.coding_agent_cli import CodingAgentCLIProposer


class OpenCodeProposer(CodingAgentCLIProposer):
    """Proposer using the OpenCode CLI.

    Uses ``opencode run`` which is the non-interactive execution mode.
    The ``run`` command accepts a message, processes it, and exits —
    no TUI, no interactive session.
    """

    def __init__(self, model: str | None = None):
        cli_args: list[str] = []
        if model:
            cli_args.extend(["--model", model])
        super().__init__(
            cli_command="opencode",
            cli_args=cli_args,
        )

    def build_command(self, candidate_dir: Path, instruction: str) -> list[str]:
        cmd = [
            self.cli_command,
            "run",
            *self.cli_args,
            "--dir",
            str(candidate_dir),
            "--format",
            "json",
        ]
        cmd.append(instruction)
        return cmd