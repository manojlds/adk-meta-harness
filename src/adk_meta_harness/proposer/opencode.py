"""OpenCode CLI proposer."""

from __future__ import annotations

from pathlib import Path

from adk_meta_harness.proposer.coding_agent_cli import CodingAgentCLIProposer


class OpenCodeProposer(CodingAgentCLIProposer):
    """Proposer using the OpenCode CLI."""

    def __init__(self, model: str | None = None):
        cli_args = []
        if model:
            cli_args.extend(["--model", model])
        super().__init__(
            cli_command="opencode",
            cli_args=cli_args,
        )

    def build_command(self, candidate_dir: Path, instruction: str) -> list[str]:
        return [
            self.cli_command,
            *self.cli_args,
            "--non-interactive",
            "-p",
            instruction,
            str(candidate_dir),
        ]