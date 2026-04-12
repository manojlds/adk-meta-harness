"""Pi CLI proposer."""

from __future__ import annotations

from pathlib import Path

from adk_meta_harness.proposer.coding_agent_cli import CodingAgentCLIProposer


class PiProposer(CodingAgentCLIProposer):
    """Proposer using the Pi CLI.

    Uses ``--print`` and ``-p`` flags for non-interactive mode.
    ``--print`` causes Pi to process the prompt and exit without TUI.
    ``-p`` is the shorthand for ``--print``.
    """

    def __init__(self, model: str | None = None):
        cli_args: list[str] = []
        if model:
            cli_args.extend(["--model", model])
        super().__init__(
            cli_command="pi",
            cli_args=cli_args,
        )

    def build_command(self, candidate_dir: Path, instruction: str) -> list[str]:
        return [
            self.cli_command,
            *self.cli_args,
            "--print",
            "--mode",
            "json",
            instruction,
        ]