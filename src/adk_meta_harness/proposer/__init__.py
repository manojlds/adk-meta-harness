"""Proposer module — pluggable coding-agent-CLI adapters."""

from adk_meta_harness.proposer.base import ProposerProtocol
from adk_meta_harness.proposer.coding_agent_cli import CodingAgentCLIProposer
from adk_meta_harness.proposer.opencode import OpenCodeProposer
from adk_meta_harness.proposer.pi import PiProposer

PROPOSERS: dict[str, type[CodingAgentCLIProposer]] = {
    "opencode": OpenCodeProposer,
    "pi": PiProposer,
}


def get_proposer(name: str, **kwargs) -> CodingAgentCLIProposer:
    """Get a proposer by name.

    Args:
        name: One of 'opencode', 'pi', or a custom CLI command.
        **kwargs: Additional args passed to the proposer constructor.

    Returns:
        A CodingAgentCLIProposer instance.
    """
    if name in PROPOSERS:
        return PROPOSERS[name](**kwargs)
    return CodingAgentCLIProposer(cli_command=name, **kwargs)


__all__ = [
    "PROPOSERS",
    "CodingAgentCLIProposer",
    "OpenCodeProposer",
    "PiProposer",
    "ProposerProtocol",
    "get_proposer",
]