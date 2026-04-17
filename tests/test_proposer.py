"""Tests for proposer module."""

from adk_meta_harness.proposer import (
    CodingAgentCLIProposer,
    OpenCodeProposer,
    PiProposer,
    get_proposer,
)


def test_get_proposer_opencode():
    p = get_proposer("opencode")
    assert isinstance(p, OpenCodeProposer)
    assert p.name == "opencode"


def test_get_proposer_pi():
    p = get_proposer("pi")
    assert isinstance(p, PiProposer)
    assert p.name == "pi"


def test_get_proposer_custom():
    p = get_proposer("custom-cli")
    assert isinstance(p, CodingAgentCLIProposer)
    assert p.name == "custom-cli"
