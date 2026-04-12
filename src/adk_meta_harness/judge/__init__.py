"""Judge module — pluggable trace scoring adapters.

Judges evaluate agent execution traces and produce a score + reasoning.
Three implementations:
- LiteLLMJudge: Any model via litellm (OpenAI, Anthropic, Gemini, etc.)
- ADKJudge: An ADK agent that evaluates traces
- CodingAgentCLIJudge: OpenCode, Pi, or any coding CLI that judges traces
"""

from adk_meta_harness.judge.adk_judge import ADKJudge
from adk_meta_harness.judge.base import JudgeProtocol, JudgeResult
from adk_meta_harness.judge.cli_judge import CodingAgentCLIJudge
from adk_meta_harness.judge.litellm_judge import LiteLLMJudge

JUDGES: dict[str, type] = {
    "litellm": LiteLLMJudge,
    "adk": ADKJudge,
    "opencode": CodingAgentCLIJudge,
    "pi": CodingAgentCLIJudge,
}


def get_judge(name: str, **kwargs) -> JudgeProtocol:
    """Get a judge by name.

    Args:
        name: One of 'litellm', 'adk', 'opencode', 'pi',
              or a custom CLI command.
        **kwargs: Additional args passed to the judge constructor.

    Returns:
        A JudgeProtocol instance.
    """
    if name in JUDGES:
        if name in ("opencode", "pi"):
            return JUDGES[name](cli_command=name, **kwargs)
        return JUDGES[name](**kwargs)
    return CodingAgentCLIJudge(cli_command=name, **kwargs)


__all__ = [
    "JUDGES",
    "ADKJudge",
    "CodingAgentCLIJudge",
    "JudgeProtocol",
    "JudgeResult",
    "LiteLLMJudge",
    "get_judge",
]