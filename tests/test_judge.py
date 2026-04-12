"""Tests for judge module."""

from pathlib import Path

from adk_meta_harness.judge import JUDGES, get_judge
from adk_meta_harness.judge.base import JudgeResult
from adk_meta_harness.judge.cli_judge import CodingAgentCLIJudge
from adk_meta_harness.judge.litellm_judge import LiteLLMJudge


def test_judge_result_creation():
    result = JudgeResult(
        score=0.85,
        reasoning="Partially completed",
        model="gemini/gemini-2.5-flash",
        task_name="task_1",
    )
    assert result.score == 0.85
    assert result.task_name == "task_1"


def test_litellm_parse_score():
    text = "SCORE: 0.75\nREASONING: Good but incomplete"
    assert LiteLLMJudge._parse_score(text) == 0.75


def test_litellm_parse_score_whitespace():
    text = "SCORE:  0.5  \nREASONING: Half done"
    assert LiteLLMJudge._parse_score(text) == 0.5


def test_litellm_parse_score_missing():
    text = "No score here"
    assert LiteLLMJudge._parse_score(text) == 0.0


def test_litellm_parse_reasoning():
    text = "SCORE: 0.75\nREASONING: Good but had errors in step 3"
    assert LiteLLMJudge._parse_reasoning(text) == "Good but had errors in step 3"


def test_litellm_parse_reasoning_missing():
    text = "Some output without structured format"
    assert LiteLLMJudge._parse_reasoning(text) == text[:500]


def test_litellm_judge_name():
    judge = LiteLLMJudge(model="openai/gpt-4o")
    assert judge.name == "litellm:openai/gpt-4o"


def test_cli_judge_name():
    judge = CodingAgentCLIJudge(cli_command="opencode")
    assert judge.name == "cli:opencode"


def test_get_judge_litellm():
    judge = get_judge("litellm", model="openai/gpt-4o")
    assert isinstance(judge, LiteLLMJudge)
    assert judge.model == "openai/gpt-4o"


def test_get_judge_opencode():
    judge = get_judge("opencode")
    assert isinstance(judge, CodingAgentCLIJudge)
    assert judge.cli_command == "opencode"


def test_get_judge_custom():
    judge = get_judge("my-custom-cli")
    assert isinstance(judge, CodingAgentCLIJudge)
    assert judge.cli_command == "my-custom-cli"


def test_cli_judge_build_command_opencode():
    judge = CodingAgentCLIJudge(cli_command="opencode")
    cmd = judge._build_command("evaluate this", Path("/tmp/trace.md"))
    assert cmd[0] == "opencode"
    assert "run" in cmd
    assert "--format" in cmd
    assert "json" in cmd


def test_cli_judge_build_command_pi():
    judge = CodingAgentCLIJudge(cli_command="pi")
    cmd = judge._build_command("evaluate this", Path("/tmp/trace.md"))
    assert cmd[0] == "pi"
    assert "--print" in cmd
    assert "--mode" in cmd


def test_judges_registry():
    assert "litellm" in JUDGES
    assert "adk" in JUDGES
    assert "opencode" in JUDGES
    assert "pi" in JUDGES