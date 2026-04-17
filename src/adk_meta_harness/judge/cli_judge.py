"""Coding-agent-CLI judge — uses OpenCode, Pi, or any CLI to judge traces."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from adk_meta_harness.judge.base import JudgeResult

CLI_JUDGE_PROMPT = """\
You are an expert judge evaluating an AI agent's performance on a task.

Read the task instruction and agent trace below, then produce your evaluation.

Rate the agent's performance on a scale of 0.0 to 1.0:
- 1.0: Task fully completed correctly.
- 0.5-0.9: Partially completed or minor errors.
- 0.0-0.4: Failed or did not attempt.

You MUST respond in this exact format:
SCORE: <float between 0.0 and 1.0>
REASONING: <brief explanation of your score>
"""


class CodingAgentCLIJudge:
    """Judge traces using a coding agent CLI.

    Writes the instruction and trace to a temporary file, asks the CLI
    to evaluate it, then parses the SCORE/REASONING from the output.
    """

    def __init__(
        self,
        cli_command: str,
        cli_args: list[str] | None = None,
        model: str | None = None,
        env: dict[str, str] | None = None,
    ):
        self.cli_command = cli_command
        self.cli_args = cli_args or []
        self.model = model
        self.env = env

    @property
    def name(self) -> str:
        return f"cli:{self.cli_command}"

    async def judge_trace(
        self,
        task_instruction: str,
        trace: str,
        task_name: str = "",
        expected_outcome: str | None = None,
    ) -> JudgeResult:
        content = f"# Task Instruction\n{task_instruction}\n\n"
        if expected_outcome:
            content += f"# Expected Outcome\n{expected_outcome}\n\n"
        content += f"# Agent Trace\n{trace}"

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "trace.md"
            trace_file.write_text(content)

            prompt = (
                f"{CLI_JUDGE_PROMPT}\n\n"
                f"Read the file at {trace_file} and evaluate the agent's performance."
            )

            cmd = self._build_command(prompt, trace_file)
            env = {**os.environ, **(self.env or {})}
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                env=env,
                timeout=1800,
            )

        text = result.stdout or result.stderr or ""
        score = self._parse_score(text)
        reasoning = self._parse_reasoning(text)

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            model=f"cli:{self.cli_command}",
            task_name=task_name,
            raw_output=text,
        )

    async def judge_traces(
        self,
        traces: list[tuple[str, str, str]],
        expected_outcomes: dict[str, str] | None = None,
    ) -> list[JudgeResult]:
        expected_outcomes = expected_outcomes or {}
        results = []
        for task_name, instruction, trace in traces:
            result = await self.judge_trace(
                task_instruction=instruction,
                trace=trace,
                task_name=task_name,
                expected_outcome=expected_outcomes.get(task_name),
            )
            results.append(result)
        return results

    def _build_command(self, prompt: str, trace_file: Path) -> list[str]:
        if self.cli_command == "opencode":
            args = ["run", "--format", "json"]
            if self.model:
                args.extend(["--model", self.model])
            args.append(prompt)
            return [self.cli_command, *args]

        if self.cli_command == "pi":
            args = ["--print", "--mode", "json"]
            if self.model:
                args.extend(["--model", self.model])
            return [self.cli_command, *args, prompt]

        return [self.cli_command, *self.cli_args, prompt]

    @staticmethod
    def _parse_score(text: str) -> float:
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    return float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    continue
        return 0.0

    @staticmethod
    def _parse_reasoning(text: str) -> str:
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("REASONING:"):
                return line.split(":", 1)[1].strip()
        return text[:500]
