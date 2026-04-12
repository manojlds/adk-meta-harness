"""LLM judge for scoring unlabeled traces.

When a task has no programmatic verifier, the LLM judge reads the agent's
execution trace and produces a score. Following canvas-org/meta-agent's approach
where judge-based search can outperform labeled search.
"""

from __future__ import annotations

from dataclasses import dataclass

JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating an AI agent's performance on a task.

You will receive:
1. The task instruction given to the agent.
2. The agent's execution trace (tool calls, responses, errors).
3. The expected outcome (if available).

Rate the agent's performance on a scale of 0.0 to 1.0:
- 1.0: Task fully completed correctly.
- 0.5-0.9: Partially completed or minor errors.
- 0.0-0.4: Failed or did not attempt.

Consider:
- Did the agent complete the task as instructed?
- Were there unnecessary errors or wasted steps?
- Did the agent use tools correctly and efficiently?
- Is the final output correct and complete?

Respond in this exact format:
SCORE: <float between 0.0 and 1.0>
REASONING: <brief explanation of your score>
"""


@dataclass
class JudgeResult:
    score: float
    reasoning: str
    model: str
    task_name: str


class LLMJudge:
    """Score unlabeled agent traces using an LLM.

    This is used when Harbor tasks don't have programmatic verifiers,
    or as a supplement to binary pass/fail for richer signal.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
    ):
        self.model = model
        self.system_prompt = system_prompt

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
        from google import genai

        client = genai.Client()

        user_content = f"## Task Instruction\n{task_instruction}\n\n"
        if expected_outcome:
            user_content += f"## Expected Outcome\n{expected_outcome}\n\n"
        user_content += f"## Agent Trace\n{trace}"

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
            ),
        )

        text = response.text
        score = self._parse_score(text)
        reasoning = self._parse_reasoning(text)

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            model=self.model,
            task_name=task_name,
        )

    async def judge_traces(
        self,
        traces: list[tuple[str, str, str]],  # (task_name, instruction, trace)
        expected_outcomes: dict[str, str] | None = None,
    ) -> list[JudgeResult]:
        """Judge multiple traces.

        Args:
            traces: List of (task_name, instruction, trace) tuples.
            expected_outcomes: Optional mapping of task_name -> expected outcome.

        Returns:
            List of JudgeResult objects.
        """
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

    @staticmethod
    def _parse_score(text: str) -> float:
        """Parse SCORE: X.X from judge output."""
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
        """Parse REASONING: ... from judge output."""
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("REASONING:"):
                return line.split(":", 1)[1].strip()
        return text[:500]