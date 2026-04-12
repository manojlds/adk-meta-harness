"""LiteLLM-based judge — uses any model via litellm completion API."""

from __future__ import annotations

from adk_meta_harness.judge.base import JudgeResult

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


class LiteLLMJudge:
    """Score traces using any model via litellm.

    Supports OpenAI, Anthropic, Gemini, Mistral, and 100+ other
    providers through the litellm unified interface.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
    ):
        self.model = model
        self.system_prompt = system_prompt

    @property
    def name(self) -> str:
        return f"litellm:{self.model}"

    async def judge_trace(
        self,
        task_instruction: str,
        trace: str,
        task_name: str = "",
        expected_outcome: str | None = None,
    ) -> JudgeResult:
        from litellm import acompletion

        user_content = f"## Task Instruction\n{task_instruction}\n\n"
        if expected_outcome:
            user_content += f"## Expected Outcome\n{expected_outcome}\n\n"
        user_content += f"## Agent Trace\n{trace}"

        response = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )

        text = response.choices[0].message.content or ""
        score = self._parse_score(text)
        reasoning = self._parse_reasoning(text)

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            model=self.model,
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