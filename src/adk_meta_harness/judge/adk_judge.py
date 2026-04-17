"""ADK-based judge — an ADK agent that evaluates traces."""

from __future__ import annotations

from adk_meta_harness.judge.base import JudgeResult

JUDGE_INSTRUCTION = """\
You are an expert judge evaluating an AI agent's performance on a task.

You will be given a task instruction and the agent's execution trace.
Optionally, you may also receive the expected outcome.

Rate the agent's performance on a scale of 0.0 to 1.0:
- 1.0: Task fully completed correctly.
- 0.5-0.9: Partially completed or minor errors.
- 0.0-0.4: Failed or did not attempt.

Consider:
- Did the agent complete the task as instructed?
- Were there unnecessary errors or wasted steps?
- Did the agent use tools correctly and efficiently?
- Is the final output correct and complete?

You MUST respond in this exact format:
SCORE: <float between 0.0 and 1.0>
REASONING: <brief explanation of your score>
"""


class ADKJudge:
    """Score traces using an ADK agent.

    This creates an ADK LlmAgent with the judge instruction and runs it
    via the ADK Runner. The model is specified via litellm model strings
    so any provider can be used (e.g. "gemini/gemini-2.5-flash",
    "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514").
    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        instruction: str = JUDGE_INSTRUCTION,
    ):
        self.model = model
        self.instruction = instruction

    @property
    def name(self) -> str:
        return f"adk:{self.model}"

    async def judge_trace(
        self,
        task_instruction: str,
        trace: str,
        task_name: str = "",
        expected_outcome: str | None = None,
    ) -> JudgeResult:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        user_content = f"## Task Instruction\n{task_instruction}\n\n"
        if expected_outcome:
            user_content += f"## Expected Outcome\n{expected_outcome}\n\n"
        user_content += f"## Agent Trace\n{trace}"

        agent = LlmAgent(
            name="judge",
            model=self.model,
            instruction=self.instruction,
            tools=[],
        )

        runner = Runner(
            agent=agent,
            session_service=InMemorySessionService(),
            app_name="adk-meta-harness-judge",
        )

        response_parts = []
        async for event in runner.run_async(
            user_id="judge",
            session_id=task_name or "default",
            new_message={"parts": [{"text": user_content}]},
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_parts.append(part.text)

        text = "\n".join(response_parts)
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
