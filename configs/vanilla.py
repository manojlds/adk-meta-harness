"""Vanilla ADK agent harness — minimal baseline."""

from google.adk.agents import Agent


def create_agent(model: str = "gemini-2.5-flash") -> Agent:
    """Create a minimal ADK agent with no skills, tools, or callbacks."""
    system_prompt = "You are a helpful assistant. Complete the task as instructed."

    agent = Agent(
        name="vanilla-agent",
        model=model,
        instruction=system_prompt,
        tools=[],
    )
    return agent


agent = create_agent()