"""ADK agent harness with adk-skills and adk-tool-search."""

from google.adk.agents import Agent
from adk_skills_agent import SkillsRegistry


def create_agent(model: str = "gemini-2.5-flash") -> Agent:
    """Create an ADK agent with skills discovery."""
    registry = SkillsRegistry()
    registry.discover(["./skills"])

    agent = Agent(
        name="skills-agent",
        model=model,
        instruction="You are a helpful assistant. Use your skills when relevant.",
        tools=[
            registry.create_use_skill_tool(),
            registry.create_run_script_tool(),
        ],
    )
    return agent


agent = create_agent()