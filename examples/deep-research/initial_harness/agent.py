"""Deep research agent — initial harness for optimization."""

from google.adk.agents import Agent
from adk_skills_agent import SkillsRegistry
from adk_tool_search import (
    ToolRegistry,
    create_search_and_load_tools,
    create_session_scoped_loader_callbacks,
)


def create_agent(model: str = "gemini-2.5-pro") -> Agent:
    """Create a deep research agent with skills discovery and dynamic tool search."""
    skills_registry = SkillsRegistry()
    skills_registry.discover(["./skills"])

    tool_registry = ToolRegistry()
    search_tool, load_tool = create_search_and_load_tools(tool_registry)
    before_cb, after_cb = create_session_scoped_loader_callbacks(tool_registry)

    agent = Agent(
        name="deep-research-agent",
        model=model,
        instruction=open("system_prompt.md").read(),
        tools=[
            skills_registry.create_use_skill_tool(),
            skills_registry.create_run_script_tool(),
            search_tool,
            load_tool,
        ],
        before_model_callback=before_cb,
        after_tool_callback=after_cb,
    )
    return agent


agent = create_agent()