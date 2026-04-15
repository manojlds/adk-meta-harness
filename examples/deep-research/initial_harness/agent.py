"""Deep research agent — initial harness for optimization."""

import yaml
from pathlib import Path

from google.adk.agents import Agent
from adk_skills_agent import SkillsRegistry
from adk_tool_search import (
    ToolRegistry,
    create_search_and_load_tools,
    create_session_scoped_loader_callbacks,
)


def create_agent(model: str | None = None) -> Agent:
    """Create a deep research agent with skills discovery and dynamic tool search.

    Model precedence: model parameter > config.yaml > default.
    """
    if model is None:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text()) or {}
            model = config.get("model", "gemini-2.5-flash")

    system_prompt = (Path(__file__).parent / "system_prompt.md").read_text()
    skills_registry = SkillsRegistry()
    skills_registry.discover(["./skills"])

    tool_registry = ToolRegistry()
    search_tool, load_tool = create_search_and_load_tools(tool_registry)
    before_cb, after_cb = create_session_scoped_loader_callbacks(tool_registry)

    return Agent(
        name="deep_research_agent",
        model=model,
        instruction=system_prompt,
        tools=[
            skills_registry.create_use_skill_tool(),
            skills_registry.create_run_script_tool(),
            search_tool,
            load_tool,
        ],
        before_model_callback=before_cb,
        after_tool_callback=after_cb,
    )


agent = create_agent()