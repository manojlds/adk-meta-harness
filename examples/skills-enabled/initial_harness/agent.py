"""ADK agent harness with adk-skills."""

import yaml
from pathlib import Path

from google.adk.agents import Agent
from adk_skills_agent import SkillsRegistry


def create_agent(model: str | None = None) -> Agent:
    """Create an ADK agent with skills discovery.

    Model precedence: model parameter > config.yaml > default.
    """
    if model is None:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text()) or {}
            model = config.get("model", "gemini-2.5-flash")

    system_prompt = (Path(__file__).parent / "system_prompt.md").read_text()
    registry = SkillsRegistry()
    registry.discover(["./skills"])

    return Agent(
        name="skills_agent",
        model=model,
        instruction=system_prompt,
        tools=[
            registry.create_use_skill_tool(),
            registry.create_run_script_tool(),
        ],
    )


agent = create_agent()