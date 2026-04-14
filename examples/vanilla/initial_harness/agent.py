"""Vanilla ADK agent harness — minimal baseline."""

import yaml
from pathlib import Path

from google.adk.agents import Agent


def create_agent(model: str | None = None) -> Agent:
    """Create a minimal ADK agent with no skills, tools, or callbacks.

    Model precedence: model parameter > config.yaml > default.
    """
    if model is None:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text()) or {}
            model = config.get("model", "gemini-2.5-flash")

    system_prompt = (Path(__file__).parent / "system_prompt.md").read_text()

    return Agent(
        name="vanilla-agent",
        model=model,
        instruction=system_prompt,
        tools=[],
    )


agent = create_agent()