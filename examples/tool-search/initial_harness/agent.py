"""ADK agent harness with adk-tool-search for dynamic tool discovery."""

import yaml
from pathlib import Path

from google.adk.agents import LlmAgent
from adk_tool_search import (
    ToolRegistry,
    create_search_and_load_tools,
    create_session_scoped_loader_callbacks,
)


def create_agent(
    model: str | None = None,
    registry: ToolRegistry | None = None,
) -> LlmAgent:
    """Create an ADK agent with dynamic tool search.

    Model precedence: model parameter > config.yaml > default.
    """
    if model is None:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text()) or {}
            model = config.get("model", "gemini-2.5-flash")

    if registry is None:
        registry = ToolRegistry()

    search_tool, load_tool = create_search_and_load_tools(registry)
    before_cb, after_cb = create_session_scoped_loader_callbacks(registry)

    return LlmAgent(
        name="tool-search-agent",
        model=model,
        instruction=(
            "You are a helpful assistant. Use search_tools to find relevant "
            "tools, then load_tool to activate them before calling."
        ),
        tools=[search_tool, load_tool],
        before_model_callback=before_cb,
        after_tool_callback=after_cb,
    )


agent = create_agent()