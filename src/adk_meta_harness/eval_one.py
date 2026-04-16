from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_meta_harness.harbor_adapter import _ensure_importable, _read_instruction
from adk_meta_harness.trace.otel_to_atif import OtelToAtifConverter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adk-meta-harness.eval_one",
        description="Evaluate an ADK harness on a single instruction. "
        "Intended to run inside a Harbor container.",
    )
    parser.add_argument(
        "--harness",
        type=Path,
        required=True,
        help="Path to the harness directory inside the container",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="The task instruction to send to the agent",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write trajectory.json and response.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override (reads from harness config.yaml if not set)",
    )

    args = parser.parse_args()
    harness_dir: Path = args.harness.resolve()
    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from adk_meta_harness.harbor_adapter import load_model_from_config

    model = args.model or load_model_from_config(harness_dir) or "gemini-2.5-flash"
    _ensure_importable(harness_dir)

    root_agent, app = _load_agent(harness_dir, model)

    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)
    session = asyncio.get_event_loop().run_until_complete(
        session_service.create_session(
            app_name=getattr(app, "name", "adk-meta-harness"),
            user_id="meta_harness",
            session_id="eval-one",
        )
    )

    content = types.Content(parts=[types.Part(text=args.instruction)], role="user")

    events = []

    async def _run():
        async for event in runner.run_async(
            user_id="meta_harness",
            session_id="eval-one",
            new_message=content,
        ):
            events.append(event)

    asyncio.get_event_loop().run_until_complete(_run())

    converter = OtelToAtifConverter()
    trajectory = converter.adk_events_to_atif(
        events,
        agent_name=getattr(root_agent, "name", "adk-agent"),
        model_name=model,
    )

    trajectory.to_json_file(output_dir / "trajectory.json")

    last_message = ""
    for step in reversed(trajectory.steps):
        if step.message and step.source == "agent":
            last_message = step.message
            break
    (output_dir / "response.txt").write_text(last_message)


def _load_agent(candidate_dir: Path, model: str) -> tuple:
    from google.adk.apps.app import App
    from google.adk.cli.utils.agent_loader import AgentLoader
    from google.adk.agents.base_agent import BaseAgent

    parent_dir = str(candidate_dir.parent)
    agent_name = candidate_dir.name
    loader = AgentLoader(agents_dir=parent_dir)
    agent_or_app = loader.load_agent(agent_name)

    if isinstance(agent_or_app, App):
        app = agent_or_app
        root_agent = app.root_agent
    elif isinstance(agent_or_app, BaseAgent):
        root_agent = agent_or_app
        app = App(name=agent_name, root_agent=root_agent)
    else:
        msg = f"Expected BaseAgent or App from {candidate_dir}, got {type(agent_or_app)}"
        raise TypeError(msg)

    if not getattr(root_agent, "model", None):
        root_agent.model = model

    return root_agent, app


if __name__ == "__main__":
    main()
