"""Single-task evaluator for inside Harbor containers.

Runs an ADK harness on one instruction and writes trajectory.json and
response.txt to the output directory.  Intended to be executed inside
a Harbor container by ``AdkHarborAgent.run()``:

    python -m adk_meta_harness.eval_one \\
        --harness /app/harness \\
        --instruction "Read hello.txt" \\
        --output /logs/agent \\
        --model openai/glm-5.1

Trace collection uses a ``FileSpanExporter`` that writes OTel spans to
``<output>/otel_spans.json`` which is then converted to ATIF.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_meta_harness.harbor_adapter import (
    _ensure_importable,
    _ensure_user_instruction_step,
    load_model_from_config,
)
from adk_meta_harness.trace.atif import AtifAgent, AtifStep, AtifTrajectory
from adk_meta_harness.trace.file_exporter import (
    FileSpanExporter,
    setup_file_exporter,
    teardown_file_exporter,
)
from adk_meta_harness.trace.otel_to_atif import OtelToAtifConverter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adk-meta-harness.eval_one",
        description="Evaluate an ADK harness on a single instruction inside a Harbor container.",
    )
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)

    args = parser.parse_args()
    harness_dir: Path = args.harness.resolve()
    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = args.model or load_model_from_config(harness_dir) or "gemini-2.5-flash"
    _ensure_importable(harness_dir)

    root_agent, app = _load_agent(harness_dir, model)

    # Set up per-task file exporter for OTel spans.
    span_path = output_dir / "otel_spans.json"
    exporter: FileSpanExporter | None = None
    try:
        exporter = setup_file_exporter(span_path)
    except Exception:
        exporter = None

    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        session_service.create_session(
            app_name=getattr(app, "name", "adk-meta-harness"),
            user_id="meta_harness",
            session_id="eval-one",
        )
    )

    content = types.Content(parts=[types.Part(text=args.instruction)], role="user")

    run_error: Exception | None = None
    try:

        async def _run():
            async for _ in runner.run_async(
                user_id="meta_harness",
                session_id="eval-one",
                new_message=content,
            ):
                pass

        loop.run_until_complete(_run())
    except Exception as e:
        run_error = e

    # Flush spans to file.
    if exporter is not None:
        try:
            teardown_file_exporter(exporter)
        except Exception:
            pass

    # Convert OTel spans → ATIF trajectory.
    converter = OtelToAtifConverter()
    trajectory: AtifTrajectory | None = None
    if span_path.exists():
        try:
            trajectory = converter.convert_file(span_path)
        except Exception:
            trajectory = None

    if trajectory is None:
        trajectory = AtifTrajectory(
            agent=AtifAgent(
                name=getattr(root_agent, "name", "adk-agent"), version="1.0", model_name=model
            ),
        )

    trajectory = _ensure_user_instruction_step(trajectory, args.instruction)

    if run_error and not trajectory.steps:
        trajectory.steps.append(
            AtifStep(
                step_id="step-error",
                timestamp="",
                source="system",
                message=f"Run error: {run_error}",
            )
        )
        trajectory.compute_final_metrics()

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
