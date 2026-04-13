"""OTel → ATIF converter.

Converts OpenTelemetry spans emitted by ADK into ATIF trajectories.
Designed to run as a sidecar collector during Harbor evaluation runs.

Architecture:
    ADK Agent (in Harbor container)
        │
        ├── emits OTel spans (native ADK)
        │
        ▼
    OTel Collector (sidecar)
        │
        ├── exports to Jaeger (dev observability)
        ├── converts → ATIF trajectory.json → /logs/agent/
        │
        ▼
    Harbor verifier reads /logs/agent/trajectory.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adk_meta_harness.trace.atif import (
    AtifAgent,
    AtifMetrics,
    AtifObservation,
    AtifStep,
    AtifToolCall,
    AtifTrajectory,
)


class OtelToAtifConverter:
    """Converts OpenTelemetry spans to ATIF trajectories.

    ADK emits OTel spans with the following relevant attributes:
    - gen_ai.system: "google_adk"
    - gen_ai.operation.name: "generate", "tool_call"
    - gen_ai.request.model: model name
    - gen_ai.usage.input_tokens, gen_ai.usage.output_tokens
    - gen_ai.prompt: user message (if enabled)
    - gen_ai.completion: assistant response (if enabled)
    - tool.call.name: tool function name
    - tool.call.arguments: tool arguments

    This converter maps those spans to ATIF steps.
    """

    def convert_spans(self, spans: list[dict[str, Any]]) -> AtifTrajectory:
        """Convert a list of OTel span dicts to an ATIF trajectory.

        Args:
            spans: List of OTel span dictionaries. Each span should have
                'attributes', 'name', 'start_time', 'events', etc.

        Returns:
            AtifTrajectory with steps derived from the spans.
        """
        trajectory = AtifTrajectory()
        agent_info = self._extract_agent_info(spans)
        if agent_info:
            trajectory.agent = agent_info

        step_map: dict[str, AtifStep] = {}

        for span in sorted(spans, key=lambda s: s.get("start_time", "")):
            step = self._span_to_step(span, step_map)
            if step:
                step_map[step.step_id] = step

        trajectory.steps = list(step_map.values())
        trajectory.compute_final_metrics()
        return trajectory

    def convert_file(
        self,
        input_path: Path,
        output_path: Path | None = None,
    ) -> AtifTrajectory:
        """Read OTel spans from a JSON file and convert to ATIF.

        Args:
            input_path: Path to file containing OTel spans as JSON.
                Can be a single span, a list of spans, or an OTel
                export format with a 'resourceSpans' key.
            output_path: If provided, write the ATIF trajectory here.

        Returns:
            AtifTrajectory.
        """
        data = json.loads(input_path.read_text())
        spans = self._extract_spans(data)
        trajectory = self.convert_spans(spans)

        if output_path:
            trajectory.to_json_file(output_path)

        return trajectory

    def _extract_spans(self, data: dict[str, Any] | list) -> list[dict[str, Any]]:
        """Extract spans from various OTel export formats."""
        if isinstance(data, list):
            return data

        if "resourceSpans" in data:
            spans = []
            for resource_span in data["resourceSpans"]:
                for scope_span in resource_span.get("scopeSpans", []):
                    spans.extend(scope_span.get("spans", []))
            return spans

        if "spans" in data:
            return data["spans"]

        if "attributes" in data:
            return [data]

        return data if isinstance(data, list) else [data]

    def _extract_agent_info(self, spans: list[dict[str, Any]]) -> AtifAgent | None:
        """Extract agent metadata from spans."""
        model_name = ""
        for span in spans:
            attrs = span.get("attributes", {})
            if "gen_ai.request.model" in attrs:
                model_name = attrs["gen_ai.request.model"]
                break
            if "gen_ai.model" in attrs:
                model_name = attrs["gen_ai.model"]
                break

        if not model_name:
            return None

        return AtifAgent(
            name="adk-agent",
            version="1.0",
            model_name=model_name,
        )

    def _span_to_step(
        self,
        span: dict[str, Any],
        step_map: dict[str, AtifStep],
    ) -> AtifStep | None:
        """Convert a single OTel span to an ATIF step."""
        attrs = span.get("attributes", {})
        span_name = span.get("name", "")

        operation = attrs.get("gen_ai.operation.name", "")

        if "tool" in span_name.lower() or "tool_call" in operation.lower():
            return self._tool_call_span_to_step(span, step_map)
        if operation in ("generate", "chat", "invoke"):
            return self._generation_span_to_step(span)
        if "gen_ai.system" in attrs:
            return self._generation_span_to_step(span)

        return None

    def _generation_span_to_step(self, span: dict[str, Any]) -> AtifStep | None:
        """Convert a generation/invoke span to an ATIF step."""
        attrs = span.get("attributes", {})
        span_id = span.get("span_id", span.get("id", ""))

        message = ""
        if "gen_ai.completion" in attrs:
            message = attrs["gen_ai.completion"]
        elif "gen_ai.prompt" in attrs:
            message = attrs["gen_ai.prompt"]

        reasoning = attrs.get("gen_ai.reasoning_content", "")

        metrics = None
        if "gen_ai.usage.input_tokens" in attrs or "gen_ai.usage.output_tokens" in attrs:
            metrics = AtifMetrics(
                prompt_tokens=int(attrs.get("gen_ai.usage.input_tokens", 0)),
                completion_tokens=int(attrs.get("gen_ai.usage.output_tokens", 0)),
                cached_tokens=int(attrs.get("gen_ai.usage.cached_tokens", 0)),
                cost_usd=float(attrs.get("gen_ai.usage.cost_usd", 0.0)),
            )

        return AtifStep(
            step_id=str(span_id),
            timestamp=span.get("start_time", ""),
            source="agent",
            message=message,
            reasoning_content=reasoning,
            metrics=metrics,
        )

    def _tool_call_span_to_step(
        self,
        span: dict[str, Any],
        step_map: dict[str, AtifStep],
    ) -> AtifStep | None:
        """Convert a tool_call span to an ATIF step.

        Tool call spans get merged into the parent step if one exists,
        or create a new step.
        """
        attrs = span.get("attributes", {})
        span_id = span.get("span_id", span.get("id", ""))
        parent_id = span.get("parent_span_id", "")

        tool_call = AtifToolCall(
            tool_call_id=str(span_id),
            function_name=attrs.get("tool.call.name", attrs.get("gen_ai.tool.name", "")),
            arguments=attrs.get("tool.call.arguments", attrs.get("gen_ai.tool.arguments", "")),
        )

        if parent_id and parent_id in step_map:
            step_map[parent_id].tool_calls.append(tool_call)
            return None

        message = attrs.get("gen_ai.completion", "")
        return AtifStep(
            step_id=str(span_id) or f"step-tool-{len(step_map)}",
            timestamp=span.get("start_time", ""),
            source="agent",
            message=message,
            tool_calls=[tool_call],
        )

    def adk_events_to_atif(
        self,
        events: list[Any],
        agent_name: str = "adk-agent",
        model_name: str = "",
    ) -> AtifTrajectory:
        """Convert ADK event stream objects to an ATIF trajectory.

        This is the direct conversion path — used when we have access
        to the ADK Runner's event stream during evaluation, rather than
        going through OTel. It converts ADK Event objects directly.

        Args:
            events: List of ADK Event objects from runner.run_async().
            agent_name: Name of the agent.
            model_name: Model used by the agent.

        Returns:
            AtifTrajectory.
        """
        trajectory = AtifTrajectory(
            agent=AtifAgent(name=agent_name, version="1.0", model_name=model_name),
        )

        for i, event in enumerate(events):
            step = self._adk_event_to_step(event, i)
            if step:
                trajectory.add_step(step)

        trajectory.compute_final_metrics()
        return trajectory

    def _adk_event_to_step(self, event: Any, index: int) -> AtifStep | None:
        """Convert a single ADK event to an ATIF step."""
        if not hasattr(event, "content") or not event.content:
            return None

        step_id = f"step-{index:04d}"
        timestamp = ""
        if hasattr(event, "timestamp") and event.timestamp:
            timestamp = str(event.timestamp)

        message = ""
        tool_calls = []
        observation = None

        if hasattr(event.content, "parts") and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    message += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        AtifToolCall(
                            tool_call_id=getattr(fc, "id", f"tc-{index}"),
                            function_name=getattr(fc, "name", ""),
                            arguments=json.dumps(getattr(fc, "args", {})),
                        )
                    )
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    observation = AtifObservation(
                        source_call_id=getattr(fr, "id", f"tc-{index}"),
                        content=json.dumps(getattr(fr, "response", {})),
                    )

        if not message and not tool_calls and not observation:
            return None

        source = "agent"
        if hasattr(event, "author"):
            if event.author == "user":
                source = "user"
            elif event.author == "system":
                source = "system"

        return AtifStep(
            step_id=step_id,
            timestamp=timestamp,
            source=source,
            message=message,
            tool_calls=tool_calls,
            observation=observation,
        )