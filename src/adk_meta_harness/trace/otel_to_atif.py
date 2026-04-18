"""OTel → ATIF converter.

Converts OpenTelemetry spans emitted by ADK into ATIF trajectories.
Designed to run during local task evaluation runs.

Architecture:
    ADK Agent
        │
        ├── emits OTel spans (native ADK)
        │
        ▼
    OTel Collector / file exporter
        │
        ├── exports to Jaeger (dev observability)
        ├── converts → ATIF trajectory.json → /logs/agent/
        │
        ▼
    verifier reads /logs/agent/trajectory.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adk_meta_harness.trace.atif import (
    AtifAgent,
    AtifMetrics,
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
            llm_request = attrs.get("gcp.vertex.agent.llm_request")
            if llm_request:
                req = self._safe_json_loads(llm_request)
                if isinstance(req, dict) and req.get("model"):
                    model_name = str(req["model"])
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
        source = "agent"
        tool_calls: list[AtifToolCall] = []
        if "gen_ai.completion" in attrs:
            message = attrs["gen_ai.completion"]
        elif "gen_ai.prompt" in attrs:
            message = attrs["gen_ai.prompt"]
            source = "user"

        # Parse ADK's structured request/response payloads when available.
        llm_request = attrs.get("gcp.vertex.agent.llm_request")
        llm_response = attrs.get("gcp.vertex.agent.llm_response")

        if llm_response:
            resp = self._safe_json_loads(llm_response)
            if isinstance(resp, dict):
                if not message:
                    completion_text = self._extract_text_from_content(resp.get("content"))
                    if completion_text:
                        message = completion_text
                        source = "agent"

                tool_calls.extend(self._extract_tool_calls_from_response(resp, str(span_id)))

        if not message and llm_request:
            req = self._safe_json_loads(llm_request)
            prompt_text = self._extract_prompt_from_request(req)
            if prompt_text:
                message = prompt_text
                source = "user"

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
            source=source,
            message=message,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            metrics=metrics,
        )

    def _safe_json_loads(self, value: Any) -> dict[str, Any] | list | None:
        """Safely parse JSON values that may already be structured."""
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            return None
        return None

    def _extract_prompt_from_request(self, request: Any) -> str:
        """Extract user prompt text from gcp.vertex.agent.llm_request payload."""
        if not isinstance(request, dict):
            return ""
        contents = request.get("contents", [])
        if not isinstance(contents, list):
            return ""

        parts_text: list[str] = []
        for content in contents:
            if not isinstance(content, dict):
                continue
            if content.get("role") != "user":
                continue
            parts = content.get("parts", [])
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    parts_text.append(str(part["text"]))

        return "\n".join(parts_text).strip()

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract textual model response from a response content payload."""
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            return ""

        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                text_parts.append(str(part["text"]))

        return "\n".join(text_parts).strip()

    def _extract_tool_calls_from_response(
        self,
        response: dict[str, Any],
        default_span_id: str,
    ) -> list[AtifToolCall]:
        """Extract tool calls from gcp.vertex.agent.llm_response payload."""
        content = response.get("content")
        if not isinstance(content, dict):
            return []
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            return []

        tool_calls: list[AtifToolCall] = []
        for i, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            function_call = part.get("function_call")
            if not isinstance(function_call, dict):
                continue

            args = function_call.get("args", {})
            try:
                args_json = json.dumps(args)
            except Exception:
                args_json = str(args)

            tool_calls.append(
                AtifToolCall(
                    tool_call_id=str(function_call.get("id", f"{default_span_id}-tc-{i}")),
                    function_name=str(function_call.get("name", "")),
                    arguments=args_json,
                )
            )

        return tool_calls

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
