"""ATIF — Agent Trajectory Interchange Format.

Standardized JSON format for logging the complete interaction history of
autonomous LLM agents. Based on ATIF-v1.4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AtifMetrics:
    """Per-step token and cost metrics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtifMetrics:
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
        )


@dataclass
class AtifToolCall:
    """A tool call made by the agent in a step."""

    tool_call_id: str = ""
    function_name: str = ""
    arguments: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "function_name": self.function_name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtifToolCall:
        return cls(
            tool_call_id=data.get("tool_call_id", ""),
            function_name=data.get("function_name", ""),
            arguments=data.get("arguments", ""),
        )


@dataclass
class AtifObservation:
    """Environment feedback returned to the agent after a tool call."""

    source_call_id: str = ""
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_call_id": self.source_call_id,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtifObservation:
        return cls(
            source_call_id=data.get("source_call_id", ""),
            content=data.get("content", ""),
        )


@dataclass
class AtifStep:
    """A single step in an agent trajectory.

    Each step represents one turn: the agent's message, any tool calls,
    and the observations returned by the environment.
    """

    step_id: str = ""
    timestamp: str = ""
    source: str = "agent"
    message: str = ""
    reasoning_content: str = ""
    tool_calls: list[AtifToolCall] = field(default_factory=list)
    observation: AtifObservation | None = None
    metrics: AtifMetrics | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "message": self.message,
        }
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.observation:
            d["observation"] = self.observation.to_dict()
        if self.metrics:
            d["metrics"] = self.metrics.to_dict()
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtifStep:
        tool_calls = [AtifToolCall.from_dict(tc) for tc in data.get("tool_calls", [])]
        observation_data = data.get("observation")
        observation = AtifObservation.from_dict(observation_data) if observation_data else None
        metrics_data = data.get("metrics")
        metrics = AtifMetrics.from_dict(metrics_data) if metrics_data else None
        return cls(
            step_id=data.get("step_id", ""),
            timestamp=data.get("timestamp", ""),
            source=data.get("source", "agent"),
            message=data.get("message", ""),
            reasoning_content=data.get("reasoning_content", ""),
            tool_calls=tool_calls,
            observation=observation,
            metrics=metrics,
            extra=data.get("extra", {}),
        )


@dataclass
class AtifFinalMetrics:
    """Aggregate metrics for the full trajectory."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtifFinalMetrics:
        return cls(
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
            total_cached_tokens=data.get("total_cached_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_steps=data.get("total_steps", 0),
        )


@dataclass
class AtifAgent:
    """Agent metadata."""

    name: str = ""
    version: str = ""
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtifAgent:
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            model_name=data.get("model_name", ""),
        )


@dataclass
class AtifTrajectory:
    """A complete agent trajectory in ATIF format.

    This is the canonical trace format stored in candidate directories
    and read by proposers to diagnose failures.
    """

    schema_version: str = "ATIF-v1.4"
    agent: AtifAgent | None = None
    steps: list[AtifStep] = field(default_factory=list)
    final_metrics: AtifFinalMetrics | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"schema_version": self.schema_version}
        if self.agent:
            d["agent"] = self.agent.to_dict()
        d["steps"] = [s.to_dict() for s in self.steps]
        if self.final_metrics:
            d["final_metrics"] = self.final_metrics.to_dict()
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AtifTrajectory:
        agent_data = data.get("agent")
        agent = AtifAgent.from_dict(agent_data) if agent_data else None
        steps = [AtifStep.from_dict(s) for s in data.get("steps", [])]
        final_metrics_data = data.get("final_metrics")
        final_metrics = (
            AtifFinalMetrics.from_dict(final_metrics_data) if final_metrics_data else None
        )
        return cls(
            schema_version=data.get("schema_version", "ATIF-v1.4"),
            agent=agent,
            steps=steps,
            final_metrics=final_metrics,
            extra=data.get("extra", {}),
        )

    def add_step(self, step: AtifStep) -> None:
        self.steps.append(step)
        if self.final_metrics:
            self.final_metrics.total_steps = len(self.steps)

    def compute_final_metrics(self) -> None:
        """Compute final metrics from per-step metrics."""
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        total_cost = 0.0
        for step in self.steps:
            if step.metrics:
                total_prompt += step.metrics.prompt_tokens
                total_completion += step.metrics.completion_tokens
                total_cached += step.metrics.cached_tokens
                total_cost += step.metrics.cost_usd
        self.final_metrics = AtifFinalMetrics(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached,
            total_cost_usd=total_cost,
            total_steps=len(self.steps),
        )

    def to_json_file(self, path: Path) -> None:
        """Write trajectory to a JSON file."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2))

    @classmethod
    def from_json_file(cls, path: Path) -> AtifTrajectory:
        """Read trajectory from a JSON file."""

        data = json.loads(path.read_text())
        return cls.from_dict(data)
