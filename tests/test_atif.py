"""Tests for ATIF trace models."""

import tempfile
from pathlib import Path

from adk_meta_harness.trace.atif import (
    AtifAgent,
    AtifMetrics,
    AtifObservation,
    AtifStep,
    AtifToolCall,
    AtifTrajectory,
)


def test_atif_trajectory_roundtrip():
    traj = AtifTrajectory(
        agent=AtifAgent(name="test-agent", version="1.0", model_name="gemini-2.5-flash"),
        steps=[
            AtifStep(
                step_id="step-0000",
                timestamp="2026-01-01T00:00:00Z",
                source="agent",
                message="I'll look up that order.",
                tool_calls=[
                    AtifToolCall(
                        tool_call_id="tc-0",
                        function_name="lookup_order",
                        arguments='{"order_id": "1234"}',
                    )
                ],
            ),
            AtifStep(
                step_id="step-0001",
                timestamp="2026-01-01T00:00:01Z",
                source="agent",
                message="",
                observation=AtifObservation(
                    source_call_id="tc-0",
                    content='{"status": "shipped"}',
                ),
            ),
        ],
    )
    traj.compute_final_metrics()

    d = tempfile.mkdtemp()
    path = Path(d) / "trajectory.json"
    traj.to_json_file(path)

    loaded = AtifTrajectory.from_json_file(path)
    assert loaded.schema_version == "ATIF-v1.4"
    assert len(loaded.steps) == 2
    assert loaded.steps[0].tool_calls[0].function_name == "lookup_order"
    assert loaded.steps[1].observation.source_call_id == "tc-0"
    assert loaded.final_metrics is not None
    assert loaded.final_metrics.total_steps == 2


def test_atif_metrics_from_dict():
    data = {"prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 20, "cost_usd": 0.05}
    m = AtifMetrics.from_dict(data)
    assert m.prompt_tokens == 100
    assert m.completion_tokens == 50


def test_atif_step_with_metrics():
    step = AtifStep(
        step_id="s1",
        timestamp="now",
        source="agent",
        message="hello",
        metrics=AtifMetrics(prompt_tokens=200, completion_tokens=100),
    )
    d = step.to_dict()
    assert d["metrics"]["prompt_tokens"] == 200
    loaded = AtifStep.from_dict(d)
    assert loaded.metrics.prompt_tokens == 200
