"""Tests for OTel → ATIF converter."""

from adk_meta_harness.trace.atif import AtifTrajectory
from adk_meta_harness.trace.otel_to_atif import OtelToAtifConverter


def test_convert_generation_span():
    converter = OtelToAtifConverter()
    spans = [
        {
            "span_id": "abc123",
            "start_time": "2026-01-01T00:00:00Z",
            "name": "gen_ai.generate",
            "attributes": {
                "gen_ai.system": "google_adk",
                "gen_ai.operation.name": "generate",
                "gen_ai.request.model": "gemini-2.5-flash",
                "gen_ai.usage.input_tokens": 500,
                "gen_ai.usage.output_tokens": 100,
                "gen_ai.completion": "I'll help you with that.",
            },
        }
    ]
    traj = converter.convert_spans(spans)
    assert isinstance(traj, AtifTrajectory)
    assert traj.agent is not None
    assert traj.agent.model_name == "gemini-2.5-flash"
    assert len(traj.steps) == 1
    assert traj.steps[0].message == "I'll help you with that."
    assert traj.steps[0].metrics is not None
    assert traj.steps[0].metrics.prompt_tokens == 500


def test_convert_tool_call_span():
    converter = OtelToAtifConverter()
    spans = [
        {
            "span_id": "def456",
            "start_time": "2026-01-01T00:00:01Z",
            "name": "tool_call",
            "attributes": {
                "tool.call.name": "lookup_order",
                "tool.call.arguments": '{"order_id": "1234"}',
            },
        }
    ]
    traj = converter.convert_spans(spans)
    assert len(traj.steps) == 1
    assert traj.steps[0].tool_calls[0].function_name == "lookup_order"


def test_convert_file(tmp_path):
    import json

    converter = OtelToAtifConverter()
    spans = [
        {
            "span_id": "abc",
            "start_time": "2026-01-01T00:00:00Z",
            "name": "gen_ai.generate",
            "attributes": {
                "gen_ai.system": "google_adk",
                "gen_ai.operation.name": "generate",
                "gen_ai.request.model": "gemini-2.5-pro",
                "gen_ai.completion": "Done.",
            },
        }
    ]
    input_path = tmp_path / "spans.json"
    input_path.write_text(json.dumps(spans))

    output_path = tmp_path / "trajectory.json"
    traj = converter.convert_file(input_path, output_path)

    assert len(traj.steps) == 1
    assert output_path.exists()
    loaded = AtifTrajectory.from_json_file(output_path)
    assert loaded.steps[0].message == "Done."


def test_adk_events_to_atif():
    converter = OtelToAtifConverter()

    class MockPart:
        def __init__(self, text=None, function_call=None, function_response=None):
            self.text = text
            self.function_call = function_call
            self.function_response = function_response

    class MockContent:
        def __init__(self, parts):
            self.parts = parts

    class MockEvent:
        def __init__(self, content, author="agent"):
            self.content = content
            self.author = author
            self.timestamp = "2026-01-01T00:00:00Z"

    events = [
        MockEvent(MockContent([MockPart(text="Looking up order.")])),
        MockEvent(
            MockContent([
                MockPart(
                    function_call=type(
                        "FC", (), {"id": "tc-1", "name": "lookup_order", "args": {"id": "1234"}}
                    )()
                )
            ])
        ),
    ]

    traj = converter.adk_events_to_atif(
        events,
        agent_name="test-agent",
        model_name="gemini-2.5-flash",
    )
    assert len(traj.steps) == 2
    assert traj.steps[0].message == "Looking up order."
    assert traj.steps[1].tool_calls[0].function_name == "lookup_order"
    assert traj.agent.name == "test-agent"