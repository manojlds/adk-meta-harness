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


def test_convert_span_with_llm_request_includes_user_prompt():
    converter = OtelToAtifConverter()
    spans = [
        {
            "span_id": "req001",
            "start_time": "2026-01-01T00:00:00Z",
            "name": "call_llm",
            "attributes": {
                "gen_ai.system": "google_adk",
                "gen_ai.operation.name": "invoke",
                "gcp.vertex.agent.llm_request": (
                    '{"model": "openai/glm-5", "contents": '
                    '[{"parts": [{"text": "Read hello.txt"}], "role": "user"}]}'
                ),
            },
        }
    ]

    traj = converter.convert_spans(spans)

    assert traj.agent is not None
    assert traj.agent.model_name == "openai/glm-5"
    assert len(traj.steps) == 1
    assert traj.steps[0].source == "user"
    assert traj.steps[0].message == "Read hello.txt"


def test_convert_span_with_llm_response_extracts_tool_call():
    converter = OtelToAtifConverter()
    spans = [
        {
            "span_id": "resp001",
            "start_time": "2026-01-01T00:00:01Z",
            "name": "call_llm",
            "attributes": {
                "gen_ai.system": "google_adk",
                "gen_ai.operation.name": "invoke",
                "gcp.vertex.agent.llm_response": (
                    '{"content": {"parts": ['
                    '{"text": "I will read the file."}, '
                    '{"function_call": {"id": "tc-1", "name": "read_file", '
                    '"args": {"path": "hello.txt"}}}], '
                    '"role": "model"}}'
                ),
            },
        }
    ]

    traj = converter.convert_spans(spans)

    assert len(traj.steps) == 1
    assert traj.steps[0].source == "agent"
    assert "I will read the file." in traj.steps[0].message
    assert len(traj.steps[0].tool_calls) == 1
    assert traj.steps[0].tool_calls[0].function_name == "read_file"
