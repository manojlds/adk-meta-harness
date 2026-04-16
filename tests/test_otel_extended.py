from __future__ import annotations

import json
import pytest

from adk_meta_harness.trace.otel_to_atif import OtelToAtifConverter


class TestSafeJsonLoads:
    def test_dict_string(self):
        converter = OtelToAtifConverter()
        result = converter._safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_list_string(self):
        converter = OtelToAtifConverter()
        result = converter._safe_json_loads("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_already_dict(self):
        converter = OtelToAtifConverter()
        result = converter._safe_json_loads({"key": "value"})
        assert result == {"key": "value"}

    def test_already_list(self):
        converter = OtelToAtifConverter()
        result = converter._safe_json_loads([1, 2])
        assert result == [1, 2]

    def test_empty_string(self):
        converter = OtelToAtifConverter()
        assert converter._safe_json_loads("") is None

    def test_whitespace_string(self):
        converter = OtelToAtifConverter()
        assert converter._safe_json_loads("   ") is None

    def test_invalid_json(self):
        converter = OtelToAtifConverter()
        assert converter._safe_json_loads("not json") is None

    def test_json_string(self):
        converter = OtelToAtifConverter()
        result = converter._safe_json_loads('"hello"')
        assert result is None


class TestExtractPromptFromRequest:
    def test_extracts_user_text(self):
        converter = OtelToAtifConverter()
        req = {
            "contents": [
                {"role": "user", "parts": [{"text": "Read hello.txt"}]},
                {"role": "model", "parts": [{"text": "I will help."}]},
            ]
        }
        result = converter._extract_prompt_from_request(req)
        assert result == "Read hello.txt"

    def test_empty_request(self):
        converter = OtelToAtifConverter()
        assert converter._extract_prompt_from_request({}) == ""

    def test_no_user_role(self):
        converter = OtelToAtifConverter()
        req = {"contents": [{"role": "model", "parts": [{"text": "response"}]}]}
        assert converter._extract_prompt_from_request(req) == ""


class TestExtractTextFromContent:
    def test_extracts_text_parts(self):
        converter = OtelToAtifConverter()
        content = {"parts": [{"text": "Hello"}, {"text": "World"}]}
        assert converter._extract_text_from_content(content) == "Hello\nWorld"

    def test_empty_content(self):
        converter = OtelToAtifConverter()
        assert converter._extract_text_from_content({}) == ""
        assert converter._extract_text_from_content({"parts": []}) == ""


class TestToolCallMerging:
    def test_tool_call_span_creates_step_with_tool_calls(self):
        converter = OtelToAtifConverter()
        spans = [
            {
                "span_id": "s1",
                "start_time": "2026-01-01T00:00:00Z",
                "name": "call_llm",
                "attributes": {
                    "gen_ai.system": "google_adk",
                    "gen_ai.operation.name": "invoke",
                    "gcp.vertex.agent.llm_response": json.dumps(
                        {
                            "content": {
                                "parts": [
                                    {"text": "Let me search for that."},
                                    {
                                        "function_call": {
                                            "id": "tc-1",
                                            "name": "search",
                                            "args": {"query": "test"},
                                        }
                                    },
                                ],
                                "role": "model",
                            }
                        }
                    ),
                },
            },
        ]
        traj = converter.convert_spans(spans)

        assert len(traj.steps) == 1
        assert len(traj.steps[0].tool_calls) == 1
        assert traj.steps[0].tool_calls[0].function_name == "search"
