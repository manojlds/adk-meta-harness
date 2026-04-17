"""Tests for FileSpanExporter and setup/teardown helpers."""

import json
from unittest.mock import MagicMock

from adk_meta_harness.trace.file_exporter import (
    FileSpanExporter,
    _normalize_otel_value,
    _readable_span_to_dict,
)


class TestNormalizeOtelValue:
    def test_primitives(self):
        assert _normalize_otel_value(None) is None
        assert _normalize_otel_value("hello") == "hello"
        assert _normalize_otel_value(42) == 42
        assert _normalize_otel_value(3.14) == 3.14
        assert _normalize_otel_value(True) is True

    def test_bytes(self):
        assert _normalize_otel_value(b"hello") == "hello"

    def test_list(self):
        assert _normalize_otel_value([1, 2, 3]) == [1, 2, 3]

    def test_nested_dict(self):
        result = _normalize_otel_value({"key": {"nested": True}})
        assert result == {"key": {"nested": True}}

    def test_mixed_iterable(self):
        result = _normalize_otel_value([1, "two", b"three"])
        assert result == [1, "two", "three"]

    def test_empty_string(self):
        assert _normalize_otel_value("") == ""


class TestReadableSpanToDict:
    def test_basic_span(self):
        span = MagicMock()
        span.attributes = {"gen_ai.system": "google_adk", "gen_ai.completion": "Hello"}
        span.context = MagicMock()
        span.context.span_id = 0xABCD
        span.parent = None
        span.start_time = 1700000000000000000
        span.name = "gen_ai.generate"

        result = _readable_span_to_dict(span)
        assert result["name"] == "gen_ai.generate"
        assert result["span_id"] == "abcd"
        assert result["parent_span_id"] == ""
        assert result["attributes"]["gen_ai.system"] == "google_adk"

    def test_span_with_parent(self):
        span = MagicMock()
        span.attributes = {}
        span.context = MagicMock()
        span.context.span_id = 0x1234
        span.parent = MagicMock()
        span.parent.span_id = 0x5678
        span.start_time = 0
        span.name = "tool_call"

        result = _readable_span_to_dict(span)
        assert result["parent_span_id"] == "5678"


class TestFileSpanExporter:
    def test_export_and_flush(self, tmp_path):
        output = tmp_path / "otel_spans.json"
        exporter = FileSpanExporter(output)

        span = MagicMock()
        span.attributes = {"key": "value"}
        span.context = MagicMock()
        span.context.span_id = 0xAAAA
        span.parent = None
        span.start_time = 1000
        span.name = "test_span"

        exporter.export([span])
        assert not output.exists()

        exporter.flush()
        assert output.exists()

        data = json.loads(output.read_text())
        assert len(data) == 1
        assert data[0]["name"] == "test_span"
        assert data[0]["attributes"]["key"] == "value"

    def test_flush_with_no_spans(self, tmp_path):
        output = tmp_path / "otel_spans.json"
        exporter = FileSpanExporter(output)
        exporter.flush()
        assert not output.exists()

    def test_clear(self, tmp_path):
        output = tmp_path / "otel_spans.json"
        exporter = FileSpanExporter(output)

        span = MagicMock()
        span.attributes = {}
        span.context = MagicMock()
        span.context.span_id = 1
        span.parent = None
        span.start_time = 0
        span.name = "s1"

        exporter.export([span])
        exporter.clear()
        exporter.flush()
        assert not output.exists()

    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "deep" / "nested" / "otel_spans.json"
        exporter = FileSpanExporter(output)

        span = MagicMock()
        span.attributes = {}
        span.context = MagicMock()
        span.context.span_id = 1
        span.parent = None
        span.start_time = 0
        span.name = "s1"

        exporter.export([span])
        exporter.flush()
        assert output.exists()
