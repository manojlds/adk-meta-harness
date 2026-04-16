"""OTel File Span Exporter — writes spans to a per-task JSON file.

This replaces the need for a separate OTel Collector process. The ADK agent
emits spans via the standard OTel Python SDK. A ``FileSpanProcessor`` wraps
a ``FileSpanExporter`` that accumulates spans in-memory and writes them as a
JSON array on ``flush()``. The resulting file is consumed by
``OtelToAtifConverter.convert_file()``.

Usage::

    from adk_meta_harness.trace.file_exporter import FileSpanExporter, setup_file_exporter

    exporter = setup_file_exporter(Path("evaluation/read-file/otel_spans.json"))
    try:
        ...  # run agent
    finally:
        exporter.flush_and_teardown()
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_otel_value(value: Any) -> Any:
    """Normalize OTel attribute values into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, dict):
        return {str(k): _normalize_otel_value(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_normalize_otel_value(v) for v in value]
    return str(value)


def _readable_span_to_dict(span: Any) -> dict[str, Any]:
    """Convert an OpenTelemetry ReadableSpan into a dict for JSON serialization."""
    attrs = {}
    for k, v in dict(getattr(span, "attributes", {}) or {}).items():
        attrs[str(k)] = _normalize_otel_value(v)

    span_id = ""
    parent_span_id = ""
    context = getattr(span, "context", None)
    if context is not None and getattr(context, "span_id", None) is not None:
        span_id = format(context.span_id, "x")

    parent = getattr(span, "parent", None)
    if parent is not None and getattr(parent, "span_id", None) is not None:
        parent_span_id = format(parent.span_id, "x")

    return {
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "start_time": int(getattr(span, "start_time", 0) or 0),
        "name": str(getattr(span, "name", "")),
        "attributes": attrs,
    }


class FileSpanExporter:
    """OTel SpanExporter that accumulates spans and writes them to a JSON file.

    The file format is a JSON array of span dicts, compatible with
    ``OtelToAtifConverter.convert_file()``.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._spans: list[Any] = []

    def export(self, spans: list[Any]) -> None:
        self._spans.extend(spans)

    def flush(self) -> None:
        if not self._spans:
            return
        data = [_readable_span_to_dict(s) for s in self._spans]
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(json.dumps(data, indent=2))
        logger.debug("Wrote %d spans to %s", len(data), self._output_path)

    def shutdown(self) -> None:
        self.flush()

    def clear(self) -> None:
        self._spans.clear()


def setup_file_exporter(output_path: Path) -> FileSpanExporter:
    """Set up a FileSpanExporter with the global tracer provider.

    Creates a ``SimpleSpanProcessor`` wrapping a ``FileSpanExporter`` and
    registers it with the current ``TracerProvider``. If the provider is not
    an ``SDKTracerProvider``, a new one is created and set as the global
    provider.

    Returns the ``FileSpanExporter`` so callers can call ``flush()`` after
    the agent run completes.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = FileSpanExporter(output_path)

    provider = trace.get_tracer_provider()
    if not isinstance(provider, SDKTracerProvider):
        provider = SDKTracerProvider()
        trace.set_tracer_provider(provider)

    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    logger.debug("FileSpanExporter registered writing to %s", output_path)

    return exporter


def teardown_file_exporter(exporter: FileSpanExporter) -> None:
    """Flush and remove the file exporter's processor from the tracer provider.

    Safe to call even if setup failed or the provider doesn't support removal.
    """
    try:
        exporter.flush()
    except Exception:
        logger.debug("Failed to flush FileSpanExporter", exc_info=True)
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

        provider = trace.get_tracer_provider()
        if isinstance(provider, SDKTracerProvider):
            for p in getattr(provider, "_span_processors", []):
                if getattr(p, "span_exporter", None) is exporter:
                    provider._span_processors.remove(p)
                    break
    except Exception:
        logger.debug("Failed to remove FileSpanProcessor", exc_info=True)
