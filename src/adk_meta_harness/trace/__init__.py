"""Trace module — ATIF trajectory models and OTel conversion.

ATIF (Agent Trajectory Interchange Format) is a standardized JSON
format for logging agent execution traces. We use it as the canonical
trace format that the proposer reads from prior candidates.

The pipeline is:
    ADK Agent (OTel spans) → OTel Collector → OTel → ATIF converter → trajectory.json

The proposer reads trajectory.json from prior candidate directories
to diagnose failures.
"""

from adk_meta_harness.trace.atif import (
    AtifAgent,
    AtifFinalMetrics,
    AtifMetrics,
    AtifStep,
    AtifTrajectory,
)
from adk_meta_harness.trace.otel_to_atif import OtelToAtifConverter
from adk_meta_harness.trace.reward import parse_reward

__all__ = [
    "AtifAgent",
    "AtifFinalMetrics",
    "AtifMetrics",
    "AtifStep",
    "AtifTrajectory",
    "OtelToAtifConverter",
    "parse_reward",
]
