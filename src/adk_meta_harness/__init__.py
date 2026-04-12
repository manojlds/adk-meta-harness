"""adk-meta-harness: Meta-harness optimization for Google ADK agents."""

from adk_meta_harness.candidate import Candidate, CandidateDiff
from adk_meta_harness.gate import GateResult
from adk_meta_harness.outer_loop import optimize

__all__ = ["Candidate", "CandidateDiff", "GateResult", "optimize"]
__version__ = "0.1.0"