"""Gate logic — decide whether to keep or discard a proposed harness change.

Following Meta-Harness and canvas-org/meta-agent:
- Keep if holdout score improves.
- Keep if same holdout score and the harness is simpler.
- Discard otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    kept: bool
    reason: str
    previous_holdout: float | None
    current_holdout: float | None
    previous_search: float | None
    current_search: float | None

    @property
    def holdout_delta(self) -> float | None:
        if self.current_holdout is not None and self.previous_holdout is not None:
            return self.current_holdout - self.previous_holdout
        return None

    @property
    def search_delta(self) -> float | None:
        if self.current_search is not None and self.previous_search is not None:
            return self.current_search - self.previous_search
        return None


def gate_decision(
    current_holdout: float | None,
    previous_holdout: float | None,
    current_search: float | None = None,
    previous_search: float | None = None,
    current_complexity: int | None = None,
    previous_complexity: int | None = None,
    tolerance: float = 0.001,
) -> GateResult:
    """Decide whether to keep or discard a proposed harness change.

    Args:
        current_holdout: Holdout score of the proposed harness.
        previous_holdout: Holdout score of the previous best harness.
        current_search: Score on the search (training) set for proposed harness.
        previous_search: Score on the search set for previous best harness.
        current_complexity: Optional complexity metric (e.g., file count, line count).
        previous_complexity: Optional complexity metric for previous best.
        tolerance: Score differences below this are considered equal.

    Returns:
        GateResult with the decision and reasoning.
    """
    if current_holdout is None:
        return GateResult(
            kept=False,
            reason="Holdout score is None (eval failed or not run)",
            previous_holdout=previous_holdout,
            current_holdout=current_holdout,
            previous_search=previous_search,
            current_search=current_search,
        )

    if previous_holdout is None:
        return GateResult(
            kept=True,
            reason="First candidate with a holdout score",
            previous_holdout=previous_holdout,
            current_holdout=current_holdout,
            previous_search=previous_search,
            current_search=current_search,
        )

    # Strictly better holdout
    if current_holdout > previous_holdout + tolerance:
        return GateResult(
            kept=True,
            reason=f"Holdout improved: {previous_holdout:.4f} -> {current_holdout:.4f}",
            previous_holdout=previous_holdout,
            current_holdout=current_holdout,
            previous_search=previous_search,
            current_search=current_search,
        )

    # Same holdout but simpler
    if abs(current_holdout - previous_holdout) <= tolerance:
        if (
            current_complexity is not None
            and previous_complexity is not None
            and current_complexity < previous_complexity
        ):
            return GateResult(
                kept=True,
                reason=(
                    f"Holdout equal ({current_holdout:.4f}), "
                    f"but simpler ({current_complexity} vs "
                    f"{previous_complexity} components)"
                ),
                previous_holdout=previous_holdout,
                current_holdout=current_holdout,
                previous_search=previous_search,
                current_search=current_search,
            )
        return GateResult(
            kept=False,
            reason=(
                f"Holdout not improved "
                f"({previous_holdout:.4f} -> {current_holdout:.4f}), "
                f"no simplification"
            ),
            previous_holdout=previous_holdout,
            current_holdout=current_holdout,
            previous_search=previous_search,
            current_search=current_search,
        )

    # Worse holdout
    return GateResult(
        kept=False,
        reason=f"Holdout regressed: {previous_holdout:.4f} -> {current_holdout:.4f}",
        previous_holdout=previous_holdout,
        current_holdout=current_holdout,
        previous_search=previous_search,
        current_search=current_search,
    )