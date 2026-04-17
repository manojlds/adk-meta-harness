"""Learnings accumulator — persistent memory across optimization iterations.

Following the pattern from auto-harness and meta-agent: the proposer reads
learnings.md before each iteration to avoid repeating failed changes and to
build on successful ones.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


class Learnings:
    """Accumulates insights from each optimization iteration.

    Written to disk as learnings.md so the proposer can read it.
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[str] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        content = self.path.read_text()
        entries = []
        current = []
        for line in content.split("\n"):
            if line.startswith("## Iteration") or line.startswith("## Baseline"):
                if current:
                    entries.append("\n".join(current))
                current = [line]
            elif current:
                current.append(line)
        if current:
            entries.append("\n".join(current))
        self.entries = entries

    def add(
        self,
        iteration: int,
        description: str,
        kept: bool,
        holdout_score: float | None,
        search_score: float | None,
        failure_patterns: list[str] | None = None,
        insights: list[str] | None = None,
    ) -> None:
        """Add a learning entry from an iteration."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        status = "KEPT" if kept else "DISCARDED"
        lines = [
            f"## Iteration {iteration} ({status}) — {timestamp}",
            f"Change: {description}",
        ]
        if holdout_score is not None:
            lines.append(f"Holdout score: {holdout_score:.4f}")
        if search_score is not None:
            lines.append(f"Search score: {search_score:.4f}")
        if failure_patterns:
            lines.append("Failure patterns:")
            for pattern in failure_patterns:
                lines.append(f"  - {pattern}")
        if insights:
            lines.append("Insights:")
            for insight in insights:
                lines.append(f"  - {insight}")
        lines.append("")
        entry = "\n".join(lines)
        self.entries.append(entry)
        self._write()

    def completed_iterations(self) -> int:
        """Return the number of completed iterations recorded in learnings.

        Each iteration (kept, discarded, or validation-failed) writes an
        entry.  This is the most reliable count because it includes
        validation-failed iterations whose candidate dirs were deleted
        and excludes interrupted iterations that never finished.

        The baseline (iteration 0) is not counted.
        """
        import re

        count = 0
        for entry in self.entries:
            m = re.match(r"## Iteration (\d+)", entry)
            if m and int(m.group(1)) > 0:
                count += 1
        return count

    def get_content(self) -> str:
        """Get the full learnings content for the proposer."""
        return self.path.read_text() if self.path.exists() else ""

    def _write(self) -> None:
        header = (
            "# Learnings\n\n"
            "Accumulated insights from harness optimization iterations.\n"
            "The proposer reads this before each iteration to avoid repeating\n"
            "failed changes and to build on successful ones.\n\n"
        )
        self.path.write_text(header + "\n\n".join(self.entries) + "\n")
