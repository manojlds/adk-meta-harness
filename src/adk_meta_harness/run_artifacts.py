"""Run-scoped artifact management for optimization runs.

Provides directory layout creation, frontier/evolution persistence,
pending-eval tracking, and helpers for resumable outer-loop execution.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL_ITERATION_STATUSES = {"validation_failed", "kept", "discarded"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    root_dir: Path
    run_dir: Path
    candidates_dir: Path
    task_splits_path: Path
    pending_eval_path: Path
    frontier_path: Path
    evolution_summary_path: Path


def init_run_artifacts(root_dir: Path, run_id: str) -> RunArtifacts:
    run_id = _sanitize_run_id(run_id)
    run_dir = root_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        run_id=run_id,
        root_dir=root_dir,
        run_dir=run_dir,
        candidates_dir=candidates_dir,
        task_splits_path=run_dir / "task_splits.json",
        pending_eval_path=run_dir / "pending_eval.json",
        frontier_path=run_dir / "frontier_val.json",
        evolution_summary_path=run_dir / "evolution_summary.jsonl",
    )


def reset_run_state(artifacts: RunArtifacts) -> None:
    """Clear mutable run state so a run can start fresh."""
    for path in (
        artifacts.frontier_path,
        artifacts.pending_eval_path,
        artifacts.evolution_summary_path,
    ):
        path.unlink(missing_ok=True)

    if artifacts.candidates_dir.exists():
        for entry in artifacts.candidates_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)


def write_pending_eval(artifacts: RunArtifacts, payload: dict[str, Any]) -> None:
    enriched = {
        "run_id": artifacts.run_id,
        "updated_at": _utc_now(),
        **payload,
    }
    artifacts.pending_eval_path.write_text(json.dumps(enriched, indent=2, sort_keys=True))


def update_frontier(artifacts: RunArtifacts, payload: dict[str, Any]) -> None:
    enriched = {
        "run_id": artifacts.run_id,
        "updated_at": _utc_now(),
        **payload,
    }
    artifacts.frontier_path.write_text(json.dumps(enriched, indent=2, sort_keys=True))


def load_frontier(artifacts: RunArtifacts) -> dict[str, Any] | None:
    if not artifacts.frontier_path.exists():
        return None
    data = json.loads(artifacts.frontier_path.read_text())
    if not isinstance(data, dict):
        return None
    return data


def append_evolution_row(artifacts: RunArtifacts, row: dict[str, Any]) -> None:
    payload = {
        "run_id": artifacts.run_id,
        "timestamp": _utc_now(),
        **row,
    }
    with artifacts.evolution_summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def read_evolution_rows(artifacts: RunArtifacts) -> list[dict[str, Any]]:
    if not artifacts.evolution_summary_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in artifacts.evolution_summary_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def max_completed_iteration(rows: list[dict[str, Any]]) -> int:
    completed = 0
    for row in rows:
        status = str(row.get("status", ""))
        if status not in TERMINAL_ITERATION_STATUSES:
            continue
        try:
            iteration = int(row.get("iteration", 0))
        except (TypeError, ValueError):
            continue
        if iteration > completed:
            completed = iteration
    return completed


def completed_iterations(rows: list[dict[str, Any]]) -> int:
    """Backward-compatible alias for max_completed_iteration()."""
    return max_completed_iteration(rows)


def latest_final_test_score(rows: list[dict[str, Any]]) -> float | None:
    for row in reversed(rows):
        if row.get("status") != "final_test":
            continue
        score = row.get("test_score")
        try:
            if score is not None:
                return float(score)
        except (TypeError, ValueError):
            continue
    return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sanitize_run_id(run_id: str) -> str:
    candidate = run_id.strip()
    if candidate in {"", ".", ".."}:
        msg = f"Invalid run_id: {run_id!r}"
        raise ValueError(msg)
    if "/" in candidate or "\\" in candidate:
        msg = f"Invalid run_id: {run_id!r}"
        raise ValueError(msg)
    if not RUN_ID_PATTERN.fullmatch(candidate):
        msg = f"Invalid run_id: {run_id!r}"
        raise ValueError(msg)
    return candidate
