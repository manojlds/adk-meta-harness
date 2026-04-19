from __future__ import annotations

from pathlib import Path

import pytest

from adk_meta_harness.runner import temporal_runner
from adk_meta_harness.runner.temporal_runner import (
    TemporalOptimizeInput,
    TemporalOptimizeOutput,
    TemporalTaskRunner,
    start_optimize_workflow,
)
from adk_meta_harness.task_executor import EvalOutput, EvalResult


def test_temporal_optimize_input_payload_roundtrip():
    original = TemporalOptimizeInput(
        dataset="/tmp/tasks",
        initial_harness="/tmp/harness",
        proposer="opencode",
        proposer_model="openai/gpt-4.1",
        model="openai/gpt-4.1-mini",
        iterations=12,
        holdout_ratio=0.25,
        candidates_dir="/tmp/candidates",
        judge="litellm",
        judge_model="openai/gpt-4.1-mini",
        timeout=120,
    )

    payload = original.to_payload()
    restored = TemporalOptimizeInput.from_payload(payload)

    assert restored == original


def test_temporal_optimize_output_payload_roundtrip():
    original = TemporalOptimizeOutput(
        best_candidate_path="/tmp/candidates/v0007",
        best_holdout=0.8,
        best_search=0.9,
        iterations_completed=7,
        candidates_dir="/tmp/candidates",
    )

    payload = original.to_payload()
    restored = TemporalOptimizeOutput.from_payload(payload)

    assert restored == original


@pytest.mark.asyncio
async def test_temporal_runner_evaluate_delegates_to_task_executor(monkeypatch, tmp_path):
    called: dict[str, object] = {}

    async def fake_evaluate_candidate(**kwargs):
        called.update(kwargs)
        return EvalOutput(search_results=[EvalResult(task_name="t1", passed=True, score=1.0)])

    monkeypatch.setattr(temporal_runner, "evaluate_candidate", fake_evaluate_candidate)

    runner = TemporalTaskRunner()
    candidate_dir = tmp_path / "candidate"
    tasks_dir = tmp_path / "tasks"
    out = await runner.evaluate(
        candidate_dir=candidate_dir,
        tasks_dir=tasks_dir,
        model="openai/gpt-4.1-mini",
        timeout=42,
    )

    assert called["candidate_dir"] == candidate_dir
    assert called["tasks_dir"] == tasks_dir
    assert called["model"] == "openai/gpt-4.1-mini"
    assert called["timeout"] == 42
    assert out.search_results[0].task_name == "t1"


@pytest.mark.asyncio
async def test_start_optimize_workflow_requires_temporal_dependency(monkeypatch):
    monkeypatch.setattr(temporal_runner, "_TEMPORAL_AVAILABLE", False)
    monkeypatch.setattr(temporal_runner, "_TEMPORAL_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(RuntimeError, match="Temporal support requires optional dependency"):
        await start_optimize_workflow(
            TemporalOptimizeInput(
                dataset=str(Path("/tmp/tasks")),
                initial_harness=str(Path("/tmp/harness")),
            )
        )
