from __future__ import annotations

import json

import pytest

from adk_meta_harness.run_artifacts import (
    append_evolution_row,
    init_run_artifacts,
    latest_final_test_score,
    load_frontier,
    max_completed_iteration,
    read_evolution_rows,
    reset_run_state,
    update_frontier,
    write_pending_eval,
)


def test_init_run_artifacts_creates_run_and_candidate_dirs(tmp_path):
    artifacts = init_run_artifacts(tmp_path / "candidates", "run-1")

    assert artifacts.run_dir.exists()
    assert artifacts.candidates_dir.exists()
    assert artifacts.frontier_path.name == "frontier_val.json"
    assert artifacts.evolution_summary_path.name == "evolution_summary.jsonl"


def test_frontier_pending_and_evolution_roundtrip(tmp_path):
    artifacts = init_run_artifacts(tmp_path / "candidates", "run-2")

    write_pending_eval(
        artifacts,
        {
            "iteration": 1,
            "proposal": {"description": "tighten prompt"},
        },
    )
    pending = json.loads(artifacts.pending_eval_path.read_text())
    assert pending["run_id"] == "run-2"
    assert pending["iteration"] == 1

    update_frontier(
        artifacts,
        {
            "iterations_completed": 3,
            "best": {
                "version": 7,
                "candidate_path": "/tmp/candidates/v0007",
                "holdout_score": 0.8,
                "search_score": 0.9,
            },
        },
    )
    frontier = load_frontier(artifacts)
    assert frontier is not None
    assert frontier["run_id"] == "run-2"
    assert frontier["best"]["version"] == 7

    append_evolution_row(artifacts, {"iteration": 0, "status": "baseline", "version": 0})
    append_evolution_row(artifacts, {"iteration": 1, "status": "kept", "version": 1})
    append_evolution_row(
        artifacts,
        {"iteration": 1, "status": "final_test", "version": 1, "test_score": 0.75},
    )
    rows = read_evolution_rows(artifacts)

    assert len(rows) == 3
    assert rows[0]["status"] == "baseline"
    assert max_completed_iteration(rows) == 1
    assert latest_final_test_score(rows) == 0.75


def test_init_run_artifacts_rejects_unsafe_run_id(tmp_path):
    with pytest.raises(ValueError, match="Invalid run_id"):
        init_run_artifacts(tmp_path / "candidates", "../../etc")


def test_reset_run_state_clears_mutable_artifacts_and_candidates(tmp_path):
    artifacts = init_run_artifacts(tmp_path / "candidates", "run-3")

    write_pending_eval(artifacts, {"iteration": 1})
    update_frontier(artifacts, {"iterations_completed": 1, "best": {"version": 1}})
    append_evolution_row(artifacts, {"iteration": 1, "status": "kept", "version": 1})
    candidate_dir = artifacts.candidates_dir / "v0001"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "meta.json").write_text("{}")

    reset_run_state(artifacts)

    assert not artifacts.pending_eval_path.exists()
    assert not artifacts.frontier_path.exists()
    assert not artifacts.evolution_summary_path.exists()
    assert list(artifacts.candidates_dir.iterdir()) == []
