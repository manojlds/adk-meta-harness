from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adk_meta_harness.gate import GateResult
from adk_meta_harness.outer_loop import (
    _build_proposer_instruction,
    _cleanup_proposer_files,
    _compute_score,
    _count_harness_files,
    _extract_failure_patterns,
    _link_traces_to_candidate,
)
from adk_meta_harness.learnings import Learnings


class TestComputeScore:
    def test_empty_results(self):
        assert _compute_score([], []) == {
            "combined": 0.0,
            "search": 0.0,
            "holdout": 0.0,
            "passed": 0,
            "total": 0,
        }

    def test_all_passed(self):
        results = [MagicMock(passed=True), MagicMock(passed=True), MagicMock(passed=True)]
        score = _compute_score(results, [])
        assert score["combined"] == 1.0
        assert score["search"] == 1.0
        assert score["holdout"] == 1.0
        assert score["passed"] == 3
        assert score["total"] == 3

    def test_mixed_results(self):
        search = [MagicMock(passed=True), MagicMock(passed=False)]
        holdout = [MagicMock(passed=False)]
        score = _compute_score(search, holdout)
        assert score["combined"] == 1.0 / 3
        assert score["search"] == 0.5
        assert score["holdout"] == 0.0

    def test_no_holdout_falls_back_to_search(self):
        search = [MagicMock(passed=True), MagicMock(passed=True)]
        score = _compute_score(search, [])
        assert score["holdout"] == score["search"]


class TestLinkTracesToCandidate:
    def test_copies_trajectory_json(self, tmp_path):
        cand_dir = tmp_path / "v0001"
        cand_dir.mkdir()
        eval_dir = cand_dir / "evaluation" / "read-file"
        eval_dir.mkdir(parents=True)
        traj = {"schema_version": "ATIF-v1.4", "steps": []}
        (eval_dir / "trajectory.json").write_text(json.dumps(traj))

        _link_traces_to_candidate(cand_dir)

        dest = cand_dir / "traces" / "read-file.json"
        assert dest.exists()
        assert json.loads(dest.read_text()) == traj

    def test_skips_non_dir_entries(self, tmp_path):
        cand_dir = tmp_path / "v0001"
        cand_dir.mkdir()
        eval_dir = cand_dir / "evaluation"
        eval_dir.mkdir()
        (eval_dir / "notes.txt").write_text("not a dir")
        (cand_dir / "traces").mkdir()

        _link_traces_to_candidate(cand_dir)

        traces = list((cand_dir / "traces").iterdir())
        assert traces == []

    def test_skips_missing_trajectory(self, tmp_path):
        cand_dir = tmp_path / "v0001"
        cand_dir.mkdir()
        eval_dir = cand_dir / "evaluation" / "task1"
        eval_dir.mkdir(parents=True)
        (cand_dir / "traces").mkdir()

        _link_traces_to_candidate(cand_dir)

        assert not (cand_dir / "traces" / "task1.json").exists()

    def test_no_evaluation_dir(self, tmp_path):
        cand_dir = tmp_path / "v0001"
        cand_dir.mkdir()

        _link_traces_to_candidate(cand_dir)

        assert (cand_dir / "traces").is_dir()
        assert list((cand_dir / "traces").iterdir()) == []


class TestBuildProposerInstruction:
    def test_includes_iteration_and_score(self, tmp_path):
        learnings = Learnings(tmp_path / "learnings.md")
        result = _build_proposer_instruction(3, 0.75, learnings)
        assert "Iteration 3" in result
        assert "0.7500" in result

    def test_includes_must_edit(self, tmp_path):
        learnings = Learnings(tmp_path / "learnings2.md")
        result = _build_proposer_instruction(1, 0.0, learnings)
        assert "MUST make at least one concrete file edit" in result


class TestExtractFailurePatterns:
    def test_extracts_error_from_failed_results(self):
        r1 = MagicMock(passed=True, error=None)
        r2 = MagicMock(passed=False, error="timeout: exceeded 300s")
        r3 = MagicMock(passed=False, error="import error: module not found")
        patterns = _extract_failure_patterns([r1, r2, r3])
        assert len(patterns) == 2
        assert "timeout" in patterns[0]
        assert "import error" in patterns[1]

    def test_limits_to_ten(self):
        results = [MagicMock(passed=False, error=f"err{i}") for i in range(15)]
        patterns = _extract_failure_patterns(results)
        assert len(patterns) == 10

    def test_no_failures(self):
        results = [MagicMock(passed=True, error=None) for _ in range(5)]
        assert _extract_failure_patterns(results) == []

    def test_truncates_long_errors(self):
        r = MagicMock(passed=False, error="x" * 200)
        r.task_name = "mytask"
        patterns = _extract_failure_patterns([r])
        assert len(patterns) == 1
        assert patterns[0].startswith("mytask:")
        assert len(patterns[0]) <= 100 + len("mytask") + 2


class TestCountHarnessFiles:
    def test_counts_relevant_files(self, tmp_path):
        (tmp_path / "agent.py").write_text("")
        (tmp_path / "system_prompt.md").write_text("")
        (tmp_path / "config.yaml").write_text("")
        (tmp_path / "notes.txt").write_text("")
        (tmp_path / "data.json").write_text("")
        _pycache = tmp_path / "__pycache__"
        _pycache.mkdir()
        (_pycache / "foo.pyc").write_bytes(b"bytecode")

        assert _count_harness_files(tmp_path) == 3

    def test_empty_dir(self, tmp_path):
        assert _count_harness_files(tmp_path) == 0


class TestCleanupProposerFiles:
    def test_removes_proposer_and_learnings_files(self, tmp_path):
        proposer = tmp_path / "PROPOSER.md"
        learnings = tmp_path / "learnings.md"
        proposer.write_text("instructions")
        learnings.write_text("insights")

        _cleanup_proposer_files(tmp_path)

        assert not proposer.exists()
        assert not learnings.exists()

    def test_no_files_is_noop(self, tmp_path):
        _cleanup_proposer_files(tmp_path)


class TestComputeScoreWithRealObjects:
    def test_with_eval_result_objects(self):
        from adk_meta_harness.harbor_adapter import EvalResult

        search = [
            EvalResult(task_name="t1", passed=True, score=1.0),
            EvalResult(task_name="t2", passed=False, score=0.0),
        ]
        holdout = [
            EvalResult(task_name="t3", passed=True, score=1.0),
        ]
        score = _compute_score(search, holdout)
        assert score["search"] == 0.5
        assert score["holdout"] == 1.0
        assert score["combined"] == pytest.approx(2 / 3)
