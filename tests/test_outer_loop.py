from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from adk_meta_harness.learnings import Learnings
from adk_meta_harness.outer_loop import (
    OptimizeConfig,
    _build_proposer_instruction,
    _cleanup_proposer_files,
    _compute_score,
    _count_harness_files,
    _extract_failure_patterns,
    _link_traces_to_candidate,
    optimize,
)
from adk_meta_harness.splits import split_task_names
from adk_meta_harness.task_executor import EvalOutput, EvalResult
from adk_meta_harness.validate import ValidationResult


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
        from adk_meta_harness.task_executor import EvalResult

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


@pytest.mark.asyncio
async def test_optimize_uses_search_holdout_splits_and_runs_final_test_once(
    monkeypatch,
    tmp_path,
):
    dataset = tmp_path / "tasks"
    dataset.mkdir()
    for i in range(10):
        task_dir = dataset / f"task-{i:02d}"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("Do task")

    initial_harness = tmp_path / "initial_harness"
    initial_harness.mkdir()
    (initial_harness / "agent.py").write_text("agent = object()\nroot_agent = agent\n")
    (initial_harness / "system_prompt.md").write_text("You are an agent")
    (initial_harness / "config.yaml").write_text("model: gemini-2.5-flash\n")

    expected = split_task_names(
        [f"task-{i:02d}" for i in range(10)],
        holdout_ratio=0.3,
        test_ratio=0.2,
        seed=123,
    )

    class FakeRunner:
        def __init__(self):
            self.calls: list[dict] = []

        async def evaluate(self, **kwargs):
            self.calls.append(kwargs)
            search_names = kwargs.get("search_task_names") or []
            holdout_names = kwargs.get("holdout_task_names") or []
            return EvalOutput(
                search_results=[
                    EvalResult(task_name=name, passed=True, score=1.0) for name in search_names
                ],
                holdout_results=[
                    EvalResult(task_name=name, passed=True, score=1.0) for name in holdout_names
                ],
            )

    class FakeProposer:
        name = "fake"

        async def propose_edit(self, **kwargs):
            return {
                "description": "no-op",
                "change_type": "harness",
            }

    fake_runner = FakeRunner()
    monkeypatch.setattr("adk_meta_harness.runner.get_runner", lambda *_a, **_k: fake_runner)
    monkeypatch.setattr(
        "adk_meta_harness.outer_loop.get_proposer", lambda *_a, **_k: FakeProposer()
    )
    monkeypatch.setattr(
        "adk_meta_harness.outer_loop.validate_candidate",
        lambda *_a, **_k: ValidationResult(valid=True),
    )

    result = await optimize(
        OptimizeConfig(
            dataset=dataset,
            initial_harness=initial_harness,
            proposer="opencode",
            model="gemini-2.5-flash",
            iterations=1,
            holdout_ratio=0.3,
            test_ratio=0.2,
            split_seed=123,
            candidates_dir=tmp_path / "candidates",
            run_id="phase1-test",
        )
    )

    assert len(fake_runner.calls) == 3

    baseline_call = fake_runner.calls[0]
    iter_call = fake_runner.calls[1]
    final_call = fake_runner.calls[2]

    assert set(baseline_call["search_task_names"]) == set(expected.search_task_names)
    assert set(baseline_call["holdout_task_names"]) == set(expected.holdout_task_names)

    assert set(iter_call["search_task_names"]) == set(expected.search_task_names)
    assert set(iter_call["holdout_task_names"]) == set(expected.holdout_task_names)

    assert set(final_call["search_task_names"]) == set(expected.test_task_names)
    assert final_call["holdout_task_names"] == []

    assert result.best_test == 1.0
    assert result.run_id == "phase1-test"

    split_manifest = tmp_path / "candidates" / "runs" / "phase1-test" / "task_splits.json"
    assert split_manifest.exists()
    payload = json.loads(split_manifest.read_text())
    assert payload["splits"]["seed"] == 123

    run_dir = tmp_path / "candidates" / "runs" / "phase1-test"
    assert (run_dir / "pending_eval.json").exists()
    assert (run_dir / "frontier_val.json").exists()
    assert (run_dir / "evolution_summary.jsonl").exists()
    lines = (run_dir / "evolution_summary.jsonl").read_text().strip().splitlines()
    statuses = [json.loads(line)["status"] for line in lines]
    assert statuses[0] == "baseline"
    assert "final_test" in statuses


@pytest.mark.asyncio
async def test_optimize_resume_uses_run_artifacts_without_rerunning_eval(monkeypatch, tmp_path):
    dataset = tmp_path / "tasks"
    dataset.mkdir()
    for i in range(8):
        task_dir = dataset / f"task-{i:02d}"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("Do task")

    initial_harness = tmp_path / "initial_harness"
    initial_harness.mkdir()
    (initial_harness / "agent.py").write_text("agent = object()\nroot_agent = agent\n")
    (initial_harness / "system_prompt.md").write_text("You are an agent")
    (initial_harness / "config.yaml").write_text("model: gemini-2.5-flash\n")

    class FakeRunner:
        def __init__(self):
            self.calls: list[dict] = []

        async def evaluate(self, **kwargs):
            self.calls.append(kwargs)
            search_names = kwargs.get("search_task_names") or []
            holdout_names = kwargs.get("holdout_task_names") or []
            return EvalOutput(
                search_results=[
                    EvalResult(task_name=name, passed=True, score=1.0) for name in search_names
                ],
                holdout_results=[
                    EvalResult(task_name=name, passed=True, score=1.0) for name in holdout_names
                ],
            )

    class FakeProposer:
        name = "fake"

        async def propose_edit(self, **kwargs):
            return {
                "description": "no-op",
                "change_type": "harness",
            }

    fake_runner = FakeRunner()
    monkeypatch.setattr("adk_meta_harness.runner.get_runner", lambda *_a, **_k: fake_runner)
    monkeypatch.setattr(
        "adk_meta_harness.outer_loop.get_proposer", lambda *_a, **_k: FakeProposer()
    )
    monkeypatch.setattr(
        "adk_meta_harness.outer_loop.validate_candidate",
        lambda *_a, **_k: ValidationResult(valid=True),
    )

    await optimize(
        OptimizeConfig(
            dataset=dataset,
            initial_harness=initial_harness,
            proposer="opencode",
            model="gemini-2.5-flash",
            iterations=1,
            holdout_ratio=0.25,
            test_ratio=0.25,
            split_seed=7,
            candidates_dir=tmp_path / "candidates",
            run_id="resume-test",
        )
    )
    assert len(fake_runner.calls) == 3

    fake_runner.calls.clear()
    resumed = await optimize(
        OptimizeConfig(
            dataset=dataset,
            initial_harness=initial_harness,
            proposer="opencode",
            model="gemini-2.5-flash",
            iterations=1,
            holdout_ratio=0.25,
            test_ratio=0.25,
            split_seed=7,
            candidates_dir=tmp_path / "candidates",
            run_id="resume-test",
        )
    )

    assert fake_runner.calls == []
    assert resumed.iterations_completed == 1
    assert resumed.best_test == 1.0
