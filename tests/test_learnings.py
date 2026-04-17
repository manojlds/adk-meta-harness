"""Tests for learnings module."""

import tempfile
from pathlib import Path

from adk_meta_harness.learnings import Learnings


def test_learnings_create():
    path = Path(tempfile.mktemp(suffix=".md"))
    learnings = Learnings(path)
    assert learnings.entries == []


def test_learnings_add():
    path = Path(tempfile.mktemp(suffix=".md"))
    learnings = Learnings(path)
    learnings.add(
        iteration=1,
        description="Added skills",
        kept=True,
        holdout_score=0.75,
        search_score=0.78,
        failure_patterns=["task_a: timeout"],
        insights=["Skills help with multi-step tasks"],
    )

    content = learnings.get_content()
    assert "Added skills" in content
    assert "0.75" in content
    assert "KEPT" in content
    assert "timeout" in content


def test_learnings_persistence():
    path = Path(tempfile.mktemp(suffix=".md"))
    learnings = Learnings(path)
    learnings.add(iteration=1, description="test1", kept=True, holdout_score=0.5, search_score=0.6)
    learnings.add(
        iteration=2, description="test2", kept=False, holdout_score=0.45, search_score=0.55
    )

    loaded = Learnings(path)
    assert len(loaded.entries) == 2
    assert "test1" in loaded.entries[0]
    assert "test2" in loaded.entries[1]


class TestCompletedIterations:
    def test_empty_learnings(self, tmp_path):
        learnings = Learnings(tmp_path / "learnings.md")
        assert learnings.completed_iterations() == 0

    def test_baseline_only(self, tmp_path):
        learnings = Learnings(tmp_path / "learnings.md")
        learnings.add(
            iteration=0, description="Baseline", kept=True, holdout_score=0.5, search_score=0.5
        )
        assert learnings.completed_iterations() == 0

    def test_counts_non_baseline(self, tmp_path):
        learnings = Learnings(tmp_path / "learnings.md")
        learnings.add(
            iteration=0, description="Baseline", kept=True, holdout_score=0.5, search_score=0.5
        )
        learnings.add(
            iteration=1, description="edit1", kept=True, holdout_score=0.6, search_score=0.6
        )
        learnings.add(
            iteration=2, description="edit2", kept=False, holdout_score=0.4, search_score=0.4
        )
        assert learnings.completed_iterations() == 2

    def test_counts_validation_failed(self, tmp_path):
        learnings = Learnings(tmp_path / "learnings.md")
        learnings.add(
            iteration=0, description="Baseline", kept=True, holdout_score=0.5, search_score=0.5
        )
        learnings.add(
            iteration=1,
            description="VALIDATION FAILED: syntax error",
            kept=False,
            holdout_score=0.0,
            search_score=0.0,
        )
        learnings.add(
            iteration=2, description="edit2", kept=True, holdout_score=0.7, search_score=0.7
        )
        assert learnings.completed_iterations() == 2

    def test_survives_reload(self, tmp_path):
        path = tmp_path / "learnings.md"
        learnings = Learnings(path)
        learnings.add(
            iteration=0, description="Baseline", kept=True, holdout_score=0.5, search_score=0.5
        )
        learnings.add(
            iteration=1, description="edit1", kept=True, holdout_score=0.6, search_score=0.6
        )
        learnings.add(
            iteration=2, description="edit2", kept=False, holdout_score=0.4, search_score=0.4
        )

        reloaded = Learnings(path)
        assert reloaded.completed_iterations() == 2
