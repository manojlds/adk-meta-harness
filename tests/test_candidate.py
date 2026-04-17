"""Tests for candidate module."""

import tempfile
from pathlib import Path

from adk_meta_harness.candidate import (
    Candidate,
    CandidateDiff,
    create_candidate,
    init_candidates_dir,
)


def test_candidate_paths():
    d = Path(tempfile.mkdtemp())
    (d / "agent.py").write_text("# agent")
    (d / "system_prompt.md").write_text("hello")
    (d / "config.yaml").write_text("model: test")

    candidate = Candidate(version=0, path=d)
    assert candidate.agent_py == d / "agent.py"
    assert candidate.config_yaml == d / "config.yaml"
    assert candidate.system_prompt_md == d / "system_prompt.md"
    assert candidate.skills_dir == d / "skills"


def test_candidate_meta_roundtrip():
    d = Path(tempfile.mkdtemp())
    (d / "agent.py").write_text("# agent")

    diff = CandidateDiff(
        creation_path=d,
        score=0.75,
        holdout_score=0.70,
        search_score=0.78,
        passed=7,
        total=10,
        kept=True,
        description="test candidate",
    )
    candidate = Candidate(version=2, path=d, diff=diff)
    candidate.write_meta()

    loaded = Candidate.load_meta(d)
    assert loaded.version == 2
    assert loaded.diff is not None
    assert loaded.diff.score == 0.75
    assert loaded.diff.holdout_score == 0.70
    assert loaded.diff.passed == 7
    assert loaded.diff.total == 10
    assert loaded.diff.kept is True


def test_create_candidate():
    src = Path(tempfile.mkdtemp())
    (src / "agent.py").write_text("# original")
    (src / "system_prompt.md").write_text("be helpful")

    exp_dir = Path(tempfile.mkdtemp())
    candidate = create_candidate(
        candidates_dir=exp_dir,
        source=src,
        version=0,
        description="baseline",
    )

    assert (candidate.path / "agent.py").read_text() == "# original"
    assert (candidate.path / "system_prompt.md").read_text() == "be helpful"
    assert candidate.version == 0


def test_init_candidates_dir():
    src = Path(tempfile.mkdtemp())
    (src / "agent.py").write_text("# agent")

    exp_dir = Path(tempfile.mkdtemp()) / "candidates"
    candidate = init_candidates_dir(exp_dir, src)

    assert candidate.version == 0
    assert (exp_dir / "results.tsv").exists()
    assert (exp_dir / "learnings.md").exists()
    assert (candidate.path / "agent.py").read_text() == "# agent"
