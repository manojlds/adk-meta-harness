"""Tests for candidate module."""

import tempfile
from pathlib import Path

from adk_meta_harness.candidate import (
    Candidate,
    CandidateDiff,
    create_candidate,
    discover_candidates,
    find_best_candidate,
    init_candidates_dir,
    max_version,
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


def _make_candidate(candidates_dir, version, kept, holdout_score, search_score=0.5):
    """Helper to create a candidate with meta.json on disk."""
    d = candidates_dir / f"v{version:04d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "agent.py").write_text("# agent")
    diff = CandidateDiff(
        creation_path=d,
        score=holdout_score,
        holdout_score=holdout_score,
        search_score=search_score,
        passed=int(holdout_score * 10),
        total=10,
        kept=kept,
        description=f"iteration {version}",
    )
    c = Candidate(version=version, path=d, diff=diff)
    c.write_meta()
    return c


class TestDiscoverCandidates:
    def test_empty_dir(self, tmp_path):
        assert discover_candidates(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert discover_candidates(tmp_path / "nope") == []

    def test_discovers_candidates(self, tmp_path):
        _make_candidate(tmp_path, 0, True, 0.5)
        _make_candidate(tmp_path, 1, True, 0.6)
        _make_candidate(tmp_path, 2, False, 0.4)

        found = discover_candidates(tmp_path)
        assert len(found) == 3
        assert [c.version for c in found] == [0, 1, 2]

    def test_skips_dirs_without_meta(self, tmp_path):
        _make_candidate(tmp_path, 0, True, 0.5)
        # A directory without meta.json should be skipped
        (tmp_path / "v0001").mkdir()
        (tmp_path / "v0001" / "agent.py").write_text("# no meta")

        found = discover_candidates(tmp_path)
        assert len(found) == 1
        assert found[0].version == 0

    def test_skips_non_candidate_dirs(self, tmp_path):
        _make_candidate(tmp_path, 0, True, 0.5)
        # Non-candidate directories (no v prefix)
        (tmp_path / "traces").mkdir()
        (tmp_path / "results.tsv").write_text("header\n")

        found = discover_candidates(tmp_path)
        assert len(found) == 1

    def test_sorted_by_version(self, tmp_path):
        _make_candidate(tmp_path, 3, False, 0.3)
        _make_candidate(tmp_path, 0, True, 0.5)
        _make_candidate(tmp_path, 1, True, 0.6)

        found = discover_candidates(tmp_path)
        assert [c.version for c in found] == [0, 1, 3]


class TestFindBestCandidate:
    def test_no_candidates(self):
        assert find_best_candidate([]) is None

    def test_picks_highest_holdout(self, tmp_path):
        _make_candidate(tmp_path, 0, True, 0.5)
        _make_candidate(tmp_path, 1, True, 0.8)
        _make_candidate(tmp_path, 2, True, 0.6)

        found = discover_candidates(tmp_path)
        best = find_best_candidate(found)
        assert best is not None
        assert best.version == 1

    def test_ignores_discarded(self, tmp_path):
        _make_candidate(tmp_path, 0, True, 0.5)
        _make_candidate(tmp_path, 1, False, 0.9)  # discarded

        found = discover_candidates(tmp_path)
        best = find_best_candidate(found)
        assert best is not None
        assert best.version == 0

    def test_all_discarded(self, tmp_path):
        _make_candidate(tmp_path, 0, False, 0.5)
        _make_candidate(tmp_path, 1, False, 0.3)

        found = discover_candidates(tmp_path)
        assert find_best_candidate(found) is None

    def test_tie_prefers_newest(self, tmp_path):
        """When holdout scores are equal, prefer the newest candidate.

        The gate keeps equal-holdout candidates only when they are simpler,
        so the newest kept candidate at the same score is the simplest.
        """
        _make_candidate(tmp_path, 0, True, 0.7)
        _make_candidate(tmp_path, 1, True, 0.7)  # same holdout, kept = simpler
        _make_candidate(tmp_path, 3, True, 0.7)  # same holdout, kept = even simpler

        found = discover_candidates(tmp_path)
        best = find_best_candidate(found)
        assert best is not None
        assert best.version == 3


class TestMaxVersion:
    def test_empty(self):
        assert max_version([]) == -1

    def test_single(self, tmp_path):
        _make_candidate(tmp_path, 0, True, 0.5)
        found = discover_candidates(tmp_path)
        assert max_version(found) == 0

    def test_multiple(self, tmp_path):
        _make_candidate(tmp_path, 0, True, 0.5)
        _make_candidate(tmp_path, 1, True, 0.6)
        _make_candidate(tmp_path, 5, False, 0.3)

        found = discover_candidates(tmp_path)
        assert max_version(found) == 5
