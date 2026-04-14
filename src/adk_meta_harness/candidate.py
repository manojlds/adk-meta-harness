"""Candidate harness representation.

A candidate is a snapshot of all mutable harness files at a given iteration.
It lives on disk as a directory the proposer can read and edit directly.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

CANDIDATE_DIR_PREFIX = "v"


@dataclass
class CandidateDiff:
    creation_path: Path
    score: float | None = None
    holdout_score: float | None = None
    search_score: float | None = None
    passed: int = 0
    total: int = 0
    description: str = ""
    kept: bool | None = None

    def summary_row(self) -> str:
        kept_str = "keep" if self.kept else "discard" if self.kept is not None else "pending"
        return (
            f"{self.creation_path.name}\t"
            f"{self.score:.4f}\t"
            f"{self.holdout_score:.4f}\t"
            f"{self.search_score:.4f}\t"
            f"{self.passed}/{self.total}\t"
            f"{kept_str}\t"
            f"{self.description}"
        )


@dataclass
class Candidate:
    version: int
    path: Path
    parent_version: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    diff: CandidateDiff | None = None

    @property
    def agent_py(self) -> Path:
        return self.path / "agent.py"

    @property
    def config_yaml(self) -> Path:
        return self.path / "config.yaml"

    @property
    def system_prompt_md(self) -> Path:
        return self.path / "system_prompt.md"

    @property
    def skills_dir(self) -> Path:
        return self.path / "skills"

    @property
    def tools_dir(self) -> Path:
        return self.path / "tools"

    @property
    def callbacks_dir(self) -> Path:
        return self.path / "callbacks"

    @property
    def routing_dir(self) -> Path:
        return self.path / "routing"

    @property
    def traces_dir(self) -> Path:
        return self.path / "traces"

    @property
    def evaluation_dir(self) -> Path:
        return self.path / "evaluation"

    @property
    def validation_dir(self) -> Path:
        return self.path / "validation"

    @property
    def proposal_dir(self) -> Path:
        return self.path / "proposal"

    @property
    def meta_json(self) -> Path:
        return self.path / "meta.json"

    def write_meta(self) -> None:
        data = {
            "version": self.version,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
        }
        if self.diff:
            data["diff"] = {
                "score": self.diff.score,
                "holdout_score": self.diff.holdout_score,
                "search_score": self.diff.search_score,
                "passed": self.diff.passed,
                "total": self.diff.total,
                "kept": self.diff.kept,
                "description": self.diff.description,
            }
        self.meta_json.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_meta(cls, path: Path) -> Candidate:
        data = json.loads((path / "meta.json").read_text())
        diff_data = data.get("diff")
        diff = None
        if diff_data:
            diff = CandidateDiff(
                creation_path=path,
                score=diff_data.get("score"),
                holdout_score=diff_data.get("holdout_score"),
                search_score=diff_data.get("search_score"),
                passed=diff_data.get("passed", 0),
                total=diff_data.get("total", 0),
                kept=diff_data.get("kept"),
                description=diff_data.get("description", ""),
            )
        return cls(
            version=data["version"],
            path=path,
            parent_version=data.get("parent_version"),
            created_at=data.get("created_at", ""),
            diff=diff,
        )


def create_candidate(
    candidates_dir: Path,
    source: Path,
    version: int,
    parent_version: int | None = None,
    description: str = "",
) -> Candidate:
    dest = candidates_dir / f"{CANDIDATE_DIR_PREFIX}{version:04d}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    candidate = Candidate(
        version=version,
        path=dest,
        parent_version=parent_version,
        diff=CandidateDiff(creation_path=dest, description=description),
    )
    candidate.write_meta()
    return candidate


def init_candidates_dir(candidates_dir: Path, initial_harness: Path) -> Candidate:
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / "traces").mkdir(exist_ok=True)
    candidate = create_candidate(
        candidates_dir=candidates_dir,
        source=initial_harness,
        version=0,
        description="Initial baseline harness",
    )
    # Create standard subdirectories in the candidate
    for subdir in ("traces", "evaluation", "validation", "proposal"):
        (candidate.path / subdir).mkdir(exist_ok=True)
    results_path = candidates_dir / "results.tsv"
    if not results_path.exists():
        results_path.write_text(
            "version\tscore\tholdout_score\tsearch_score\tpassed\tkept\tdescription\n"
        )
    learnings_path = candidates_dir / "learnings.md"
    if not learnings_path.exists():
        learnings_path.write_text(
            "# Learnings\n\nAccumulated insights from harness optimization.\n\n"
        )
    return candidate