"""Core optimization outer loop — propose, evaluate, gate, repeat.

This is the heart of adk-meta-harness. Following the Meta-Harness paper:
1. Proposer reads filesystem of all prior candidates, traces, and scores.
2. Proposer proposes one targeted harness change.
3. We evaluate on search + holdout tasks.
4. Gate decides keep/discard based on holdout score.
5. Learnings are accumulated for the next iteration.
6. Repeat.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from adk_meta_harness.candidate import (
    Candidate,
    CandidateDiff,
    create_candidate,
    init_candidates_dir,
)
from adk_meta_harness.gate import gate_decision
from adk_meta_harness.harbor_adapter import evaluate_candidate
from adk_meta_harness.learnings import Learnings
from adk_meta_harness.proposer import get_proposer

if TYPE_CHECKING:
    from adk_meta_harness.judge.base import JudgeProtocol


@dataclass
class OptimizeConfig:
    dataset: Path
    initial_harness: Path
    proposer: str = "opencode"
    proposer_model: str | None = None
    model: str = "gemini-2.5-flash"
    iterations: int = 10
    holdout_ratio: float = 0.3
    candidates_dir: Path | None = None
    judge: JudgeProtocol | None = None
    timeout: int = 300


@dataclass
class OptimizeResult:
    best_candidate: Candidate
    best_holdout: float
    best_search: float
    iterations_completed: int
    candidates_dir: Path
    all_results: list[dict]


async def optimize(config: OptimizeConfig) -> OptimizeResult:
    """Run the meta-harness optimization loop.

    Args:
        config: Optimization configuration.

    Returns:
        OptimizeResult with the best candidate and full history.
    """
    candidates_dir = config.candidates_dir or config.dataset / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    proposer = get_proposer(config.proposer, model=config.proposer_model)
    learnings = Learnings(candidates_dir / "learnings.md")

    # Initialize baseline candidate
    baseline = init_candidates_dir(candidates_dir, config.initial_harness)
    print(f"[v{baseline.version}] Baseline harness initialized")

    # Evaluate baseline
    eval_output = await evaluate_candidate(
        candidate_dir=baseline.path,
        tasks_dir=config.dataset,
        model=config.model,
        timeout=config.timeout,
        judge=config.judge,
    )
    search_results = eval_output.search_results
    holdout_results = eval_output.holdout_results

    baseline_score = _compute_score(search_results, holdout_results)
    search_score = baseline_score.get("search", 0.0)
    holdout_score = baseline_score.get("holdout", search_score)

    baseline.diff = CandidateDiff(
        creation_path=baseline.path,
        score=baseline_score["combined"],
        holdout_score=holdout_score,
        search_score=search_score,
        passed=baseline_score["passed"],
        total=baseline_score["total"],
        description="Initial baseline",
        kept=True,
    )
    baseline.write_meta()
    _link_traces_to_candidate(baseline.path)
    _append_results(candidates_dir, baseline.diff)
    learnings.add(
        iteration=0,
        description="Baseline evaluation",
        kept=True,
        holdout_score=holdout_score,
        search_score=search_score,
        failure_patterns=_extract_failure_patterns(search_results),
    )

    best_candidate = baseline
    best_holdout = holdout_score
    best_search = search_score

    all_results = [baseline_score]

    for iteration in range(1, config.iterations + 1):
        print(f"\n=== Iteration {iteration}/{config.iterations} ===")

        new_version = baseline.version + iteration
        new_candidate = create_candidate(
            candidates_dir=candidates_dir,
            source=best_candidate.path,
            version=new_version,
            parent_version=best_candidate.version,
            description=f"Iteration {iteration}",
        )

        # Generate instruction for the proposer
        instruction = _build_proposer_instruction(iteration, best_holdout, learnings)

        # Proposer edits the harness
        print(f"  Proposing edit with {proposer.name}...")
        edit_result = await proposer.propose_edit(
            candidate_dir=new_candidate.path,
            filesystem_dir=candidates_dir,
            learnings=learnings.get_content(),
            instruction=instruction,
        )
        print(f"  Change: {edit_result.get('description', 'N/A')}")
        print(f"  Type: {edit_result.get('change_type', 'N/A')}")

        # Clear proposer-injected files from the candidate before eval
        _cleanup_proposer_files(new_candidate.path)

        # Evaluate the proposed harness
        eval_output = await evaluate_candidate(
            candidate_dir=new_candidate.path,
            tasks_dir=config.dataset,
            model=config.model,
            timeout=config.timeout,
            judge=config.judge,
        )
        search_results = eval_output.search_results
        holdout_results = eval_output.holdout_results

        proposed_score = _compute_score(search_results, holdout_results)
        proposed_search = proposed_score.get("search", 0.0)
        proposed_holdout = proposed_score.get("holdout", proposed_search)

        _link_traces_to_candidate(new_candidate.path)

        # Gate decision
        complexity_current = _count_harness_files(new_candidate.path)
        complexity_best = _count_harness_files(best_candidate.path)

        gate = gate_decision(
            current_holdout=proposed_holdout,
            previous_holdout=best_holdout,
            current_search=proposed_search,
            previous_search=best_search,
            current_complexity=complexity_current,
            previous_complexity=complexity_best,
        )

        new_candidate.diff = CandidateDiff(
            creation_path=new_candidate.path,
            score=proposed_score["combined"],
            holdout_score=proposed_holdout,
            search_score=proposed_search,
            passed=proposed_score["passed"],
            total=proposed_score["total"],
            description=edit_result.get("description", ""),
            kept=gate.kept,
        )
        new_candidate.write_meta()
        _append_results(candidates_dir, new_candidate.diff)

        print(f"  Holdout: {proposed_holdout:.4f} (prev: {best_holdout:.4f})")
        print(f"  Search: {proposed_search:.4f} (prev: {best_search:.4f})")
        print(f"  Gate: {'KEPT' if gate.kept else 'DISCARDED'} — {gate.reason}")

        # Update learnings
        failure_patterns = _extract_failure_patterns(search_results)
        learnings.add(
            iteration=iteration,
            description=edit_result.get("description", ""),
            kept=gate.kept,
            holdout_score=proposed_holdout,
            search_score=proposed_search,
            failure_patterns=failure_patterns,
        )

        # Update best if kept
        if gate.kept:
            best_candidate = new_candidate
            best_holdout = proposed_holdout
            best_search = proposed_search
        else:
            # Remove the discarded candidate directory
            shutil.rmtree(new_candidate.path, ignore_errors=True)

        all_results.append(proposed_score)

    return OptimizeResult(
        best_candidate=best_candidate,
        best_holdout=best_holdout,
        best_search=best_search,
        iterations_completed=config.iterations,
        candidates_dir=candidates_dir,
        all_results=all_results,
    )


def _compute_score(search_results: list, holdout_results: list) -> dict:
    """Compute combined score from search and holdout results."""
    all_results = search_results + holdout_results
    if not all_results:
        return {"combined": 0.0, "search": 0.0, "holdout": 0.0, "passed": 0, "total": 0}

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    combined = passed / total if total > 0 else 0.0

    search_passed = sum(1 for r in search_results if r.passed)
    search_total = len(search_results)
    search_score = search_passed / search_total if search_total > 0 else 0.0

    holdout_passed = sum(1 for r in holdout_results if r.passed)
    holdout_total = len(holdout_results)
    holdout_score = holdout_passed / holdout_total if holdout_total > 0 else search_score

    return {
        "combined": combined,
        "search": search_score,
        "holdout": holdout_score,
        "passed": passed,
        "total": total,
    }


def _build_proposer_instruction(iteration: int, best_score: float, learnings: Learnings) -> str:
    """Build the instruction for the proposer."""
    return (
        f"Iteration {iteration}. Current best holdout score: {best_score:.4f}.\n\n"
        "Read traces from previous candidates to diagnose failure patterns. "
        "Propose ONE targeted harness change that fixes a class of failures.\n\n"
        "You MUST make at least one concrete file edit in this candidate directory "
        "(agent.py, system_prompt.md, config.yaml, tools/, skills/, callbacks/, or routing/). "
        "Do not return analysis-only output.\n\n"
        "Focus on: system prompts, skills, tool definitions, callbacks, routing, "
        "or agent configuration. Make the smallest effective change.\n\n"
        "After your change, the harness will be evaluated on a holdout set "
        "that you cannot see. Overfitting to specific tasks will be penalized."
    )


def _append_results(candidates_dir: Path, diff: CandidateDiff) -> None:
    """Append a row to results.tsv."""
    results_path = candidates_dir / "results.tsv"
    kept_str = "keep" if diff.kept else "discard" if diff.kept is not None else "pending"
    row = (
        f"{diff.creation_path.name}\t{diff.score:.4f}\t{diff.holdout_score:.4f}\t"
        f"{diff.search_score:.4f}\t{diff.passed}/{diff.total}\t"
        f"{kept_str}\t{diff.description}\n"
    )
    with open(results_path, "a") as f:
        f.write(row)


def _count_harness_files(candidate_dir: Path) -> int:
    """Count the number of harness files (complexity metric)."""
    count = 0
    for f in candidate_dir.rglob("*"):
        if (
            f.is_file()
            and not f.name.startswith(".")
            and "__pycache__" not in f.parts
            and f.suffix in (".py", ".md", ".yaml", ".yml", ".toml")
        ):
            count += 1
    return count


def _extract_failure_patterns(search_results: list) -> list[str]:
    """Extract brief failure patterns from eval results."""
    patterns = []
    for r in search_results:
        if not r.passed and r.error:
            patterns.append(f"{r.task_name}: {r.error[:100]}")
    return patterns[:10]


def _link_traces_to_candidate(candidate_dir: Path) -> None:
    """Copy trajectory.json from evaluation/<task>/ into traces/<task>.json.

    The proposer template instructs the agent to browse ``traces/`` for
    prior trajectory data.  The evaluate_candidate function writes
    ``evaluation/<task>/trajectory.json``, so we mirror those files into
    the ``traces/`` directory so the proposer can actually find them.
    """
    traces_dir = candidate_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    eval_dir = candidate_dir / "evaluation"
    if not eval_dir.exists():
        return
    for task_dir in eval_dir.iterdir():
        if not task_dir.is_dir():
            continue
        traj = task_dir / "trajectory.json"
        if traj.exists():
            dest = traces_dir / f"{task_dir.name}.json"
            dest.write_text(traj.read_text())


def _cleanup_proposer_files(candidate_dir: Path) -> None:
    """Remove proposer-injected files from the candidate directory."""
    for name in ("PROPOSER.md", "learnings.md"):
        p = candidate_dir / name
        if p.exists():
            p.unlink()
