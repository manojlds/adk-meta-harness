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

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from adk_meta_harness.candidate import (
    Candidate,
    CandidateDiff,
    create_candidate,
    discover_candidates,
    init_candidates_dir,
    max_version,
)
from adk_meta_harness.gate import gate_decision
from adk_meta_harness.learnings import Learnings
from adk_meta_harness.proposer import get_proposer
from adk_meta_harness.run_artifacts import (
    append_evolution_row,
    init_run_artifacts,
    latest_final_test_score,
    load_frontier,
    max_completed_iteration,
    read_evolution_rows,
    update_frontier,
    write_pending_eval,
)
from adk_meta_harness.splits import TaskSplits, split_task_names
from adk_meta_harness.task import discover_tasks
from adk_meta_harness.validate import validate_candidate

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
    test_ratio: float = 0.2
    split_seed: int = 42
    run_id: str | None = None
    candidates_dir: Path | None = None
    judge: JudgeProtocol | None = None
    timeout: int = 300
    runner: str = "local"
    runner_kwargs: dict | None = None


@dataclass
class OptimizeResult:
    best_candidate: Candidate
    best_holdout: float
    best_search: float
    best_test: float | None
    iterations_completed: int
    candidates_dir: Path
    run_id: str
    all_results: list[dict]


async def optimize(config: OptimizeConfig) -> OptimizeResult:
    """Run the meta-harness optimization loop.

    Supports resuming a crashed or interrupted run via run-scoped
    artifacts in ``candidates/runs/<run_id>/``. If frontier/evolution
    artifacts are present for the run, the loop resumes from the last
    completed iteration instead of re-running baseline and prior steps.

    Args:
        config: Optimization configuration.

    Returns:
        OptimizeResult with the best candidate and full history.
    """
    candidates_root_dir = config.candidates_dir or config.dataset / "candidates"
    candidates_root_dir.mkdir(parents=True, exist_ok=True)

    run_id = config.run_id or _default_run_id()
    artifacts = init_run_artifacts(candidates_root_dir, run_id)

    all_task_names = [task.name for task in discover_tasks(config.dataset)]
    task_splits = _load_or_create_task_splits(
        run_dir=artifacts.run_dir,
        task_names=all_task_names,
        holdout_ratio=config.holdout_ratio,
        test_ratio=config.test_ratio,
        split_seed=config.split_seed,
    )
    print(
        "Task splits: "
        f"search={len(task_splits.search_task_names)} "
        f"holdout={len(task_splits.holdout_task_names)} "
        f"test={len(task_splits.test_task_names)} "
        f"(seed={task_splits.seed})"
    )

    from adk_meta_harness.runner import get_runner

    task_runner = get_runner(config.runner, **(config.runner_kwargs or {}))
    proposer = get_proposer(config.proposer, model=config.proposer_model)
    learnings = Learnings(artifacts.run_dir / "learnings.md")

    existing = discover_candidates(artifacts.candidates_dir)
    rows = read_evolution_rows(artifacts)
    all_results = _results_from_evolution_rows(rows)
    iterations_completed = max_completed_iteration(rows)
    start_iteration = iterations_completed + 1
    best_test: float | None = latest_final_test_score(rows)
    best_candidate_changed = False

    frontier = load_frontier(artifacts)
    resumed = False
    if frontier:
        best_payload = frontier.get("best")
        if isinstance(best_payload, dict):
            candidate_path = best_payload.get("candidate_path")
            if candidate_path:
                best_path = Path(str(candidate_path))
                if (best_path / "meta.json").exists():
                    best_candidate = Candidate.load_meta(best_path)
                    best_holdout = float(best_payload.get("holdout_score", 0.0))
                    best_search = float(best_payload.get("search_score", 0.0))
                    resumed = True
                    print(
                        f"Resuming run {run_id} at iteration {start_iteration} "
                        f"(best v{best_candidate.version}, holdout={best_holdout:.4f})"
                    )

    if not resumed:
        # Fresh run — initialize baseline candidate
        baseline = init_candidates_dir(artifacts.candidates_dir, config.initial_harness)
        print(f"[v{baseline.version}] Baseline harness initialized")

        # Evaluate baseline
        eval_output = await task_runner.evaluate(
            candidate_dir=baseline.path,
            tasks_dir=config.dataset,
            model=config.model,
            timeout=config.timeout,
            search_task_names=task_splits.search_task_names,
            holdout_task_names=task_splits.holdout_task_names,
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
        append_evolution_row(
            artifacts,
            {
                "iteration": 0,
                "status": "baseline",
                "version": baseline.version,
                "parent_version": baseline.parent_version,
                "candidate_path": str(baseline.path.resolve()),
                "description": "Initial baseline",
                "change_type": "baseline",
                "combined_score": baseline_score["combined"],
                "search_score": search_score,
                "holdout_score": holdout_score,
                "passed": baseline_score["passed"],
                "total": baseline_score["total"],
                "gate_kept": True,
                "gate_reason": "Baseline candidate",
            },
        )
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
        iterations_completed = 0
        start_iteration = 1
        best_candidate_changed = True
        update_frontier(
            artifacts,
            _frontier_payload(
                best_candidate=best_candidate,
                best_holdout=best_holdout,
                best_search=best_search,
                best_test=best_test,
                iterations_completed=iterations_completed,
            ),
        )

    # Determine the next version number within this run.
    existing = discover_candidates(artifacts.candidates_dir)
    next_version = max_version(existing) + 1 if existing else 1

    for iteration in range(start_iteration, config.iterations + 1):
        print(f"\n=== Iteration {iteration}/{config.iterations} ===")

        new_version = next_version + (iteration - start_iteration)
        new_candidate = create_candidate(
            candidates_dir=artifacts.candidates_dir,
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
            filesystem_dir=artifacts.run_dir,
            learnings=learnings.get_content(),
            instruction=instruction,
        )
        print(f"  Change: {edit_result.get('description', 'N/A')}")
        print(f"  Type: {edit_result.get('change_type', 'N/A')}")
        write_pending_eval(
            artifacts,
            {
                "iteration": iteration,
                "parent_version": best_candidate.version,
                "candidate": {
                    "version": new_candidate.version,
                    "path": str(new_candidate.path.resolve()),
                },
                "proposal": {
                    "description": edit_result.get("description", ""),
                    "change_type": edit_result.get("change_type", "unknown"),
                    "diff_summary": edit_result.get("diff_summary", ""),
                },
            },
        )

        # Clear proposer-injected files from the candidate before eval
        _cleanup_proposer_files(new_candidate.path)

        # Validate candidate before expensive evaluation
        validation = validate_candidate(new_candidate.path)
        if not validation.valid:
            print("  Validation FAILED:")
            for err in validation.errors:
                print(f"    {err}")
            new_candidate.diff = CandidateDiff(
                creation_path=new_candidate.path,
                score=0.0,
                holdout_score=0.0,
                search_score=0.0,
                passed=0,
                total=0,
                description=f"Validation failed: {'; '.join(validation.errors[:3])}",
                kept=False,
            )
            new_candidate.write_meta()
            append_evolution_row(
                artifacts,
                {
                    "iteration": iteration,
                    "status": "validation_failed",
                    "version": new_candidate.version,
                    "parent_version": new_candidate.parent_version,
                    "candidate_path": str(new_candidate.path.resolve()),
                    "description": edit_result.get("description", ""),
                    "change_type": edit_result.get("change_type", "unknown"),
                    "diff_summary": edit_result.get("diff_summary", ""),
                    "combined_score": 0.0,
                    "search_score": 0.0,
                    "holdout_score": 0.0,
                    "passed": 0,
                    "total": 0,
                    "validation_status": "failed",
                    "validation_errors": validation.errors,
                    "gate_kept": False,
                    "gate_reason": "Validation failed",
                },
            )
            learnings.add(
                iteration=iteration,
                description=f"VALIDATION FAILED: {'; '.join(validation.errors[:3])}",
                kept=False,
                holdout_score=0.0,
                search_score=0.0,
                failure_patterns=[f"validation: {e}" for e in validation.errors],
            )
            all_results.append(
                {
                    "combined": 0.0,
                    "search": 0.0,
                    "holdout": 0.0,
                    "passed": 0,
                    "total": 0,
                }
            )
            iterations_completed = iteration
            continue
        if validation.warnings:
            for warn in validation.warnings:
                print(f"  Validation warning: {warn}")

        # Evaluate the proposed harness
        eval_output = await task_runner.evaluate(
            candidate_dir=new_candidate.path,
            tasks_dir=config.dataset,
            model=config.model,
            timeout=config.timeout,
            search_task_names=task_splits.search_task_names,
            holdout_task_names=task_splits.holdout_task_names,
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
        append_evolution_row(
            artifacts,
            {
                "iteration": iteration,
                "status": "kept" if gate.kept else "discarded",
                "version": new_candidate.version,
                "parent_version": new_candidate.parent_version,
                "candidate_path": str(new_candidate.path.resolve()),
                "description": edit_result.get("description", ""),
                "change_type": edit_result.get("change_type", "unknown"),
                "diff_summary": edit_result.get("diff_summary", ""),
                "combined_score": proposed_score["combined"],
                "search_score": proposed_search,
                "holdout_score": proposed_holdout,
                "passed": proposed_score["passed"],
                "total": proposed_score["total"],
                "validation_status": "passed",
                "gate_kept": gate.kept,
                "gate_reason": gate.reason,
                "search_delta": gate.search_delta,
                "holdout_delta": gate.holdout_delta,
            },
        )

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
            best_candidate_changed = True
            update_frontier(
                artifacts,
                _frontier_payload(
                    best_candidate=best_candidate,
                    best_holdout=best_holdout,
                    best_search=best_search,
                    best_test=best_test,
                    iterations_completed=iteration,
                ),
            )
        else:
            # Keep discarded candidates on disk so the proposer can
            # learn from failed attempts (traces, diffs, scores).
            pass

        all_results.append(proposed_score)
        iterations_completed = iteration

    should_run_final_test = bool(task_splits.test_task_names) and (
        best_test is None or best_candidate_changed
    )
    if should_run_final_test:
        print(f"\n=== Final test evaluation ({len(task_splits.test_task_names)} tasks) ===")
        # evaluate() has search/holdout slots only. We pass test tasks through
        # search_task_names here and interpret the resulting search score as test score.
        final_output = await task_runner.evaluate(
            candidate_dir=best_candidate.path,
            tasks_dir=config.dataset,
            model=config.model,
            timeout=config.timeout,
            search_task_names=task_splits.test_task_names,
            holdout_task_names=[],
            judge=config.judge,
        )
        final_score = _compute_score(final_output.search_results, [])
        test_score_from_search = final_score["search"]
        best_test = test_score_from_search
        print(f"  Final test: {best_test:.4f}")
        append_evolution_row(
            artifacts,
            {
                "iteration": iterations_completed,
                "status": "final_test",
                "version": best_candidate.version,
                "candidate_path": str(best_candidate.path.resolve()),
                "test_score": best_test,
                "task_count": len(task_splits.test_task_names),
            },
        )
    elif best_test is not None:
        print(f"\nFinal test already recorded: {best_test:.4f}")

    update_frontier(
        artifacts,
        _frontier_payload(
            best_candidate=best_candidate,
            best_holdout=best_holdout,
            best_search=best_search,
            best_test=best_test,
            iterations_completed=iterations_completed,
        ),
    )

    return OptimizeResult(
        best_candidate=best_candidate,
        best_holdout=best_holdout,
        best_search=best_search,
        best_test=best_test,
        iterations_completed=iterations_completed,
        candidates_dir=artifacts.candidates_dir,
        run_id=run_id,
        all_results=all_results,
    )


def _default_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid4().hex[:8]}"


def _load_or_create_task_splits(
    *,
    run_dir: Path,
    task_names: list[str],
    holdout_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> TaskSplits:
    manifest_path = run_dir / "task_splits.json"
    normalized_task_names = sorted({name for name in task_names if name})

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        splits_payload = payload.get("splits", payload)
        cached_task_names = payload.get("task_names", []) if isinstance(payload, dict) else []
        cached_task_names_set = sorted({str(name) for name in cached_task_names if name})
        cached_splits = TaskSplits.from_dict(splits_payload)
        if (
            cached_splits.holdout_ratio != holdout_ratio
            or cached_splits.test_ratio != test_ratio
            or cached_splits.seed != split_seed
        ):
            msg = (
                "Existing task_splits.json uses different split parameters "
                "(holdout_ratio/test_ratio/split_seed). Use a new run_id."
            )
            raise ValueError(msg)
        if cached_task_names_set != normalized_task_names:
            msg = (
                "Existing task_splits.json does not match current dataset task names. "
                "Use a new run_id for changed task sets."
            )
            raise ValueError(msg)
        return cached_splits

    splits = split_task_names(
        normalized_task_names,
        holdout_ratio=holdout_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "task_count": len(normalized_task_names),
        "task_names": normalized_task_names,
        "splits": splits.to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return splits


def _results_from_evolution_rows(rows: list[dict]) -> list[dict]:
    statuses = {"baseline", "validation_failed", "kept", "discarded"}
    results: list[dict] = []
    for row in rows:
        if row.get("status") not in statuses:
            continue
        try:
            results.append(
                {
                    "combined": float(row.get("combined_score", 0.0)),
                    "search": float(row.get("search_score", 0.0)),
                    "holdout": float(row.get("holdout_score", 0.0)),
                    "passed": int(row.get("passed", 0)),
                    "total": int(row.get("total", 0)),
                }
            )
        except (TypeError, ValueError):
            continue
    return results


def _frontier_payload(
    *,
    best_candidate: Candidate,
    best_holdout: float,
    best_search: float,
    best_test: float | None,
    iterations_completed: int,
) -> dict:
    return {
        "iterations_completed": iterations_completed,
        "best_test": best_test,
        "best": {
            "version": best_candidate.version,
            "candidate_path": str(best_candidate.path.resolve()),
            "holdout_score": best_holdout,
            "search_score": best_search,
        },
    }


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
