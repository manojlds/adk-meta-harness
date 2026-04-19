"""CLI entry point for adk-meta-harness.

Also callable as `amh` via the alias in pyproject.toml.

Model precedence:
    --model CLI flag (runtime override) > config.yaml (harness) > agent default
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    from adk_meta_harness.runner.temporal_runner import (
        DEFAULT_TEMPORAL_SERVER_URL,
        DEFAULT_TEMPORAL_TASK_QUEUE,
    )

    parser = argparse.ArgumentParser(
        prog="adk-meta-harness",
        description="Meta-harness optimization for Google ADK agents",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    judge_help = (
        "Judge backend: litellm, adk, opencode, pi, or a custom CLI command (default: litellm)"
    )

    # optimize subcommand
    opt = subparsers.add_parser("optimize", help="Run the optimization loop")
    opt.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to task/dataset directory",
    )
    opt.add_argument(
        "--initial-harness",
        type=Path,
        required=True,
        help="Path to the initial harness candidate directory",
    )
    opt.add_argument(
        "--proposer",
        default="opencode",
        choices=["opencode", "pi"],
        help="Proposer CLI to use (default: opencode)",
    )
    opt.add_argument(
        "--proposer-model",
        default=None,
        help="Model override for the proposer CLI",
    )
    opt.add_argument(
        "--model",
        default=None,
        help=(
            "Runtime override for the ADK agent model. "
            "If not set, reads from config.yaml in the harness. "
            "If config.yaml has no model, uses the agent's default."
        ),
    )
    opt.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of optimization iterations (default: 10)",
    )
    opt.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.3,
        help="Fraction of tasks to hold out for gating (default: 0.3)",
    )
    opt.add_argument(
        "--candidates-dir",
        type=Path,
        default=None,
        help="Directory to store candidates, traces, and learnings",
    )
    opt.add_argument(
        "--judge",
        default="litellm",
        help=judge_help,
    )
    opt.add_argument(
        "--judge-model",
        default=None,
        help="Model for the judge (e.g. openai/glm-5). Defaults depend on the judge backend.",
    )
    opt.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds (default: 300)",
    )
    opt.add_argument(
        "--runner",
        default="local",
        choices=["local", "temporal"],
        help="Task runner (default: local)",
    )
    opt.add_argument(
        "--server",
        default=DEFAULT_TEMPORAL_SERVER_URL,
        help=(
            "Temporal server address for --runner temporal "
            f"(default: {DEFAULT_TEMPORAL_SERVER_URL})"
        ),
    )
    opt.add_argument(
        "--task-queue",
        default=DEFAULT_TEMPORAL_TASK_QUEUE,
        help=(
            f"Temporal task queue for --runner temporal (default: {DEFAULT_TEMPORAL_TASK_QUEUE})"
        ),
    )
    opt.add_argument(
        "--workflow-id",
        default=None,
        help="Optional Temporal workflow ID for --runner temporal",
    )

    # eval subcommand
    ev = subparsers.add_parser("eval", help="Evaluate a single harness candidate")
    ev.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to the harness candidate directory",
    )
    ev.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to task/dataset directory",
    )
    ev.add_argument(
        "--judge",
        default="litellm",
        help=judge_help,
    )
    ev.add_argument(
        "--judge-model",
        default=None,
        help="Model for the judge (e.g. openai/glm-5). Defaults depend on the judge backend.",
    )
    ev.add_argument(
        "--model",
        default=None,
        help=(
            "Runtime override for the ADK agent model. "
            "If not set, reads from config.yaml. "
            "If config.yaml has no model, uses the agent's default."
        ),
    )
    ev.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds (default: 300)",
    )
    ev.add_argument(
        "--runner",
        default="local",
        choices=["local", "temporal"],
        help="Task runner (default: local)",
    )
    ev.add_argument(
        "--server",
        default=DEFAULT_TEMPORAL_SERVER_URL,
        help=(
            "Temporal server address for --runner temporal "
            f"(default: {DEFAULT_TEMPORAL_SERVER_URL})"
        ),
    )
    ev.add_argument(
        "--task-queue",
        default=DEFAULT_TEMPORAL_TASK_QUEUE,
        help=(
            f"Temporal task queue for --runner temporal (default: {DEFAULT_TEMPORAL_TASK_QUEUE})"
        ),
    )

    # worker subcommand
    wk = subparsers.add_parser("worker", help="Run a Temporal worker")
    wk.add_argument(
        "--server",
        default=DEFAULT_TEMPORAL_SERVER_URL,
        help=f"Temporal server address (default: {DEFAULT_TEMPORAL_SERVER_URL})",
    )
    wk.add_argument(
        "--task-queue",
        default=DEFAULT_TEMPORAL_TASK_QUEUE,
        help=f"Temporal task queue (default: {DEFAULT_TEMPORAL_TASK_QUEUE})",
    )

    args = parser.parse_args()

    if args.command == "optimize":
        if args.runner == "temporal":
            from adk_meta_harness.runner.temporal_runner import (
                TemporalOptimizeInput,
                start_optimize_workflow,
            )

            workflow_id, run_id = asyncio.run(
                start_optimize_workflow(
                    TemporalOptimizeInput(
                        dataset=str(args.dataset),
                        initial_harness=str(args.initial_harness),
                        proposer=args.proposer,
                        proposer_model=args.proposer_model,
                        model=args.model,
                        iterations=args.iterations,
                        holdout_ratio=args.holdout_ratio,
                        candidates_dir=str(args.candidates_dir) if args.candidates_dir else None,
                        judge=args.judge,
                        judge_model=args.judge_model,
                        timeout=args.timeout,
                    ),
                    server_url=args.server,
                    task_queue=args.task_queue,
                    workflow_id=args.workflow_id,
                )
            )
            print("\nStarted Temporal optimization workflow")
            print(f"Workflow ID: {workflow_id}")
            if run_id:
                print(f"Run ID: {run_id}")
            return

        from adk_meta_harness.judge import get_judge
        from adk_meta_harness.outer_loop import OptimizeConfig, optimize

        judge = get_judge(args.judge, model=args.judge_model)

        runner_kwargs = {}

        config = OptimizeConfig(
            dataset=args.dataset,
            initial_harness=args.initial_harness,
            proposer=args.proposer,
            proposer_model=args.proposer_model,
            model=args.model,
            iterations=args.iterations,
            holdout_ratio=args.holdout_ratio,
            candidates_dir=args.candidates_dir,
            timeout=args.timeout,
            judge=judge,
            runner=args.runner,
            runner_kwargs=runner_kwargs,
        )
        result = asyncio.run(optimize(config))
        print("\nOptimization complete!")
        print(f"Best holdout: {result.best_holdout:.4f}")
        print(f"Best search: {result.best_search:.4f}")
        print(f"Best candidate: {result.best_candidate.path}")
        print(f"Iterations: {result.iterations_completed}")

    elif args.command == "eval":
        from adk_meta_harness.judge import get_judge
        from adk_meta_harness.runner import get_runner

        judge = get_judge(args.judge, model=args.judge_model)
        runner_kwargs = {}
        if args.runner == "temporal":
            runner_kwargs = {
                "server_url": args.server,
                "task_queue": args.task_queue,
            }
        task_runner = get_runner(args.runner, **runner_kwargs)

        eval_output = asyncio.run(
            task_runner.evaluate(
                candidate_dir=args.candidate,
                tasks_dir=args.dataset,
                model=args.model,
                timeout=args.timeout,
                judge=judge,
            )
        )
        all_results = eval_output.search_results + eval_output.holdout_results
        passed = sum(1 for r in all_results if r.passed)
        total = len(all_results)
        print(f"\nResults: {passed}/{total} passed ({passed / total:.1%})")
        for r in all_results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.task_name}")

    elif args.command == "worker":
        from adk_meta_harness.runner.temporal_runner import run_worker

        print(f"Starting Temporal worker on {args.server} (task queue: {args.task_queue})")
        asyncio.run(run_worker(server_url=args.server, task_queue=args.task_queue))


if __name__ == "__main__":
    main()
