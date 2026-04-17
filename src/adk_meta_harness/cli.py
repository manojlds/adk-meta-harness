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
        help="Path to Harbor task/dataset directory",
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
        choices=["local", "harbor"],
        help="Task runner: local (in-process) or harbor (Docker containers, default: local)",
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
        help="Path to Harbor task/dataset directory",
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
        choices=["local", "harbor"],
        help="Task runner: local (in-process) or harbor (Docker containers, default: local)",
    )

    args = parser.parse_args()

    if args.command == "optimize":
        from adk_meta_harness.judge import get_judge
        from adk_meta_harness.outer_loop import OptimizeConfig, optimize

        judge = get_judge(args.judge, model=args.judge_model)

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
        task_runner = get_runner(args.runner)

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


if __name__ == "__main__":
    main()
