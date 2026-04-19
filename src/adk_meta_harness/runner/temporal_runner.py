from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from adk_meta_harness.task import DEFAULT_AGENT_TIMEOUT_SEC
from adk_meta_harness.task_executor import EvalOutput, evaluate_candidate

DEFAULT_TEMPORAL_SERVER_URL = "localhost:7233"
DEFAULT_TEMPORAL_TASK_QUEUE = "amh-tasks"

logger = logging.getLogger(__name__)

_TEMPORAL_IMPORT_ERROR: Exception | None = None
try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.worker import Worker

    _TEMPORAL_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised via runtime checks.
    _TEMPORAL_AVAILABLE = False
    _TEMPORAL_IMPORT_ERROR = exc
    activity = None  # type: ignore[assignment]
    workflow = None  # type: ignore[assignment]
    Client = None  # type: ignore[assignment]
    Worker = None  # type: ignore[assignment]


def _ensure_temporal_available() -> None:
    if _TEMPORAL_AVAILABLE:
        return
    msg = (
        "Temporal support requires optional dependency 'temporalio'. "
        "Install it with: uv sync --extra temporal"
    )
    raise RuntimeError(msg) from _TEMPORAL_IMPORT_ERROR


@dataclass
class TemporalOptimizeInput:
    dataset: str
    initial_harness: str
    proposer: str = "opencode"
    proposer_model: str | None = None
    model: str | None = None
    iterations: int = 10
    holdout_ratio: float = 0.3
    candidates_dir: str | None = None
    judge: str = "litellm"
    judge_model: str | None = None
    timeout: int = DEFAULT_AGENT_TIMEOUT_SEC

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "initial_harness": self.initial_harness,
            "proposer": self.proposer,
            "proposer_model": self.proposer_model,
            "model": self.model,
            "iterations": self.iterations,
            "holdout_ratio": self.holdout_ratio,
            "candidates_dir": self.candidates_dir,
            "judge": self.judge,
            "judge_model": self.judge_model,
            "timeout": self.timeout,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TemporalOptimizeInput:
        return cls(
            dataset=str(payload["dataset"]),
            initial_harness=str(payload["initial_harness"]),
            proposer=str(payload.get("proposer", "opencode")),
            proposer_model=_optional_str(payload.get("proposer_model")),
            model=_optional_str(payload.get("model")),
            iterations=int(payload.get("iterations", 10)),
            holdout_ratio=float(payload.get("holdout_ratio", 0.3)),
            candidates_dir=_optional_str(payload.get("candidates_dir")),
            judge=str(payload.get("judge", "litellm")),
            judge_model=_optional_str(payload.get("judge_model")),
            timeout=int(payload.get("timeout", DEFAULT_AGENT_TIMEOUT_SEC)),
        )


@dataclass
class TemporalOptimizeOutput:
    best_candidate_path: str
    best_holdout: float
    best_search: float
    iterations_completed: int
    candidates_dir: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "best_candidate_path": self.best_candidate_path,
            "best_holdout": self.best_holdout,
            "best_search": self.best_search,
            "iterations_completed": self.iterations_completed,
            "candidates_dir": self.candidates_dir,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TemporalOptimizeOutput:
        return cls(
            best_candidate_path=str(payload["best_candidate_path"]),
            best_holdout=float(payload["best_holdout"]),
            best_search=float(payload["best_search"]),
            iterations_completed=int(payload["iterations_completed"]),
            candidates_dir=str(payload["candidates_dir"]),
        )


if _TEMPORAL_AVAILABLE:

    @activity.defn
    async def optimize_activity(config_payload: dict[str, Any]) -> dict[str, Any]:
        """Run the full optimization loop in a worker activity."""
        from adk_meta_harness.judge import get_judge
        from adk_meta_harness.outer_loop import OptimizeConfig, optimize

        optimize_input = TemporalOptimizeInput.from_payload(config_payload)
        judge = get_judge(optimize_input.judge, model=optimize_input.judge_model)

        config = OptimizeConfig(
            dataset=Path(optimize_input.dataset),
            initial_harness=Path(optimize_input.initial_harness),
            proposer=optimize_input.proposer,
            proposer_model=optimize_input.proposer_model,
            model=optimize_input.model or "gemini-2.5-flash",
            iterations=optimize_input.iterations,
            holdout_ratio=optimize_input.holdout_ratio,
            candidates_dir=Path(optimize_input.candidates_dir)
            if optimize_input.candidates_dir
            else None,
            judge=judge,
            timeout=optimize_input.timeout,
            # Activities execute inside worker processes and should use the
            # in-process local runner there. Temporal orchestration happens at
            # the workflow/activity level, not by nesting Temporal runners.
            runner="local",
        )

        result = await optimize(config)
        return TemporalOptimizeOutput(
            best_candidate_path=str(result.best_candidate.path),
            best_holdout=result.best_holdout,
            best_search=result.best_search,
            iterations_completed=result.iterations_completed,
            candidates_dir=str(result.candidates_dir),
        ).to_payload()

    @workflow.defn
    class OptimizeWorkflow:
        @workflow.run
        async def run(self, config_payload: dict[str, Any]) -> dict[str, Any]:
            """Execute one optimization run via activities."""
            return await workflow.execute_activity(
                optimize_activity,
                config_payload,
                start_to_close_timeout=timedelta(hours=24),
            )

else:

    async def optimize_activity(config_payload: dict[str, Any]) -> dict[str, Any]:  # type: ignore[misc]
        _ensure_temporal_available()
        msg = "Temporal runtime unavailable"
        raise RuntimeError(msg)

    class OptimizeWorkflow:  # type: ignore[no-redef]
        pass


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _default_workflow_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"amh-optimize-{ts}"


async def start_optimize_workflow(
    optimize_input: TemporalOptimizeInput,
    *,
    server_url: str = DEFAULT_TEMPORAL_SERVER_URL,
    task_queue: str = DEFAULT_TEMPORAL_TASK_QUEUE,
    workflow_id: str | None = None,
) -> tuple[str, str]:
    """Start the optimize workflow and return (workflow_id, run_id)."""
    _ensure_temporal_available()

    client = await Client.connect(server_url)
    handle = await client.start_workflow(
        OptimizeWorkflow.run,
        optimize_input.to_payload(),
        id=workflow_id or _default_workflow_id(),
        task_queue=task_queue,
    )
    run_id = handle.first_execution_run_id
    return handle.id, run_id


async def run_worker(
    *,
    server_url: str = DEFAULT_TEMPORAL_SERVER_URL,
    task_queue: str = DEFAULT_TEMPORAL_TASK_QUEUE,
) -> None:
    """Run a Temporal worker for optimization workflows."""
    _ensure_temporal_available()

    client = await Client.connect(server_url)
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[OptimizeWorkflow],
        activities=[optimize_activity],
    )
    await worker.run()


class TemporalTaskRunner:
    """Temporal-backed runner.

    For `amh optimize --runner temporal`, the CLI starts `OptimizeWorkflow`
    and exits immediately. The workflow runs on Temporal workers.

    For `amh eval --runner temporal`, evaluation still runs locally via
    `evaluate_candidate()` to preserve the TaskRunner interface.
    """

    def __init__(
        self,
        *,
        server_url: str = DEFAULT_TEMPORAL_SERVER_URL,
        task_queue: str = DEFAULT_TEMPORAL_TASK_QUEUE,
    ) -> None:
        self._server_url = server_url
        self._task_queue = task_queue

    @property
    def name(self) -> str:
        return "temporal"

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def task_queue(self) -> str:
        return self._task_queue

    async def evaluate(
        self,
        candidate_dir: Path,
        tasks_dir: Path,
        *,
        model: str | None = None,
        timeout: int = DEFAULT_AGENT_TIMEOUT_SEC,
        search_task_names: list[str] | None = None,
        holdout_task_names: list[str] | None = None,
        judge: object | None = None,
    ) -> EvalOutput:
        if (
            self._server_url != DEFAULT_TEMPORAL_SERVER_URL
            or self._task_queue != DEFAULT_TEMPORAL_TASK_QUEUE
        ):
            logger.warning(
                "TemporalTaskRunner.evaluate runs locally; server_url=%s and "
                "task_queue=%s are ignored",
                self._server_url,
                self._task_queue,
            )

        return await evaluate_candidate(
            candidate_dir=candidate_dir,
            tasks_dir=tasks_dir,
            model=model,
            timeout=timeout,
            search_task_names=search_task_names,
            holdout_task_names=holdout_task_names,
            judge=judge,
        )


__all__ = [
    "DEFAULT_TEMPORAL_SERVER_URL",
    "DEFAULT_TEMPORAL_TASK_QUEUE",
    "OptimizeWorkflow",
    "TemporalOptimizeInput",
    "TemporalOptimizeOutput",
    "TemporalTaskRunner",
    "optimize_activity",
    "run_worker",
    "start_optimize_workflow",
]
