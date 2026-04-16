from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harbor import (
    AgentConfig,
    EnvironmentConfig,
    EnvironmentType,
    TaskConfig,
    Trial,
    TrialConfig,
    VerifierConfig,
)

from adk_meta_harness.harbor_adapter import (
    EvalOutput,
    EvalResult,
    _discover_tasks,
    _resolve_task_path,
)
from adk_meta_harness.runner.harbor_agent import AdkHarborAgent
from adk_meta_harness.trace.atif import AtifTrajectory

if TYPE_CHECKING:
    from adk_meta_harness.judge.base import JudgeProtocol


_DOCKER_IMAGE_NAME = "adk-meta-harness:latest"


class HarborTaskRunner:
    """Runs tasks inside Docker containers via the Harbor SDK.

    Each task gets its own container built from the task's
    ``environment/Dockerfile`` (which should use ``adk-meta-harness:latest``
    as its base image).  The harness is uploaded, the ADK agent runs
    inside the container, and the verifier also runs inside.

    Requires Docker and the ``harbor`` package.
    """

    def __init__(
        self,
        *,
        base_image: str | None = None,
        env_type: EnvironmentType = EnvironmentType.DOCKER,
    ) -> None:
        self._base_image = base_image or os.environ.get("AMH_BASE_IMAGE", _DOCKER_IMAGE_NAME)
        self._env_type = env_type

    @property
    def name(self) -> str:
        return "harbor"

    async def evaluate(
        self,
        candidate_dir: Path,
        tasks_dir: Path,
        *,
        model: str | None = None,
        timeout: int = 300,
        search_task_names: list[str] | None = None,
        holdout_task_names: list[str] | None = None,
        judge: JudgeProtocol | None = None,
    ) -> EvalOutput:
        all_tasks = _discover_tasks(tasks_dir)
        search_set = search_task_names or [t for t in all_tasks]
        holdout_set = holdout_task_names or []

        output = EvalOutput()

        for task_name in all_tasks:
            task_path = _resolve_task_path(tasks_dir, task_name)
            if not task_path.exists():
                continue

            result = await self._run_trial(
                candidate_dir=candidate_dir,
                task_path=task_path,
                task_name=task_name,
                model=model,
                timeout=timeout,
            )

            if task_name in holdout_set:
                output.holdout_results.append(result)
            else:
                output.search_results.append(result)

        return output

    async def _run_trial(
        self,
        candidate_dir: Path,
        task_path: Path,
        task_name: str,
        model: str | None,
        timeout: int,
    ) -> EvalResult:
        agent_config = AgentConfig(
            import_path=AdkHarborAgent.import_path(),
            model_name=model,
            extra_env={"AMH_HARNESS_DIR": str(candidate_dir.resolve())},
        )

        env_config = EnvironmentConfig(
            type=self._env_type,
            docker_image=self._base_image,
        )

        verifier_config = VerifierConfig(
            timeout_sec=float(timeout),
        )

        trial_config = TrialConfig(
            task=TaskConfig(path=task_path.resolve(), name=task_name),
            agent=agent_config,
            environment=env_config,
            verifier=verifier_config,
            trial_name=f"amh-{task_name}",
        )

        trial = Trial(trial_config)
        trial_result = await trial.run()

        trajectory: AtifTrajectory | None = None
        if trial_result.trial_uri:
            trial_dir = Path(trial_result.trial_uri)
            traj_file = trial_dir / "agent" / "trajectory.json"
            if traj_file.exists():
                try:
                    trajectory = AtifTrajectory.from_json_file(traj_file)
                except Exception:
                    pass

        reward = 0.0
        passed = False
        if trial_result.verifier_result and trial_result.verifier_result.rewards:
            reward = float(trial_result.verifier_result.rewards.get("reward", 0.0))
            passed = reward >= 0.5

        error = None
        if trial_result.exception_info:
            error = str(trial_result.exception_info)

        return EvalResult(
            task_name=task_name,
            passed=passed,
            score=reward,
            trajectory=trajectory,
            error=error,
        )
