from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent, AgentContext
from harbor.environments.base import BaseEnvironment


class AdkHarborAgent(BaseAgent):
    """Harbor agent that runs an ADK harness inside the container.

    The base Docker image (``adk-meta-harness``) has Python, google-adk,
    and adk-meta-harness pre-installed.  The harness files (agent.py,
    config.yaml, system_prompt.md, tools/, etc.) are uploaded into
    ``/app/harness/`` inside the container.

    On ``setup()`` the harness files are uploaded.

    On ``run()`` we execute::

        python -m adk_meta_harness.eval_one \\
            --harness /app/harness \\
            --instruction "<instruction>" \\
            --output /logs/agent \\
            --model <model>

    The ``eval_one`` module runs the agent on a single instruction and
    writes ``trajectory.json`` + ``response.txt`` to ``/logs/agent/``.
    """

    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        harness_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._harness_dir = harness_dir or os.environ.get("AMH_HARNESS_DIR", "/app/harness")

    @staticmethod
    def name() -> str:
        return "adk-meta-harness"

    def version(self) -> str | None:
        try:
            from importlib.metadata import version as pkg_version

            return pkg_version("adk-meta-harness")
        except Exception:
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        harness_src = Path(self._harness_dir)
        if not harness_src.exists():
            harness_src = Path(os.environ.get("AMH_HARNESS_SRC", ""))
        if harness_src.exists():
            await environment.exec("mkdir -p /app/harness", user="root")
            environment.upload_dir(str(harness_src), "/app/harness")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        env: dict[str, str] = {}
        if self.model_name:
            env["AMH_MODEL"] = self.model_name

        cmd = (
            f"python -m adk_meta_harness.eval_one "
            f"--harness /app/harness "
            f"--instruction {escaped_instruction} "
            f"--output /logs/agent"
        )
        if self.model_name:
            cmd += f" --model {shlex.quote(self.model_name)}"

        result = await environment.exec(
            command=cmd,
            cwd="/app",
            env=env,
            timeout_sec=None,
        )

        (self.logs_dir / "exit_code.txt").write_text(str(result.return_code))

    def populate_context_post_run(self, context: AgentContext) -> None:
        trajectory_path = self.logs_dir / "agent" / "trajectory.json"
        if trajectory_path.exists():
            try:
                data = json.loads(trajectory_path.read_text())
                if data.get("final_metrics"):
                    fm = data["final_metrics"]
                    context.n_input_tokens = fm.get("total_prompt_tokens")
                    context.n_output_tokens = fm.get("total_completion_tokens")
                    context.cost_usd = fm.get("total_cost_usd")
            except Exception:
                pass
