from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_AGENT_TIMEOUT_SEC = 300
DEFAULT_VERIFIER_TIMEOUT_SEC = 300
DEFAULT_SETUP_TIMEOUT_SEC = 60
DEFAULT_TEARDOWN_TIMEOUT_SEC = 60


@dataclass
class TaskConfig:
    name: str
    path: Path
    instruction: str
    agent_timeout: int = DEFAULT_AGENT_TIMEOUT_SEC
    verifier_timeout: int = DEFAULT_VERIFIER_TIMEOUT_SEC
    setup_timeout: int = DEFAULT_SETUP_TIMEOUT_SEC
    teardown_timeout: int = DEFAULT_TEARDOWN_TIMEOUT_SEC
    env: dict[str, str] = field(default_factory=dict)
    setup_script: Path | None = None
    teardown_script: Path | None = None
    verifier_script: Path | None = None
    fixtures_dir: Path | None = None
    description: str = ""

    @classmethod
    def from_path(cls, task_path: Path, name: str) -> TaskConfig:
        task_path = task_path.resolve()
        task_toml = _load_task_toml(task_path / "task.toml")

        metadata = _as_dict(task_toml.get("metadata"))
        agent = _as_dict(task_toml.get("agent"))
        verifier = _as_dict(task_toml.get("verifier"))
        scripts = _as_dict(task_toml.get("scripts"))

        # New unified [env] section with fallback support for Harbor-era sections.
        merged_env: dict[str, str] = {}
        for section_name in ("verifier", "environment", "solution"):
            section = _as_dict(task_toml.get(section_name))
            merged_env.update(_to_str_dict(section.get("env")))
        merged_env.update(_to_str_dict(task_toml.get("env")))

        setup_script = (task_path / "scripts" / "setup.sh").resolve()
        teardown_script = (task_path / "scripts" / "teardown.sh").resolve()
        verifier_script = (task_path / "tests" / "test.sh").resolve()
        fixtures_dir = (task_path / "fixtures").resolve()

        return cls(
            name=name,
            path=task_path,
            instruction=read_instruction(task_path),
            agent_timeout=_read_timeout(agent, "timeout_sec", DEFAULT_AGENT_TIMEOUT_SEC),
            verifier_timeout=_read_timeout(verifier, "timeout_sec", DEFAULT_VERIFIER_TIMEOUT_SEC),
            setup_timeout=_read_timeout(scripts, "setup_timeout_sec", DEFAULT_SETUP_TIMEOUT_SEC),
            teardown_timeout=_read_timeout(
                scripts,
                "teardown_timeout_sec",
                DEFAULT_TEARDOWN_TIMEOUT_SEC,
            ),
            env=merged_env,
            setup_script=setup_script if setup_script.exists() else None,
            teardown_script=teardown_script if teardown_script.exists() else None,
            verifier_script=verifier_script if verifier_script.exists() else None,
            fixtures_dir=fixtures_dir if fixtures_dir.is_dir() else None,
            description=str(metadata.get("description", "")),
        )


def discover_tasks(tasks_dir: Path) -> list[TaskConfig]:
    if not tasks_dir.exists() or not tasks_dir.is_dir():
        return []

    tasks: list[TaskConfig] = []
    for entry in sorted(tasks_dir.iterdir()):
        if not entry.is_dir():
            continue
        task_path = resolve_task_path(tasks_dir, entry.name)
        if task_path.exists() and _has_task_files(task_path):
            tasks.append(TaskConfig.from_path(task_path, entry.name))
    return tasks


def resolve_task_path(tasks_dir: Path, task_name: str) -> Path:
    """Resolve flat or Harbor-era nested task directories."""
    flat = tasks_dir / task_name
    nested = tasks_dir / task_name / task_name
    if _has_task_files(flat):
        return flat
    if _has_task_files(nested):
        return nested
    return flat


def read_instruction(task_path: Path) -> str:
    instruction_file = task_path / "instruction.md"
    if instruction_file.exists() and instruction_file.is_file():
        return instruction_file.read_text()
    return ""


def _has_task_files(task_path: Path) -> bool:
    return (task_path / "instruction.md").exists() or (task_path / "task.toml").exists()


def _load_task_toml(task_toml_path: Path) -> dict[str, Any]:
    if not task_toml_path.exists() or not task_toml_path.is_file():
        return {}

    try:
        data = tomllib.loads(task_toml_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        logger.warning("Failed to parse task TOML at %s: %s", task_toml_path, exc)
        return {}
    except OSError as exc:
        logger.warning("Failed to read task TOML at %s: %s", task_toml_path, exc)
        return {}

    if isinstance(data, dict):
        return data
    return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, item in value.items():
        if item is None:
            continue
        out[str(key)] = str(item)
    return out


def _read_timeout(data: dict[str, Any], key: str, default: int) -> int:
    raw = data.get(key, default)
    try:
        parsed = int(float(raw))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
