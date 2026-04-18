from __future__ import annotations

import os
import sys

import pytest

from adk_meta_harness.task import TaskConfig
from adk_meta_harness.task_executor import (
    _build_script_env,
    _discover_tasks,
    _ensure_importable,
    _ensure_user_instruction_step,
    _read_instruction,
    _resolve_task_path,
    _temporary_cwd,
    _temporary_env,
    run_setup,
    run_teardown,
)
from adk_meta_harness.trace.atif import AtifStep, AtifTrajectory


class TestEnsureImportable:
    def test_creates_init_py(self, tmp_path):
        d = tmp_path / "agent_dir"
        d.mkdir()
        (d / "agent.py").write_text("agent = None")
        assert not (d / "__init__.py").exists()

        _ensure_importable(d)

        assert (d / "__init__.py").exists()

    def test_does_not_overwrite_init_py(self, tmp_path):
        d = tmp_path / "agent_dir"
        d.mkdir()
        (d / "__init__.py").write_text("# existing")

        _ensure_importable(d)

        assert (d / "__init__.py").read_text() == "# existing"

    def test_adds_candidate_dir_to_sys_path(self, tmp_path):
        d = tmp_path / "agent_dir"
        d.mkdir()
        (d / "agent.py").write_text("agent = None")

        original_path = sys.path.copy()
        _ensure_importable(d)
        resolved = str(d.resolve())

        assert resolved in sys.path

        sys.path[:] = original_path

    def test_resolves_relative_path(self, tmp_path):
        d = tmp_path / "agent_dir"
        d.mkdir()
        (d / "agent.py").write_text("agent = None")

        original_path = sys.path.copy()
        _ensure_importable(d)
        assert str(d.resolve()) in sys.path

        sys.path[:] = original_path


class TestDiscoverTasks:
    def test_finds_flat_tasks(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        for name in ["read-file", "write-file"]:
            task_d = tasks_dir / name
            task_d.mkdir(parents=True)
            (task_d / "task.toml").write_text("schema_version = '1.1'")
        tasks = _discover_tasks(tasks_dir)
        assert sorted(tasks) == ["read-file", "write-file"]

    def test_finds_nested_tasks(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        task_d = tasks_dir / "read-file" / "read-file"
        task_d.mkdir(parents=True)
        (task_d / "task.toml").write_text("schema_version = '1.1'")
        tasks = _discover_tasks(tasks_dir)
        assert tasks == ["read-file"]


class TestResolveTaskPath:
    def test_resolves_flat_task(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        task_d = tasks_dir / "read-file"
        task_d.mkdir(parents=True)
        (task_d / "task.toml").write_text("")

        result = _resolve_task_path(tasks_dir, "read-file")
        assert result == task_d

    def test_resolves_nested_task(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        task_d = tasks_dir / "read-file" / "read-file"
        task_d.mkdir(parents=True)
        (task_d / "task.toml").write_text("schema_version = '1.1'")

        result = _resolve_task_path(tasks_dir, "read-file")
        assert result == task_d


class TestReadInstruction:
    def test_reads_instruction_md(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("Read the file hello.txt")

        result = _read_instruction(task_dir)
        assert result == "Read the file hello.txt"

    def test_returns_empty_string_when_missing(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()

        result = _read_instruction(task_dir)
        assert result == ""


class TestEnsureUserInstructionStep:
    def test_prepends_user_prompt(self):
        traj = AtifTrajectory(
            steps=[AtifStep(step_id="s1", source="agent", message="I will help.")]
        )
        updated = _ensure_user_instruction_step(traj, "Do the thing")
        assert len(updated.steps) == 2
        assert updated.steps[0].source == "user"
        assert updated.steps[0].message == "Do the thing"

    def test_no_duplicate(self):
        traj = AtifTrajectory(steps=[AtifStep(step_id="u1", source="user", message="Do the thing")])
        updated = _ensure_user_instruction_step(traj, "Do the thing")
        assert len(updated.steps) == 1

    def test_empty_instruction_returns_unchanged(self):
        traj = AtifTrajectory(steps=[AtifStep(step_id="s1", source="agent", message="hi")])
        updated = _ensure_user_instruction_step(traj, "")
        assert len(updated.steps) == 1


@pytest.mark.asyncio
async def test_run_setup_executes_script_with_task_env(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)

    setup_script = tmp_path / "setup.sh"
    setup_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\' "$SETUP_FLAG" > "$WORK_DIR/from_setup.txt"\n'
    )

    task = TaskConfig(
        name="sample",
        path=tmp_path,
        instruction="",
        env={"SETUP_FLAG": "ok"},
        setup_script=setup_script,
    )

    error = await run_setup(task, logs_dir, work_dir)

    assert error is None
    assert (work_dir / "from_setup.txt").read_text() == "ok"


@pytest.mark.asyncio
async def test_run_teardown_executes_script(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)

    teardown_script = tmp_path / "teardown.sh"
    teardown_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'done' > \"$LOGS_DIR/teardown_done.txt\"\n"
    )

    task = TaskConfig(
        name="sample",
        path=tmp_path,
        instruction="",
        teardown_script=teardown_script,
    )

    error = await run_teardown(task, work_dir, logs_dir)

    assert error is None
    assert (logs_dir / "teardown_done.txt").read_text() == "done"


def test_temporary_env_restores_when_cwd_setup_fails(tmp_path):
    key = "AMH_TEST_TEMP_ENV"
    original = os.environ.get(key)
    os.environ.pop(key, None)

    bad_work_dir = tmp_path / "bad-work-dir"
    bad_work_dir.write_text("not-a-directory")

    with (
        pytest.raises(FileExistsError),
        _temporary_env({key: "temp"}),
        _temporary_cwd(bad_work_dir),
    ):
        pass

    if original is None:
        assert key not in os.environ
    else:
        assert os.environ[key] == original


def test_build_script_env_infra_vars_override_task_env(tmp_path):
    logs_dir = tmp_path / "logs"
    work_dir = tmp_path / "work"
    task = TaskConfig(
        name="sample",
        path=tmp_path,
        instruction="",
        env={
            "LOGS_DIR": "bad-logs",
            "REWARD_DIR": "bad-reward",
            "AGENT_DIR": "bad-agent",
            "AGENT_RESPONSE_FILE": "bad-response",
            "WORK_DIR": "bad-work",
            "CUSTOM_FLAG": "ok",
        },
    )

    env = _build_script_env(task, logs_dir, work_dir)

    logs_root = logs_dir.resolve()
    work_root = work_dir.resolve()
    assert env["LOGS_DIR"] == str(logs_root)
    assert env["REWARD_DIR"] == str(logs_root / "verifier")
    assert env["AGENT_DIR"] == str(logs_root / "agent")
    assert env["AGENT_RESPONSE_FILE"] == str(logs_root / "agent" / "response.txt")
    assert env["WORK_DIR"] == str(work_root)
    assert env["CUSTOM_FLAG"] == "ok"
