from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from adk_meta_harness.harbor_adapter import (
    _discover_tasks,
    _ensure_importable,
    _ensure_user_instruction_step,
    _normalize_otel_value,
    _read_instruction,
    _resolve_task_path,
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


class TestNormalizeOtelValue:
    def test_primitives(self):
        assert _normalize_otel_value(None) is None
        assert _normalize_otel_value("hello") == "hello"
        assert _normalize_otel_value(42) == 42
        assert _normalize_otel_value(3.14) == 3.14
        assert _normalize_otel_value(True) is True

    def test_bytes(self):
        assert _normalize_otel_value(b"hello") == "hello"

    def test_list(self):
        assert _normalize_otel_value([1, 2, 3]) == [1, 2, 3]

    def test_nested_dict(self):
        result = _normalize_otel_value({"key": {"nested": True}})
        assert result == {"key": {"nested": True}}

    def test_mixed_iterable(self):
        result = _normalize_otel_value([1, "two", b"three"])
        assert result == [1, "two", "three"]


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
