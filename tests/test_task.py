from __future__ import annotations

import logging

from adk_meta_harness.task import TaskConfig, discover_tasks


def test_discover_tasks_supports_flat_and_nested_layouts(tmp_path):
    tasks_dir = tmp_path / "tasks"

    flat = tasks_dir / "flat-task"
    flat.mkdir(parents=True)
    (flat / "instruction.md").write_text("Do flat")

    nested = tasks_dir / "nested-task" / "nested-task"
    nested.mkdir(parents=True)
    (nested / "instruction.md").write_text("Do nested")

    tasks = discover_tasks(tasks_dir)

    assert [task.name for task in tasks] == ["flat-task", "nested-task"]
    assert tasks[0].path == flat.resolve()
    assert tasks[1].path == nested.resolve()


def test_task_config_from_path_reads_timeouts_and_env(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True)

    (task_dir / "instruction.md").write_text("Do the thing")
    (task_dir / "task.toml").write_text(
        "[agent]\n"
        "timeout_sec = 42\n"
        "[verifier]\n"
        "timeout_sec = 77\n"
        "[scripts]\n"
        "setup_timeout_sec = 12\n"
        "teardown_timeout_sec = 34\n"
        "[env]\n"
        "TASK_FLAG = 'on'\n"
    )

    (task_dir / "scripts").mkdir()
    (task_dir / "scripts" / "setup.sh").write_text("#!/usr/bin/env bash\n")
    (task_dir / "scripts" / "teardown.sh").write_text("#!/usr/bin/env bash\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/usr/bin/env bash\n")
    (task_dir / "fixtures").mkdir()

    config = TaskConfig.from_path(task_dir, "task")

    assert config.name == "task"
    assert config.instruction == "Do the thing"
    assert config.agent_timeout == 42
    assert config.verifier_timeout == 77
    assert config.setup_timeout == 12
    assert config.teardown_timeout == 34
    assert config.env == {"TASK_FLAG": "on"}
    assert config.setup_script == (task_dir / "scripts" / "setup.sh").resolve()
    assert config.teardown_script == (task_dir / "scripts" / "teardown.sh").resolve()
    assert config.verifier_script == (task_dir / "tests" / "test.sh").resolve()
    assert config.fixtures_dir == (task_dir / "fixtures").resolve()


def test_task_config_logs_warning_for_invalid_toml(tmp_path, caplog):
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do the thing")
    (task_dir / "task.toml").write_text("[agent\ntimeout_sec = 10")

    caplog.set_level(logging.WARNING)
    _ = TaskConfig.from_path(task_dir, "task")

    assert any("Failed to parse task TOML" in record.message for record in caplog.records)
