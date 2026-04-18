from __future__ import annotations

import pytest

from adk_meta_harness.runner import get_runner
from adk_meta_harness.runner.base import TaskRunner
from adk_meta_harness.runner.local import LocalTaskRunner


def test_get_runner_local():
    runner = get_runner("local")
    assert isinstance(runner, LocalTaskRunner)
    assert runner.name == "local"


def test_get_runner_unknown_raises():
    with pytest.raises(ValueError, match="Unknown runner"):
        get_runner("nonexistent")


def test_local_runner_is_task_runner():
    runner = get_runner("local")
    assert isinstance(runner, TaskRunner)


def test_local_runner_evaluate_signature():
    runner = LocalTaskRunner()
    assert callable(runner.evaluate)
    import inspect

    sig = inspect.signature(runner.evaluate)
    params = list(sig.parameters.keys())
    assert "candidate_dir" in params
    assert "tasks_dir" in params
    assert "model" in params
    assert "timeout" in params
