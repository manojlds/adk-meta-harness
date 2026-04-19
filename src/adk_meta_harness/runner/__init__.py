from adk_meta_harness.runner.base import TaskRunner
from adk_meta_harness.runner.local import LocalTaskRunner
from adk_meta_harness.runner.temporal_runner import TemporalTaskRunner

RUNNERS: dict[str, type[TaskRunner]] = {
    "local": LocalTaskRunner,
    "temporal": TemporalTaskRunner,
}


def get_runner(name: str, **kwargs) -> TaskRunner:
    if name in RUNNERS:
        return RUNNERS[name](**kwargs)
    msg = f"Unknown runner: {name!r}. Available: local, temporal"
    raise ValueError(msg)


__all__ = [
    "RUNNERS",
    "LocalTaskRunner",
    "TaskRunner",
    "TemporalTaskRunner",
    "get_runner",
]
