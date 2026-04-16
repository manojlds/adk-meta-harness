from adk_meta_harness.runner.base import TaskRunner
from adk_meta_harness.runner.local import LocalTaskRunner

RUNNERS: dict[str, type[TaskRunner]] = {
    "local": LocalTaskRunner,
}


def get_runner(name: str, **kwargs) -> TaskRunner:
    if name == "harbor":
        from adk_meta_harness.runner.harbor_runner import HarborTaskRunner

        return HarborTaskRunner(**kwargs)
    if name in RUNNERS:
        return RUNNERS[name](**kwargs)
    msg = f"Unknown runner: {name!r}. Available: local, harbor"
    raise ValueError(msg)


__all__ = [
    "LocalTaskRunner",
    "RUNNERS",
    "TaskRunner",
    "get_runner",
]
