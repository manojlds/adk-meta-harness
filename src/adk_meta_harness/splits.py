from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSplits:
    search_task_names: list[str]
    holdout_task_names: list[str]
    test_task_names: list[str]
    holdout_ratio: float
    test_ratio: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_task_names": list(self.search_task_names),
            "holdout_task_names": list(self.holdout_task_names),
            "test_task_names": list(self.test_task_names),
            "holdout_ratio": self.holdout_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "counts": {
                "search": len(self.search_task_names),
                "holdout": len(self.holdout_task_names),
                "test": len(self.test_task_names),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskSplits:
        return cls(
            search_task_names=_to_str_list(payload.get("search_task_names")),
            holdout_task_names=_to_str_list(payload.get("holdout_task_names")),
            test_task_names=_to_str_list(payload.get("test_task_names")),
            holdout_ratio=float(payload.get("holdout_ratio", 0.3)),
            test_ratio=float(payload.get("test_ratio", 0.2)),
            seed=int(payload.get("seed", 42)),
        )


def split_task_names(
    task_names: list[str],
    *,
    holdout_ratio: float = 0.3,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> TaskSplits:
    """Deterministically split tasks into search, holdout, and test sets."""
    _validate_ratios(holdout_ratio, test_ratio)

    names = sorted({name for name in task_names if name})
    if not names:
        return TaskSplits(
            search_task_names=[],
            holdout_task_names=[],
            test_task_names=[],
            holdout_ratio=holdout_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    n_tasks = len(names)
    n_holdout = _ratio_count(n_tasks, holdout_ratio)
    n_test = _ratio_count(n_tasks, test_ratio)

    if holdout_ratio > 0 and n_holdout == 0 and n_tasks >= 2:
        n_holdout = 1
    if test_ratio > 0 and n_test == 0 and n_tasks >= 3:
        n_test = 1

    max_reserved = max(0, n_tasks - 1)
    while n_holdout + n_test > max_reserved:
        if n_test >= n_holdout and n_test > 0:
            n_test -= 1
        elif n_holdout > 0:
            n_holdout -= 1
        else:
            break

    n_search = n_tasks - n_holdout - n_test
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)

    search = sorted(shuffled[:n_search])
    holdout = sorted(shuffled[n_search : n_search + n_holdout])
    test = sorted(shuffled[n_search + n_holdout :])

    return TaskSplits(
        search_task_names=search,
        holdout_task_names=holdout,
        test_task_names=test,
        holdout_ratio=holdout_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def _validate_ratios(holdout_ratio: float, test_ratio: float) -> None:
    if not 0.0 <= holdout_ratio < 1.0:
        msg = "holdout_ratio must be in [0.0, 1.0)"
        raise ValueError(msg)
    if not 0.0 <= test_ratio < 1.0:
        msg = "test_ratio must be in [0.0, 1.0)"
        raise ValueError(msg)
    if holdout_ratio + test_ratio >= 1.0:
        msg = "holdout_ratio + test_ratio must be < 1.0"
        raise ValueError(msg)


def _ratio_count(total: int, ratio: float) -> int:
    return int(total * ratio + 0.5)


def _to_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
