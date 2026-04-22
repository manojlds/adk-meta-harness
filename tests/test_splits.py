from __future__ import annotations

import pytest

from adk_meta_harness.splits import split_task_names


def test_split_task_names_is_deterministic_for_same_seed():
    tasks = [f"task-{i}" for i in range(20)]

    a = split_task_names(tasks, holdout_ratio=0.3, test_ratio=0.2, seed=42)
    b = split_task_names(tasks, holdout_ratio=0.3, test_ratio=0.2, seed=42)

    assert a == b


def test_split_task_names_has_no_overlap():
    tasks = [f"task-{i}" for i in range(20)]
    splits = split_task_names(tasks, holdout_ratio=0.3, test_ratio=0.2, seed=7)

    search = set(splits.search_task_names)
    holdout = set(splits.holdout_task_names)
    test = set(splits.test_task_names)

    assert search.isdisjoint(holdout)
    assert search.isdisjoint(test)
    assert holdout.isdisjoint(test)
    assert search | holdout | test == set(tasks)


def test_split_task_names_keeps_at_least_one_search_task():
    tasks = ["only-one"]
    splits = split_task_names(tasks, holdout_ratio=0.3, test_ratio=0.2, seed=42)

    assert splits.search_task_names == ["only-one"]
    assert splits.holdout_task_names == []
    assert splits.test_task_names == []


def test_split_task_names_rejects_invalid_ratios():
    with pytest.raises(ValueError, match="holdout_ratio"):
        split_task_names(["a", "b"], holdout_ratio=-0.1, test_ratio=0.2, seed=1)

    with pytest.raises(ValueError, match="test_ratio"):
        split_task_names(["a", "b"], holdout_ratio=0.2, test_ratio=1.0, seed=1)

    with pytest.raises(ValueError, match=r"must be < 1\.0"):
        split_task_names(["a", "b"], holdout_ratio=0.7, test_ratio=0.3, seed=1)


def test_split_task_names_uses_conventional_half_up_rounding():
    tasks = [f"task-{i}" for i in range(5)]
    splits = split_task_names(tasks, holdout_ratio=0.5, test_ratio=0.0, seed=1)

    # 5 * 0.5 == 2.5, expect 3 holdout with half-up rounding.
    assert len(splits.holdout_task_names) == 3
    assert len(splits.search_task_names) == 2
