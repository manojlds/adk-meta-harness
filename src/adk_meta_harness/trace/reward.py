"""Reward file parser.

Verifier scripts write evaluation results to:
- /logs/verifier/reward.txt (single number, 0 or 1)
- /logs/verifier/reward.json (multiple metrics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Reward:
    """Parsed reward from task evaluation."""

    score: float
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if self.details:
            return self.details.get("summary", "")
        return f"score={self.score:.4f}"


def parse_reward(reward_path: Path) -> Reward:
    """Parse a reward file.

    Handles three formats:
    1. reward.txt: a single number (0 or 1)
    2. reward.json: {"accuracy": 0.95, ...}
    3. reward.json with "passed" key

    Args:
        reward_path: Path to the reward file.

    Returns:
        Reward with parsed score and metrics.
    """
    if not reward_path.exists():
        return Reward(score=0.0, passed=False)

    text = reward_path.read_text().strip()

    if reward_path.suffix == ".json":
        return _parse_reward_json(text)
    return _parse_reward_txt(text)


def parse_reward_dir(logs_dir: Path) -> Reward:
    """Parse reward from a task logs directory.

    Looks for:
    1. logs_dir/verifier/reward.json (preferred)
    2. logs_dir/verifier/reward.txt (fallback)
    3. logs_dir/reward.json (alternative location)
    4. logs_dir/reward.txt (alternative location)

    Args:
        logs_dir: Path to the logs directory.

    Returns:
        Reward with parsed score and metrics.
    """
    search_paths = [
        logs_dir / "verifier" / "reward.json",
        logs_dir / "verifier" / "reward.txt",
        logs_dir / "reward.json",
        logs_dir / "reward.txt",
    ]

    for path in search_paths:
        if path.exists():
            return parse_reward(path)

    return Reward(score=0.0, passed=False)


def _parse_reward_txt(text: str) -> Reward:
    """Parse a reward.txt file (single number)."""
    try:
        score = float(text)
        return Reward(
            score=score,
            passed=score >= 0.5,
            metrics={"score": score},
        )
    except ValueError:
        return Reward(score=0.0, passed=False)


def _parse_reward_json(text: str) -> Reward:
    """Parse a reward.json file."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Reward(score=0.0, passed=False)

    if isinstance(data, (int, float)):
        score = float(data)
        return Reward(
            score=score,
            passed=score >= 0.5,
            metrics={"score": score},
        )

    if not isinstance(data, dict):
        return Reward(score=0.0, passed=False)

    score = 0.0
    passed = False
    metrics: dict[str, float] = {}
    details: dict[str, Any] = {}

    if "passed" in data:
        passed = bool(data["passed"])
        score = 1.0 if passed else 0.0
    elif "accuracy" in data:
        score = float(data["accuracy"])
        passed = score >= 0.5
    elif "score" in data:
        score = float(data["score"])
        passed = score >= 0.5
    elif "reward" in data:
        score = float(data["reward"])
        passed = score >= 0.5

    for key, value in data.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    if "details" in data and isinstance(data["details"], dict):
        details = data["details"]

    return Reward(
        score=score,
        passed=passed,
        metrics=metrics,
        details=details,
    )
