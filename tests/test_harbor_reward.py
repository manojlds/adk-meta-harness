"""Tests for Harbor reward parsing."""

import json
import tempfile
from pathlib import Path

from adk_meta_harness.trace.harbor_reward import parse_reward, parse_reward_dir


def test_parse_reward_txt_pass():
    d = tempfile.mkdtemp()
    p = Path(d) / "reward.txt"
    p.write_text("1")
    result = parse_reward(p)
    assert result.score == 1.0
    assert result.passed is True


def test_parse_reward_txt_fail():
    d = tempfile.mkdtemp()
    p = Path(d) / "reward.txt"
    p.write_text("0")
    result = parse_reward(p)
    assert result.score == 0.0
    assert result.passed is False


def test_parse_reward_txt_float():
    d = tempfile.mkdtemp()
    p = Path(d) / "reward.txt"
    p.write_text("0.75")
    result = parse_reward(p)
    assert result.score == 0.75
    assert result.passed is True


def test_parse_reward_json_accuracy():
    d = tempfile.mkdtemp()
    p = Path(d) / "reward.json"
    p.write_text(json.dumps({"accuracy": 0.95, "runtime_sec": 1.23}))
    result = parse_reward(p)
    assert result.score == 0.95
    assert result.passed is True
    assert result.metrics["accuracy"] == 0.95
    assert result.metrics["runtime_sec"] == 1.23


def test_parse_reward_json_passed():
    d = tempfile.mkdtemp()
    p = Path(d) / "reward.json"
    p.write_text(json.dumps({"passed": True}))
    result = parse_reward(p)
    assert result.passed is True
    assert result.score == 1.0


def test_parse_reward_json_score():
    d = tempfile.mkdtemp()
    p = Path(d) / "reward.json"
    p.write_text(json.dumps({"score": 0.6}))
    result = parse_reward(p)
    assert result.score == 0.6
    assert result.passed is True


def test_parse_reward_missing_file():
    result = parse_reward(Path("/nonexistent/reward.txt"))
    assert result.score == 0.0
    assert result.passed is False


def test_parse_reward_dir():
    d = tempfile.mkdtemp()
    verifier_dir = Path(d) / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "reward.txt").write_text("1")
    result = parse_reward_dir(Path(d))
    assert result.score == 1.0
    assert result.passed is True


def test_parse_reward_dir_empty():
    d = tempfile.mkdtemp()
    result = parse_reward_dir(Path(d))
    assert result.score == 0.0
    assert result.passed is False
