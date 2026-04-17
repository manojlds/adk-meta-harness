"""Tests for coding_agent_cli proposer."""

import tempfile
from pathlib import Path

from adk_meta_harness.proposer.coding_agent_cli import (
    _compute_diff_summary,
    _detect_change_type,
    _snapshot_files,
)


def test_snapshot_files():
    d = Path(tempfile.mkdtemp())
    (d / "agent.py").write_text("agent code")
    (d / "system_prompt.md").write_text("be helpful")
    (d / "skills").mkdir()
    (d / "skills" / "test-skill").mkdir()
    (d / "skills" / "test-skill" / "SKILL.md").write_text("---\nname: test\n---\ntest")

    snapshot = _snapshot_files(d)
    assert "agent.py" in snapshot
    assert "system_prompt.md" in snapshot
    assert "skills/test-skill/SKILL.md" in snapshot
    assert snapshot["agent.py"] == "agent code"


def test_diff_summary_new_file():
    before = {"agent.py": "old"}
    after = {"agent.py": "old", "system_prompt.md": "new"}
    summary = _compute_diff_summary(before, after)
    assert "+ system_prompt.md" in summary
    assert "~ agent.py" not in summary


def test_diff_summary_modified_file():
    before = {"agent.py": "old"}
    after = {"agent.py": "new"}
    summary = _compute_diff_summary(before, after)
    assert "~ agent.py" in summary


def test_diff_summary_deleted_file():
    before = {"agent.py": "old", "system_prompt.md": "prompt"}
    after = {"agent.py": "old"}
    summary = _compute_diff_summary(before, after)
    assert "- system_prompt.md" in summary


def test_detect_change_type_skill():
    before = {"system_prompt.md": "old"}
    after = {"system_prompt.md": "old", "skills/test/SKILL.md": "new"}
    assert _detect_change_type(before, after) == "skill"


def test_detect_change_type_system_prompt():
    before = {"system_prompt.md": "old"}
    after = {"system_prompt.md": "new"}
    assert _detect_change_type(before, after) == "system_prompt"


def test_detect_change_type_harness():
    before = {"agent.py": "old"}
    after = {"agent.py": "new"}
    assert _detect_change_type(before, after) == "harness"


def test_detect_change_type_multiple():
    before = {"agent.py": "old", "system_prompt.md": "old"}
    after = {"agent.py": "new", "system_prompt.md": "new"}
    assert _detect_change_type(before, after) == "multiple"


def test_detect_change_type_none():
    before = {"agent.py": "old"}
    after = {"agent.py": "old"}
    assert _detect_change_type(before, after) == "none"
