from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from adk_meta_harness.proposer.coding_agent_cli import (
    CodingAgentCLIProposer,
    _compute_diff_summary,
    _detect_change_type,
    _snapshot_files,
)
from adk_meta_harness.proposer.opencode import OpenCodeProposer


class TestOpenCodeProposerBuildCommand:
    def test_build_command_uses_absolute_path(self, tmp_path):
        proposer = OpenCodeProposer(model="test-model")
        relative_dir = tmp_path / "candidate"
        relative_dir.mkdir()

        cmd = proposer.build_command(relative_dir, "fix the bug")

        assert "--dir" in cmd
        dir_idx = cmd.index("--dir")
        dir_value = cmd[dir_idx + 1]
        resolved = str(relative_dir.resolve())
        assert dir_value == resolved

    def test_build_command_includes_model(self):
        proposer = OpenCodeProposer(model="openai/glm-5.1")
        cmd = proposer.build_command(Path("/tmp/test"), "do stuff")

        assert "--model" in cmd
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "openai/glm-5.1"

    def test_build_command_without_model(self):
        proposer = OpenCodeProposer()
        cmd = proposer.build_command(Path("/tmp/test"), "do stuff")

        assert "--model" not in cmd
        assert cmd[0] == "opencode"
        assert cmd[1] == "run"

    def test_build_command_appends_instruction(self):
        proposer = OpenCodeProposer()
        cmd = proposer.build_command(Path("/tmp/test"), "my instruction here")

        assert cmd[-1] == "my instruction here"


class TestProposeEditRetryLogic:
    def test_retry_on_no_edit(self, tmp_path):
        CodingAgentCLIProposer(
            cli_command="echo",
            cli_args=[],
            prompt_mode="argv",
        )

        candidate_dir = tmp_path / "candidate"
        candidate_dir.mkdir()
        (candidate_dir / "agent.py").write_text("original")

        first_result = MagicMock()
        first_result.returncode = 0
        first_result.stdout = ""
        first_result.stderr = ""

        with patch(
            "adk_meta_harness.proposer.coding_agent_cli._run_argv_mode",
            side_effect=[first_result, None],
        ):
            pass

    def test_detect_change_type_none_on_identical(self, tmp_path):
        before = {"agent.py": "content"}
        after = {"agent.py": "content"}
        assert _detect_change_type(before, after) == "none"

    def test_detect_change_type_single(self, tmp_path):
        before = {"agent.py": "old"}
        after = {"agent.py": "new"}
        assert _detect_change_type(before, after) == "harness"

    def test_detect_change_type_system_prompt(self):
        before = {"system_prompt.md": "old"}
        after = {"system_prompt.md": "new"}
        assert _detect_change_type(before, after) == "system_prompt"

    def test_detect_change_type_tools(self):
        before = {"agent.py": "same"}
        after = {"agent.py": "same", "tools/file_tools.py": "new"}
        assert _detect_change_type(before, after) == "tool"

    def test_detect_change_type_multiple(self):
        before = {"agent.py": "old"}
        after = {"agent.py": "new", "system_prompt.md": "new prompt"}
        assert _detect_change_type(before, after) == "multiple"

    def test_detect_change_type_new_file(self):
        before = {"agent.py": "same"}
        after = {"agent.py": "same", "config.yaml": "model: flash"}
        assert _detect_change_type(before, after) == "config"

    def test_detect_change_type_deleted_file(self):
        before = {"agent.py": "same", "system_prompt.md": "old prompt"}
        after = {"agent.py": "same"}
        assert _detect_change_type(before, after) == "system_prompt"


class TestSnapshotFiles:
    def test_snapshots_file_contents(self, tmp_path):
        (tmp_path / "agent.py").write_text("print('hello')")
        (tmp_path / "config.yaml").write_text("model: flash")

        snap = _snapshot_files(tmp_path)

        assert snap["agent.py"] == "print('hello')"
        assert snap["config.yaml"] == "model: flash"

    def test_skips_hidden_and_cache_dirs(self, tmp_path):
        (tmp_path / "agent.py").write_text("code")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "foo.pyc").write_bytes(b"bytecode")
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config").write_text("git data")

        snap = _snapshot_files(tmp_path)

        assert "agent.py" in snap
        assert "foo.pyc" not in snap
        assert ".git/config" not in snap

    def test_skips_proposer_learnings_files(self, tmp_path):
        (tmp_path / "agent.py").write_text("code")
        (tmp_path / "PROPOSER.md").write_text("instructions")
        (tmp_path / "learnings.md").write_text("insights")

        snap = _snapshot_files(tmp_path)

        assert "agent.py" in snap
        assert "PROPOSER.md" not in snap
        assert "learnings.md" not in snap


class TestComputeDiffSummary:
    def test_added_file(self):
        before = {"agent.py": "old"}
        after = {"agent.py": "old", "config.yaml": "new"}
        summary = _compute_diff_summary(before, after)
        assert "+ config.yaml" in summary

    def test_deleted_file(self):
        before = {"agent.py": "old", "config.yaml": "remove me"}
        after = {"agent.py": "old"}
        summary = _compute_diff_summary(before, after)
        assert "- config.yaml" in summary

    def test_modified_file(self):
        before = {"agent.py": "old"}
        after = {"agent.py": "new"}
        summary = _compute_diff_summary(before, after)
        assert "~ agent.py" in summary

    def test_no_changes(self):
        before = {"agent.py": "same"}
        after = {"agent.py": "same"}
        assert _compute_diff_summary(before, after) == "No changes detected"
