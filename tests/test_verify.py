"""Tests for deterministic task verification."""

import tempfile
from pathlib import Path

from adk_meta_harness.trace.atif import AtifStep, AtifTrajectory
from adk_meta_harness.verify import verify_task


def test_verify_task_unknown_returns_none():
    d = Path(tempfile.mkdtemp())
    (d / "instruction.md").write_text("Do something unrelated.")

    result = verify_task("unknown-task", d, None)

    assert result is None


def test_verify_read_file_passes_on_expected_response_text():
    d = Path(tempfile.mkdtemp())
    (d / "instruction.md").write_text("Read a file and report its contents.")
    trajectory = AtifTrajectory(
        steps=[
            AtifStep(
                step_id="step-1",
                source="agent",
                message="The file says: hello world from the test file",
            )
        ]
    )

    result = verify_task("read-file", d, trajectory)

    assert result == (True, 1.0)


def test_verify_write_file_passes_when_output_file_matches():
    task_dir = Path(tempfile.mkdtemp())
    (task_dir / "instruction.md").write_text("Write text to output.txt.")
    work_dir = Path(tempfile.mkdtemp())
    (work_dir / "output.txt").write_text("The quick brown fox jumps over the lazy dog")

    result = verify_task("write-file", task_dir, None, working_dir=work_dir)

    assert result == (True, 1.0)
