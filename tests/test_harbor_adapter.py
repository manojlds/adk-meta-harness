"""Tests for Harbor adapter helper behavior."""

from adk_meta_harness.harbor_adapter import _ensure_user_instruction_step
from adk_meta_harness.trace.atif import AtifStep, AtifTrajectory


def test_ensure_user_instruction_step_prepends_user_prompt():
    traj = AtifTrajectory(
        steps=[
            AtifStep(step_id="step-0001", source="agent", message="I cannot access files."),
        ]
    )

    updated = _ensure_user_instruction_step(traj, "Read hello.txt")

    assert len(updated.steps) == 2
    assert updated.steps[0].source == "user"
    assert updated.steps[0].message == "Read hello.txt"


def test_ensure_user_instruction_step_no_duplicate_when_present():
    traj = AtifTrajectory(
        steps=[
            AtifStep(step_id="step-user", source="user", message="Read hello.txt"),
            AtifStep(step_id="step-0001", source="agent", message="Trying now."),
        ]
    )

    updated = _ensure_user_instruction_step(traj, "Read hello.txt")

    assert len(updated.steps) == 2
    assert updated.steps[0].source == "user"
    assert updated.steps[0].message == "Read hello.txt"
