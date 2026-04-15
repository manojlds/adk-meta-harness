"""Deterministic task verification for file I/O tasks.

Checks the agent's trajectory (tool calls and responses) against expected
outcomes. This replaces the LLM judge for tasks with clear right/wrong answers.

For tasks that write files, also checks the filesystem for the expected output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adk_meta_harness.trace.atif import AtifTrajectory


def verify_task(
    task_name: str,
    task_path: Path,
    trajectory: AtifTrajectory | None,
    agent_response: str = "",
    working_dir: Path | None = None,
) -> tuple[bool, float] | None:
    """Verify a task deterministically based on its type.

    Returns (passed, score) where score is 0.0-1.0.

    Returns None if no deterministic verifier can be resolved for this task.

    Checks:
    1. The agent's final response text for expected content
    2. Tool calls in the trajectory for expected actions
    3. The filesystem for expected output files (for write tasks)
    """
    # Collect all tool calls and observations from the trajectory
    tool_calls: list[dict] = []
    if trajectory and trajectory.steps:
        for step in trajectory.steps:
            for tc in step.tool_calls:
                tool_calls.append({
                    "name": tc.function_name,
                    "arguments": tc.arguments,
                })
            if step.observation and step.observation.content:
                tool_calls.append({
                    "name": "_observation",
                    "content": step.observation.content,
                })

    # Collect agent messages
    messages: list[str] = []
    if trajectory and trajectory.steps:
        for step in trajectory.steps:
            if step.message:
                messages.append(step.message)

    full_response = agent_response or " ".join(messages)

    # Route to task-specific verifier
    verifier = _get_verifier(task_name, task_path)
    if verifier:
        return verifier(full_response, tool_calls, working_dir)

    # No verifier found
    return None


def _get_verifier(task_name: str, task_path: Path):
    """Get a verifier function for a task.

    Checks for a verify.py in the task directory, then falls back
    to built-in verifiers for known task patterns.
    """
    # Check for task-specific verify.py
    verify_py = task_path / "verify.py"
    if verify_py.exists():
        return _load_verify_py(verify_py)

    # Built-in verifiers for common patterns
    name_lower = task_name.lower().replace("-", "_")
    if "read" in name_lower and "file" in name_lower:
        return _verify_read_file
    if "write" in name_lower and "file" in name_lower:
        return _verify_write_file
    if "list" in name_lower and ("dir" in name_lower or "file" in name_lower):
        return _verify_list_dir

    # Check the instruction for hints
    instruction_file = task_path / "instruction.md"
    if instruction_file.exists():
        instruction = instruction_file.read_text().lower()
        if "read" in instruction and "file" in instruction and "contents" in instruction:
            return _verify_read_file
        if "write" in instruction and "file" in instruction:
            return _verify_write_file
        if "list" in instruction and ("file" in instruction or "directory" in instruction):
            return _verify_list_dir

    return None


def _verify_read_file(
    response: str, tool_calls: list[dict], working_dir: Path | None
) -> tuple[bool, float]:
    """Verify a read-file task: agent should report file contents."""
    # Check if any tool call reads a file
    read_calls = [
        tc for tc in tool_calls
        if tc.get("name", "") in ("read_file", "file_read", "read")
        or "file_path" in tc.get("arguments", "")
        or "path" in tc.get("arguments", "")
    ]

    # Check if the response mentions common test file content
    # Look for key phrases that indicate the file was actually read
    content_found = False
    key_phrases = ["hello world", "hello world from the test file"]
    for phrase in key_phrases:
        if phrase.lower() in response.lower():
            content_found = True
            break

    if content_found:
        return True, 1.0

    # Partial credit if the agent at least tried to read the file
    if read_calls:
        # Check if any observation contains file-like content
        for tc in tool_calls:
            content = tc.get("content", "") or tc.get("arguments", "")
            if "hello world" in content.lower():
                return True, 1.0
        return False, 0.3

    return False, 0.0


def _verify_write_file(
    response: str, tool_calls: list[dict], working_dir: Path | None
) -> tuple[bool, float]:
    """Verify a write-file task: agent should create a file with specific content."""
    expected_content = "the quick brown fox jumps over the lazy dog"

    # Check filesystem first (most reliable)
    if working_dir:
        output_file = working_dir / "output.txt"
        if output_file.exists():
            content = output_file.read_text().strip().lower().replace("\n", "").replace(" ", "")
            expected = expected_content.lower().replace(" ", "")
            if content == expected:
                return True, 1.0
            # Partial match
            if any(word in content for word in ["quick", "brown", "fox", "lazy", "dog"]):
                return False, 0.5

    # Check if agent called a write tool
    write_calls = [
        tc for tc in tool_calls
        if tc.get("name", "") in ("write_file", "file_write", "write")
        or "write" in tc.get("name", "").lower()
    ]

    if write_calls:
        # Check if the expected content appears in the arguments
        for tc in write_calls:
            args = tc.get("arguments", "").lower()
            if expected_content[:20].lower() in args or "quick brown fox" in args:
                return True, 0.8

    # Check response for the expected text
    if expected_content[:20].lower() in response.lower():
        return True, 0.5

    return False, 0.0


def _verify_list_dir(
    response: str, tool_calls: list[dict], working_dir: Path | None
) -> tuple[bool, float]:
    """Verify a list-directory task: agent should report files in the directory."""
    # Common test files we expect to see listed
    expected_files = ["alpha", "beta"]

    found = 0
    for name in expected_files:
        if name in response.lower():
            found += 1

    # Check tool calls for list_directory
    list_calls = [
        tc for tc in tool_calls
        if tc.get("name", "") in ("list_directory", "list_dir", "ls", "dir")
        or "list" in tc.get("name", "").lower()
    ]

    if found >= 2:
        return True, 1.0
    if found == 1:
        return False, 0.5
    if list_calls:
        return False, 0.3

    return False, 0.0


def _load_verify_py(verify_path: Path):
    """Load a custom verify.py from the task directory.

    The verify.py should expose a verify(response, tool_calls, working_dir)
    function that returns (passed: bool, score: float).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify", verify_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None

    if hasattr(module, "verify"):
        return module.verify
    return None
