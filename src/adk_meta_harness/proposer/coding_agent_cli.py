"""Generic coding-agent-CLI proposer adapter.

Works with any coding agent CLI that:
1. Operates on a filesystem (reads/writes files)
2. Can be given a prompt/instruction
3. Produces file edits in the working directory

Tested with: OpenCode, Pi
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROPOSER_TEMPLATE = """\
# Meta-Harness Proposer Instructions

You are an autonomous agent harness engineer. Your job is to improve the
ADK agent harness in this directory so the agent performs better on tasks.

## What You Can Modify

Everything in this directory is mutable:
- `agent.py` — ADK Agent construction (model, tools, skills, callbacks)
- `system_prompt.md` — System prompt/instruction
- `config.yaml` — Model, max_turns, stop_conditions
- `skills/` — Agent skills (SKILL.md + scripts/ + references/)
- `tools/` — Custom tool definitions
- `callbacks/` — before_model_callback, after_tool_callback
- `routing/` — Multi-agent transfer rules

## What You Must NOT Do

- Do not change the model unless the human explicitly requests it.
- Do not add task-specific hacks or benchmark-specific keyword rules.
- Do not modify files outside this directory.

## Goal

Maximize holdout accuracy. Use `passed` as the primary metric.

## How to Work

1. Read `learnings.md` for accumulated insights from prior iterations.
2. Read `PROPOSER.md` for the optimization directive.
3. Browse `../` for prior candidate directories, their traces, and scores.
4. Diagnose failure patterns from traces.
5. Make ONE targeted harness change at a time.
6. Prefer changes that fix a *class* of failures, not a single task.

## Simplicity Criterion

All else being equal, simpler is better. If a change achieves the same result
with a simpler harness, keep the simpler one.

## Overfitting Rule

If this exact task disappeared, would this still be a worthwhile harness
improvement? If not, it is probably overfitting. Do NOT do it.

## Never Stop

Once you start, do NOT stop to ask whether you should continue. Keep
iterating until told to stop.
"""


class CodingAgentCLIProposer:
    """Generic adapter for CLI-based coding agents.

    Subclasses override `build_command` to customize the CLI invocation.
    """

    def __init__(
        self,
        cli_command: str,
        cli_args: list[str] | None = None,
        proposer_template: str = PROPOSER_TEMPLATE,
        env: dict[str, str] | None = None,
    ):
        self.cli_command = cli_command
        self.cli_args = cli_args or []
        self.proposer_template = proposer_template
        self.env = env

    @property
    def name(self) -> str:
        return self.cli_command

    async def propose_edit(
        self,
        candidate_dir: Path,
        filesystem_dir: Path,
        learnings: str,
        instruction: str,
    ) -> dict[str, str]:
        # Write PROPOSER.md into the candidate directory
        proposer_md = candidate_dir / "PROPOSER.md"
        proposer_content = self.proposer_template + "\n\n## Current Task\n\n" + instruction
        proposer_md.write_text(proposer_content)

        # Write learnings.md into the candidate directory
        learnings_file = candidate_dir / "learnings.md"
        learnings_file.write_text(learnings)

        # Snapshot before
        before = _snapshot_files(candidate_dir)

        # Build and run the CLI command
        cmd = self.build_command(candidate_dir, instruction)
        env = {**os.environ, **(self.env or {})}
        result = subprocess.run(
            cmd,
            cwd=str(candidate_dir),
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )

        # Snapshot after
        after = _snapshot_files(candidate_dir)

        # Compute diff summary
        diff_summary = _compute_diff_summary(before, after)

        # Detect change type
        change_type = _detect_change_type(before, after)

        # Clean up proposer files
        proposer_md.unlink(missing_ok=True)
        learnings_file.unlink(missing_ok=True)

        description = instruction[:200] if len(instruction) > 200 else instruction

        return {
            "description": description,
            "change_type": change_type,
            "diff_summary": diff_summary,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    def build_command(self, candidate_dir: Path, instruction: str) -> list[str]:
        """Build the command to invoke the coding agent CLI.

        Subclasses should override this for CLI-specific flags.
        """
        return [self.cli_command, *self.cli_args]


def _snapshot_files(directory: Path) -> dict[str, str]:
    """Snapshot all files in a directory (relative paths -> content)."""
    snapshot = {}
    for f in directory.rglob("*"):
        if f.is_file() and f.name not in ("PROPOSER.md", "learnings.md"):
            if ".git" in f.parts or "__pycache__" in f.parts:
                continue
            rel = str(f.relative_to(directory))
            try:
                snapshot[rel] = f.read_text(errors="replace")
            except Exception:
                snapshot[rel] = f.read_bytes().hex()
    return snapshot


def _compute_diff_summary(before: dict[str, str], after: dict[str, str]) -> str:
    """Summarize what changed between two snapshots."""
    changes = []
    all_keys = set(before) | set(after)
    for key in sorted(all_keys):
        if key not in before:
            changes.append(f"  + {key}")
        elif key not in after:
            changes.append(f"  - {key}")
        elif before[key] != after[key]:
            changes.append(f"  ~ {key}")
    if not changes:
        return "No changes detected"
    return "\n".join(changes)


def _detect_change_type(before: dict[str, str], after: dict[str, str]) -> str:
    """Detect what type of harness component changed."""
    changed = set()
    all_keys = set(before) | set(after)
    for key in all_keys:
        if key not in before or key not in after or before.get(key) != after.get(key):
            parts = Path(key).parts
            if parts[0] == "skills" if parts else False:
                changed.add("skill")
            elif parts[0] == "tools" if parts else False:
                changed.add("tool")
            elif parts[0] == "callbacks" if parts else False:
                changed.add("callback")
            elif parts[0] == "routing" if parts else False:
                changed.add("routing")
            elif key == "system_prompt.md":
                changed.add("system_prompt")
            elif key == "config.yaml":
                changed.add("config")
            elif key == "agent.py":
                changed.add("harness")

    if len(changed) == 1:
        return changed.pop()
    if changed:
        return "multiple"
    return "none"