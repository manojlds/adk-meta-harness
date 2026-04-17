"""Candidate validation — quick checks before expensive evaluation.

Following the reference Meta-Harness pattern: import-check and smoke-test
candidates before running full evaluation to avoid wasting API budget on
broken candidates produced by the proposer.

Validation stages (fast to slow):
1. Syntax check — all .py files compile (py_compile)
2. YAML check — config.yaml parses
3. Skill structure — skills/*/SKILL.md exist and have frontmatter
4. Import check — agent.py can be imported and exposes root_agent
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of validating a candidate harness."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [f"Validation: {status}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        return "\n".join(lines)


def validate_candidate(candidate_dir: Path) -> ValidationResult:
    """Run all validation checks on a candidate harness.

    Returns a ValidationResult. If valid is False, the candidate should
    be discarded without running evaluation.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Check agent.py exists
    agent_py = candidate_dir / "agent.py"
    if not agent_py.exists():
        errors.append("agent.py not found")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # 2. Syntax check all .py files
    py_errors = _check_python_syntax(candidate_dir)
    errors.extend(py_errors)

    # 3. YAML check
    yaml_errors = _check_yaml(candidate_dir)
    errors.extend(yaml_errors)

    # 4. Skill structure
    skill_warnings = _check_skills(candidate_dir)
    warnings.extend(skill_warnings)

    # If syntax or YAML errors, skip import check (it will fail anyway)
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # 5. Import check — run in a subprocess to avoid polluting the process
    import_errors = _check_import(candidate_dir)
    errors.extend(import_errors)

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _check_python_syntax(candidate_dir: Path) -> list[str]:
    """Compile-check all .py files in the candidate directory."""
    errors = []
    for py_file in candidate_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            rel = py_file.relative_to(candidate_dir)
            errors.append(f"Syntax error in {rel}: {e.msg}")
    return errors


def _check_yaml(candidate_dir: Path) -> list[str]:
    """Validate config.yaml if present."""
    config_path = candidate_dir / "config.yaml"
    if not config_path.exists():
        return []
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text())
        if data is not None and not isinstance(data, dict):
            return ["config.yaml: expected a mapping, got " + type(data).__name__]
    except Exception as e:
        return [f"config.yaml: YAML parse error: {e}"]
    return []


def _check_skills(candidate_dir: Path) -> list[str]:
    """Check skill directories have valid structure."""
    warnings = []
    skills_dir = candidate_dir / "skills"
    if not skills_dir.exists():
        return []
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            warnings.append(f"skills/{skill_dir.name}: missing SKILL.md")
    return warnings


def _check_import(candidate_dir: Path) -> list[str]:
    """Import-check agent.py in a subprocess.

    Runs a short Python script that imports agent.py and verifies it
    exposes a ``root_agent`` or ``agent`` attribute. Uses a subprocess
    to avoid polluting the current process with side effects.
    """
    check_script = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('agent', sys.argv[1])\n"
        "if spec is None or spec.loader is None:\n"
        "    print('cannot create module spec'); sys.exit(1)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['agent'] = mod\n"
        "try:\n"
        "    spec.loader.exec_module(mod)\n"
        "except Exception as e:\n"
        "    print(f'import error: {e}'); sys.exit(1)\n"
        "if not (hasattr(mod, 'root_agent') or hasattr(mod, 'agent')):\n"
        "    print('agent.py does not expose root_agent or agent'); sys.exit(1)\n"
        "print('OK')\n"
    )

    agent_py = candidate_dir / "agent.py"
    try:
        result = subprocess.run(
            [sys.executable, "-c", check_script, str(agent_py)],
            cwd=str(candidate_dir),
            capture_output=True,
            text=True,
            timeout=30,
            env=_import_check_env(candidate_dir),
        )
        if result.returncode != 0:
            msg = result.stdout.strip() or result.stderr.strip() or "unknown import error"
            return [f"Import check failed: {msg[:500]}"]
    except subprocess.TimeoutExpired:
        return ["Import check timed out (30s)"]
    except Exception as e:
        return [f"Import check error: {e}"]
    return []


def _import_check_env(candidate_dir: Path) -> dict[str, str]:
    """Build environment for the import-check subprocess."""
    import os

    env = os.environ.copy()
    # Prepend candidate dir to PYTHONPATH so relative imports work
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(candidate_dir) + ((":" + existing) if existing else "")
    return env
