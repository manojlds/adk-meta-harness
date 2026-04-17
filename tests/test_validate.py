"""Tests for candidate validation."""

from adk_meta_harness.validate import (
    ValidationResult,
    _check_python_syntax,
    _check_skills,
    _check_yaml,
    validate_candidate,
)


class TestCheckPythonSyntax:
    def test_valid_python(self, tmp_path):
        (tmp_path / "agent.py").write_text("x = 1\n")
        assert _check_python_syntax(tmp_path) == []

    def test_syntax_error(self, tmp_path):
        (tmp_path / "agent.py").write_text("def f(\n")
        errors = _check_python_syntax(tmp_path)
        assert len(errors) == 1
        assert "agent.py" in errors[0]

    def test_multiple_files(self, tmp_path):
        (tmp_path / "agent.py").write_text("x = 1\n")
        tools = tmp_path / "tools"
        tools.mkdir()
        (tools / "search.py").write_text("def search():\n  return 1\n")
        (tools / "broken.py").write_text("if True\n")
        errors = _check_python_syntax(tmp_path)
        assert len(errors) == 1
        assert "broken.py" in errors[0]

    def test_skips_pycache(self, tmp_path):
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "foo.py").write_text("this is not valid python {{{{")
        assert _check_python_syntax(tmp_path) == []


class TestCheckYaml:
    def test_no_config(self, tmp_path):
        assert _check_yaml(tmp_path) == []

    def test_valid_yaml(self, tmp_path):
        (tmp_path / "config.yaml").write_text("model: gemini-2.5-flash\nmax_turns: 10\n")
        assert _check_yaml(tmp_path) == []

    def test_invalid_yaml(self, tmp_path):
        (tmp_path / "config.yaml").write_text(":\n  - :\n  bad: [")
        errors = _check_yaml(tmp_path)
        assert len(errors) == 1
        assert "YAML parse error" in errors[0]

    def test_non_dict_yaml(self, tmp_path):
        (tmp_path / "config.yaml").write_text("- item1\n- item2\n")
        errors = _check_yaml(tmp_path)
        assert len(errors) == 1
        assert "expected a mapping" in errors[0]

    def test_empty_yaml(self, tmp_path):
        (tmp_path / "config.yaml").write_text("")
        assert _check_yaml(tmp_path) == []


class TestCheckSkills:
    def test_no_skills_dir(self, tmp_path):
        assert _check_skills(tmp_path) == []

    def test_valid_skill(self, tmp_path):
        skill = tmp_path / "skills" / "search"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("# Search skill\n")
        assert _check_skills(tmp_path) == []

    def test_missing_skill_md(self, tmp_path):
        skill = tmp_path / "skills" / "broken-skill"
        skill.mkdir(parents=True)
        (skill / "readme.txt").write_text("not a skill")
        warnings = _check_skills(tmp_path)
        assert len(warnings) == 1
        assert "broken-skill" in warnings[0]
        assert "missing SKILL.md" in warnings[0]

    def test_skips_non_dirs(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "notes.txt").write_text("ignore me")
        assert _check_skills(tmp_path) == []


class TestValidateCandidate:
    def test_valid_candidate(self, tmp_path):
        (tmp_path / "agent.py").write_text(
            "class Agent: pass\nagent = Agent()\nroot_agent = agent\n"
        )
        (tmp_path / "config.yaml").write_text("model: test\n")
        result = validate_candidate(tmp_path)
        assert result.valid

    def test_missing_agent_py(self, tmp_path):
        (tmp_path / "config.yaml").write_text("model: test\n")
        result = validate_candidate(tmp_path)
        assert not result.valid
        assert any("agent.py not found" in e for e in result.errors)

    def test_syntax_error_skips_import(self, tmp_path):
        (tmp_path / "agent.py").write_text("def f(\n")
        result = validate_candidate(tmp_path)
        assert not result.valid
        assert any("Syntax error" in e for e in result.errors)
        # Should NOT have an import error — we skip import check on syntax errors
        assert not any("Import check" in e for e in result.errors)

    def test_yaml_error_skips_import(self, tmp_path):
        (tmp_path / "agent.py").write_text("x = 1\n")
        (tmp_path / "config.yaml").write_text("bad: [")
        result = validate_candidate(tmp_path)
        assert not result.valid
        assert any("YAML" in e for e in result.errors)

    def test_import_check_no_root_agent(self, tmp_path):
        (tmp_path / "agent.py").write_text("x = 42\n")
        result = validate_candidate(tmp_path)
        assert not result.valid
        assert any("does not expose" in e for e in result.errors)

    def test_import_check_with_import_error(self, tmp_path):
        (tmp_path / "agent.py").write_text("import nonexistent_library_xyz_999\nagent = None\n")
        result = validate_candidate(tmp_path)
        assert not result.valid
        assert any("Import check failed" in e for e in result.errors)

    def test_skill_warnings_dont_block(self, tmp_path):
        (tmp_path / "agent.py").write_text("agent = 1\nroot_agent = agent\n")
        skill = tmp_path / "skills" / "broken"
        skill.mkdir(parents=True)
        result = validate_candidate(tmp_path)
        assert result.valid
        assert len(result.warnings) == 1
        assert "missing SKILL.md" in result.warnings[0]


class TestValidationResult:
    def test_summary_valid(self):
        r = ValidationResult(valid=True)
        assert "VALID" in r.summary()

    def test_summary_invalid(self):
        r = ValidationResult(valid=False, errors=["bad thing"])
        s = r.summary()
        assert "INVALID" in s
        assert "bad thing" in s

    def test_summary_with_warnings(self):
        r = ValidationResult(valid=True, warnings=["minor issue"])
        s = r.summary()
        assert "VALID" in s
        assert "minor issue" in s
