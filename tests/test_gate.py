"""Tests for gate module."""

from adk_meta_harness.gate import gate_decision


def test_gate_first_candidate():
    result = gate_decision(
        current_holdout=0.5,
        previous_holdout=None,
    )
    assert result.kept is True
    assert "First" in result.reason


def test_gate_improvement():
    result = gate_decision(
        current_holdout=0.75,
        previous_holdout=0.70,
    )
    assert result.kept is True
    assert "improved" in result.reason.lower()


def test_gate_regression():
    result = gate_decision(
        current_holdout=0.60,
        previous_holdout=0.70,
    )
    assert result.kept is False
    assert "regressed" in result.reason.lower()


def test_gate_same_score_simpler():
    result = gate_decision(
        current_holdout=0.70,
        previous_holdout=0.70,
        current_complexity=3,
        previous_complexity=5,
    )
    assert result.kept is True
    assert "simplpler" in result.reason.lower() or "simpler" in result.reason.lower()


def test_gate_same_score_not_simpler():
    result = gate_decision(
        current_holdout=0.70,
        previous_holdout=0.70,
        current_complexity=5,
        previous_complexity=3,
    )
    assert result.kept is False
    assert "not improved" in result.reason.lower()


def test_gate_same_score_no_complexity():
    result = gate_decision(
        current_holdout=0.70,
        previous_holdout=0.70,
    )
    assert result.kept is False


def test_gate_none_holdout():
    result = gate_decision(
        current_holdout=None,
        previous_holdout=0.5,
    )
    assert result.kept is False


def test_gate_result_deltas():
    result = gate_decision(
        current_holdout=0.75,
        previous_holdout=0.70,
        current_search=0.80,
        previous_search=0.73,
    )
    assert abs(result.holdout_delta - 0.05) < 1e-9
    assert abs(result.search_delta - 0.07) < 1e-9