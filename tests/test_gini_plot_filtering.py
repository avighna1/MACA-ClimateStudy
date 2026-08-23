"""
Tests that Gini plots include ONLY the negotiating agents (residents +
functional) and EXCLUDE Judge/Improver turns.

Why this matters: Judge and Improver messages carry no allocation, so
their recorded Gini value is just a repeat of the preceding negotiating
turn. Including them made the trajectory plots show flat "staircase"
steps that looked like the allocation had stalled when it simply hadn't
been updated yet, and dragged the averages toward those repeated values.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from metrics import TurnMetrics
from logger import (
    NEGOTIATING_ROLES,
    negotiating_turns,
    plot_average_relative_gini,
    plot_gini_trajectories,
    plot_relative_gini_with_interventions,
)


def _mixed_sequence():
    """A realistic interleaving: each negotiating turn followed by a
    judge review, with one improver intervention."""
    return [
        TurnMetrics(1, "Resident-New York", {}, 0.10, 0.20, 1.0, False, role="resident"),
        TurnMetrics(2, "JudgeAgent", {}, 0.10, 0.20, 1.0, False, role="judge"),
        TurnMetrics(3, "ImproverAgent", {}, 0.10, 0.20, 1.0, True, role="improver"),
        TurnMetrics(4, "Resident-San Francisco", {}, 0.30, 0.40, 1.0, False, role="resident"),
        TurnMetrics(5, "JudgeAgent", {}, 0.30, 0.40, 1.0, False, role="judge"),
        TurnMetrics(6, "InfrastructureAgent", {}, 0.20, 0.30, 1.0, False, role="functional"),
        TurnMetrics(7, "JudgeAgent", {}, 0.20, 0.30, 1.0, False, role="judge"),
        TurnMetrics(8, "FinancialAgent", {}, 0.05, 0.15, 1.0, False, role="functional"),
        TurnMetrics(9, "JudgeAgent", {}, 0.05, 0.15, 1.0, False, role="judge"),
    ]


def test_negotiating_roles_are_resident_and_functional_only():
    assert NEGOTIATING_ROLES == {"resident", "functional"}


def test_filter_drops_judge_and_improver_turns():
    turns = _mixed_sequence()
    kept = negotiating_turns(turns)

    assert len(kept) == 4   # 2 residents + 2 functional
    assert all(t.role in NEGOTIATING_ROLES for t in kept)
    assert not any("Judge" in t.speaker for t in kept)
    assert not any("Improver" in t.speaker for t in kept)
    assert [t.turn_index for t in kept] == [1, 4, 6, 8]


def test_filtering_actually_changes_the_average():
    """If judge/improver turns didn't skew the numbers there'd be no
    point filtering -- confirm they do."""
    turns = _mixed_sequence()
    all_mean = float(np.mean([t.relative_gini_score for t in turns]))
    kept_mean = float(np.mean([t.relative_gini_score for t in negotiating_turns(turns)]))
    assert abs(all_mean - kept_mean) > 1e-9


def test_legacy_turns_without_role_still_plot():
    """Traces written before the `role` field existed must still render
    rather than silently producing an empty plot."""
    legacy = [
        TurnMetrics(1, "SomeAgent", {}, 0.1, 0.2, 1.0, False),
        TurnMetrics(2, "OtherAgent", {}, 0.2, 0.3, 1.0, False),
    ]
    kept = negotiating_turns(legacy)
    assert len(kept) == 2   # falls back to plotting everything


def test_empty_input_returns_empty():
    assert negotiating_turns([]) == []


def test_all_three_plots_render_from_filtered_data():
    """End-to-end: every plot function must produce a file using only
    negotiating turns, without crashing on judge/improver-heavy input."""
    results = {
        "star": _mixed_sequence(),
        "parallel": _mixed_sequence(),
        "ring": _mixed_sequence(),
    }
    with tempfile.TemporaryDirectory() as d:
        p1 = plot_gini_trajectories(results, study_name="t", save_path=os.path.join(d, "traj.png"))
        p2 = plot_average_relative_gini(results, study_name="t", save_path=os.path.join(d, "avg.png"))
        p3 = plot_relative_gini_with_interventions(
            results, study_name="t", save_path=os.path.join(d, "interv.png")
        )
        for p in (p1, p2, p3):
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000   # a real rendered image, not a stub


def test_plots_handle_topology_with_only_judge_turns():
    """A topology whose turns are ALL judge/improver filters to empty --
    must be skipped gracefully rather than crashing the whole figure."""
    results = {
        "star": _mixed_sequence(),
        "parallel": [
            TurnMetrics(1, "JudgeAgent", {}, 0.1, 0.2, 1.0, False, role="judge"),
            TurnMetrics(2, "ImproverAgent", {}, 0.1, 0.2, 1.0, True, role="improver"),
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        p = plot_gini_trajectories(results, study_name="t", save_path=os.path.join(d, "x.png"))
        assert os.path.exists(p)


def test_role_is_serialized_into_trace_dict():
    """The role must survive into JSON traces, or evaluate.py's
    reload-from-disk plotting can't filter."""
    tm = TurnMetrics(1, "Resident-New York", {}, 0.1, 0.2, 1.0, False, role="resident")
    d = tm.to_dict()
    assert d["role"] == "resident"
