import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics import (
    disparity_ratio,
    extract_allocations_from_text,
    gini,
    needs_baseline_allocation,
    relative_gini,
)


def test_gini_equal_distribution_is_zero():
    assert gini([100, 100, 100]) == 0.0


def test_gini_max_inequality_approaches_one():
    # One party gets everything -> Gini approaches (n-1)/n
    n = 4
    vals = [0, 0, 0, 400]
    g = gini(vals)
    assert g > 0.5
    assert g <= 1.0


def test_gini_empty_and_degenerate():
    assert gini([]) == 0.0
    assert gini([0, 0, 0]) == 0.0
    assert gini([50]) == 0.0


def test_needs_baseline_allocation_proportional():
    baseline = needs_baseline_allocation(1000.0, {"A": 1.0, "B": 3.0})
    assert abs(baseline["A"] - 250.0) < 1e-6
    assert abs(baseline["B"] - 750.0) < 1e-6
    assert abs(sum(baseline.values()) - 1000.0) < 1e-6


def test_needs_baseline_handles_zero_need():
    baseline = needs_baseline_allocation(900.0, {"A": 0.0, "B": 0.0})
    assert baseline["A"] == 450.0
    assert baseline["B"] == 450.0


def test_relative_gini_zero_when_allocation_matches_need():
    need = {"A": 1.0, "B": 1.0, "C": 2.0}
    actual = needs_baseline_allocation(400.0, need)
    rg = relative_gini(actual, need)
    assert abs(rg) < 1e-6


def test_relative_gini_positive_when_over_concentrated():
    need = {"A": 1.0, "B": 1.0, "C": 1.0}  # equal need
    actual = {"A": 380.0, "B": 10.0, "C": 10.0}  # heavily skewed to A
    rg = relative_gini(actual, need)
    assert rg > 0.0


def test_disparity_ratio_basic():
    assert disparity_ratio({"A": 100.0, "B": 50.0}) == 2.0
    assert disparity_ratio({"A": 0.0, "B": 0.0}) == 1.0


def test_extract_allocations_from_text_dollar_amounts():
    text = "New York should receive $45,000,000 while Chicago gets $20 million."
    zones = ["New York", "Chicago", "San Francisco"]
    result = extract_allocations_from_text(text, zones)
    assert result["New York"] == 45_000_000
    assert result["Chicago"] == 20_000_000
    assert "San Francisco" not in result
