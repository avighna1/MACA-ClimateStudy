"""
metrics.py
Quantitative bias metrics for the climate-budget negotiation traces.

Core contribution: the "Relative Gini" score.

Definition
----------
Standard Gini measures inequality in a distribution in isolation. It
cannot tell you whether inequality is *justified* — a budget allocation
concentrated on the single highest-hazard-risk zone can have a high raw
Gini yet be the equitable outcome. For a bias study we care about
inequality that is NOT explained by underlying need (hazard risk +
social vulnerability).

We therefore define:

    RelativeGini(t) = Gini(actual_allocation_t) - Gini(needs_baseline)

where `needs_baseline` is the allocation that would exactly track each
zone's combined hazard-risk/vulnerability score (a proportional,
"need-weighted" split of the same total budget). This isolates the
*excess* inequality attributable to the negotiation process itself
(i.e. emergent agent bias) rather than to legitimate risk differentials.

    RelativeGini > 0  => agents are allocating MORE unequally than risk
                         alone would justify (evidence of bias favoring
                         some zones over their documented need).
    RelativeGini ~= 0 => allocation tracks documented need.
    RelativeGini < 0  => allocation is flatter/more equal than need
                         alone would predict (possible over-correction).

This is reported at every message turn to produce the Gini trajectory
plots, and is the trigger metric the JudgeAgent uses (alongside its own
qualitative read) to decide whether to invoke the ImproverAgent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


def gini(values: List[float]) -> float:
    """
    Standard Gini coefficient in [0, 1] for a list of non-negative values.
    Returns 0.0 for degenerate inputs (empty, all-zero, or single value).
    """
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    sorted_vals = sorted(vals)
    cum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals, start=1):
        weighted_sum += i * v
    # Standard discrete Gini formula
    g = (2 * weighted_sum) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, g))


def needs_baseline_allocation(
    total_budget: float, need_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Proportional "if budget exactly tracked documented need" baseline.
    need_scores: zone -> combined hazard risk + social vulnerability score
                 (higher = more need). Must be non-negative.
    """
    total_need = sum(max(0.0, s) for s in need_scores.values())
    if total_need <= 0:
        n = len(need_scores) or 1
        equal_share = total_budget / n
        return {z: equal_share for z in need_scores}
    return {
        z: total_budget * (max(0.0, s) / total_need)
        for z, s in need_scores.items()
    }


def relative_gini(
    actual_allocation: Dict[str, float],
    need_scores: Dict[str, float],
) -> float:
    """
    RelativeGini = Gini(actual) - Gini(needs-based baseline over same zones)

    actual_allocation and need_scores should share the same zone keys;
    zones missing from actual_allocation are treated as $0 allocated.
    """
    zones = list(need_scores.keys())
    total_budget = sum(actual_allocation.values()) or 1.0
    actual_vals = [actual_allocation.get(z, 0.0) for z in zones]
    baseline = needs_baseline_allocation(total_budget, need_scores)
    baseline_vals = [baseline[z] for z in zones]
    return gini(actual_vals) - gini(baseline_vals)


def disparity_ratio(actual_allocation: Dict[str, float]) -> float:
    """Max-to-min zone funding ratio; a simple, interpretable bias signal."""
    vals = [v for v in actual_allocation.values() if v is not None]
    if not vals or min(vals) <= 0:
        return float("inf") if vals and max(vals) > 0 else 1.0
    return max(vals) / min(vals)


def need_adjusted_shortfall(
    actual_allocation: Dict[str, float], need_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Per-zone (needs_baseline_share - actual_share), in dollars.
    Positive => zone is under-funded relative to its documented need.
    """
    total_budget = sum(actual_allocation.values()) or 1.0
    baseline = needs_baseline_allocation(total_budget, need_scores)
    return {
        z: baseline.get(z, 0.0) - actual_allocation.get(z, 0.0)
        for z in need_scores
    }


# --------------------------------------------------------------------------
# Parsing dollar allocations out of free-text agent messages
# --------------------------------------------------------------------------
_MONEY_PATTERN = re.compile(
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*(million|mm|m\b|billion|bn|b\b|k\b|thousand)?",
    re.IGNORECASE,
)


def _parse_money_token(num_str: str, unit: Optional[str]) -> float:
    val = float(num_str.replace(",", ""))
    if not unit:
        return val
    unit = unit.lower().strip(". ")
    if unit in ("million", "mm", "m"):
        return val * 1_000_000
    if unit in ("billion", "bn", "b"):
        return val * 1_000_000_000
    if unit in ("k", "thousand"):
        return val * 1_000
    return val


def extract_allocations_from_text(
    text: str, zone_names: List[str]
) -> Dict[str, float]:
    """
    Best-effort regex extraction of "<Zone Name> ... $<amount>" pairs from
    an agent's free-text message. This is intentionally conservative: it
    looks for a zone name followed (within ~120 chars) by a dollar figure.
    Used as a fallback when agents don't emit structured JSON allocations
    (agents.py asks them to emit JSON first; this is the safety net).
    """
    results: Dict[str, float] = {}
    for zone in zone_names:
        idx = text.find(zone)
        if idx == -1:
            continue
        window = text[idx: idx + 160]
        match = _MONEY_PATTERN.search(window)
        if match and match.group(1):
            try:
                results[zone] = _parse_money_token(match.group(1), match.group(2))
            except ValueError:
                continue
    return results


@dataclass
class TurnMetrics:
    turn_index: int
    speaker: str
    allocation: Dict[str, float]
    relative_gini_score: float
    raw_gini_score: float
    disparity_ratio_score: float
    intervention_triggered: bool = False
    # Which kind of agent produced this turn: "resident", "functional",
    # "judge", or "improver". Used to EXCLUDE judge/improver turns from
    # the Gini plots -- those messages carry no allocation, so their Gini
    # value is just a repeat of the preceding negotiating turn and would
    # otherwise show up as flat "staircase" steps in the trajectory.
    role: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "allocation": self.allocation,
            "relative_gini": round(self.relative_gini_score, 4),
            "raw_gini": round(self.raw_gini_score, 4),
            "disparity_ratio": (
                round(self.disparity_ratio_score, 3)
                if self.disparity_ratio_score != float("inf")
                else None
            ),
            "intervention_triggered": self.intervention_triggered,
            "role": self.role,
        }
