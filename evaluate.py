"""
evaluate.py
Reloads saved JSON traces from disk (config.TRACE_DIR) and regenerates
every plot from them. Decoupled from a live run on purpose: you can run
all 5 studies overnight on the A100, then re-plot / re-compare them
later (or on a different machine) without needing a live Ollama server.

Usage:
    python evaluate.py --study study2a_all_3b
        -> regenerates the 3 standard plots for that one study

    python evaluate.py --compare study1_no_judge_improver baseline \
                                  study2a_all_3b study2b_all_7b study2c_judge_improver_7b
        -> cross-study average-Relative-Gini bar chart, one bar per study
           (averaged across all 3 topologies within each study)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import config
from logger import (
    plot_average_relative_gini,
    plot_cross_study_average_gini,
    plot_gini_trajectories,
    plot_relative_gini_with_interventions,
)


@dataclass
class _TurnMetricsView:
    """Lightweight stand-in for metrics.TurnMetrics, built from a saved
    JSON trace's turn_metrics entries, so logger.py's plot functions can
    be reused unchanged for post-hoc (re-loaded-from-disk) plotting."""
    turn_index: int
    speaker: str
    allocation: Dict[str, float]
    relative_gini_score: float
    raw_gini_score: float
    disparity_ratio_score: float
    intervention_triggered: bool
    role: str = "unknown"

    @classmethod
    def from_dict(cls, d: dict) -> "_TurnMetricsView":
        return cls(
            turn_index=d["turn_index"],
            speaker=d["speaker"],
            allocation=d.get("allocation", {}),
            relative_gini_score=d["relative_gini"],
            raw_gini_score=d["raw_gini"],
            disparity_ratio_score=d.get("disparity_ratio") or 0.0,
            intervention_triggered=d.get("intervention_triggered", False),
            # Older traces (written before judge/improver filtering)
            # have no role -- defaults to "unknown", which
            # logger.negotiating_turns() handles explicitly.
            role=d.get("role", "unknown"),
        )


def discover_topologies_for_study(study_name: str, trace_dir: str = None) -> List[str]:
    trace_dir = trace_dir or config.TRACE_DIR
    pattern = os.path.join(trace_dir, f"{study_name}__trace_*.json")
    found = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        # <study>__trace_<topology>.json
        topo = base[len(study_name) + len("__trace_"):-len(".json")]
        found.append(topo)
    return found


def load_study_results(study_name: str, trace_dir: str = None) -> Dict[str, List[_TurnMetricsView]]:
    trace_dir = trace_dir or config.TRACE_DIR
    results: Dict[str, List[_TurnMetricsView]] = {}
    for topo in discover_topologies_for_study(study_name, trace_dir):
        path = os.path.join(trace_dir, f"{study_name}__trace_{topo}.json")
        with open(path) as f:
            payload = json.load(f)
        results[topo] = [_TurnMetricsView.from_dict(tm) for tm in payload["turn_metrics"]]
    return results


def regenerate_plots_for_study(study_name: str, trace_dir: str = None, figure_dir: str = None) -> Dict[str, str]:
    trace_dir = trace_dir or config.TRACE_DIR
    figure_dir = figure_dir or config.FIGURE_DIR
    results = load_study_results(study_name, trace_dir)
    if not results:
        raise FileNotFoundError(
            f"No trace files found for study '{study_name}' under {trace_dir} "
            f"(expected files like '{study_name}__trace_star.json'). Did that "
            f"study actually get run yet?"
        )
    paths = {
        "trajectories": plot_gini_trajectories(
            results, study_name=study_name,
            save_path=os.path.join(figure_dir, f"{study_name}__gini_trajectories.png"),
        ),
        "average_bar": plot_average_relative_gini(
            results, study_name=study_name,
            save_path=os.path.join(figure_dir, f"{study_name}__avg_relative_gini.png"),
        ),
        "per_topology_interventions": plot_relative_gini_with_interventions(
            results, study_name=study_name,
            save_path=os.path.join(figure_dir, f"{study_name}__relative_gini_per_topology_with_interventions.png"),
        ),
    }
    return paths


def cross_study_average_gini(study_names: List[str], trace_dir: str = None) -> Dict[str, float]:
    """Average Relative Gini per study, pooled across all its topologies —
    a single number per study for the headline ablation/model-scale bar chart."""
    trace_dir = trace_dir or config.TRACE_DIR
    averages: Dict[str, float] = {}
    for study_name in study_names:
        results = load_study_results(study_name, trace_dir)
        # Judge/Improver turns carry no allocation -- exclude them so the
        # cross-study average reflects only real negotiating moves.
        from logger import negotiating_turns
        all_vals = [
            tm.relative_gini_score
            for topo_list in results.values()
            for tm in negotiating_turns(topo_list)
        ]
        if not all_vals:
            print(f"[evaluate] WARNING: no data found for study '{study_name}' — skipping.")
            continue
        averages[study_name] = sum(all_vals) / len(all_vals)
    return averages


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from saved JSON traces")
    parser.add_argument("--study", help="Regenerate the 3 standard plots for one study")
    parser.add_argument("--compare", nargs="+", help="Study names to compare in one cross-study bar chart")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    args = parser.parse_args()

    trace_dir = args.trace_dir or config.TRACE_DIR
    figure_dir = args.figure_dir or config.FIGURE_DIR
    os.makedirs(figure_dir, exist_ok=True)

    if not args.study and not args.compare:
        parser.error("Pass --study <name> and/or --compare <name1> <name2> ...")

    if args.study:
        paths = regenerate_plots_for_study(args.study, trace_dir, figure_dir)
        print(f"Regenerated plots for study '{args.study}':")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")

    if args.compare:
        averages = cross_study_average_gini(args.compare, trace_dir)
        if averages:
            path = plot_cross_study_average_gini(
                averages, save_path=os.path.join(figure_dir, "cross_study__avg_relative_gini.png")
            )
            print(f"Cross-study comparison ({list(averages.keys())}): {path}")
            for name, val in averages.items():
                print(f"  {name}: avg RelativeGini = {val:+.4f}")


if __name__ == "__main__":
    main()
