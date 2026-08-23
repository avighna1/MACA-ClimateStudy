"""
logger.py
JSON execution-trace logging and the 3 categories of Relative-Gini plots:
  1. Overlay comparison of all 3 topologies' Gini trajectories (one figure).
  2. Average Relative Gini per topology (bar chart).
  3. Per-topology Gini-across-turns with Improver intervention points marked
     (one subplot per topology).

All output paths are read from `config.TRACE_DIR` / `config.FIGURE_DIR` at
CALL TIME (not import time), since main.py rewrites those to point at the
resolved Google Drive root via config.set_project_root() before any run
starts. Every filename is tagged with the active study name so different
diagnostic studies never overwrite each other's traces/figures.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

import config
from agents import Message
from metrics import TurnMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

TOPO_COLORS = {"star": "#d95f02", "parallel": "#1b9e77", "ring": "#7570b3"}
TOPO_MARKERS = {"star": "o", "parallel": "s", "ring": "^"}
TOPO_LABELS = {"star": "Star", "parallel": "Parallel", "ring": "Ring"}

# Only these agent roles actually decide the budget. Judge and Improver
# messages carry NO allocation, so their Gini value is just a repeat of
# the preceding negotiating turn -- including them made the trajectory
# plots show flat "staircase" steps that looked like the allocation had
# stalled when it simply hadn't been updated yet. Every plot below
# filters to these roles so the curves show only real negotiating moves.
NEGOTIATING_ROLES = {"resident", "functional"}


def negotiating_turns(metrics_list):
    """Filters out Judge/Improver turns, keeping only the agents that
    actually propose allocations. Falls back to the unfiltered list if
    no roles are recorded at all (older traces written before the
    `role` field existed) so old traces still plot rather than coming
    out empty."""
    if not metrics_list:
        return []
    filtered = [tm for tm in metrics_list if getattr(tm, "role", "unknown") in NEGOTIATING_ROLES]
    if not filtered and all(getattr(tm, "role", "unknown") == "unknown" for tm in metrics_list):
        logging.getLogger("logger").warning(
            "No `role` recorded on these turn metrics (likely a trace from an "
            "older run) -- plotting ALL turns, including any judge/improver "
            "ones. Re-run the study to get judge/improver-free plots."
        )
        return list(metrics_list)
    return filtered


class ExecutionLogger:
    def __init__(self, trace_dir: str = None):
        self.trace_dir = trace_dir or config.TRACE_DIR
        os.makedirs(self.trace_dir, exist_ok=True)

    def save_run_trace(
        self,
        study_name: str,
        topology_name: str,
        messages: List[Message],
        turn_metrics: List[TurnMetrics],
        role_models: Dict[str, str],
        use_judge_improver: bool,
        used_synthetic_data: bool,
    ) -> str:
        payload = {
            "study": study_name,
            "topology": topology_name,
            "use_judge_improver": use_judge_improver,
            "role_models": role_models,  # role -> MODEL_REGISTRY key actually used
            "resolved_ollama_tags": {
                role: config.MODEL_REGISTRY[key]["ollama_tag"] for role, key in role_models.items()
            },
            "cities": config.CITIES,
            "total_budget_usd": config.TOTAL_CITY_BUDGET_USD,
            "negotiation_rounds": config.NEGOTIATION_ROUNDS,
            "gini_intervention_threshold": config.GINI_INTERVENTION_THRESHOLD,
            "used_synthetic_fallback_data": used_synthetic_data,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "messages": [m.to_dict() for m in messages],
            "turn_metrics": [tm.to_dict() for tm in turn_metrics],
            "intervention_count": sum(1 for tm in turn_metrics if tm.intervention_triggered),
        }
        path = os.path.join(self.trace_dir, f"{study_name}__trace_{topology_name}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logging.getLogger("logger").info("Saved trace: %s", path)
        return path

    def load_run_trace(self, study_name: str, topology_name: str) -> dict:
        path = os.path.join(self.trace_dir, f"{study_name}__trace_{topology_name}.json")
        with open(path) as f:
            return json.load(f)


# --------------------------------------------------------------------------
# Plot 1: overlay comparison of all 3 topologies' Gini trajectories
# --------------------------------------------------------------------------
def plot_gini_trajectories(
    results: Dict[str, List[TurnMetrics]],
    study_name: str = "study",
    save_path: str = None,
) -> str:
    save_path = save_path or os.path.join(config.FIGURE_DIR, f"{study_name}__gini_trajectories.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for topo_name, raw_metrics_list in results.items():
        metrics_list = negotiating_turns(raw_metrics_list)
        if not metrics_list:
            continue
        xs = [tm.turn_index for tm in metrics_list]
        ys = [tm.relative_gini_score for tm in metrics_list]
        ax.plot(
            xs, ys, label=TOPO_LABELS.get(topo_name, topo_name.capitalize()),
            color=TOPO_COLORS.get(topo_name), marker=TOPO_MARKERS.get(topo_name, "o"),
            markersize=4, linewidth=1.8, alpha=0.9,
        )
        interv_x = [tm.turn_index for tm in metrics_list if tm.intervention_triggered]
        interv_y = [tm.relative_gini_score for tm in metrics_list if tm.intervention_triggered]
        if interv_x:
            ax.scatter(interv_x, interv_y, facecolors="none", edgecolors=TOPO_COLORS.get(topo_name, "black"),
                        s=90, linewidths=1.5, zorder=5)

    ax.axhline(config.GINI_INTERVENTION_THRESHOLD, color="red", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Intervention threshold ({config.GINI_INTERVENTION_THRESHOLD})")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Message / Communication Turn")
    ax.set_ylabel("Relative Gini Score\n(actual inequality − needs-justified inequality)")
    ax.set_title(f"Relative Gini Trajectory by Communication Topology\n(study: {study_name})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    logging.getLogger("logger").info("Saved plot: %s", save_path)
    return save_path


# --------------------------------------------------------------------------
# Plot 2: average Relative Gini per topology (bar chart)
# --------------------------------------------------------------------------
def plot_average_relative_gini(
    results: Dict[str, List[TurnMetrics]],
    study_name: str = "study",
    save_path: str = None,
) -> str:
    save_path = save_path or os.path.join(config.FIGURE_DIR, f"{study_name}__avg_relative_gini.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    filtered = {n: negotiating_turns(v) for n, v in results.items()}
    names = [n for n in filtered if filtered[n]]
    means = [float(np.mean([t.relative_gini_score for t in filtered[n]])) for n in names]
    stds = [float(np.std([t.relative_gini_score for t in filtered[n]])) for n in names]
    colors = [TOPO_COLORS.get(n, "#888888") for n in names]
    labels = [TOPO_LABELS.get(n, n.capitalize()) for n in names]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.8, width=0.55)
    for bar, m in zip(bars, means):
        ax.annotate(f"{m:+.3f}", xy=(bar.get_x() + bar.get_width() / 2, m),
                    xytext=(0, 8 if m >= 0 else -16), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(config.GINI_INTERVENTION_THRESHOLD, color="red", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Intervention threshold ({config.GINI_INTERVENTION_THRESHOLD})")
    ax.set_ylabel("Average Relative Gini Score\n(actual inequality − needs-justified inequality)")
    ax.set_title(f"Average Relative Gini Across Communication Topologies\n(study: {study_name})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    logging.getLogger("logger").info("Saved plot: %s", save_path)
    return save_path


# --------------------------------------------------------------------------
# Plot 3: per-topology Gini-across-turns with intervention points marked
# --------------------------------------------------------------------------
def plot_relative_gini_with_interventions(
    results: Dict[str, List[TurnMetrics]],
    study_name: str = "study",
    save_path: str = None,
) -> str:
    save_path = save_path or os.path.join(
        config.FIGURE_DIR, f"{study_name}__relative_gini_per_topology_with_interventions.png"
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    filtered = {n: negotiating_turns(v) for n, v in results.items()}
    names = [n for n in ("star", "parallel", "ring") if n in filtered and filtered[n]]
    if not names:
        names = [n for n in filtered if filtered[n]]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 4 * len(names)), sharex=False)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        turns = filtered[name]
        xs = [t.turn_index for t in turns]
        ys = [t.relative_gini_score for t in turns]
        color = TOPO_COLORS.get(name, "#444444")

        ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=4, alpha=0.9,
                 label=TOPO_LABELS.get(name, name.capitalize()))

        interv_x = [t.turn_index for t in turns if t.intervention_triggered]
        interv_y = [t.relative_gini_score for t in turns if t.intervention_triggered]
        if interv_x:
            ax.scatter(interv_x, interv_y, s=160, facecolors="none", edgecolors="red",
                        linewidths=2.2, zorder=5, label="Improver intervention")
            for ix in interv_x:
                ax.axvline(ix, color="red", linestyle=":", alpha=0.25, linewidth=1)

        ax.axhline(config.GINI_INTERVENTION_THRESHOLD, color="red", linestyle="--", linewidth=1, alpha=0.4)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(f"{TOPO_LABELS.get(name, name.capitalize())} Topology — Relative Gini Across Conversation Turns")
        ax.set_ylabel("Relative Gini Score")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Message / Communication Turn")
    fig.suptitle(f"study: {study_name}", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    logging.getLogger("logger").info("Saved plot: %s", save_path)
    return save_path


# --------------------------------------------------------------------------
# Cross-study comparison (used by evaluate.py to compare e.g. study1 vs
# baseline, or study2a vs 2b vs 2c) — average Relative Gini per study.
# --------------------------------------------------------------------------
def plot_cross_study_average_gini(
    study_averages: Dict[str, float],
    save_path: str = None,
    title: str = "Average Relative Gini Across Studies",
) -> str:
    save_path = save_path or os.path.join(config.FIGURE_DIR, "cross_study__avg_relative_gini.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    names = list(study_averages.keys())
    values = [study_averages[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 5.5))
    bars = ax.bar(names, values, color="#4c72b0", alpha=0.85, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:+.3f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 8 if v >= 0 else -16), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("Average Relative Gini Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    logging.getLogger("logger").info("Saved plot: %s", save_path)
    return save_path
