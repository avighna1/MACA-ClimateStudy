"""
run_experiments.py
Runs several diagnostic studies back to back in one command, saving each
one's results to its own set of files, then writes a combined comparison.

Every artifact is already tagged with the study name by main.py, so
studies never overwrite each other:

    traces/<study>__trace_<topology>.json
    transcripts/<study>__<topology>_transcript.txt
    figures/<study>__gini_trajectories.png
    figures/<study>__avg_relative_gini.png
    figures/<study>__relative_gini_per_topology_with_interventions.png

This runner adds, on top of those:

    results/<study>__summary.json      per-study summary (per topology + overall)
    results/comparison.json            all studies side by side
    results/comparison.csv             same, spreadsheet-friendly
    figures/cross_study__avg_relative_gini.png

Usage:
    # everything
    python run_experiments.py --all

    # just the ablation comparison
    python run_experiments.py --studies study1_no_judge_improver baseline

    # repeat each study N times (different seeds) -- see note below
    python run_experiments.py --all --repeats 3

A NOTE ON --repeats AND WHAT THIS CAN ACTUALLY SHOW:
A single run per study is not enough to conclude that the Judge/Improver
reduces bias. LLM negotiations are stochastic, and run-to-run variance
can easily be larger than the effect you're looking for -- so one
baseline run beating one ablation run (or vice versa) is not evidence
either way. --repeats runs each study multiple times and reports mean,
standard deviation, and min/max across runs, which is the minimum needed
to say anything defensible. If the intervals for two studies overlap
heavily, the honest conclusion is "no detectable difference at this
sample size", not a win for whichever mean happens to be lower.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Dict, List

import config


def _results_dir() -> str:
    d = os.path.join(config.PROJECT_ROOT, "results")
    os.makedirs(d, exist_ok=True)
    return d


def run_one_study(study: str, extra_args: List[str]) -> bool:
    """Invokes main.py as a subprocess so each study gets a clean
    interpreter state (no config mutations leaking between studies)."""
    cmd = [sys.executable, "main.py", "--study", study] + extra_args
    print(f"\n{'=' * 78}\n>>> RUNNING STUDY: {study}\n>>> {' '.join(cmd)}\n{'=' * 78}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n*** Study '{study}' exited with code {result.returncode} -- see output above. ***")
        return False
    return True


def summarize_study(study: str) -> Dict:
    """Reads the study's trace files and computes per-topology and overall
    stats. Judge/Improver turns are excluded (they carry no allocation, so
    their Gini just repeats the preceding turn)."""
    from evaluate import load_study_results
    from logger import negotiating_turns

    results = load_study_results(study, trace_dir=config.TRACE_DIR)
    if not results:
        return {"study": study, "error": "no trace files found"}

    per_topology = {}
    pooled: List[float] = []
    total_interventions = 0

    for topo, turns in results.items():
        agent_turns = negotiating_turns(turns)
        vals = [t.relative_gini_score for t in agent_turns]
        interventions = sum(1 for t in turns if t.intervention_triggered)
        total_interventions += interventions
        if not vals:
            continue
        pooled.extend(vals)
        per_topology[topo] = {
            "n_agent_turns": len(vals),
            "mean_relative_gini": round(mean(vals), 5),
            "final_relative_gini": round(vals[-1], 5),
            "max_relative_gini": round(max(vals), 5),
            "min_relative_gini": round(min(vals), 5),
            "interventions": interventions,
        }

    if not pooled:
        return {"study": study, "error": "no agent turns found in traces"}

    cfg = config.STUDY_CONFIGS.get(study, {})
    return {
        "study": study,
        "description": cfg.get("description", ""),
        "use_judge_improver": cfg.get("use_judge_improver"),
        "role_models": cfg.get("role_models", {}),
        "negotiation_rounds": config.NEGOTIATION_ROUNDS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_topology": per_topology,
        "overall": {
            "n_agent_turns": len(pooled),
            "mean_relative_gini": round(mean(pooled), 5),
            "abs_mean_relative_gini": round(mean(abs(v) for v in pooled), 5),
            "std_relative_gini": round(pstdev(pooled), 5) if len(pooled) > 1 else 0.0,
            "total_interventions": total_interventions,
        },
    }


def write_comparison(summaries: List[Dict], repeats: int) -> None:
    rd = _results_dir()

    # Group repeated runs of the same study
    by_study: Dict[str, List[Dict]] = {}
    for s in summaries:
        if "error" in s:
            continue
        by_study.setdefault(s["study"], []).append(s)

    comparison = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repeats_per_study": repeats,
        "negotiation_rounds": config.NEGOTIATION_ROUNDS,
        "cities": config.CITIES,
        "studies": {},
    }

    for study, runs in by_study.items():
        means = [r["overall"]["mean_relative_gini"] for r in runs]
        abs_means = [r["overall"]["abs_mean_relative_gini"] for r in runs]
        intervs = [r["overall"]["total_interventions"] for r in runs]
        comparison["studies"][study] = {
            "n_runs": len(runs),
            "use_judge_improver": runs[0].get("use_judge_improver"),
            "mean_relative_gini": {
                "mean": round(mean(means), 5),
                "std": round(pstdev(means), 5) if len(means) > 1 else 0.0,
                "min": round(min(means), 5),
                "max": round(max(means), 5),
                "runs": [round(m, 5) for m in means],
            },
            "abs_mean_relative_gini": {
                "mean": round(mean(abs_means), 5),
                "std": round(pstdev(abs_means), 5) if len(abs_means) > 1 else 0.0,
            },
            "total_interventions": {"mean": round(mean(intervs), 2), "runs": intervs},
        }

    path = os.path.join(rd, "comparison.json")
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)

    csv_path = os.path.join(rd, "comparison.csv")
    with open(csv_path, "w") as f:
        f.write("study,use_judge_improver,n_runs,mean_relative_gini,std,min,max,"
                "abs_mean_relative_gini,mean_interventions\n")
        for study, d in comparison["studies"].items():
            g = d["mean_relative_gini"]
            f.write(
                f"{study},{d['use_judge_improver']},{d['n_runs']},{g['mean']},{g['std']},"
                f"{g['min']},{g['max']},{d['abs_mean_relative_gini']['mean']},"
                f"{d['total_interventions']['mean']}\n"
            )

    print(f"\nWrote {path}\nWrote {csv_path}")
    _print_comparison_table(comparison, repeats)


def _print_comparison_table(comparison: Dict, repeats: int) -> None:
    print(f"\n{'=' * 92}")
    print("CROSS-STUDY COMPARISON  (Relative Gini: 0.0 = allocation tracks documented need;")
    print("                         positive = MORE unequal than need justifies -- lower |value| is better)")
    print("=" * 92)
    print(f"{'study':<34}{'judge?':<9}{'runs':<6}{'mean':>10}{'std':>9}{'|mean|':>10}{'interv':>9}")
    print("-" * 92)
    for study, d in comparison["studies"].items():
        g = d["mean_relative_gini"]
        print(
            f"{study:<34}{str(d['use_judge_improver']):<9}{d['n_runs']:<6}"
            f"{g['mean']:>+10.4f}{g['std']:>9.4f}"
            f"{d['abs_mean_relative_gini']['mean']:>10.4f}"
            f"{d['total_interventions']['mean']:>9.1f}"
        )
    print("=" * 92)

    if repeats < 3:
        print(
            "\nINTERPRETATION WARNING: with "
            f"{repeats} run(s) per study there is no meaningful variance estimate.\n"
            "LLM negotiations are stochastic and run-to-run spread is often larger than\n"
            "the effect being measured, so differences in the table above may be noise.\n"
            "Re-run with --repeats 5 (or more) before drawing any conclusion about\n"
            "whether the Judge/Improver reduces bias."
        )
    else:
        print(
            "\nInterpretation: compare each study's mean against the SPREAD (std, min-max)\n"
            "across its runs. If two studies' ranges overlap substantially, the honest\n"
            "conclusion is 'no detectable difference at this sample size' -- not a win\n"
            "for whichever mean is lower."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run several studies consecutively, saving each to its own result files"
    )
    parser.add_argument("--studies", nargs="+", help="Study names to run, in order")
    parser.add_argument("--all", action="store_true", help="Run every study in config.STUDY_CONFIGS")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run each study this many times (needed for any variance estimate)")
    parser.add_argument("--rounds", type=int, default=None, help="Override negotiation rounds")
    parser.add_argument("--topology", default=None, help="Restrict to one topology")
    parser.add_argument("--drive-root", default=None)
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Skip running; just re-summarize existing traces")
    args = parser.parse_args()

    if args.all:
        studies = list(config.STUDY_CONFIGS.keys())
    elif args.studies:
        unknown = [s for s in args.studies if s not in config.STUDY_CONFIGS]
        if unknown:
            print(f"Unknown study name(s): {unknown}\nAvailable: {list(config.STUDY_CONFIGS)}")
            return 1
        studies = args.studies
    else:
        parser.error("Pass --all or --studies <name> [<name> ...]")

    if args.drive_root:
        import drive_utils
        config.set_project_root(drive_utils.mount_and_resolve_drive_root(args.drive_root))

    extra: List[str] = []
    if args.rounds is not None:
        extra += ["--rounds", str(args.rounds)]
    if args.topology:
        extra += ["--topology", args.topology]
    if args.drive_root:
        extra += ["--drive-root", args.drive_root]
    if args.skip_server_check:
        extra.append("--skip-server-check")

    print(f"Studies to run: {studies}")
    print(f"Repeats per study: {args.repeats}")

    summaries: List[Dict] = []
    failures: List[str] = []

    for rep in range(1, args.repeats + 1):
        for study in studies:
            if not args.summarize_only:
                if args.repeats > 1:
                    print(f"\n### repeat {rep}/{args.repeats} ###")
                if not run_one_study(study, extra):
                    failures.append(f"{study} (repeat {rep})")
                    continue

            summary = summarize_study(study)
            if "error" in summary:
                print(f"  [{study}] summary skipped: {summary['error']}")
                failures.append(f"{study} (repeat {rep}): {summary['error']}")
                continue

            rd = _results_dir()
            suffix = f"__run{rep}" if args.repeats > 1 else ""
            path = os.path.join(rd, f"{study}{suffix}__summary.json")
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  [{study}] summary -> {path}")
            summaries.append(summary)

    if summaries:
        write_comparison(summaries, args.repeats)
        try:
            from evaluate import cross_study_average_gini
            from logger import plot_cross_study_average_gini

            averages = cross_study_average_gini(studies, trace_dir=config.TRACE_DIR)
            if averages:
                p = plot_cross_study_average_gini(
                    averages,
                    save_path=os.path.join(config.FIGURE_DIR, "cross_study__avg_relative_gini.png"),
                )
                print(f"Wrote {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"(cross-study plot skipped: {exc})")

    if failures:
        print(f"\n*** {len(failures)} run(s) failed or produced no data: {failures} ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
