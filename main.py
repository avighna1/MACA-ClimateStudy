
from __future__ import annotations

import argparse
import logging
import sys

import config
import drive_utils
import setup_ollama
from agents import build_role_clients
from logger import (
    ExecutionLogger,
    plot_average_relative_gini,
    plot_gini_trajectories,
    plot_relative_gini_with_interventions,
)
from image_data import any_synthetic_risk, build_city_data, cities_without_images
from topologies import TOPOLOGY_REGISTRY
from transcript import TranscriptLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-agent climate budget bias study runner")
    p.add_argument(
        "--study", choices=list(config.STUDY_CONFIGS.keys()),
        default=config.ACTIVE_STUDY_DEFAULT,
        help="Which diagnostic study to run (see config.STUDY_CONFIGS)",
    )
    p.add_argument(
        "--topology", choices=list(TOPOLOGY_REGISTRY.keys()) + ["all"],
        default="all", help="Which topology to run (default: all 3)",
    )
    p.add_argument(
        "--rounds", type=int, default=config.NEGOTIATION_ROUNDS,
    )
    p.add_argument("--budget", type=float, default=config.TOTAL_CITY_BUDGET_USD)
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument(
        "--drive-root", default=None,
        help="Explicit Google Drive (or other) output root. If omitted, auto-detected "
             "(Colab mount, pre-mounted Drive path, or a clearly-labeled local fallback). "
             "Google Drive is the only persistence backend this project uses.",
    )
    p.add_argument(
        "--skip-server-check", action="store_true",
        help="Skip the pre-flight Ollama health/model check (not recommended — a failed "
             "check means calls will error out mid-run instead of failing fast).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config.NEGOTIATION_ROUNDS = args.rounds
    config.TOTAL_CITY_BUDGET_USD = args.budget
    study_name = args.study
    study_cfg = config.STUDY_CONFIGS[study_name]

    
    root = drive_utils.mount_and_resolve_drive_root(args.drive_root)
    config.set_project_root(root)
    is_real_drive = "LOCAL_NOT_DRIVE" not in root

    print("=" * 78)
    print(f"Multi-Agent Climate Budget Allocation — Emergent Bias Study")
    print(f"Study:              {study_name}")
    print(f"  {study_cfg['description']}")
    print(f"Judge/Improver:     {'ENABLED' if study_cfg['use_judge_improver'] else 'DISABLED (ablation)'}")
    print(f"Ollama endpoint:    {config.OLLAMA_HOST}")
    print(f"Role -> model tag:")
    for role, key in study_cfg["role_models"].items():
        print(f"    {role:16s} -> {config.ollama_tag_for(key)}")
    print(f"Cities:             {', '.join(config.CITIES)}")
    print(f"Total budget:       ${config.TOTAL_CITY_BUDGET_USD:,.0f}")
    print(f"Rounds/topology:    {config.NEGOTIATION_ROUNDS}")
    print(f"Output root:        {config.PROJECT_ROOT}"
          f"{'  (Google Drive)' if is_real_drive else '  (LOCAL — NOT Drive, see warning above)'}")
    print("=" * 78)

    
    if not args.skip_server_check:
        if not setup_ollama.is_server_running():
            print(
                f"\n*** ERROR: Ollama is not reachable at {config.OLLAMA_HOST}.\n"
                f"    Start it first (a system service, the Ollama app, or "
                f"`ollama serve` in its own terminal), then run:\n"
                f"        python setup_ollama.py --study {study_name}\n"
                f"    and re-run main.py.\n"
                f"    (Skip this check with --skip-server-check if you know "
                f"what you're doing — NOT recommended.) ***\n"
            )
            sys.exit(1)
        missing = setup_ollama.missing_models(study_name)
        if missing:
            print(
                f"\n*** ERROR: study '{study_name}' needs these Ollama model tags, "
                f"which aren't pulled yet: {missing}\n"
                f"    Pull them first with:\n"
                f"        python setup_ollama.py --study {study_name}\n"
                f"    and re-run main.py.\n"
                f"    (Skip this check with --skip-server-check if you know "
                f"what you're doing — NOT recommended.) ***\n"
            )
            sys.exit(1)
        print("[main] Pre-flight check passed: Ollama is running and all required models are pulled.\n")

    
    logger.info("Loading CityLens images from local disk + FEMA NRI risk data...")
    city_data = build_city_data(config.CITIES)

    for city, cd in city_data.items():
        print(
            f"  {city:16s} {len(cd.images)} image(s)"
            f"{' from ' + cd.image_source_dir if cd.image_source_dir else ' (no directory found)'}"
            f" | hazard={cd.hazard_risk_score:.1f} sovi={cd.social_vulnerability_index:.2f}"
            f"{'  [SYNTHETIC RISK]' if cd.synthetic_risk else ''}"
        )

    missing_images = cities_without_images(city_data)
    if missing_images:
        print(
            f"\n*** WARNING: no images were loaded for: {missing_images}. Those "
            f"Resident agents will run with FEMA risk text only. Check the paths "
            f"in config.CITYLENS_IMAGE_DIRS (see the log lines above for the "
            f"exact directories checked and the current working directory). ***\n"
        )

    used_synthetic = any_synthetic_risk(city_data)
    if used_synthetic:
        print(
            "\n*** WARNING: FEMA NRI risk data fell back to SYNTHETIC placeholder "
            "values for one or more cities (the CSV couldn't be found or matched). "
            "Do not report results from this run as real FEMA NRI findings. This "
            "is recorded in every trace's 'used_synthetic_fallback_data' flag. ***\n"
        )

    
    clients = build_role_clients(study_cfg["role_models"], allow_offline_fallback=False)

    topologies_to_run = list(TOPOLOGY_REGISTRY.keys()) if args.topology == "all" else [args.topology]

    exec_logger = ExecutionLogger()
    all_metrics = {}

    for topo_name in topologies_to_run:
        topo_cls = TOPOLOGY_REGISTRY[topo_name]
        transcript_path = f"{config.TRANSCRIPT_DIR}/{study_name}__{topo_name}_transcript.txt"
        transcript = TranscriptLogger(transcript_path, echo_to_terminal=True)

        run = topo_cls(
            city_data=city_data,
            clients=clients,
            transcript=transcript,
            use_judge_improver=study_cfg["use_judge_improver"],
        )
        try:
            messages, turn_metrics = run.run(rounds=config.NEGOTIATION_ROUNDS)
        except Exception as exc:  # noqa: BLE001
            logger.error("Topology '%s' failed with an unrecoverable error: %s", topo_name, exc)
            transcript.log_note(f"RUN FAILED: {exc}")
            transcript.close()
            continue
        transcript.close()

        exec_logger.save_run_trace(
            study_name=study_name,
            topology_name=topo_name,
            messages=messages,
            turn_metrics=turn_metrics,
            role_models=study_cfg["role_models"],
            use_judge_improver=study_cfg["use_judge_improver"],
            used_synthetic_data=used_synthetic,
        )
        all_metrics[topo_name] = turn_metrics

        interventions = sum(1 for tm in turn_metrics if tm.intervention_triggered)
        final_gini = turn_metrics[-1].relative_gini_score if turn_metrics else float("nan")
        print(
            f"\n>>> [{topo_name.upper()}] complete: {len(messages)} messages, "
            f"{interventions} improver interventions, final RelativeGini={final_gini:+.3f}\n"
            f">>> Transcript saved to: {transcript_path}\n"
        )

    if not args.skip_plot and all_metrics:
        p1 = plot_gini_trajectories(all_metrics, study_name=study_name)
        p2 = plot_average_relative_gini(all_metrics, study_name=study_name)
        p3 = plot_relative_gini_with_interventions(all_metrics, study_name=study_name)
        print(f"\nPlots saved:\n  {p1}\n  {p2}\n  {p3}")

    print(f"\nDone. All traces, transcripts, and figures are under: {config.PROJECT_ROOT}"
          f"{' (Google Drive)' if is_real_drive else ' (LOCAL — NOT Drive)'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
