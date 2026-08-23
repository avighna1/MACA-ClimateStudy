"""
Smoke test for the full pipeline WITHOUT a live Ollama server.

We monkeypatch OllamaClient.chat with a deterministic stub that returns
plausible negotiation text + a JSON allocations block, so we can verify
that RAG loading (offline synthetic fallback), the role -> client
registry, transcript logging, topology routing (including the Judge/
Improver ablation switch), metrics computation, JSON trace logging, and
all 3 plot types wire together correctly end to end across MULTIPLE
diagnostic studies.

This does NOT validate real model output quality, nor real Ollama
behavior -- see README for how to run the real pipeline against a live
Ollama server.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

# NOTE: every run.run(...) call in this file passes rounds=1 explicitly
# (kept fast on purpose) -- do NOT set config.NEGOTIATION_ROUNDS at
# module level here. A prior version did, which mutated the GLOBAL
# default permanently for the rest of the test process (module-level
# code runs once at import time and is never undone), silently breaking
# any other test file that checks config.NEGOTIATION_ROUNDS's actual
# default value when run in the same process after this one.

from agents import OllamaClient, build_role_clients
from evaluate import load_study_results, regenerate_plots_for_study
from logger import ExecutionLogger, plot_gini_trajectories
from image_data import CityData
from topologies import TOPOLOGY_REGISTRY
from transcript import TranscriptLogger


class StubOllamaClient(OllamaClient):
    """Bypasses real HTTP calls; returns deterministic, parseable text."""

    def __init__(self, model="stub-model"):
        self.base_url = "stub://"
        self.api_key = "stub"
        self.model = model
        self.max_retries = 1
        self.allow_offline_fallback = True
        self._client = "stub"  # truthy sentinel so chat() doesn't reinit

    def chat(self, system_prompt: str, user_prompt: str, temperature=None, max_tokens=None, images=None) -> str:
        is_judge = "JudgeAgent" in system_prompt
        is_improver = "ImproverAgent" in system_prompt
        if is_judge:
            return "No bias evident in this message; allocation tracks documented need."
        if is_improver:
            return "Corrective guidance: re-align with documented need scores."
        rng = random.Random(hash(system_prompt) % (2**32))
        allocations = {city: round(rng.uniform(20_000_000, 120_000_000), 2) for city in config.CITIES}
        return (
            "This is a stubbed negotiation message referencing retrieved hazard "
            "and vulnerability evidence for the discussion.\n"
            + json.dumps({"allocations": allocations, "rationale": "stubbed rationale"})
        )


def _stub_role_clients():
    """Same shape as agents.build_role_clients(), but every entry is a stub."""
    return {role: StubOllamaClient(model=f"stub-{role}") for role in config.AGENT_ROLES}


def _stub_city_data():
    """Minimal per-city bundles (a tiny fake image + risk numbers) --
    no disk reads, no network, no RAG."""
    return {
        city: CityData(
            city=city,
            images=[("ZmFrZWltYWdl", "image/png")],
            hazard_risk_score=50.0 + 10 * i,
            social_vulnerability_index=0.4 + 0.1 * i,
            dominant_hazard="Coastal Flooding",
            n_counties_matched=1,
        )
        for i, city in enumerate(config.CITIES)
    }


def test_city_data_builds_offline():
    city_data = _stub_city_data()
    assert len(city_data) == len(config.CITIES)
    from image_data import zone_need_scores
    need_scores = zone_need_scores(city_data)
    assert len(need_scores) == len(config.CITIES)
    for cd in city_data.values():
        assert 0.0 <= cd.combined_need_score() <= 1.0


def test_build_role_clients_shares_client_per_model_key():
    role_models = {
        "resident": "qwen_4b", "infrastructure": "qwen_4b", "financial": "qwen_4b",
        "sustainability": "qwen_4b", "judge": "qwen_9b", "improver": "qwen_9b",
    }
    unique_keys = set(role_models.values())
    assert unique_keys == {"qwen_4b", "qwen_9b"}


def test_topology_run_with_judge_improver_enabled(tmp_path):
    city_data = _stub_city_data()
    clients = _stub_role_clients()
    transcript = TranscriptLogger(str(tmp_path / "t.txt"), echo_to_terminal=False)

    run = TOPOLOGY_REGISTRY["ring"](city_data=city_data, clients=clients, transcript=transcript, use_judge_improver=True)
    messages, turn_metrics = run.run(rounds=1)
    transcript.close()

    assert len(messages) > 0
    roles_seen = {m.role for m in messages}
    assert "judge" in roles_seen
    for tm in turn_metrics:
        assert -1.0 <= tm.relative_gini_score <= 1.0
        assert 0.0 <= tm.raw_gini_score <= 1.0
    assert os.path.exists(str(tmp_path / "t.txt"))
    with open(str(tmp_path / "t.txt")) as f:
        content = f.read()
    assert "TRANSCRIPT START" in content and "TRANSCRIPT END" in content


def test_topology_run_with_judge_improver_disabled(tmp_path):
    city_data = _stub_city_data()
    clients = _stub_role_clients()
    transcript = TranscriptLogger(str(tmp_path / "t2.txt"), echo_to_terminal=False)

    run = TOPOLOGY_REGISTRY["star"](city_data=city_data, clients=clients, transcript=transcript, use_judge_improver=False)
    messages, turn_metrics = run.run(rounds=1)
    transcript.close()

    roles_seen = {m.role for m in messages}
    assert "judge" not in roles_seen
    assert "improver" not in roles_seen
    assert len(messages) > 0


def test_full_multi_topology_run_and_all_plots(tmp_path):
    city_data = _stub_city_data()
    clients = _stub_role_clients()

    study_name = "unit_test_study"
    trace_dir = str(tmp_path / "traces")
    figure_dir = str(tmp_path / "figures")
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    exec_logger = ExecutionLogger(trace_dir=trace_dir)
    all_metrics = {}
    for name, cls in TOPOLOGY_REGISTRY.items():
        transcript = TranscriptLogger(str(tmp_path / f"{name}.txt"), echo_to_terminal=False)
        run = cls(city_data=city_data, clients=clients, transcript=transcript, use_judge_improver=True)
        messages, turn_metrics = run.run(rounds=1)
        transcript.close()

        path = exec_logger.save_run_trace(
            study_name=study_name, topology_name=name, messages=messages,
            turn_metrics=turn_metrics,
            role_models={r: "qwen_4b" for r in config.AGENT_ROLES},
            use_judge_improver=True, used_synthetic_data=True,
        )
        assert os.path.exists(path)
        with open(path) as f:
            payload = json.load(f)
        assert payload["topology"] == name
        assert payload["study"] == study_name
        all_metrics[name] = turn_metrics

    fig_path = plot_gini_trajectories(all_metrics, study_name=study_name, save_path=str(tmp_path / "traj.png"))
    assert os.path.exists(fig_path)

    config.set_project_root(str(tmp_path))
    reloaded = load_study_results(study_name, trace_dir=trace_dir)
    assert set(reloaded.keys()) == set(TOPOLOGY_REGISTRY.keys())
    paths = regenerate_plots_for_study(study_name, trace_dir=trace_dir, figure_dir=figure_dir)
    for p in paths.values():
        assert os.path.exists(p)
