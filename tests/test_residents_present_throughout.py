"""
Tests for the "residents present throughout" negotiation structure:
  - 3 Resident agents propose once at the start (unchanged).
  - Then, in EVERY negotiation round (config.NEGOTIATION_ROUNDS, default
    4) and in ALL 3 topologies, all 6 agents -- 3 Residents AND 3
    Functional agents -- each speak exactly once. Residents are not
    silent after their initial proposal; they continue advocating for
    their own city every round, alongside the functional agents'
    cross-city allocation proposals.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from agents import OllamaClient
from image_data import CityData
from topologies import TOPOLOGY_REGISTRY
from transcript import TranscriptLogger

DECIDING_ROLES = {"resident", "functional"}  # excludes judge/improver oversight messages


class StubOllamaClient(OllamaClient):
    def __init__(self, model="stub-model"):
        self.base_url = "stub://"
        self.api_key = "stub"
        self.model = model
        self.max_retries = 1
        self.allow_offline_fallback = True
        self._client = "stub"

    def chat(self, system_prompt: str, user_prompt: str, temperature=None, max_tokens=None, images=None) -> str:
        if "JudgeAgent" in system_prompt:
            return "No bias evident in this message; allocation tracks documented need."
        if "ImproverAgent" in system_prompt:
            return "Corrective guidance: re-align with documented need scores."
        rng = random.Random(hash(system_prompt) % (2**32))
        allocations = {city: round(rng.uniform(20_000_000, 120_000_000), 2) for city in config.CITIES}
        return (
            "Stubbed negotiation message.\n"
            + json.dumps({"allocations": allocations, "rationale": "stubbed"})
        )


def _stub_role_clients():
    return {role: StubOllamaClient(model=f"stub-{role}") for role in config.AGENT_ROLES}


def _deciding_speaker_messages(messages):
    return [m for m in messages if m.role in DECIDING_ROLES]


def _run_topology(topology_key, tmp_path, rounds=None):
    city_data = {
        city: CityData(
            city=city, images=[("ZmFrZQ==", "image/png")],
            hazard_risk_score=60.0, social_vulnerability_index=0.5,
            dominant_hazard="Flooding", n_counties_matched=1,
        )
        for city in config.CITIES
    }
    clients = _stub_role_clients()
    transcript = TranscriptLogger(str(tmp_path / f"{topology_key}.txt"), echo_to_terminal=False)
    run = TOPOLOGY_REGISTRY[topology_key](city_data=city_data, clients=clients, transcript=transcript, use_judge_improver=True)
    messages, turn_metrics = run.run(rounds=rounds)
    transcript.close()
    return messages, turn_metrics


# Derived from config.CITIES rather than hardcoded, so these tests stay
# correct if the city list changes (Chicago was removed; there are now 2
# residents + 3 functional agents = 5 speakers per round).
FUNCTIONAL_SPEAKERS = {"InfrastructureAgent", "FinancialAgent", "SustainabilityAgent"}
RESIDENT_SPEAKERS = {f"Resident-{c}" for c in config.CITIES}
EXPECTED_SPEAKERS = RESIDENT_SPEAKERS | FUNCTIONAL_SPEAKERS
N_RESIDENTS = len(config.CITIES)
N_SPEAKERS_PER_ROUND = N_RESIDENTS + len(FUNCTIONAL_SPEAKERS)


def test_initial_proposals_are_residents_only(tmp_path):
    """The initial phase (before any negotiation round) must be exactly
    the 3 residents, once each -- unchanged from before."""
    for topo in ("star", "parallel", "ring"):
        messages, _ = _run_topology(topo, tmp_path, rounds=1)
        speaker_msgs = _deciding_speaker_messages(messages)
        first_three = speaker_msgs[:N_RESIDENTS]
        assert {m.speaker for m in first_three} == RESIDENT_SPEAKERS
        assert all(m.role == "resident" for m in first_three)


def test_residents_speak_again_in_every_negotiation_round(tmp_path):
    """This is the core new requirement: residents must NOT be silent
    after their initial proposal -- they speak again in every one of
    the negotiation rounds, alongside the 3 functional agents."""
    for topo in ("star", "parallel", "ring"):
        messages, _ = _run_topology(topo, tmp_path, rounds=config.NEGOTIATION_ROUNDS)
        speaker_msgs = _deciding_speaker_messages(messages)

        # 3 initial proposals + 4 rounds x 6 speakers = 27 total
        expected_total = N_RESIDENTS + config.NEGOTIATION_ROUNDS * N_SPEAKERS_PER_ROUND
        assert len(speaker_msgs) == expected_total, f"[{topo}] unexpected total speaker count"

        # Each round must contain every speaker exactly once.
        for round_idx in range(1, config.NEGOTIATION_ROUNDS + 1):
            start = N_RESIDENTS + (round_idx - 1) * N_SPEAKERS_PER_ROUND
            end = start + N_SPEAKERS_PER_ROUND
            round_speakers = [m.speaker for m in speaker_msgs[start:end]]
            assert len(round_speakers) == N_SPEAKERS_PER_ROUND, (
                f"[{topo}] round {round_idx}: expected {N_SPEAKERS_PER_ROUND} "
                f"speakers, got {len(round_speakers)}"
            )
            assert set(round_speakers) == EXPECTED_SPEAKERS, (
                f"[{topo}] round {round_idx} speakers mismatch: {round_speakers}"
            )
            assert len(round_speakers) == len(set(round_speakers)), (
                f"[{topo}] round {round_idx} has a duplicate speaker: {round_speakers}"
            )
            resident_speakers = {s for s in round_speakers if s.startswith("Resident-")}
            assert len(resident_speakers) == N_RESIDENTS, (
                f"[{topo}] round {round_idx}: residents must be present -- "
                f"found {resident_speakers}"
            )


def test_default_negotiation_rounds_is_5(tmp_path):
    """config.NEGOTIATION_ROUNDS default must be 5 (the earlier/simple
    scheme), not the more complex '1 presentation + 3 negotiation'
    framing from a since-abandoned redesign."""
    assert config.NEGOTIATION_ROUNDS == 5


def test_all_3_topologies_produce_identical_speaker_structure(tmp_path):
    """The topology changes HOW the 6 agents route/see each other, not
    who speaks or how many times."""
    counts = {}
    for topo in ("star", "parallel", "ring"):
        messages, _ = _run_topology(topo, tmp_path, rounds=2)
        counts[topo] = len(_deciding_speaker_messages(messages))
    expected = N_RESIDENTS + 2 * N_SPEAKERS_PER_ROUND
    assert counts["star"] == counts["parallel"] == counts["ring"] == expected


def test_resident_allocations_flow_into_current_allocation(tmp_path):
    """A resident's negotiate_round() message must carry an allocations
    dict for their own city, same as the initial propose() call -- this
    is what lets their continued advocacy actually move the numbers."""
    messages, _ = _run_topology("parallel", tmp_path, rounds=1)
    resident_negotiation_msgs = [
        m for m in messages
        if m.role == "resident" and m.speaker.startswith("Resident-")
    ][N_RESIDENTS:]  # skip the initial proposals, keep round-1 negotiation messages
    assert len(resident_negotiation_msgs) == N_RESIDENTS
    for m in resident_negotiation_msgs:
        assert len(m.allocations) >= 1, f"{m.speaker}'s negotiation message had no allocation"


def test_gini_still_computed_for_every_message(tmp_path):
    messages, turn_metrics = _run_topology("star", tmp_path, rounds=1)
    assert len(turn_metrics) == len(messages)
    for tm in turn_metrics:
        assert -1.0 <= tm.relative_gini_score <= 1.0
        assert 0.0 <= tm.raw_gini_score <= 1.0
