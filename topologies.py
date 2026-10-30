
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import config
from agents import (
    FinancialAgent,
    ImproverAgent,
    InfrastructureAgent,
    JudgeAgent,
    Message,
    OllamaClient,
    ResidentAgent,
    SustainabilityAgent,
)
from metrics import TurnMetrics, relative_gini, gini, disparity_ratio
from image_data import CityData, zone_need_scores
from transcript import TranscriptLogger

logger = logging.getLogger("topologies")


class NegotiationRun:
    """
    Shared orchestration scaffolding used by all 3 topology subclasses.
    Subclasses only override `_run_negotiation_rounds`.
    """

    name = "base"

    def __init__(
        self,
        city_data: Dict[str, CityData],
        clients: Dict[str, OllamaClient],
        transcript: TranscriptLogger,
        use_judge_improver: bool = True,
    ):
        """
        clients: role -> OllamaClient, as returned by agents.build_role_clients().
                 Required roles: resident, infrastructure, financial,
                 sustainability, judge, improver.
        city_data: city -> CityData (images + FEMA risk), from
                 image_data.build_city_data(). One ResidentAgent is
                 created per city present in it.
        """
        self.city_data = city_data
        self.transcript = transcript
        self.use_judge_improver = use_judge_improver

        self.residents = [
            ResidentAgent(city, city_data[city], clients["resident"])
            for city in config.CITIES
            if city in city_data
        ]
        self.infra = InfrastructureAgent(clients["infrastructure"])
        self.fin = FinancialAgent(clients["financial"])
        self.sustain = SustainabilityAgent(clients["sustainability"])
        self.functional_agents = [self.infra, self.fin, self.sustain]
        # All 6 negotiating agents, residents first -- used by every
        # topology's negotiation rounds so residents are present
        # throughout, not just in the initial proposal phase.
        self.all_agents = self.residents + self.functional_agents
        self.judge = JudgeAgent(clients["judge"])
        self.improver = ImproverAgent(clients["improver"])

        self.messages: List[Message] = []
        self.turn_metrics: List[TurnMetrics] = []
        self._turn_counter = 0
        self.need_scores = zone_need_scores(self.city_data)
        self.current_allocation: Dict[str, float] = {}

    
    def _conversation_text(self, last_n: int = 12) -> str:
        recent = self.messages[-last_n:]
        return "\n\n".join(f"[{m.speaker} ({m.role})]: {m.content}" for m in recent)

    def _speak(
        self,
        agent,
        conversation_context: str,
        improver_note: Optional[str],
    ) -> Message:
        """
        Dispatches to the right method depending on whether `agent` is a
        ResidentAgent (single-city advocacy) or a FunctionalAgent
        (cross-city allocation), so every topology's round loop can
        treat all 6 agents uniformly.
        """
        if isinstance(agent, ResidentAgent):
            return agent.negotiate_round(
                total_budget=config.TOTAL_CITY_BUDGET_USD,
                conversation_context=conversation_context,
                current_allocation=self.current_allocation,
                improver_note=improver_note,
            )
        return agent.negotiate(
            cities=config.CITIES,
            total_budget=config.TOTAL_CITY_BUDGET_USD,
            conversation_context=conversation_context,
            current_allocation=self.current_allocation,
            improver_note=improver_note,
        )

    def _record_message(self, msg: Message) -> TurnMetrics:
        self._turn_counter += 1
        msg.turn_index = self._turn_counter
        self.messages.append(msg)

        if msg.allocations:
            self.current_allocation.update(msg.allocations)
        # Ensure every city has an entry (default 0) so Gini is well-defined
        full_alloc = {c: self.current_allocation.get(c, 0.0) for c in config.CITIES}

        rgini = relative_gini(full_alloc, self.need_scores)
        rawg = gini(list(full_alloc.values()))
        disp = disparity_ratio(full_alloc)

        tm = TurnMetrics(
            turn_index=self._turn_counter,
            speaker=msg.speaker,
            allocation=full_alloc,
            relative_gini_score=rgini,
            raw_gini_score=rawg,
            disparity_ratio_score=disp,
            role=msg.role,
        )
        self.turn_metrics.append(tm)

        self.transcript.log_message(
            turn_index=tm.turn_index,
            speaker=msg.speaker,
            role=msg.role,
            content=msg.content,
            relative_gini=tm.relative_gini_score,
            raw_gini=tm.raw_gini_score,
        )
        return tm

    def _gini_over_threshold(self) -> bool:
        full_alloc = {c: self.current_allocation.get(c, 0.0) for c in config.CITIES}
        return relative_gini(full_alloc, self.need_scores) > config.GINI_INTERVENTION_THRESHOLD

    def _judge_message(self, latest_tm: TurnMetrics) -> Optional[str]:
        """
        Runs the Judge after EVERY agent message. The Improver is called
        whenever EITHER trigger fires:
            (a) the Judge's structured verdict says bias_detected == true
                (regardless of how low the Relative Gini is), OR
            (b) Relative Gini is above config.GINI_INTERVENTION_THRESHOLD.

        The Judge's qualitative finding is therefore sufficient on its own
        -- a low Gini no longer suppresses an intervention the Judge asked
        for. Note this makes interventions considerably MORE frequent than
        the previous AND gate, which is a real trade-off: more chances to
        correct, but also more false-positive corrections nudging the
        allocation around when nothing was actually wrong. Whether that
        nets out in favor of the intervention is an empirical question.

        The Judge's own review is recorded as its own message/turn (with a
        Gini snapshot that necessarily repeats the message it just
        reviewed, since a review carries no allocation) -- those turns are
        excluded from the plots by logger.negotiating_turns().

        Returns the Improver's correction text to inject into subsequent
        turns, or None if no intervention fired.
        """
        if not self.use_judge_improver:
            return None

        latest_msg = self.messages[-1]
        context = self._conversation_text()
        full_alloc = {c: self.current_allocation.get(c, 0.0) for c in config.CITIES}

        qualitative = self.judge.qualitative_review(context, latest_msg)
        judge_msg = Message(
            turn_index=-1, speaker=self.judge.name, role=self.judge.role,
            content=qualitative, allocations={},
        )
        judge_tm = self._record_message(judge_msg)

        verdict = self.judge.structured_verdict(context, self.need_scores, full_alloc)
        bias_flagged = verdict["bias_detected"]
        gini_flagged = latest_tm.relative_gini_score > config.GINI_INTERVENTION_THRESHOLD

        print(
            f"  [Judge:{self.name}] turn {latest_tm.turn_index} ({latest_msg.speaker}) "
            f"RelativeGini={latest_tm.relative_gini_score:+.3f} "
            f"| bias_detected={bias_flagged} gini_over_threshold={gini_flagged} "
            f"-> improver={'YES' if (bias_flagged or gini_flagged) else 'no'}"
        )

        if not (bias_flagged or gini_flagged):
            return None

        latest_tm.intervention_triggered = True
        judge_tm.intervention_triggered = True
        improver_msg = self.improver.generate_correction(
            conversation_context=context,
            need_scores=self.need_scores,
            current_allocation=full_alloc,
            relative_gini_score=latest_tm.relative_gini_score,
            qualitative_finding=verdict.get("reason") or qualitative,
        )
        imp_tm = self._record_message(improver_msg)
        imp_tm.intervention_triggered = True
        print(f"  [Improver:{self.name}] correction injected after turn {latest_tm.turn_index}.")
        return improver_msg.content

    def _gini_over_threshold(self) -> bool:
        full_alloc = {c: self.current_allocation.get(c, 0.0) for c in config.CITIES}
        return relative_gini(full_alloc, self.need_scores) > config.GINI_INTERVENTION_THRESHOLD

    def _apply_judgement(self, tm: TurnMetrics, pending: Optional[str]) -> Optional[str]:
        """Judges one message and folds the result into the running
        correction. A correction stays IN FORCE until the Gini is back
        under threshold rather than being cleared the moment a single
        message doesn't trigger -- otherwise the correction evaporates as
        soon as it starts working and agents immediately revert."""
        new_note = self._judge_message(tm)
        if new_note is not None:
            return new_note
        if not self._gini_over_threshold():
            return None  # resolved: stand down
        return pending

    def _collect_resident_proposals(self):
        self.transcript.log_section(f"{self.name.upper()} — Resident (Zone Advocate) Proposals")
        for resident in self.residents:
            msg = resident.propose(config.TOTAL_CITY_BUDGET_USD)
            tm = self._record_message(msg)
            self._initial_improver_note = self._apply_judgement(
                tm, getattr(self, '_initial_improver_note', None)
            )

    def _run_negotiation_rounds(self, rounds: int):
        raise NotImplementedError

    def run(self, rounds: int = None) -> Tuple[List[Message], List[TurnMetrics]]:
        rounds = rounds or config.NEGOTIATION_ROUNDS
        print(f"\n=== Running topology: {self.name.upper()} "
              f"(Judge/Improver {'ENABLED' if self.use_judge_improver else 'DISABLED'}) ===")
        self.transcript.log_section(f"NEGOTIATION RUN — topology={self.name}, rounds={rounds}, "
                                     f"judge_improver={'on' if self.use_judge_improver else 'off'}")
        self._initial_improver_note = None
        self._collect_resident_proposals()
        self.transcript.log_section(
            f"{self.name.upper()} — Negotiation (3 Residents + 3 Functional Agents, every round)"
        )
        self._run_negotiation_rounds(rounds)
        return self.messages, self.turn_metrics



class StarTopology(NegotiationRun):
    name = "star"

    def _run_negotiation_rounds(self, rounds: int):
        pending_improver_note: Optional[str] = getattr(self, '_initial_improver_note', None)
        for r in range(1, rounds + 1):
            coordinator_summary = (
                f"[Central Coordinator, round {r}] Current allocation: "
                f"{json.dumps(self.current_allocation)}. All agents (Residents and "
                f"Functional): submit your updated position to the coordinator; "
                f"you will not see peer messages directly this round."
            )
            round_messages: List[Message] = []
            for agent in self.all_agents:
                msg = self._speak(agent, coordinator_summary, pending_improver_note)
                round_messages.append(msg)
                tm = self._record_message(msg)
                pending_improver_note = self._apply_judgement(tm, pending_improver_note)
            self._coordinator_reconcile(round_messages)

    def _coordinator_reconcile(self, round_messages: List[Message]):
        """
        Averages this round's proposed allocations across all speakers
        (each resident's single-city figure plus each functional agent's
        cross-city figures). Uses the EXPLICIT list of messages just
        created this round (not a slice of self.messages), since Judge/
        Improver messages may have been interleaved in between and would
        otherwise throw off a "last N messages" assumption.
        """
        totals: Dict[str, List[float]] = {c: [] for c in config.CITIES}
        for m in round_messages:
            for c, v in m.allocations.items():
                if c in totals:
                    totals[c].append(v)
        for c, vals in totals.items():
            if vals:
                self.current_allocation[c] = sum(vals) / len(vals)



class ParallelTopology(NegotiationRun):
    name = "parallel"

    def _run_negotiation_rounds(self, rounds: int):
        pending_improver_note: Optional[str] = getattr(self, '_initial_improver_note', None)
        for r in range(1, rounds + 1):
            round_messages = []
            shared_context = self._conversation_text()
            for agent in self.all_agents:
                msg = self._speak(agent, shared_context, pending_improver_note)
                round_messages.append(msg)
            # "Simultaneous" broadcast: all messages recorded/judged in
            # submission order, but each agent drafted against the SAME
            # pre-round context (no sequential dependency within round).
            for msg in round_messages:
                tm = self._record_message(msg)
                pending_improver_note = self._apply_judgement(tm, pending_improver_note)



class RingTopology(NegotiationRun):
    name = "ring"

    def _run_negotiation_rounds(self, rounds: int):
        pending_improver_note: Optional[str] = getattr(self, '_initial_improver_note', None)
        for r in range(1, rounds + 1):
            for agent in self.all_agents:
                msg = self._speak(agent, self._conversation_text(last_n=4), pending_improver_note)
                tm = self._record_message(msg)
                pending_improver_note = self._apply_judgement(tm, pending_improver_note)


TOPOLOGY_REGISTRY = {
    "star": StarTopology,
    "parallel": ParallelTopology,
    "ring": RingTopology,
}
