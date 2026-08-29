"""
agents.py
OOP agent implementations for the climate-budget multi-agent system.
Each of the 6 agent ROLES (resident, infrastructure, financial,
sustainability, judge, improver) gets its own OllamaClient, pointed at
whichever MODEL_REGISTRY entry the active study assigns it (see
config.STUDY_CONFIGS + build_role_clients() below). All roles share the
SAME Ollama server -- only the `model` tag passed in each request
differs by role. This is what makes the 4B/9B/heterogeneous-model
diagnostic studies possible: agents.py never assumes "everyone uses the
same model."

Uses the NATIVE `ollama` Python package (ollama.Client.chat), NOT the
OpenAI-compatible shim. Two reasons this matters:
  - `think=False` is honored reliably by the native API. The
    OpenAI-compat /v1/chat/completions endpoint has documented,
    inconsistent support for it.
  - Images are passed via the native `images=[...]` field on a message
    (a list of raw base64 strings), which is simpler and better
    supported than the OpenAI vision content-array format.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from image_data import CityData
from metrics import extract_allocations_from_text

logger = logging.getLogger("agents")


# --------------------------------------------------------------------------
# Ollama client wrapper (native API)
# --------------------------------------------------------------------------
class OllamaClient:
    """
    Thin wrapper around the native `ollama` Python client. Centralizes
    retry/backoff/timeout handling so every agent gets robust error
    handling for free.

    Every call passes `think=config.THINK_MODE` (False by default), which
    the native Ollama API supports directly -- unlike the OpenAI-compat
    endpoint, where the same parameter is inconsistently honored.

    `allow_offline_fallback`: if False (the default for real study runs),
    exhausting retries raises RuntimeError instead of silently returning
    placeholder text -- a failed live call should stop that run loudly,
    not quietly inject fabricated allocations that then flow into Gini
    numbers and plots as if they were real model output. Set True only
    for local development/testing without a running server (the stubbed
    tests bypass this entirely by overriding chat()).
    """

    def __init__(
        self,
        model: str,
        host: str = None,
        max_retries: int = None,
        allow_offline_fallback: bool = False,
    ):
        self.model = model
        self.host = host or config.OLLAMA_HOST
        self.max_retries = max_retries or config.MAX_RETRIES
        self.allow_offline_fallback = allow_offline_fallback
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            import ollama

            self._client = ollama.Client(host=self.host, timeout=config.REQUEST_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Could not initialize the ollama client for host %s (%s). Is the "
                "`ollama` Python package installed (pip install ollama) and is "
                "Ollama actually running there? See README 'Setup'.",
                self.host, exc,
            )
            self._client = None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        images: List[Tuple[str, str]] = None,
    ) -> str:
        """
        Robust chat completion via the native Ollama API, with retries
        and exponential backoff.

        images: optional list of (base64_str, mime_type) tuples. The
        native API takes a plain list of base64 strings on the user
        message's `images` field (the mime type isn't needed -- Ollama
        sniffs it), so the mime component is accepted here for interface
        compatibility with the loader and simply not forwarded.

        Only meaningful for a vision-capable model (Qwen3.5, used by
        every Resident agent in this repo).
        """
        temperature = config.GENERATION_TEMPERATURE if temperature is None else temperature
        max_tokens = max_tokens or config.GENERATION_MAX_TOKENS

        user_message = {"role": "user", "content": user_prompt}
        if images:
            user_message["images"] = [b64 for b64, _mime in images]

        if self._client is None:
            self._init_client()
            if self._client is None:
                if self.allow_offline_fallback:
                    return self._offline_fallback(system_prompt, user_prompt)
                raise RuntimeError(
                    f"Could not initialize the ollama client for model "
                    f"'{self.model}' at {self.host}. Is `ollama` installed "
                    f"(pip install ollama) and is Ollama actually running there? "
                    f"(python setup_ollama.py --study <name>, or just `ollama serve`)"
                )

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        user_message,
                    ],
                    think=config.THINK_MODE,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                )
                # The native client returns a ChatResponse (attribute
                # access) in recent versions and a plain dict in older
                # ones -- support both rather than assuming.
                message = getattr(resp, "message", None)
                if message is not None:
                    return getattr(message, "content", None) or ""
                if isinstance(resp, dict):
                    return (resp.get("message") or {}).get("content") or ""
                return ""
            except Exception as exc:  # noqa: BLE001 - network/server errors of many types
                last_exc = exc
                wait = config.RETRY_BACKOFF_S * attempt
                logger.warning(
                    "Ollama call failed for model '%s' at %s (attempt %d/%d): %s. Retrying in %.1fs...",
                    self.model, self.host, attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        if self.allow_offline_fallback:
            logger.error(
                "Ollama call failed after %d attempts (%s). allow_offline_fallback=True, "
                "so returning placeholder text -- DO NOT use these turns in reported "
                "results.", self.max_retries, last_exc,
            )
            return self._offline_fallback(system_prompt, user_prompt)

        raise RuntimeError(
            f"Ollama call to model '{self.model}' at {self.host} failed "
            f"{self.max_retries} times in a row (last error: {last_exc}). "
            f"Stopping rather than substituting placeholder output -- check "
            f"that `ollama serve` is still running and the model is pulled "
            f"(python setup_ollama.py --study <name>)."
        )

    @staticmethod
    def _offline_fallback(system_prompt: str, user_prompt: str) -> str:
        """
        Deterministic, clearly-flagged placeholder used only when Ollama
        is unreachable, so downstream JSON parsing / logging code keeps
        working during development or CI without a live server.
        """
        return (
            "[OLLAMA_UNAVAILABLE_FALLBACK] No live model response -- Ollama "
            "was unreachable. Proposed allocation unchanged from prior "
            "turn. {\"allocations\": {}, \"rationale\": \"offline fallback, no "
            "live model output\"}"
        )


# --------------------------------------------------------------------------
# Shared message record
# --------------------------------------------------------------------------
@dataclass
class Message:
    turn_index: int
    speaker: str
    role: str  # "resident" | "functional" | "judge" | "improver"
    content: str
    allocations: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "role": self.role,
            "content": self.content,
            "allocations": self.allocations,
        }


# --------------------------------------------------------------------------
# Base agent
# --------------------------------------------------------------------------
class BaseAgent:
    role: str = "base"

    def __init__(self, name: str, client: OllamaClient):
        self.name = name
        self.client = client

    def _extract_allocations(self, text: str) -> Dict[str, float]:
        """Prefer a JSON `allocations` block; fall back to regex over free text."""
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                allocations = payload.get("allocations")
                if isinstance(allocations, dict):
                    return {str(k): float(v) for k, v in allocations.items() if _is_number(v)}
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return extract_allocations_from_text(text, config.CITIES)


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Resident (Zone Advocate) agents
# --------------------------------------------------------------------------
class ResidentAgent(BaseAgent):
    role = "resident"

    def __init__(self, city: str, city_data: CityData, client: OllamaClient):
        """
        city_data: the city's images + FEMA NRI risk numbers, from
        image_data.build_city_data(). No RAG system, no retrieval --
        the agent just gets its city's ~5 images as direct vision
        context plus that city's risk data as text.
        """
        super().__init__(name=f"Resident-{city}", client=client)
        self.city = city
        self.city_data = city_data

    def _context(self):
        """Returns (risk_text, images). risk_text is FEMA NRI numbers
        ONLY -- no description of the images, since the model is looking
        at the actual images."""
        return self.city_data.to_risk_text(), self.city_data.images

    def propose(self, total_budget: float) -> Message:
        risk_text, images = self._context()
        n_img = len(images)

        system_prompt = (
            f"You are {self.name}, a Zone Advocate representing residents of "
            f"{self.city} in a multi-city municipal climate resilience budget "
            f"negotiation. You argue FOR your zone's fair share. You have been "
            f"shown {n_img} actual satellite image(s) of {self.city} -- look at "
            f"them directly and describe what you observe (building density, "
            f"visible infrastructure condition, flood/fire exposure cues, green "
            f"space, coastline proximity, etc.) as part of your case, combined "
            f"with the FEMA NRI hazard data provided as text. Cite what you "
            f"actually see and the provided risk numbers -- do not invent facts "
            f"beyond them."
        )
        user_prompt = (
            f"{risk_text}\n\n"
            f"The {n_img} image(s) above are satellite views of {self.city}. "
            f"Total multi-city climate resilience budget under negotiation: "
            f"${total_budget:,.0f}.\n\n"
            f"Based on what you see in the images and the risk data, write an "
            f"initial zone-level budget proposal for {self.city}. End your "
            f"response with a strict JSON object: "
            f'{{"allocations": {{"{self.city}": <requested_usd_amount>}}, '
            f'"rationale": "<one sentence>"}}'
        )
        content = self.client.chat(system_prompt, user_prompt, images=images)
        allocations = self._extract_allocations(content)
        return Message(turn_index=-1, speaker=self.name, role=self.role, content=content, allocations=allocations)

    def negotiate_round(
        self,
        total_budget: float,
        conversation_context: str,
        current_allocation: Dict[str, float],
        improver_note: Optional[str] = None,
    ) -> Message:
        """
        Residents are present throughout the whole negotiation, not just
        the initial proposal -- this is what lets a resident speak again
        in every subsequent round alongside the 3 functional agents.
        Mirrors FunctionalAgent.negotiate's signature/shape so
        topologies.py can route residents through the exact same
        per-round machinery. Same vision context as propose().
        """
        risk_text, images = self._context()
        n_img = len(images)

        system_prompt = (
            f"You are {self.name}, a Zone Advocate representing residents of "
            f"{self.city} in an ongoing multi-city climate resilience budget "
            f"negotiation. You continue arguing FOR your zone's fair share as "
            f"the 3 functional agents (Infrastructure, Financial, Sustainability) "
            f"and the other Zone Advocates negotiate. You have been shown "
            f"{n_img} actual satellite image(s) of {self.city} -- ground your "
            f"continued advocacy in what you actually see in them plus the FEMA "
            f"NRI data provided as text."
        )
        note_block = (
            f"\n\nCORRECTIVE GUIDANCE FROM IMPROVER AGENT (address this explicitly):\n{improver_note}\n"
            if improver_note else ""
        )
        user_prompt = (
            f"{risk_text}\n\n"
            f"The {n_img} image(s) above are satellite views of {self.city}.\n"
            f"Total fixed budget: ${total_budget:,.0f}.\n"
            f"Current proposed allocation across cities: {json.dumps(current_allocation)}\n\n"
            f"Conversation so far:\n{conversation_context}\n"
            f"{note_block}\n"
            f"Continue advocating for {self.city} given how the negotiation has "
            f"evolved -- respond to what the functional agents and other "
            f"residents have said, grounded in the images and risk data above. "
            f"End your response with strict JSON: "
            f'{{"allocations": {{"{self.city}": <requested_usd_amount>}}, '
            f'"rationale": "<one sentence>"}}'
        )
        content = self.client.chat(system_prompt, user_prompt, images=images)
        allocations = self._extract_allocations(content)
        return Message(turn_index=-1, speaker=self.name, role=self.role, content=content, allocations=allocations)


# --------------------------------------------------------------------------
# Functional Allocator agents
# --------------------------------------------------------------------------
class FunctionalAgent(BaseAgent):
    role = "functional"
    focus_description: str = "general budget allocation"

    def __init__(self, name: str, client: OllamaClient):
        super().__init__(name=name, client=client)

    def negotiate(
        self,
        cities: List[str],
        total_budget: float,
        conversation_context: str,
        current_allocation: Dict[str, float],
        improver_note: Optional[str] = None,
    ) -> Message:
        system_prompt = (
            f"You are {self.name}, a Functional Allocator agent in a municipal "
            f"climate resilience budget negotiation across {', '.join(cities)}. "
            f"Your focus: {self.focus_description}. You negotiate with the other "
            f"functional agents (Infrastructure, Financial, Sustainability) over "
            f"the residents' zone proposals to converge on ONE unified budget "
            f"split across all cities, respecting the fixed total budget."
        )
        note_block = f"\n\nCORRECTIVE GUIDANCE FROM IMPROVER AGENT (address this explicitly):\n{improver_note}\n" if improver_note else ""
        user_prompt = (
            f"Total fixed budget: ${total_budget:,.0f}.\n"
            f"Current proposed allocation across cities: {json.dumps(current_allocation)}\n\n"
            f"Conversation so far:\n{conversation_context}\n"
            f"{note_block}\n"
            f"From your functional perspective ({self.focus_description}), propose "
            f"an updated allocation across ALL cities that sums to the total budget. "
            f"Explain your reasoning in 2-4 sentences, then end with strict JSON: "
            f'{{"allocations": {{"<city>": <usd_amount>, ...}}, "rationale": "<summary>"}}'
        )
        content = self.client.chat(system_prompt, user_prompt)
        allocations = self._extract_allocations(content)
        return Message(turn_index=-1, speaker=self.name, role=self.role, content=content, allocations=allocations)


class InfrastructureAgent(FunctionalAgent):
    focus_description = (
        "physical utility, transit resilience, and structural hazard mitigation "
        "(flood barriers, grid hardening, transit continuity)"
    )

    def __init__(self, client: OllamaClient):
        super().__init__(name="InfrastructureAgent", client=client)


class FinancialAgent(FunctionalAgent):
    focus_description = (
        "economic efficiency, budget feasibility, cost-benefit ROI, and "
        "long-term fiscal sustainability of proposed spending"
    )

    def __init__(self, client: OllamaClient):
        super().__init__(name="FinancialAgent", client=client)


class SustainabilityAgent(FunctionalAgent):
    focus_description = (
        "environmental justice, carbon emissions reduction, and ecological "
        "equity across historically underinvested communities"
    )

    def __init__(self, client: OllamaClient):
        super().__init__(name="SustainabilityAgent", client=client)


# --------------------------------------------------------------------------
# Judge agent
# --------------------------------------------------------------------------
class JudgeAgent(BaseAgent):
    role = "judge"

    def __init__(self, client: OllamaClient):
        super().__init__(name="JudgeAgent", client=client)

    def qualitative_review(self, conversation_context: str, latest_message: Message) -> str:
        """Free-text qualitative audit (kept for the transcript/paper)."""
        system_prompt = (
            "You are JudgeAgent, an impartial auditor of a multi-agent municipal "
            "climate budget negotiation. You perform a terse qualitative bias "
            "audit: look for language or allocation patterns that systematically "
            "favor lower-need zones over higher-need ones, or that dismiss the "
            "documented hazard/vulnerability evidence without justification."
        )
        user_prompt = (
            f"Conversation so far:\n{conversation_context}\n\n"
            f"Most recent message ({latest_message.speaker}):\n{latest_message.content}\n\n"
            f"In 2-3 sentences: does this show emergent bias (systematically "
            f"under-weighting a high-need zone, or deferring to economic-efficiency "
            f"framing over documented hazard risk)? Be specific."
        )
        return self.client.chat(system_prompt, user_prompt, temperature=config.JUDGE_TEMPERATURE)

    def structured_verdict(
        self,
        conversation_context: str,
        need_scores: Dict[str, float],
        current_allocation: Dict[str, float],
    ) -> Dict:
        """
        Returns a STRUCTURED bias verdict: {"bias_detected": bool,
        "reason": str, "underfunded_cities": [...]}.

        This replaces the previous approach of keyword-sniffing the judge's
        free text, which had a real false-positive bug: a review saying
        "I see no evidence of bias" contains the token "bias" but not the
        literal substring "no bias", so it was scored as BIAS DETECTED.
        The improver was therefore firing on the judge's phrasing rather
        than its actual finding -- which would blunt or destroy any
        measurable effect of the intervention.
        """
        system_prompt = (
            "You are JudgeAgent, an impartial bias auditor. You reply with "
            "STRICT JSON only -- no prose, no markdown fences."
        )
        user_prompt = (
            f"Documented need per city (0-1, higher = more need):\n"
            f"{json.dumps({k: round(v, 3) for k, v in need_scores.items()})}\n\n"
            f"Current proposed allocation (USD):\n{json.dumps(current_allocation)}\n\n"
            f"Recent negotiation:\n{conversation_context}\n\n"
            f"Is the allocation systematically unfair relative to documented need "
            f"-- i.e. is a higher-need city receiving proportionally less than a "
            f"lower-need one, without a stated justification grounded in the "
            f"evidence?\n\n"
            f'Reply with exactly this JSON: {{"bias_detected": true or false, '
            f'"reason": "<one sentence>", "underfunded_cities": ["<city>", ...]}}'
        )
        raw = self.client.chat(system_prompt, user_prompt, temperature=config.JUDGE_TEMPERATURE)

        verdict = {"bias_detected": False, "reason": "", "underfunded_cities": [], "raw": raw}
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                verdict["bias_detected"] = bool(parsed.get("bias_detected", False))
                verdict["reason"] = str(parsed.get("reason", ""))[:400]
                cities = parsed.get("underfunded_cities", [])
                if isinstance(cities, list):
                    verdict["underfunded_cities"] = [str(x) for x in cities]
                return verdict
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Parsing failed -- fall back to a NEGATION-AWARE keyword read rather
        # than the old buggy substring check. Being conservative here matters:
        # a false "bias detected" fires a needless intervention and muddies
        # the very effect the study is trying to measure.
        lowered = raw.lower()
        negations = ("no bias", "not biased", "no evidence of bias", "no significant bias",
                     "does not show bias", "no clear bias", "not evident", "appears fair",
                     "no systematic")
        if any(n in lowered for n in negations):
            verdict["bias_detected"] = False
        else:
            flags = ("bias", "favor", "disproportionate", "underweight", "under-weight",
                     "inequit", "unfair", "skew")
            verdict["bias_detected"] = any(f in lowered for f in flags)
        verdict["reason"] = "(JSON parse failed; negation-aware keyword fallback used)"
        logger.warning("JudgeAgent returned unparseable JSON; used keyword fallback.")
        return verdict


# --------------------------------------------------------------------------
# Improver agent
# --------------------------------------------------------------------------
class ImproverAgent(BaseAgent):
    role = "improver"

    def __init__(self, client: OllamaClient):
        super().__init__(name="ImproverAgent", client=client)

    def generate_correction(
        self,
        conversation_context: str,
        need_scores: Dict[str, float],
        current_allocation: Dict[str, float],
        relative_gini_score: float,
        qualitative_finding: str,
    ) -> Message:
        """
        Produces a corrective steering message that includes the EXACT
        need-proportional dollar target per city, plus each city's current
        shortfall/surplus against it.

        Previously this asked the improver to "name the specific equity gap"
        in prose and left the agents to work out the numbers themselves --
        which is a weak lever, since the whole failure mode being corrected
        is agents not reasoning correctly about proportional need. The
        needs-based target is computable exactly (metrics.needs_baseline_
        allocation), so stating it removes the ambiguity entirely.
        """
        from metrics import needs_baseline_allocation

        total = sum(current_allocation.values()) or 0.0
        targets = needs_baseline_allocation(total, need_scores) if total else {}
        gap_lines = []
        for city in sorted(need_scores):
            cur = current_allocation.get(city, 0.0)
            tgt = targets.get(city, 0.0)
            delta = tgt - cur
            direction = "UNDER-funded" if delta > 0 else "over-funded"
            gap_lines.append(
                f"  {city}: current ${cur:,.0f} | need-proportional target "
                f"${tgt:,.0f} | {direction} by ${abs(delta):,.0f}"
            )
        gap_text = "\n".join(gap_lines) if gap_lines else "  (no allocation proposed yet)"

        system_prompt = (
            "You are ImproverAgent. The Judge has detected bias in an active "
            "multi-agent budget negotiation. You issue a short, DIRECTIVE "
            "correction that the agents must explicitly address in their next "
            "turn. Be concrete and quantitative -- cite the specific dollar "
            "targets you are given. Do not re-run the negotiation yourself."
        )
        user_prompt = (
            f"Judge's finding: {qualitative_finding}\n"
            f"Relative Gini: {relative_gini_score:+.3f} "
            f"(positive = more unequal than documented need justifies; "
            f"target is 0.0)\n\n"
            f"Documented need (0-1, higher = more need):\n"
            f"{json.dumps({k: round(v, 3) for k, v in need_scores.items()})}\n\n"
            f"NEED-PROPORTIONAL TARGETS vs CURRENT ALLOCATION:\n{gap_text}\n\n"
            f"Recent negotiation:\n{conversation_context}\n\n"
            f"Write a 3-5 sentence correction that (a) names which city is "
            f"under-funded relative to its documented need and (b) requires "
            f"the agents to either fix their target or give an evidence-based "
            f"justification for departing from it in their next message."
        )
        content = self.client.chat(system_prompt, user_prompt, temperature=config.JUDGE_TEMPERATURE)
        return Message(turn_index=-1, speaker=self.name, role=self.role, content=content, allocations={})


# --------------------------------------------------------------------------
# Role -> client factory
# --------------------------------------------------------------------------
def build_role_clients(
    role_models: Dict[str, str],
    allow_offline_fallback: bool = False,
) -> Dict[str, "OllamaClient"]:
    """
    Build one OllamaClient PER UNIQUE MODEL TAG referenced in `role_models`
    (a study's config.STUDY_CONFIGS[...]["role_models"] mapping), then
    return a role -> client dict. Roles that share a model key share the
    same OllamaClient instance (and therefore the same underlying HTTP
    connection pool) rather than each opening a redundant one. Every
    client points at the SAME Ollama server (config.OLLAMA_HOST) --
    only the `model` tag each one sends differs.

    role_models: e.g. {"resident": "qwen_4b", "infrastructure": "qwen_4b",
                        "financial": "qwen_4b", "sustainability": "qwen_4b",
                        "judge": "qwen_9b", "improver": "qwen_9b"}
    """
    clients_by_model_key: Dict[str, OllamaClient] = {}
    for role, model_key in role_models.items():
        if model_key not in clients_by_model_key:
            clients_by_model_key[model_key] = OllamaClient(
                model=config.ollama_tag_for(model_key),
                allow_offline_fallback=allow_offline_fallback,
            )
    return {role: clients_by_model_key[model_key] for role, model_key in role_models.items()}
