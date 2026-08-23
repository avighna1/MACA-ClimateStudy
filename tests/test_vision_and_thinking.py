"""
Tests for:
  1. OllamaClient using the NATIVE ollama client and passing
     `think=config.THINK_MODE` (False by default) on every call --
     the native API honors this reliably, unlike the OpenAI-compat
     endpoint this repo used previously.
  2. Images being passed via the native `images=[...]` field on the user
     message (a plain list of base64 strings).
  3. ResidentAgent sending its city's actual images plus FEMA NRI
     risk-only text -- with no textual description of the images.

The real `ollama` package isn't installed in this environment, so a fake
module is injected into sys.modules mimicking `ollama.Client` just
enough to capture the exact kwargs passed to `.chat(...)`.
"""

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


class _FakeChatResponse:
    """Mimics the modern ollama ChatResponse (attribute access)."""
    class _Msg:
        content = '{"allocations": {"New York": 1000000}, "rationale": "stub"}'
    message = _Msg()


class _FakeOllamaClient:
    def __init__(self, capture, host=None, timeout=None):
        self.capture = capture
        self.host = host
        self.timeout = timeout

    def chat(self, **kwargs):
        self.capture.append(kwargs)
        return _FakeChatResponse()


def _install_fake_ollama(capture: list):
    fake = types.ModuleType("ollama")
    fake.Client = lambda host=None, timeout=None: _FakeOllamaClient(capture, host, timeout)
    sys.modules["ollama"] = fake


def _restore(original):
    if original is None:
        sys.modules.pop("ollama", None)
    else:
        sys.modules["ollama"] = original


def _fresh_agents():
    import importlib
    import agents
    importlib.reload(agents)
    return agents


def test_uses_native_client_not_openai_shim():
    """agents.py must not import `openai` anymore -- the native ollama
    package is what reliably honors think=False."""
    src_path = os.path.join(os.path.dirname(__file__), "..", "agents.py")
    with open(src_path) as f:
        src = f.read()
    assert "from openai import" not in src
    assert "import openai" not in src
    assert "extra_body" not in src   # the old OpenAI-compat think workaround


def test_every_call_passes_think_false_by_default():
    original = sys.modules.get("ollama")
    original_think = config.THINK_MODE
    capture = []
    try:
        _install_fake_ollama(capture)
        agents = _fresh_agents()
        config.THINK_MODE = False

        client = agents.OllamaClient(model="qwen3.5:4b")
        client.chat("system", "user")

        assert capture[0]["think"] is False
    finally:
        _restore(original)
        config.THINK_MODE = original_think


def test_think_true_is_respected_if_configured():
    original = sys.modules.get("ollama")
    original_think = config.THINK_MODE
    capture = []
    try:
        _install_fake_ollama(capture)
        agents = _fresh_agents()
        config.THINK_MODE = True

        client = agents.OllamaClient(model="qwen3.5:4b")
        client.chat("system", "user")

        assert capture[0]["think"] is True
    finally:
        _restore(original)
        config.THINK_MODE = original_think


def test_chat_without_images_omits_images_field():
    original = sys.modules.get("ollama")
    capture = []
    try:
        _install_fake_ollama(capture)
        agents = _fresh_agents()

        client = agents.OllamaClient(model="qwen3.5:4b")
        client.chat("system prompt", "user prompt")

        messages = capture[0]["messages"]
        assert messages[1]["content"] == "user prompt"
        assert "images" not in messages[1]
    finally:
        _restore(original)


def test_chat_with_images_uses_native_images_field():
    """Native format: a plain list of base64 strings on the user message,
    NOT the OpenAI content-array format."""
    original = sys.modules.get("ollama")
    capture = []
    try:
        _install_fake_ollama(capture)
        agents = _fresh_agents()

        client = agents.OllamaClient(model="qwen3.5:4b")
        client.chat(
            "system", "look at these",
            images=[("Zmlyc3Q=", "image/png"), ("c2Vjb25k", "image/jpeg")],
        )

        messages = capture[0]["messages"]
        assert messages[1]["content"] == "look at these"   # content stays plain text
        assert messages[1]["images"] == ["Zmlyc3Q=", "c2Vjb25k"]
    finally:
        _restore(original)


def test_temperature_and_max_tokens_go_in_options():
    original = sys.modules.get("ollama")
    capture = []
    try:
        _install_fake_ollama(capture)
        agents = _fresh_agents()

        client = agents.OllamaClient(model="qwen3.5:4b")
        client.chat("system", "user", temperature=0.25, max_tokens=123)

        opts = capture[0]["options"]
        assert opts["temperature"] == 0.25
        assert opts["num_predict"] == 123
    finally:
        _restore(original)


def test_resident_sends_images_and_risk_text_only():
    """The core requirement: actual images + FEMA risk numbers as text,
    and NO textual description of what's in the images."""
    original = sys.modules.get("ollama")
    capture = []
    try:
        _install_fake_ollama(capture)
        agents = _fresh_agents()
        from image_data import CityData

        cd = CityData(
            city="New York",
            images=[("aW1nMQ==", "image/png"), ("aW1nMg==", "image/png")],
            hazard_risk_score=72.5,
            social_vulnerability_index=0.61,
            dominant_hazard="Coastal Flooding",
            n_counties_matched=5,
        )

        client = agents.OllamaClient(model="qwen3.5:4b")
        resident = agents.ResidentAgent("New York", cd, client)
        resident.propose(total_budget=100_000_000)

        messages = capture[0]["messages"]
        assert messages[1]["images"] == ["aW1nMQ==", "aW1nMg=="]

        full_text = messages[0]["content"] + " " + messages[1]["content"]
        # FEMA risk numbers must be present as text
        assert "72.5" in full_text
        assert "Coastal Flooding" in full_text
        # No canned description of the images
        for banned in ("visual survey", "the image shows", "depicts a", "photo shows"):
            assert banned.lower() not in full_text.lower()
    finally:
        _restore(original)


def test_resident_with_no_images_still_works():
    original = sys.modules.get("ollama")
    capture = []
    try:
        _install_fake_ollama(capture)
        agents = _fresh_agents()
        from image_data import CityData

        cd = CityData(city="San Francisco", images=[], hazard_risk_score=88.0,
                       social_vulnerability_index=0.7, dominant_hazard="Wildfire")
        client = agents.OllamaClient(model="qwen3.5:4b")
        resident = agents.ResidentAgent("San Francisco", cd, client)
        resident.propose(total_budget=100_000_000)

        messages = capture[0]["messages"]
        assert "images" not in messages[1]      # no images -> field omitted
        assert "88.0" in messages[1]["content"]  # risk text still present
    finally:
        _restore(original)


def test_resident_negotiate_round_also_sends_images():
    original = sys.modules.get("ollama")
    capture = []
    try:
        _install_fake_ollama(capture)
        agents = _fresh_agents()
        from image_data import CityData

        cd = CityData(city="New York", images=[("aW1n", "image/png")],
                       hazard_risk_score=70.0, social_vulnerability_index=0.5,
                       dominant_hazard="Flooding")
        client = agents.OllamaClient(model="qwen3.5:4b")
        resident = agents.ResidentAgent("New York", cd, client)
        resident.negotiate_round(
            total_budget=100_000_000,
            conversation_context="prior turns here",
            current_allocation={"New York": 50_000_000},
        )

        messages = capture[0]["messages"]
        assert messages[1]["images"] == ["aW1n"]
        assert capture[0]["think"] is False
    finally:
        _restore(original)


def test_dict_style_response_is_supported():
    """Older ollama versions return a plain dict rather than a
    ChatResponse object -- both must work."""
    original = sys.modules.get("ollama")
    capture = []
    try:
        fake = types.ModuleType("ollama")

        class _DictClient:
            def __init__(self, host=None, timeout=None):
                pass

            def chat(self, **kwargs):
                capture.append(kwargs)
                return {"message": {"content": "dict-style response"}}

        fake.Client = lambda host=None, timeout=None: _DictClient()
        sys.modules["ollama"] = fake

        agents = _fresh_agents()
        client = agents.OllamaClient(model="qwen3.5:4b")
        out = client.chat("system", "user")
        assert out == "dict-style response"
    finally:
        _restore(original)
