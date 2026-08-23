"""
config.py
Central configuration for the emergent-bias multi-agent climate budget
allocation study (NeurIPS UrbanAI workshop submission).

Target environment: JupyterLab with Ollama running locally (CPU, Apple
Metal, or CUDA GPU -- Ollama supports all three; no CUDA-only requirement
the way vLLM had). Every agent talks to Ollama's OpenAI-compatible
/v1/chat/completions endpoint, on ONE shared server -- unlike vLLM,
Ollama serves every model from a single daemon and loads/unloads models
by tag on demand, so there's no per-model port or GPU-memory-fraction
bookkeeping here.

All tunables live here so main.py, setup_ollama.py, agents.py, etc.
share one source of truth. This file is also the ONE place you should
need to edit to add a 4th/5th diagnostic study -- see STUDY_CONFIGS below.
"""

import os
from typing import Dict, List


# ==========================================================================
# 1. Ollama serving
# ==========================================================================
# Install Ollama, then either let it run as its normal background
# service, or start it yourself with `ollama serve`. Before running
# main.py, pull whatever models the selected study needs with:
#     python setup_ollama.py --study <name>
# See README.md "Setup" for the full walkthrough.
#
# NOTE: this is the NATIVE Ollama host (no "/v1" suffix). This repo now
# uses the official `ollama` Python package (ollama.chat / ollama.Client)
# instead of the OpenAI-compatible shim, specifically because the native
# API reliably honors `think=False` while the OpenAI-compat endpoint
# has documented, inconsistent support for it.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
REQUEST_TIMEOUT_S = 180
MAX_RETRIES = 3
RETRY_BACKOFF_S = 3.0

GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 700
JUDGE_TEMPERATURE = 0.2          # Judge should be more deterministic/analytic

# Passed as `think=THINK_MODE` on every native ollama.chat() call, for
# all 8 agents. Qwen3.5 (and other thinking-capable Ollama models)
# support an extended reasoning phase before the final answer; False
# skips it, giving a direct answer immediately. Unlike the previous
# OpenAI-compat approach (where `think` was passed via extra_body and
# was inconsistently honored), the native API supports this parameter
# directly and reliably.
THINK_MODE = False

# --------------------------------------------------------------------------
# Model registry: short key -> real, VERIFIED Ollama library tag.
# --------------------------------------------------------------------------
# Every tag below was checked directly against ollama.com/library before
# being used here (not assumed from memory):
#   https://ollama.com/library/qwen3.5:4b
#   https://ollama.com/library/qwen3.5:9b
#   https://ollama.com/library/llama3.1:8b
#   https://ollama.com/library/mistral   (7b-instruct-v0.3 tag family)
# `qwen3.5` is a real Qwen family that DOES ship exact 4B and 9B "Small"
# checkpoints (also 0.8B/2B/27B/35B/122B) -- unlike the earlier "3B"
# spec, these two sizes are genuinely available.
# Add a new entry here (and reference its key from a study's
# `role_models` below) for any other model you want to use.
MODEL_REGISTRY: Dict[str, Dict] = {
    "qwen_4b": {"ollama_tag": "qwen3.5:4b"},
    "qwen_9b": {"ollama_tag": "qwen3.5:9b"},
    "llama_8b": {"ollama_tag": "llama3.1:8b"},
    "mistral_7b": {"ollama_tag": "mistral:7b-instruct-v0.3"},
}


def ollama_tag_for(model_key: str) -> str:
    return MODEL_REGISTRY[model_key]["ollama_tag"]


# ==========================================================================
# 2. Agent roles + diagnostic study configurations
# ==========================================================================
AGENT_ROLES: List[str] = [
    "resident", "infrastructure", "financial", "sustainability", "judge", "improver",
]

# --------------------------------------------------------------------------
# STUDY_CONFIGS is the single editable source of truth for every diagnostic
# study. To add a new study: copy a block, change `role_models` and/or
# `use_judge_improver`, give it a new key, then run:
#     python setup_ollama.py --study <your_new_key>
#     python main.py         --study <your_new_key>
# Nothing else in the codebase needs to change.
# --------------------------------------------------------------------------
STUDY_CONFIGS: Dict[str, Dict] = {

    "baseline": {
        "description": "Default run: all 6 roles on the 4B model, Judge+Improver active.",
        "use_judge_improver": True,
        "role_models": {r: "qwen_4b" for r in AGENT_ROLES},
    },

    # ---- Study 1: Judge/Improver ablation -----------------------------
    "study1_no_judge_improver": {
        "description": (
            "ABLATION: negotiation runs with NO Judge and NO Improver agent "
            "at all -- residents and functional agents negotiate freely, "
            "with zero bias auditing or correction. Compare its Gini "
            "trajectory against 'baseline' to isolate the Judge/Improver's "
            "causal effect on emergent bias."
        ),
        "use_judge_improver": False,
        "role_models": {r: "qwen_4b" for r in AGENT_ROLES},  # judge/improver models unused when off
    },

    # ---- Study 2: model-scale settings ---------------------------------
    "study2a_all_4b": {
        "description": "All 6 roles on the 4B model (uniform small-model condition).",
        "use_judge_improver": True,
        "role_models": {r: "qwen_4b" for r in AGENT_ROLES},
    },
    "study2b_all_9b": {
        "description": "All 6 roles on the 9B model (uniform larger-model condition).",
        "use_judge_improver": True,
        "role_models": {r: "qwen_9b" for r in AGENT_ROLES},
    },
    "study2c_judge_improver_9b": {
        "description": (
            "Residents + all 3 functional agents stay on the 4B model; "
            "ONLY Judge and Improver are upgraded to 9B -- isolates whether "
            "a stronger auditor/corrector (holding the negotiators fixed) "
            "reduces emergent bias."
        ),
        "use_judge_improver": True,
        "role_models": {
            "resident": "qwen_4b", "infrastructure": "qwen_4b",
            "financial": "qwen_4b", "sustainability": "qwen_4b",
            "judge": "qwen_9b", "improver": "qwen_9b",
        },
    },

    # ---- Study 3: heterogeneous models ---------------------------------
    "study3_heterogeneous": {
        "description": (
            "Different model FAMILIES across roles (not just sizes) -- "
            "template study for cross-model-family bias dynamics. "
            "infrastructure/financial keep the same llama3.1:8b / "
            "mistral:7b-instruct-v0.3 models as before; edit role_models "
            "and/or MODEL_REGISTRY freely -- this is a starting "
            "configuration, not a validated finding. NOTE: because "
            "Ollama loads models on demand from one daemon rather than "
            "requiring every unique model resident in VRAM simultaneously "
            "(as vLLM did), this study now runs on a single GPU/CPU "
            "without the memory-overcommit problem the vLLM version had --"
            "the trade-off is possible model-swap latency between agents "
            "using different models (see README)."
        ),
        "use_judge_improver": True,
        "role_models": {
            "resident": "qwen_4b",
            "infrastructure": "llama_8b",
            "financial": "mistral_7b",
            "sustainability": "qwen_9b",
            "judge": "qwen_9b",
            "improver": "qwen_9b",
        },
    },
}

ACTIVE_STUDY_DEFAULT = "baseline"


def required_model_keys(study_name: str) -> List[str]:
    """Unique MODEL_REGISTRY keys a study needs pulled in Ollama."""
    return sorted(set(STUDY_CONFIGS[study_name]["role_models"].values()))


# ==========================================================================
# 3. Cities / zones under study
# ==========================================================================
# TWO cities (Chicago removed per explicit request -- there was no local
# CityLens image directory for it). One Resident agent per city.
CITIES: List[str] = ["New York", "San Francisco"]

CITY_TO_ZONE_LABEL = {
    "New York": "NYC Metro Tract Cluster",
    "San Francisco": "SF Bay Tract Cluster",
}

# Real FEMA NRI county-level data (the "NRI_Table_Counties.csv" file --
# what FEMA_NRI_LOCAL_PATH points at) has NO "city" column -- it's
# organized by STATE + COUNTY name. This maps each city to the real
# county/counties it falls in, so rag-free data loading can filter NRI
# rows by county instead of assuming a city column that real data
# doesn't have. New York City spans 5 counties (one per borough);
# San Francisco's is a single county.
CITY_TO_NRI_COUNTIES = {
    "New York": {
        "state": "New York",
        "counties": ["New York", "Kings", "Queens", "Bronx", "Richmond"],
    },
    "San Francisco": {
        "state": "California",
        "counties": ["San Francisco"],
    },
}

# ==========================================================================
# 4. Data sources -- ALL LOCAL, no network requests to Hugging Face
# ==========================================================================
# CityLens satellite/streetview images are read directly off local disk.
# NOTHING in this repo makes an HTTP request to huggingface.co anymore:
# the `datasets` library is no longer imported or used, and there is no
# dataset download step at all. Point these at the directories where
# you already saved the images.
#
# ==========================================================================
# HOW IMAGES ARE SUPPLIED -- explicit, per-city, listed right here.
# ==========================================================================
# There is NO directory scanning and NO path guessing. You list exactly
# which image files each city's Resident agent gets, below. Two ways:
#
#   OPTION A (default): CITY_IMAGE_FILES -- list the 5 files per city.
#     Entries may be absolute or relative; relative ones resolve against
#     IMAGE_ROOT if set, otherwise the current working directory. Every
#     file is checked at startup and any missing one is reported by name,
#     so there's no silent fallback to "no images".
#
#   OPTION B (fully path-free): put base64 image data in embedded_images.py
#     and it is used automatically if present, taking priority over
#     Option A. Generate that file once with:
#         python make_embedded_images.py
#     which reads whatever CITY_IMAGE_FILES points at and writes the
#     bytes directly into a Python module -- after that the repo needs no
#     image files on disk at all.
#
# NOTE: image data has to come from somewhere -- either a file reference
# or bytes embedded in source. Option B is the closest thing to "no
# paths at all" (paths are used once, at generation time, then never
# again).

# Optional prefix for any RELATIVE entries in CITY_IMAGE_FILES. Leave ""
# to resolve them against the current working directory instead.
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "")

# EDIT THESE: exactly which files each city's Resident agent receives.
CITY_IMAGE_FILES: Dict[str, List[str]] = {
    "New York": [
        "12312_9646.png",
        "12312_9647.png",
        "12312_9648.png",
        "12312_9649.png",
        "12312_9650.png",
    ],
    "San Francisco": [
        "12661_5232.png",
        "12661_5233.png",
        "12661_5234.png",
        "12661_5235.png",
        "12661_5236.png",
    ],
}

# Hard cap on images sent per agent call (each one adds real request
# size/latency for a VLM). Lists longer than this are truncated.
CITYLENS_IMAGES_PER_AGENT = int(os.environ.get("CITYLENS_IMAGES_PER_AGENT", 5))

# Recognized image extensions, used only for MIME sniffing now.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

# Hardcoded directly -- no `export` needed. This is checked FIRST; the
# FEMA_NRI_LOCAL_PATH env var (if set) overrides it, so you can still
# override per-session without editing this file if you ever want to.
FEMA_NRI_LOCAL_PATH = os.environ.get(
    "FEMA_NRI_LOCAL_PATH",
    "avighna-experiments/avighna-ollama-mas/data/NRI_Table_Counties/NRI_Table_Counties.csv",
)
FEMA_NRI_SOURCE_URL = "https://hazards.fema.gov/nri/data-resources"

# ==========================================================================
# 5. Output locations -- resolved at runtime against Google Drive
# ==========================================================================
# main.py calls drive_utils.mount_and_resolve_drive_root() at startup and
# REWRITES these four module-level paths to live under the resolved
# Drive (or clearly-labeled local fallback) root before anything is
# written. Google Drive is the ONLY persistence backend this project
# uses -- don't hardcode paths elsewhere; always read config.DATA_DIR /
# config.TRACE_DIR / config.TRANSCRIPT_DIR / config.FIGURE_DIR at call time.
PROJECT_ROOT = "."
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRACE_DIR = os.path.join(PROJECT_ROOT, "traces")
TRANSCRIPT_DIR = os.path.join(PROJECT_ROOT, "transcripts")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")


def set_project_root(root: str) -> None:
    """Called once by main.py after resolving the Drive root."""
    global PROJECT_ROOT, DATA_DIR, TRACE_DIR, TRANSCRIPT_DIR, FIGURE_DIR
    PROJECT_ROOT = root
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    TRACE_DIR = os.path.join(PROJECT_ROOT, "traces")
    TRANSCRIPT_DIR = os.path.join(PROJECT_ROOT, "transcripts")
    FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
    for d in (DATA_DIR, TRACE_DIR, TRANSCRIPT_DIR, FIGURE_DIR):
        os.makedirs(d, exist_ok=True)


# ==========================================================================
# 6. Budget simulation
# ==========================================================================
TOTAL_CITY_BUDGET_USD = 300_000_000
NEGOTIATION_ROUNDS = 5

# ==========================================================================
# 7. Bias detection / correction thresholds
# ==========================================================================
GINI_INTERVENTION_THRESHOLD = 0.15

RANDOM_SEED = 42
