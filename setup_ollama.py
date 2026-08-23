"""
setup_ollama.py
Ensures Ollama is installed and running, and that every model tag the
selected study needs is pulled locally -- run this BEFORE main.py.

Unlike the old vLLM setup, there is only ONE server here: Ollama serves
every model from a single daemon (default http://localhost:11434) and
loads/unloads model weights by tag on demand, so there's no per-model
port assignment or GPU-memory-utilization fraction to budget across
processes. That also means multi-model studies (study2c, study3) no
longer risk the "not enough VRAM for all unique models at once" failure
mode vLLM had -- the trade-off is possible model-swap latency between
agents that use different models in the same run (see README "Ollama
model-swapping notes").

Usage:
    python setup_ollama.py --study baseline
    python setup_ollama.py --study study3_heterogeneous
    python setup_ollama.py --list           # show locally pulled models
    python setup_ollama.py --status         # show what's currently loaded (ollama ps)

Then, in another terminal/cell:
    python main.py --study <name>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List

import config


def check_ollama_installed() -> bool:
    """Checks the `ollama` CLI/binary is on PATH."""
    path = shutil.which("ollama")
    if path:
        print(f"[setup_ollama] Found ollama binary at: {path}")
        return True
    print(
        "[setup_ollama] WARNING: `ollama` was not found on PATH. Install it "
        "first:\n"
        "    macOS/Linux (needs sudo/root): curl -fsSL https://ollama.com/install.sh | sh\n"
        "    Windows/macOS app: download from https://ollama.com/download\n"
        "\n"
        "    If you're on a shared/managed JupyterLab box and the installer "
        "fails with something like:\n"
        "        sudo: The \"no new privileges\" flag is set, which prevents "
        "sudo from running as root\n"
        "    that's a container security setting blocking sudo entirely -- "
        "no flag or workaround fixes that from inside the container. Do a "
        "user-space install instead (no root needed):\n"
        "        ARCH=$(uname -m); [ \"$ARCH\" = \"x86_64\" ] && OLLAMA_ARCH=amd64; "
        "[ \"$ARCH\" = \"aarch64\" ] && OLLAMA_ARCH=arm64\n"
        "        mkdir -p ~/.local\n"
        "        curl -L \"https://ollama.com/download/ollama-linux-${OLLAMA_ARCH}.tar.zst\" "
        "-o /tmp/ollama.tar.zst\n"
        "        tar -C ~/.local -xf /tmp/ollama.tar.zst\n"
        "        # if that fails (no zstd binary, no root to install one), use:\n"
        "        #   pip install --user zstandard\n"
        "        #   python3 -c \"import zstandard,tarfile,os; "
        "dctx=zstandard.ZstdDecompressor(); "
        "f=open('/tmp/ollama.tar.zst','rb'); "
        "r=dctx.stream_reader(f); "
        "tarfile.open(fileobj=r, mode='r|').extractall(path=os.path.expanduser('~/.local'))\"\n"
        "        export PATH=$HOME/.local/bin:$PATH\n"
        "        export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH\n"
        "        # add both export lines to ~/.bashrc too, so new terminals keep them\n"
        "        ollama --version   # verify\n"
        "    Then start the daemon yourself in its own terminal/cell (no systemd "
        "without root): `ollama serve &`\n"
        "See README.md 'Setup' and 'Troubleshooting' for the full walkthrough."
    )
    return False


def is_server_running(timeout: float = 2.0) -> bool:
    """Checks Ollama's OpenAI-compatible /v1/models endpoint is reachable."""
    url = config.OLLAMA_HOST.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def start_server_if_needed(startup_timeout_s: int = 60) -> bool:
    """
    If Ollama isn't already reachable, tries to start it with `ollama serve`
    as a background process. On most installs (the official installer sets
    up a systemd/launchd service, and the desktop app runs it automatically)
    Ollama is ALREADY running and this is a no-op health check. Only if it's
    genuinely unreachable do we attempt to launch our own background copy.
    """
    if is_server_running():
        print(f"[setup_ollama] Ollama is already running at {config.OLLAMA_HOST}")
        return True

    if not check_ollama_installed():
        return False

    print(
        "[setup_ollama] Ollama isn't reachable yet. If you have it installed "
        "as a system service (the default on most installs), start that "
        "service instead of relying on this script, e.g.:\n"
        "    Linux (systemd): sudo systemctl start ollama\n"
        "    macOS/Windows:   launch the Ollama app\n"
        "Attempting to start `ollama serve` as a background process here as "
        "a fallback..."
    )
    log_file = open("ollama_serve.log", "w")
    subprocess.Popen(["ollama", "serve"], stdout=log_file, stderr=subprocess.STDOUT)

    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if is_server_running():
            print(f"[setup_ollama] Ollama is now running at {config.OLLAMA_HOST}")
            return True
        time.sleep(2)

    print(
        f"*** WARNING: Ollama still isn't reachable at {config.OLLAMA_HOST} "
        f"after {startup_timeout_s}s. Check ollama_serve.log, or start it "
        f"manually with `ollama serve` in its own terminal. ***"
    )
    return False


def list_local_models() -> List[str]:
    """Parses `ollama list` to see which tags are already pulled."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()[1:]  # skip header row
        return [line.split()[0] for line in lines if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        return []


def pull_model(tag: str) -> bool:
    """
    Runs `ollama pull <tag>`, streaming its native progress output
    straight to the terminal (not captured, so you see the real download
    progress bar). Idempotent -- if already pulled, this returns almost
    immediately.
    """
    print(f"[setup_ollama] Pulling '{tag}' (skips quickly if already present)...")
    result = subprocess.run(["ollama", "pull", tag])
    if result.returncode != 0:
        print(f"*** WARNING: `ollama pull {tag}` exited with code {result.returncode}. "
              f"Check the output above for the actual error. ***")
        return False
    return True


def ensure_models_for_study(study_name: str) -> bool:
    """Pulls every unique model tag the given study needs. Returns True
    iff every pull succeeded."""
    keys = config.required_model_keys(study_name)
    tags = [config.ollama_tag_for(k) for k in keys]
    print(f"[setup_ollama] Study '{study_name}': {config.STUDY_CONFIGS[study_name]['description']}")
    print(f"[setup_ollama] Requires {len(tags)} unique model tag(s): {tags}")

    if len(tags) > 1:
        print(
            "[setup_ollama] NOTE: this study uses more than one unique model. "
            "Ollama loads models into memory on demand and may unload idle "
            "ones (default after 5 minutes -- see OLLAMA_KEEP_ALIVE / "
            "OLLAMA_MAX_LOADED_MODELS in README 'Troubleshooting' to tune "
            "this), so agents that alternate between different models may "
            "see extra latency from model swapping. The run will still "
            "complete correctly either way."
        )

    ok = True
    for tag in tags:
        ok = pull_model(tag) and ok
    return ok


def warm_up_models(study_name: str, timeout_s: int = 600) -> bool:
    """
    Sends one real, minimal chat request to every unique model tag the
    study needs, with a generous timeout, so the (potentially very slow)
    first-ever load of each model into memory happens HERE -- as its own
    diagnosable step with a clear per-model progress line -- rather than
    inside main.py's first real agent call, where a cold-load taking
    longer than agents.py's normal per-call timeout looks exactly like a
    dead server ("Ollama call ... failed 3 times in a row: timed out")
    even though the server was fine and just busy loading weights.

    Uses the native `ollama` package directly (not agents.OllamaClient),
    since this needs its own much longer timeout than normal negotiation
    calls should ever need.
    """
    try:
        import ollama
    except ImportError:
        print(
            "[setup_ollama] `ollama` package not installed (pip install ollama) "
            "-- skipping warm-up. The first real call in main.py may be slow "
            "instead."
        )
        return False

    keys = config.required_model_keys(study_name)
    tags = [config.ollama_tag_for(k) for k in keys]
    client = ollama.Client(host=config.OLLAMA_HOST, timeout=timeout_s)

    print(f"[setup_ollama] Warming up {len(tags)} model(s) (timeout {timeout_s}s each -- "
          f"the FIRST call to a model can be much slower than later ones while it "
          f"loads into memory)...")
    all_ok = True
    for tag in tags:
        start = time.time()
        try:
            client.chat(
                model=tag,
                messages=[{"role": "user", "content": "Reply with one word: ready"}],
                think=False,
                options={"num_predict": 10},
            )
            elapsed = time.time() - start
            print(f"  [{tag}] warm and responding ({elapsed:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            print(f"  [{tag}] WARM-UP FAILED after {elapsed:.1f}s: {exc}")
            all_ok = False

    if not all_ok:
        print(
            "[setup_ollama] One or more models failed to warm up even with a "
            f"{timeout_s}s timeout. This points at something more than a slow "
            "cold-load -- check `ollama ps` (is it actually running the model on "
            "GPU, or silently on CPU?), `nvidia-smi`, and whether the model is "
            "genuinely pulled (`ollama list`)."
        )
    return all_ok


def missing_models(study_name: str) -> List[str]:
    """Tags required by the study that are NOT in `ollama list` output.
    Used by main.py's pre-flight check. Exact match only -- `ollama pull
    <tag>` lists exactly the tag you asked for, so fuzzy/prefix matching
    would wrongly treat different sizes of the same family (e.g.
    qwen3.5:4b vs qwen3.5:9b) as interchangeable."""
    local = set(list_local_models())
    keys = config.required_model_keys(study_name)
    required_tags = [config.ollama_tag_for(k) for k in keys]
    return [tag for tag in required_tags if tag not in local]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up Ollama + pull models for a diagnostic study")
    parser.add_argument("--study", default=config.ACTIVE_STUDY_DEFAULT, choices=list(config.STUDY_CONFIGS.keys()))
    parser.add_argument("--list", action="store_true", help="List locally pulled Ollama models and exit")
    parser.add_argument("--status", action="store_true", help="Show currently loaded models (`ollama ps`) and exit")
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="Skip the post-pull warm-up call to each model (not recommended -- the "
             "first real call in main.py may then hit its normal, much shorter "
             "per-call timeout while the model is still cold-loading).",
    )
    parser.add_argument("--warmup-timeout", type=int, default=600, help="Seconds to allow per model warm-up call.")
    args = parser.parse_args()

    if args.list:
        models = list_local_models()
        print("Locally pulled Ollama models:" if models else "No models pulled locally yet.")
        for m in models:
            print(f"  {m}")
        sys.exit(0)

    if args.status:
        subprocess.run(["ollama", "ps"])
        sys.exit(0)

    server_ok = start_server_if_needed()
    if not server_ok:
        print("\n[setup_ollama] Cannot continue without a reachable Ollama server. See warnings above.")
        sys.exit(1)

    success = ensure_models_for_study(args.study)
    if not success:
        print(
            "\n*** WARNING: one or more model pulls failed (see above). "
            "main.py's pre-flight check will catch and report any still-missing "
            "models before attempting to run. ***"
        )
        sys.exit(1)

    warm_ok = True
    if not args.skip_warmup:
        warm_ok = warm_up_models(args.study, timeout_s=args.warmup_timeout)

    if success and warm_ok:
        print(
            f"\n[setup_ollama] All models for study '{args.study}' are pulled AND warmed "
            f"up. You can now run: python main.py --study {args.study}"
        )
    elif success and not warm_ok:
        print(
            "\n*** Models are pulled but warm-up failed for at least one -- see above. "
            "Running main.py now risks the exact 'timed out' error this warm-up step "
            "exists to catch early. Diagnose with `ollama ps` / `nvidia-smi` first, or "
            "re-run with --warmup-timeout set higher if it's just a slow cold-load. ***"
        )
        sys.exit(1)
