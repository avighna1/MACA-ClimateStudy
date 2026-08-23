"""
drive_utils.py
Resolves (and, in Colab, mounts) a Google Drive-backed root directory so
every trace, transcript, and figure this repo produces lands somewhere
persistent across JupyterLab/Colab sessions.

Honesty requirement: this module NEVER reports success saving to Drive
unless it actually found/mounted a Drive path. If it can't, it falls back
to a clearly-named local directory and prints a loud warning — it does
not pretend the run was saved to Drive when it wasn't.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("drive_utils")

DEFAULT_SUBDIR = "climate-mas-bias"


def mount_and_resolve_drive_root(preferred_path: str = None) -> str:
    """
    Returns an absolute directory path to use as the project's output root.

    Resolution order:
      1. Google Colab: mount /content/drive if not already mounted, use
         <preferred_path> or .../MyDrive/climate-mas-bias under it.
      2. A Drive path that's ALREADY mounted in this JupyterLab environment
         (e.g. via Google Drive for Desktop, rclone mount, or a prior Colab
         mount in this same session) at a conventional location.
      3. `preferred_path` if explicitly given and writable, even if it
         doesn't look Drive-like (caller's explicit choice wins).
      4. Local fallback — clearly named, with a printed warning. This is
         NOT Google Drive and the function makes that explicit rather than
         silently proceeding as if it were.
    """
    # 1. Google Colab
    try:
        import google.colab  # noqa: F401
        from google.colab import drive as colab_drive

        mount_point = "/content/drive"
        already_mounted = os.path.isdir(os.path.join(mount_point, "MyDrive"))
        if not already_mounted:
            colab_drive.mount(mount_point)
            already_mounted = os.path.isdir(os.path.join(mount_point, "MyDrive"))

        if already_mounted:
            root = preferred_path or os.path.join(mount_point, "MyDrive", DEFAULT_SUBDIR)
            os.makedirs(root, exist_ok=True)
            logger.info("Google Drive mounted via Colab. Project root: %s", root)
            print(f"[drive_utils] Google Drive mounted. Saving all outputs under: {root}")
            return root
        else:
            logger.warning("Colab detected but Drive mount did not succeed as expected.")
    except ImportError:
        pass  # not running in Colab

    # 2. Already-mounted Drive path (JupyterLab-on-VM style setups)
    candidate_roots = [
        "/content/drive/MyDrive/" + DEFAULT_SUBDIR,   # leftover Colab mount in this kernel
        os.path.expanduser("~/GoogleDrive/" + DEFAULT_SUBDIR),
        os.path.expanduser("~/Google Drive/" + DEFAULT_SUBDIR),
        os.path.expanduser("~/google-drive/" + DEFAULT_SUBDIR),
        "/mnt/gdrive/" + DEFAULT_SUBDIR,
    ]
    for candidate in candidate_roots:
        parent = os.path.dirname(candidate)
        if os.path.isdir(parent):
            os.makedirs(candidate, exist_ok=True)
            logger.info("Found pre-mounted Google Drive path: %s", candidate)
            print(f"[drive_utils] Using pre-mounted Google Drive path: {candidate}")
            return candidate

    # 3. Explicit preferred_path from the caller, even if not Drive-like
    if preferred_path:
        try:
            os.makedirs(preferred_path, exist_ok=True)
            is_drive_like = "drive" in preferred_path.lower()
            if not is_drive_like:
                print(
                    f"[drive_utils] WARNING: '{preferred_path}' does not look like a "
                    f"Google Drive path and no Drive mount was detected. Using it "
                    f"as-is because it was explicitly requested, but this is "
                    f"probably NOT Google Drive."
                )
            return preferred_path
        except OSError as exc:
            logger.warning("Could not create preferred_path %s (%s); falling back.", preferred_path, exc)

    # 4. Local fallback — explicit, not silently mislabeled as Drive
    local_root = os.path.abspath("./" + DEFAULT_SUBDIR + "_LOCAL_NOT_DRIVE")
    os.makedirs(local_root, exist_ok=True)
    logger.warning(
        "Could not detect or mount Google Drive in this environment. Results "
        "will be saved LOCALLY at %s — NOT on Google Drive.", local_root,
    )
    print(
        f"\n*** WARNING: Google Drive was not detected/mounted. Saving results "
        f"to a LOCAL directory instead: {local_root}\n"
        f"    This is NOT Google Drive.\n"
        f"    If you're on a plain JupyterLab box (not Colab), there is no "
        f"automatic Google login here — mount Drive yourself first with "
        f"rclone (works on headless/remote machines):\n"
        f"        curl https://rclone.org/install.sh | sudo bash\n"
        f"        rclone config          # new remote 'gdrive' -> Google "
        f"Drive -> when asked 'Use auto config?' answer N (headless) -> "
        f"open the printed link on YOUR OWN laptop's browser, authorize, "
        f"paste the verification code back here\n"
        f"        mkdir -p ~/gdrive && rclone mount gdrive: ~/gdrive "
        f"--vfs-cache-mode writes --daemon\n"
        f"    Then re-run with: --drive-root ~/gdrive/climate-mas-bias\n"
        f"    (See README.md 'Troubleshooting' for the full walkthrough.) ***\n"
    )
    return local_root
