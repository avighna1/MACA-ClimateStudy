
from __future__ import annotations

import base64
import logging
import mimetypes
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config

logger = logging.getLogger("image_data")



@dataclass
class CityData:
    """Everything one Resident agent needs: its city's images (as raw
    base64 for vision input) and its FEMA NRI risk numbers (as text)."""

    city: str
    images: List[Tuple[str, str]] = field(default_factory=list)  # (base64, mime)
    hazard_risk_score: float = 50.0        # 0-100, FEMA NRI composite
    social_vulnerability_index: float = 0.5  # 0-1, higher = more vulnerable
    dominant_hazard: str = "Unknown"
    n_counties_matched: int = 0
    synthetic_risk: bool = False           # True if FEMA data fell back to synthetic
    image_source_dir: Optional[str] = None

    def combined_need_score(self) -> float:
        """
        Need = f(hazard risk, social vulnerability). Used as the
        needs-based baseline in metrics.relative_gini.
        Normalized hazard (0-1) weighted 0.6, vulnerability weighted 0.4.
        """
        return 0.6 * (self.hazard_risk_score / 100.0) + 0.4 * self.social_vulnerability_index

    def to_risk_text(self) -> str:
        """
        FEMA NRI risk data ONLY, as plain text. Deliberately contains NO
        description of the images -- the model looks at the actual
        images for visual context, so describing them in words here
        would defeat the point.
        """
        return (
            f"{self.city} -- FEMA National Risk Index data:\n"
            f"  Composite hazard risk score: {self.hazard_risk_score:.1f}/100\n"
            f"  Social Vulnerability Index (SoVI): {self.social_vulnerability_index:.2f} "
            f"(0 = low, 1 = high vulnerability)\n"
            f"  Dominant hazard: {self.dominant_hazard}\n"
            f"  (aggregated over {self.n_counties_matched} matched county row(s))"
            f"{chr(10) + '  [SYNTHETIC PLACEHOLDER RISK DATA -- not real FEMA NRI]' if self.synthetic_risk else ''}"
        )



def _resolve_path(entry: str) -> str:
    """Resolves one CITY_IMAGE_FILES entry. Absolute entries are used
    as-is; relative ones resolve against config.IMAGE_ROOT if set,
    otherwise the current working directory."""
    if os.path.isabs(entry):
        return entry
    root = getattr(config, "IMAGE_ROOT", "") or ""
    return os.path.join(root, entry) if root else entry


def _encode_image(path: str) -> Optional[Tuple[str, str]]:
    """Reads an image file and returns (base64_str, mime_type), or None
    if it can't be read. The raw bytes go straight to the model -- no
    decoding/parsing here, so an unusual format is passed through rather
    than crashing the run."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        logger.warning("Could not read image %s (%s); skipping.", path, exc)
        return None

    if not raw:
        logger.warning("Image file %s is empty; skipping.", path)
        return None

    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        if raw[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif raw[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        else:
            mime = "image/png"

    return base64.b64encode(raw).decode("ascii"), mime


def _load_embedded_images(city: str) -> Optional[List[Tuple[str, str]]]:
    """
    Returns images from the optional `embedded_images.py` module (base64
    bytes written directly into source by make_embedded_images.py), or
    None if that module doesn't exist / has nothing for this city.
    Takes priority over CITY_IMAGE_FILES when available -- this is the
    fully path-free option.
    """
    try:
        import embedded_images
    except ImportError:
        return None

    table = getattr(embedded_images, "CITY_IMAGES_BASE64", None)
    if not isinstance(table, dict):
        logger.warning(
            "embedded_images.py exists but has no CITY_IMAGES_BASE64 dict; "
            "ignoring it and using config.CITY_IMAGE_FILES instead."
        )
        return None

    entries = table.get(city)
    if not entries:
        return None

    out: List[Tuple[str, str]] = []
    for e in entries:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((e[0], e[1]))
        elif isinstance(e, str):
            out.append((e, "image/png"))
    return out or None


def load_images_for_city(city: str, limit: int = None) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """
    Loads the images for `city`, from embedded_images.py if present,
    otherwise from the explicit file list in config.CITY_IMAGE_FILES.
    No directory scanning and no path guessing -- exactly the files
    listed are used, and any that are missing are named individually in
    the log so the problem is obvious rather than silent.

    Returns (images, source_description).
    """
    limit = limit or config.CITYLENS_IMAGES_PER_AGENT

    embedded = _load_embedded_images(city)
    if embedded:
        imgs = embedded[:limit]
        logger.info("Loaded %d embedded image(s) for %s from embedded_images.py "
                    "(no files read).", len(imgs), city)
        return imgs, "embedded_images.py"

    entries = config.CITY_IMAGE_FILES.get(city)
    if not entries:
        logger.warning(
            "No images configured for '%s'. config.CITY_IMAGE_FILES has keys: %s. "
            "That agent will run with FEMA risk text only.",
            city, list(config.CITY_IMAGE_FILES.keys()),
        )
        return [], None

    images: List[Tuple[str, str]] = []
    missing: List[str] = []
    for entry in entries[:limit]:
        path = _resolve_path(entry)
        if not os.path.isfile(path):
            missing.append(os.path.abspath(path))
            continue
        enc = _encode_image(path)
        if enc:
            images.append(enc)

    if missing:
        logger.warning(
            "%d of %d configured image file(s) for '%s' were not found:\n  %s\n"
            "Current working directory: %s | config.IMAGE_ROOT: %r -- relative "
            "entries resolve against IMAGE_ROOT if set, else the CWD. Edit "
            "config.CITY_IMAGE_FILES (or set IMAGE_ROOT), or run "
            "`python make_embedded_images.py` to bake the bytes into source.",
            len(missing), len(entries[:limit]), city, "\n  ".join(missing),
            os.getcwd(), getattr(config, "IMAGE_ROOT", ""),
        )

    if images:
        logger.info("Loaded %d image(s) for %s from explicitly listed files.", len(images), city)
    else:
        logger.warning(
            "No images could be loaded for '%s' -- that agent will run with "
            "FEMA risk text only.", city,
        )
    return images, "config.CITY_IMAGE_FILES"



def _find_first_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Case-insensitive lookup for the first candidate column name that
    actually exists in df. Returns None if none match."""
    upper_to_real = {c.strip().upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in upper_to_real:
            return upper_to_real[cand.upper()]
    return None


def _normalize_fema_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames real FEMA NRI columns to the canonical internal names the
    rest of this pipeline expects (`hazard_risk_score`,
    `social_vulnerability_index`) if found under any of several known
    real-world naming variants -- real NRI Table CSVs call these
    RISK_SCORE/SOVI_SCORE (or similar), not our internal names.

    HONEST CAVEAT: exact FEMA NRI column names have varied slightly
    across dataset versions/redistributions, and this hasn't been
    verified against your actual downloaded file's real header row --
    it tries several plausible candidates and logs exactly what it did
    (or didn't) find, so a wrong guess is diagnosable rather than silent.

    Also rescales social vulnerability into our 0-1 convention if the
    detected column looks like it's on a 0-100 scale rather than
    already 0-1, based on its observed max value -- logged when applied.
    """
    df = df.copy()
    rename_map = {}

    risk_col = _find_first_matching_column(df, ["RISK_SCORE", "RISK_VALUE", "EAL_SCORE"])
    if risk_col and "hazard_risk_score" not in df.columns:
        rename_map[risk_col] = "hazard_risk_score"

    sovi_col = _find_first_matching_column(df, ["SOVI_SCORE", "SOVI_VALUE"])
    if sovi_col and "social_vulnerability_index" not in df.columns:
        rename_map[sovi_col] = "social_vulnerability_index"

    if rename_map:
        logger.info(
            "Mapped real FEMA NRI columns to canonical names (verify these are "
            "actually right for your file's version): %s", rename_map,
        )
        df = df.rename(columns=rename_map)
    else:
        logger.warning(
            "Could not confidently identify FEMA NRI's hazard-risk/social-"
            "vulnerability score columns. Actual columns in this file: %s. "
            "Cities will use default placeholder scores (50.0 / 0.5) for "
            "whichever weren't found -- that means the 'needs baseline' Gini "
            "calculation won't reflect real city-specific hazard differences "
            "until this is fixed. Tell me the exact real column names from the "
            "list above and I'll wire them in as recognized candidates.",
            list(df.columns),
        )

    if "social_vulnerability_index" in df.columns:
        try:
            max_val = pd.to_numeric(df["social_vulnerability_index"], errors="coerce").max()
            if pd.notna(max_val) and max_val > 1.5:
                logger.info(
                    "social_vulnerability_index looks like a 0-100ish scale (max "
                    "observed: %.2f) -- rescaling to our 0-1 convention by dividing "
                    "by 100.", max_val,
                )
                df["social_vulnerability_index"] = (
                    pd.to_numeric(df["social_vulnerability_index"], errors="coerce") / 100.0
                )
        except Exception:  # noqa: BLE001 - best-effort heuristic, never fatal
            pass

    return df


def _find_file_by_name(filename: str, search_roots: List[str], max_depth: int = 6) -> Optional[str]:
    """
    Best-effort search for a file by exact basename under a small set of
    plausible root directories, each capped at max_depth levels deep and
    skipping common huge/irrelevant directories -- so this stays fast.
    Fallback for when a configured path is wrong for a boring, common
    reason (relative path assumed a different working directory).
    """
    skip_dir_names = {
        ".git", "__pycache__", ".venv", "venv", "node_modules",
        "site-packages", ".ipynb_checkpoints",
    }
    seen_roots = set()
    for root in search_roots:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        if abs_root in seen_roots or not os.path.isdir(abs_root):
            continue
        seen_roots.add(abs_root)
        base_depth = abs_root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(abs_root):
            depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
            if depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in skip_dir_names]
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def load_fema_nri(local_path: str = None) -> Optional[pd.DataFrame]:
    """
    Loads the local FEMA National Risk Index CSV. Returns the normalized
    DataFrame, or None if it genuinely can't be found (caller then falls
    back to clearly-flagged synthetic risk scores).

    Path resolution: exact configured path -> same path with `.csv`
    appended -> a bounded filesystem search for the same basename near
    the current directory / project root / home. Whatever it finds (or
    fails to find) is logged with absolute paths so mismatches are easy
    to diagnose.
    """
    local_path = local_path or config.FEMA_NRI_LOCAL_PATH
    abs_path = os.path.abspath(local_path)
    basename = os.path.basename(local_path.rstrip("/"))
    if not basename.lower().endswith(".csv"):
        basename += ".csv"

    candidates = [local_path]
    if not local_path.lower().endswith(".csv"):
        candidates.append(local_path + ".csv")

    found_path = next((c for c in candidates if os.path.exists(c)), None)

    if not found_path:
        searched = _find_file_by_name(basename, [".", config.PROJECT_ROOT, os.path.expanduser("~")])
        if searched:
            logger.warning(
                "FEMA_NRI_LOCAL_PATH (%s) did not exist, but a file named '%s' was "
                "found nearby at: %s -- using that. Update FEMA_NRI_LOCAL_PATH in "
                "config.py to this exact path to skip the search next time.",
                abs_path, basename, searched,
            )
            found_path = searched

    if not found_path:
        parent_dir = os.path.dirname(abs_path) or "."
        listing = ""
        if os.path.isdir(parent_dir):
            try:
                entries = os.listdir(parent_dir)
                listing = f" Contents of {parent_dir}: {entries if entries else '(empty)'}."
            except OSError:
                pass
        else:
            listing = f" (directory {parent_dir} does not exist either)."
        logger.warning(
            "FEMA NRI file not found. Checked: %s.%s Current working directory: %s "
            "-- a RELATIVE FEMA_NRI_LOCAL_PATH resolves against this (an absolute "
            "path is safest). Download the county-level Table CSV from %s. Falling "
            "back to synthetic hazard scores.",
            [os.path.abspath(c) for c in candidates], listing, os.getcwd(),
            config.FEMA_NRI_SOURCE_URL,
        )
        return None

    try:
        df = pd.read_csv(found_path)
        df = _normalize_fema_columns(df)
        logger.info("Loaded FEMA NRI data from %s (%d rows).", os.path.abspath(found_path), len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to parse FEMA NRI file at %s (%s); falling back to synthetic "
            "hazard scores.", os.path.abspath(found_path), exc,
        )
        return None


def _detect_county_state_columns(nri_df: pd.DataFrame):
    """Best-effort detection of FEMA NRI's county/state name columns."""
    county_col = next(
        (c for c in nri_df.columns if c.strip().upper() in ("COUNTY", "COUNTYNAME", "COUNTY_NAME")),
        None,
    )
    state_col = next(
        (c for c in nri_df.columns if c.strip().upper() in ("STATE", "STATENAME", "STATE_NAME")),
        None,
    )
    return county_col, state_col


def _risk_for_city(nri_df: pd.DataFrame, city: str):
    """
    Averages FEMA NRI risk/vulnerability across the counties that make up
    `city` (per config.CITY_TO_NRI_COUNTIES). Real FEMA county data has
    no "city" column, so matching is by county (optionally narrowed by
    state to disambiguate same-named counties in different states).
    Returns (hazard, sovi, dominant_hazard, n_matched) or None if nothing
    matched.
    """
    county_col, state_col = _detect_county_state_columns(nri_df)
    mapping = config.CITY_TO_NRI_COUNTIES.get(city)
    if not county_col or not mapping:
        logger.warning(
            "Cannot match FEMA NRI rows for '%s' (county column found: %s; "
            "city->county mapping present: %s). Columns available: %s.",
            city, county_col, bool(mapping), list(nri_df.columns),
        )
        return None

    targets = {c.strip().lower() for c in mapping["counties"]}

    def _matches(val) -> bool:
        if not isinstance(val, str):
            return False
        v = val.strip().lower()
        if v.endswith(" county"):
            v = v[: -len(" county")]
        return v in targets

    mask = nri_df[county_col].apply(_matches)
    if state_col and mapping.get("state"):
        target_state = mapping["state"].strip().lower()
        mask &= nri_df[state_col].astype(str).str.strip().str.lower() == target_state

    rows = nri_df[mask]
    if rows.empty:
        logger.warning(
            "No FEMA NRI rows matched counties %s for '%s' in column '%s'.",
            mapping["counties"], city, county_col,
        )
        return None

    hazard = 50.0
    if "hazard_risk_score" in rows.columns:
        val = pd.to_numeric(rows["hazard_risk_score"], errors="coerce").mean()
        if pd.notna(val):
            hazard = float(val)

    sovi = 0.5
    if "social_vulnerability_index" in rows.columns:
        val = pd.to_numeric(rows["social_vulnerability_index"], errors="coerce").mean()
        if pd.notna(val):
            sovi = float(val)

    dominant = "Unknown"
    hazard_col = _find_first_matching_column(rows, ["DOMINANT_HAZARD", "HAZARD_TYPE", "HAZARD"])
    if hazard_col:
        modes = rows[hazard_col].dropna()
        if not modes.empty:
            dominant = str(modes.mode().iloc[0]) if not modes.mode().empty else str(modes.iloc[0])

    return hazard, sovi, dominant, int(len(rows))


def _synthetic_risk_for_city(city: str):
    """Deterministic, clearly-flagged placeholder risk numbers, used only
    when the real FEMA CSV is unavailable/unmatched. Seeded per city so
    runs are reproducible."""
    rng = random.Random(config.RANDOM_SEED + sum(ord(ch) for ch in city))
    hazards = ["Coastal Flooding", "Riverine Flooding", "Heat Wave", "Severe Wind/Storm", "Wildfire"]
    return (
        rng.uniform(30, 90),      # hazard_risk_score
        rng.uniform(0.2, 0.85),   # social_vulnerability_index
        rng.choice(hazards),      # dominant_hazard
    )



def build_city_data(cities: List[str] = None) -> Dict[str, CityData]:
    """
    Builds the per-city image + risk bundles every Resident agent needs.
    Entirely local: reads image files off disk and one FEMA NRI CSV. No
    HTTP requests are made to Hugging Face or anywhere else.
    """
    cities = cities or config.CITIES
    nri_df = load_fema_nri()

    out: Dict[str, CityData] = {}
    for city in cities:
        images, src_dir = load_images_for_city(city)

        risk = _risk_for_city(nri_df, city) if nri_df is not None else None
        if risk is not None:
            hazard, sovi, dominant, n_matched = risk
            synthetic_risk = False
        else:
            hazard, sovi, dominant = _synthetic_risk_for_city(city)
            n_matched = 0
            synthetic_risk = True

        out[city] = CityData(
            city=city,
            images=images,
            hazard_risk_score=hazard,
            social_vulnerability_index=sovi,
            dominant_hazard=dominant,
            n_counties_matched=n_matched,
            synthetic_risk=synthetic_risk,
            image_source_dir=src_dir,
        )

    return out


def zone_need_scores(city_data: Dict[str, CityData]) -> Dict[str, float]:
    """Per-city combined-need score, the RelativeGini baseline."""
    return {city: cd.combined_need_score() for city, cd in city_data.items()}


def any_synthetic_risk(city_data: Dict[str, CityData]) -> bool:
    return any(cd.synthetic_risk for cd in city_data.values())


def cities_without_images(city_data: Dict[str, CityData]) -> List[str]:
    return [city for city, cd in city_data.items() if not cd.images]
