"""Experimental grid, prompt construction, and ground-truth bands.

All wording and numeric ground truth lives in config.json so it can be tweaked
without touching call logic. Lock config.json before the full batch: changing
wording invalidates every cached generation.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

with open(ROOT / "config.json") as f:
    CONFIG = json.load(f)

RECIPES: dict[str, str] = CONFIG["recipes"]
ADJECTIVES: dict[str, str] = CONFIG["adjectives"]
SCALES: dict[str, str | None] = CONFIG["scales"]
GROUND_TRUTH: dict[str, dict] = CONFIG["ground_truth"]
MATCHERS: dict[str, dict] = CONFIG["ingredient_matchers"]
REPEATS: int = CONFIG["repeats"]

# Bumped whenever EXTRACTION_SYSTEM_PROMPT changes in a way score.py depends
# on. score.py refuses to score an extraction stamped with an older version
# rather than quietly scoring it under rules it was never asked to follow.
#   1 -> original
#   2 -> added option_group / is_default_option (mutually exclusive
#        alternatives) and is_optional; required a non-null unit
# Lives here, not in api.py, so the local-only stages never import openai.
EXTRACTION_SCHEMA_VERSION = 2

RECIPE_NAMES = list(RECIPES)
ADJECTIVE_NAMES = list(ADJECTIVES)
SCALE_NAMES = list(SCALES)

# Ordinal position of each scale condition, for treating scale as continuous in
# analysis. The unmodified prompt has no number, so it is not on the axis.
SCALE_N: dict[str, int | None] = {"default": None, "6": 6, "35": 35, "173": 173}


def build_prompt(recipe: str, adjective: str, scale: str) -> str:
    """Build the user-facing generation prompt.

    recipe: key of RECIPES (e.g. "pie_dough" -> "pie crust")
    adjective: key of ADJECTIVES (e.g. "control" -> "really tasty")
    scale: key of SCALES (e.g. "35" -> "for 35 people"; "default" -> "")
    """
    scale_phrase = SCALES[scale]
    return CONFIG["prompt_template"].format(
        adjective=ADJECTIVES[adjective],
        recipe=RECIPES[recipe],
        scale="" if scale_phrase is None else " " + scale_phrase,
    )


def conditions() -> list[tuple[str, str, str]]:
    """All 48 (recipe, adjective, scale) cells of the grid."""
    return [
        (recipe, adjective, scale)
        for recipe in RECIPE_NAMES
        for adjective in ADJECTIVE_NAMES
        for scale in SCALE_NAMES
    ]


def condition_id(recipe: str, adjective: str, scale: str) -> str:
    return f"{recipe}__{adjective}__{scale}"


def band(spec: dict) -> tuple[float, float, float]:
    """(low, high, center) for a ratio spec, whether it is given as an explicit
    documented range or as a center with a +/- tolerance."""
    if "center" in spec:
        center = spec["center"]
        tol = spec["tolerance_frac"]
        return center * (1 - tol), center * (1 + tol), center
    low, high = spec["low"], spec["high"]
    return low, high, (low + high) / 2
