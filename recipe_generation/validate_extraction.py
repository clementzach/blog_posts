"""Stage 3: hand-check the extractor on a stratified sample.

    python validate_extraction.py            # build the worksheet
    python validate_extraction.py --report   # summarize the filled-in judgments

Writes results/extraction_validation.md (recipe text, the extractor's JSON, and
what score.py actually did with it, one section per sampled generation) and
results/extraction_validation.csv with judgment columns to fill in by hand:

  reviewed            put 1 here once you have read this generation. Rates are
                      computed over reviewed rows only, so a row you never got
                      to is not silently counted as clean.
  missed_or_misread   extractor missed or misread a stated quantity
  false_precision     extractor resolved an ambiguous quantity ("a few eggs",
                      "2-3 cups") into a specific number instead of null
  wrong_line_semantics
                      the numbers are right but a line is classified wrong:
                      mutually exclusive alternatives not grouped (or grouped
                      when they are really separate additions), a wrong
                      is_core, or a missed "optional". These do not look like
                      errors in the JSON -- they only show up in the bucket
                      totals -- which is why the worksheet prints the buckets.

Accept/revise threshold, decided before looking at the data: accept the
extractor for the full 240 if ALL THREE error rates are <= 10% of the reviewed
generations (<= 3 of 28). Above that, revise EXTRACTION_SYSTEM_PROMPT in
api.py, delete the affected extraction files, and re-run extract.py.
"""

import argparse
import csv
import json
import random

from config import (ADJECTIVE_NAMES, GROUND_TRUTH, RECIPE_NAMES, RESULTS_DIR,
                    SCALE_NAMES)
from score import bucket_grams, ingredient_lines
from storage import iter_generations, read_json

SAMPLE_SIZE = 28
ACCEPT_THRESHOLD = 0.10
JUDGMENT_FIELDS = ["missed_or_misread", "false_precision", "wrong_line_semantics"]
TRUTHY = ("1", "y", "yes", "true")


def stratified_sample(size: int, seed: int = 0) -> list[tuple]:
    """Spread the sample across recipes, adjectives, and scales rather than
    sampling uniformly at random (which can miss a whole scale condition)."""
    records = []
    for cdir, repeat_index, gen_hash in iter_generations():
        if not (cdir / f"{gen_hash}.extraction.json").exists():
            continue
        recipe, adjective, scale = cdir.name.split("__")
        records.append((recipe, adjective, scale, repeat_index, gen_hash, cdir))

    rng = random.Random(seed)
    rng.shuffle(records)
    # Round-robin over each stratifying dimension in turn so every recipe,
    # adjective, and scale is represented before any cell is sampled twice.
    chosen, seen = [], set()
    for dimension, values in (
        (0, RECIPE_NAMES), (1, ADJECTIVE_NAMES), (2, SCALE_NAMES),
    ):
        for value in values:
            for record in records:
                if record[dimension] == value and record[4] not in seen:
                    chosen.append(record)
                    seen.add(record[4])
                    break
    for record in records:
        if len(chosen) >= size:
            break
        if record[4] not in seen:
            chosen.append(record)
            seen.add(record[4])
    return chosen[:size]


def bucket_summary(recipe: str, extraction: dict | None) -> list[str]:
    """What score.py makes of this extraction, bucket by bucket.

    The errors that survive a read of the raw JSON are the ones about which
    lines belong together, and those are only visible here. A reviewer
    comparing "45 g powdered gelatin" in the recipe against a gelatin bucket
    reading 156.25 g catches the summed-alternatives bug in one glance.
    """
    if not extraction:
        return ["(extraction failed to parse)"]
    lines, _ = ingredient_lines(extraction)
    out = []
    for ratio_name, spec in GROUND_TRUTH[recipe]["ratios"].items():
        out.append(f"{ratio_name}:")
        for role in ("numerator", "denominator"):
            bucket = bucket_grams(lines, spec[role])
            out.append(f"  {role} ({spec[role]}) = {bucket['grams']:.2f} g"
                       f"  from {bucket['matched'] or 'nothing'}")
            for field, label in (
                ("alternatives_skipped", "alternatives not counted"),
                ("optional_skipped", "optional, not counted"),
                ("non_core_skipped", "non-core, not counted"),
                ("null_quantity", "DROPPED, no usable quantity"),
                ("unconvertible", "DROPPED, unit not convertible"),
                ("from_range", "scored at the midpoint of a stated range"),
            ):
                if bucket[field]:
                    out.append(f"      {label}: {bucket[field]}")
        if bucket_grams(lines, spec["numerator"])["partial"] or \
           bucket_grams(lines, spec["denominator"])["partial"]:
            out.append("  -> PARTIAL: a matching line was dropped, so this "
                       "ratio is not scored")
    return out


def build_worksheet(size: int, seed: int) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    sample = stratified_sample(size, seed)
    if not sample:
        print("No extractions found. Run generate.py and extract.py first.")
        return

    lines = [
        "# Extraction validation worksheet",
        "",
        f"{len(sample)} stratified generations. For each, read the recipe text, "
        "check the extractor's JSON against it, check the bucket totals against "
        "the recipe, then mark `reviewed` and any error columns in "
        "`results/extraction_validation.csv`.",
        "",
        "Mark `wrong_line_semantics` when the numbers are transcribed correctly "
        "but a line is grouped or classified wrong -- alternatives summed "
        "instead of chosen, a topping counted as core, a missed `optional`. "
        "Those are invisible in the JSON and only show up in the buckets.",
        "",
    ]
    rows = []
    for recipe, adjective, scale, repeat_index, gen_hash, cdir in sample:
        generation = read_json(cdir / f"{gen_hash}.generation.json")
        extraction = read_json(cdir / f"{gen_hash}.extraction.json")
        lines += [
            f"## {recipe} / {adjective} / {scale} / repeat {repeat_index}",
            f"`{gen_hash}`",
            "",
            f"**Prompt:** {generation['prompt']}",
            "",
            "### Recipe text",
            "",
            "```",
            (generation["response_text"] or "").strip(),
            "```",
            "",
            "### Extractor output",
            "",
            "```json",
            json.dumps(extraction["extraction"], indent=2),
            "```",
            "",
            "### What score.py does with it",
            "",
            "```",
            *bucket_summary(recipe, extraction["extraction"]),
            "```",
            "",
        ]
        rows.append({
            "hash": gen_hash, "recipe": recipe, "adjective": adjective,
            "scale": scale, "repeat_index": repeat_index, "reviewed": "",
            "missed_or_misread": "", "false_precision": "",
            "wrong_line_semantics": "", "notes": "",
        })

    (RESULTS_DIR / "extraction_validation.md").write_text("\n".join(lines))
    csv_path = RESULTS_DIR / "extraction_validation.csv"
    if csv_path.exists():
        print(f"{csv_path} already exists; leaving your judgments in place.")
    else:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote worksheet for {len(sample)} generations to "
          f"{RESULTS_DIR / 'extraction_validation.md'}")
    print(f"Fill in {csv_path}, then rerun with --report.")


def report() -> None:
    csv_path = RESULTS_DIR / "extraction_validation.csv"
    if not csv_path.exists():
        print(f"{csv_path} not found; run without --report first.")
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if "reviewed" not in (rows[0] if rows else {}):
        print("This worksheet predates the `reviewed` column. Delete "
              f"{csv_path}, rebuild it, and redo the judgments -- rates over "
              "the old file counted only rows that had an error, which reads "
              "as a 100% error rate.")
        return

    # Denominator is rows explicitly marked reviewed. Deriving it from "any
    # column filled in" instead -- as this script used to -- means a reviewer
    # who ticks only the bad rows, which is the natural way to work, gets an
    # error rate of exactly 100% every time.
    reviewed = [r for r in rows if r["reviewed"].strip().lower() in TRUTHY]
    if not reviewed:
        print(f"0/{len(rows)} generations marked reviewed. Put a 1 in the "
              "`reviewed` column for each one you have checked.")
        return

    def rate(field: str) -> float:
        hits = sum(1 for r in reviewed if r[field].strip().lower() in TRUTHY)
        return hits / len(reviewed)

    print(f"Reviewed {len(reviewed)}/{len(rows)} sampled generations")
    rates = {}
    for field in JUDGMENT_FIELDS:
        rates[field] = rate(field)
        count = round(rates[field] * len(reviewed))
        print(f"  {field}: {rates[field]:.1%} ({count}/{len(reviewed)})")
    verdict = "ACCEPT" if max(rates.values()) <= ACCEPT_THRESHOLD else "REVISE"
    print(f"  threshold {ACCEPT_THRESHOLD:.0%} on every category -> {verdict}")
    if len(reviewed) < len(rows):
        print(f"  NOTE: {len(rows) - len(reviewed)} sampled generations are "
              "not yet reviewed; this verdict is provisional.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report() if args.report else build_worksheet(args.size, args.seed)


if __name__ == "__main__":
    main()
