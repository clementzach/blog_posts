"""Stage 4: compute ratios, pass/fail, and deviation magnitude.

Pure local computation over the cached extractions -- always safe to re-run,
overwrites {hash}.score.json each time.

Ingredient name matching is partial-string only for v1 (see README limitations):
each ratio-relevant bucket ("flour", "fat", "water", "egg", "cornstarch",
"gelatin", "dairy_liquid", "liquid") has include/exclude substrings in
config.json.

Three rules decide which extracted lines a bucket is allowed to add up, and
getting any of them wrong silently produces a confident, wrong ratio:

  1. MUTUALLY EXCLUSIVE ALTERNATIVES ARE CHOSEN, NOT SUMMED. A recipe that
     says "45 g powdered gelatin", then "use 50-55 g to unmold", then "or
     22-25 leaves" is offering one amount three ways. Summing them inflated
     the gelatin numerator ~3.5x. Lines sharing an `option_group` collapse to
     one, preferring the line flagged `is_default_option`.
  2. NON-CORE LINES ARE EXCLUDED, AND THE EXTRACTOR DOES NOT GET THE LAST
     WORD. `is_core` and the component denylist are ANDed: a craquelin topping
     labelled `is_core: true` still has to survive the denylist before its
     flour joins the choux flour.
  3. A PARTIALLY DROPPED BUCKET IS NOT SCORED. A bucket that lost a matching
     line to a null quantity or an unconvertible unit reports a real number
     that is missing part of itself. Those rows get status "partial", not a
     ratio.
"""

import argparse
import csv

from config import (EXTRACTION_SCHEMA_VERSION, GROUND_TRUTH, MATCHERS,
                    RESULTS_DIR, SCALE_N, band)
from storage import iter_generations, read_json, write_json
from units import to_grams


def matches(name: str, canonical: str) -> bool:
    spec = MATCHERS[canonical]
    lowered = (name or "").lower()
    if any(bad in lowered for bad in spec["exclude"]):
        return False
    return any(good in lowered for good in spec["include"])


def resolved_quantity(item: dict) -> tuple[float | None, bool]:
    """The quantity to score, and whether it came from a stated range.

    The extractor is instructed never to guess a number for a vague or ranged
    amount, so `quantity` is null there. Recipes state the pie-dough water as a
    range ("6-8 tablespoons") almost every time, though, so when the extractor
    captured explicit numeric endpoints we score their midpoint and flag it,
    rather than throwing the whole generation away. Vague amounts ("a few",
    "to taste") have no endpoints and stay unscorable.
    """
    if item.get("quantity") is not None:
        return item["quantity"], False
    low, high = item.get("quantity_low"), item.get("quantity_high")
    if low is not None and high is not None:
        return (low + high) / 2, True
    return None, False


NON_CORE_COMPONENTS = ("wash", "glaze", "garnish", "topping", "serving",
                       "dusting", "sauce", "filling", "assembly", "decorat",
                       "craquelin", "streusel", "crumble", "crumb topping",
                       "coating", "dip", "drizzle", "accompani", "variation",
                       "substitut")


def is_core(item: dict) -> bool:
    """Whether the line belongs to the preparation under test.

    Both tests have to pass. The extractor's `is_core` alone is not enough: it
    marked a black-sesame craquelin's 55 g of flour as core, which merged into
    the choux flour and moved that generation's egg:flour from 1.47 to 1.07 --
    the most extreme value in the arm, and an artifact. The denylist alone is
    not enough either, since a component can be named anything. So a line is
    core only if the extractor says so AND its component is not on the list.
    """
    if item.get("is_core") is False:
        return False
    component = (item.get("component") or "").lower()
    return not any(word in component for word in NON_CORE_COMPONENTS)


def is_optional(item: dict) -> bool:
    """Optional lines are not part of the recipe as written. "1 tbsp vodka,
    optional, replacing 1 tbsp of the water" must not be added to hydration."""
    if item.get("is_optional"):
        return True
    return "optional" in (item.get("name") or "").lower()


def _choose_from_group(entries: list[dict]) -> tuple[dict | None, list[str]]:
    """One line out of a set of mutually exclusive alternatives.

    Preference order: the extractor's flagged default, then document order.
    A default that cannot be converted to grams falls through to the next
    candidate rather than sinking the whole bucket, and the substitution is
    reported so it shows up in the hand-check.
    """
    ordered = ([e for e in entries if e["item"].get("is_default_option")]
               + [e for e in entries if not e["item"].get("is_default_option")])
    for entry in ordered:
        if entry["grams"] is not None:
            skipped = [e["name"] for e in entries if e is not entry]
            return entry, skipped
    return None, [e["name"] for e in entries]


def bucket_grams(ingredients: list[dict], canonical: str) -> dict:
    """Total grams for one ratio-relevant bucket, plus why lines were dropped.

    `partial` is the load-bearing output alongside `grams`: it is true when a
    line that belonged in this bucket could not be converted, which means the
    total is real but incomplete and must not be scored as a ratio.
    """
    candidates, non_core, optional = [], [], []
    for item in ingredients:
        name = item.get("name") or ""
        if not matches(name, canonical):
            continue
        if not is_core(item):
            non_core.append(name)
            continue
        if is_optional(item):
            optional.append(name)
            continue
        quantity, used_range = resolved_quantity(item)
        grams = (None if quantity is None
                 else to_grams(quantity, item.get("unit"), name, canonical))
        candidates.append({
            "item": item,
            "name": name,
            "quantity": quantity,
            "grams": grams,
            "used_range": used_range,
            "detail": f"{quantity} {item.get('unit')} {name}",
        })

    # Collapse each set of mutually exclusive alternatives to a single line
    # before anything is added up. Lines with no option_group are independent
    # contributions and all of them count.
    groups: dict[str, list[dict]] = {}
    selected, alternatives_skipped = [], []
    for entry in candidates:
        group = entry["item"].get("option_group")
        if group is None or group == "":
            selected.append(entry)
        else:
            groups.setdefault(str(group), []).append(entry)
    for entries in groups.values():
        chosen, skipped = _choose_from_group(entries)
        if chosen is not None:
            selected.append(chosen)
            alternatives_skipped.extend(skipped)
        else:
            # No alternative in the group converted. Carry one line through so
            # the bucket registers the drop and the ratio comes out unscorable,
            # rather than quietly losing the ingredient altogether.
            selected.append(entries[0])

    grams = 0.0
    matched, null_quantity, unconvertible, from_range = [], [], [], []
    for entry in selected:
        if entry["quantity"] is None:
            null_quantity.append(entry["name"])
            continue
        if entry["grams"] is None:
            unconvertible.append(entry["detail"])
            continue
        grams += entry["grams"]
        matched.append(entry["name"])
        if entry["used_range"]:
            from_range.append(entry["name"])

    return {
        "grams": grams,
        "matched": matched,
        "null_quantity": null_quantity,
        "unconvertible": unconvertible,
        "from_range": from_range,
        "non_core_skipped": non_core,
        "optional_skipped": optional,
        "alternatives_skipped": alternatives_skipped,
        # A dropped line means this total is missing part of itself.
        "partial": bool(null_quantity or unconvertible),
    }


def ingredient_lines(extraction: dict) -> tuple[list[dict], bool]:
    """Totals for the whole recipe, and whether batching was flagged.

    Batching never affects pass/fail: a ratio holds per batch and in total, so
    scaling by batch count leaves it unchanged. It is recorded for analysis
    because making n=173 in batches is often the correct answer, not a failure.
    """
    batching = bool(extraction.get("batching_detected"))
    lines = extraction.get("ingredients") or []
    if not lines and batching:
        per_batch = extraction.get("per_batch_ingredients") or []
        count = extraction.get("batch_count") or 1
        lines = [
            item | {"quantity": item["quantity"] * count}
            for item in per_batch
            if item.get("quantity") is not None
        ]
    return lines, batching


def score_one(extraction_record: dict) -> dict:
    recipe = extraction_record["recipe"]
    truth = GROUND_TRUTH[recipe]
    base = {
        "hash": extraction_record["hash"],
        "recipe": recipe,
        "adjective": extraction_record["adjective"],
        "scale": extraction_record["scale"],
        "repeat_index": extraction_record["repeat_index"],
        "ground_truth_basis": truth["basis"],
        "batching_detected": False,
        "ratios": {},
        "partial_bucket": False,
        "passed": None,
        "status": "scored",
    }

    extraction = extraction_record.get("extraction")
    if not extraction:
        return base | {"status": "extraction_parse_failure"}

    # An extraction made under an older prompt has no option_group, so its
    # mutually exclusive alternatives would be summed exactly as they were
    # before the fix -- and the result would look completely normal. Refuse it.
    if extraction_record.get("extraction_schema_version", 1) < EXTRACTION_SCHEMA_VERSION:
        return base | {"status": "stale_extraction"}

    lines, batching = ingredient_lines(extraction)
    base["batching_detected"] = batching
    base["batch_count"] = extraction.get("batch_count")
    if not lines:
        return base | {"status": "no_ingredients"}

    unscorable = partial = False
    for ratio_name, spec in truth["ratios"].items():
        num = bucket_grams(lines, spec["numerator"])
        den = bucket_grams(lines, spec["denominator"])
        low, high, center = band(spec)
        record = {
            "numerator": spec["numerator"],
            "denominator": spec["denominator"],
            "numerator_grams": round(num["grams"], 2),
            "denominator_grams": round(den["grams"], 2),
            "numerator_matched": num["matched"],
            "denominator_matched": den["matched"],
            "null_quantity": num["null_quantity"] + den["null_quantity"],
            "unconvertible": num["unconvertible"] + den["unconvertible"],
            "non_core_skipped": num["non_core_skipped"] + den["non_core_skipped"],
            "optional_skipped": num["optional_skipped"] + den["optional_skipped"],
            "alternatives_skipped": (num["alternatives_skipped"]
                                     + den["alternatives_skipped"]),
            "from_range": num["from_range"] + den["from_range"],
            "used_range_midpoint": bool(num["from_range"] or den["from_range"]),
            "band_low": low,
            "band_high": high,
            "band_center": center,
            "as_percent": spec["as_percent"],
        }
        if num["grams"] <= 0 or den["grams"] <= 0:
            record |= {"value": None, "passed": None, "reason": "missing_ingredient"}
            unscorable = True
        elif num["partial"] or den["partial"]:
            # The bucket totalled a real number while a line that belonged in
            # it was dropped, so the ratio is wrong by an unknown amount. It
            # would look perfectly plausible if scored, which is exactly why
            # it must not be.
            record |= {"value": None, "passed": None, "reason": "partial_bucket"}
            unscorable = partial = True
        else:
            value = num["grams"] / den["grams"]
            if spec["as_percent"]:
                value *= 100
            passed = low <= value <= high
            record |= {
                "value": value,
                "passed": passed,
                # Signed distance from the band center, so drift can be plotted
                # continuously rather than only counted as failures.
                "deviation_from_center": (value - center) / center,
                "log2_deviation": _log2_ratio(value, center),
                # 0 inside the band; relative distance past the nearest bound.
                "outside_band_by": 0.0 if passed
                else (value - high) / high if value > high else (low - value) / low,
            }
        base["ratios"][ratio_name] = record

    base["partial_bucket"] = partial
    if unscorable:
        status = "partial" if partial else "unscorable"
        return base | {"status": status, "passed": None}

    # Recipe rule: FAIL if ANY checked ratio falls outside its band (pie dough
    # in particular is not scored on the average of its two ratios).
    base["passed"] = all(r["passed"] for r in base["ratios"].values())
    return base


def _log2_ratio(value: float, center: float) -> float:
    from math import log2

    return log2(value / center)


def flatten(score: dict) -> list[dict]:
    """One CSV row per (generation, ratio) for analysis."""
    rows = []
    for ratio_name, r in score["ratios"].items():
        rows.append({
            "hash": score["hash"],
            "recipe": score["recipe"],
            "adjective": score["adjective"],
            "scale": score["scale"],
            "scale_n": SCALE_N[score["scale"]],
            "repeat_index": score["repeat_index"],
            "ground_truth_basis": score["ground_truth_basis"],
            "batching_detected": score["batching_detected"],
            "status": score["status"],
            "ratio": ratio_name,
            "value": r.get("value"),
            "band_low": r["band_low"],
            "band_high": r["band_high"],
            "band_center": r["band_center"],
            "ratio_passed": r.get("passed"),
            "deviation_from_center": r.get("deviation_from_center"),
            "log2_deviation": r.get("log2_deviation"),
            "outside_band_by": r.get("outside_band_by"),
            "used_range_midpoint": r.get("used_range_midpoint"),
            "recipe_passed": score["passed"],
            "reason": r.get("reason"),
            "n_alternatives_skipped": len(r.get("alternatives_skipped") or []),
            "n_optional_skipped": len(r.get("optional_skipped") or []),
            "n_non_core_skipped": len(r.get("non_core_skipped") or []),
        })
    if not rows:
        rows.append({
            "hash": score["hash"], "recipe": score["recipe"],
            "adjective": score["adjective"], "scale": score["scale"],
            "scale_n": SCALE_N[score["scale"]], "repeat_index": score["repeat_index"],
            "ground_truth_basis": score["ground_truth_basis"],
            "batching_detected": score["batching_detected"],
            "status": score["status"], "ratio": None, "value": None,
            "band_low": None, "band_high": None, "band_center": None,
            "ratio_passed": None, "deviation_from_center": None,
            "log2_deviation": None, "outside_band_by": None,
            "used_range_midpoint": None, "recipe_passed": None, "reason": None,
            "n_alternatives_skipped": None, "n_optional_skipped": None,
            "n_non_core_skipped": None,
        })
    return rows


def main() -> None:
    argparse.ArgumentParser().parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    rows, statuses = [], {}
    for cdir, _repeat_index, gen_hash in iter_generations():
        extraction_path = cdir / f"{gen_hash}.extraction.json"
        if not extraction_path.exists():
            continue
        score = score_one(read_json(extraction_path))
        write_json(cdir / f"{gen_hash}.score.json", score)
        rows.extend(flatten(score))
        statuses[score["status"]] = statuses.get(score["status"], 0) + 1

    csv_path = RESULTS_DIR / "scores.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Scored {sum(statuses.values())} generations -> {csv_path}")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
